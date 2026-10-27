"""Reference joint-to-anchor association and matched target-content control."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .reference_flow import FlowBlock as JointPoseBlock, graph_hops
from .reference_skeleton_xyz import JointMeshBlock, joint_relations, reference_slots, validate_reference


@dataclass(frozen=True)
class AnchorJointConfig:
    input_dim: int = 1024
    dim: int = 256
    heads: int = 8
    reference_depth: int = 2
    target_depth: int = 4
    max_joints: int = 512
    max_hop: int = 8
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for key in ('input_dim', 'dim', 'heads', 'reference_depth', 'target_depth', 'max_joints', 'max_hop'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('positive integer dimensions required')
        if self.dim % self.heads:
            raise ValueError('head dimensions must divide width')


class AnchorJointBinding(nn.Module):
    MODES = ('reference_transport', 'target_content')

    def __init__(self, config=AnchorJointConfig(), mode='reference_transport'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError('unknown association mode')
        self.config, self.mode = config, mode
        d = config.dim
        self.surface = nn.Linear(config.input_dim, d)
        self.geometry = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.joint_geometry = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        self.reference_blocks = nn.ModuleList([JointMeshBlock(config) for _ in range(config.reference_depth)])
        self.query_norm = nn.LayerNorm(d)
        self.anchor_norm = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.distance_scale = nn.Parameter(torch.zeros(config.heads))
        self.transfer_output = nn.Linear(d, d)
        self.target_blocks = nn.ModuleList([JointPoseBlock(config) for _ in range(config.target_depth)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.zeros_(self.output[-1].bias)

    def _run(self, module, *args):
        if self.training and self.config.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, features, points, normals, xyz, parents, mask, reference, *, return_association=False):
        validate_reference(xyz, parents, mask, max_joints=self.config.max_joints)
        if (features.ndim != 4 or features.shape[-1] != self.config.input_dim or features.shape[1] < 2
                or points.shape != (*features.shape[:3], 3) or normals.shape != points.shape
                or features.shape[0] != xyz.shape[0]
                or any(x.device != features.device for x in (points, normals, xyz, parents, mask, reference))):
            raise ValueError('aligned multi-pose features and geometry required')
        b, t, n, d = *features.shape[:3], self.config.dim
        j, heads = xyz.shape[1], self.config.heads
        xyz = xyz.masked_fill(~mask[..., None], 0.)
        memory = self.surface(features) + self.geometry(torch.cat((points, normals), -1))
        ref_memory, ref_points = reference_slots(memory, reference), reference_slots(points, reference)
        root = parents == -1
        offset = xyz - xyz.gather(1, parents.clamp_min(0)[..., None].expand(-1, -1, 3))
        offset = offset.masked_fill(root[..., None], 0.)
        h = self.joint_geometry(torch.cat((xyz, offset, root[..., None].to(xyz.dtype)), -1))
        relations, hops = joint_relations(parents), graph_hops(parents, mask, self.config.max_hop)
        distance = (xyz[:, :, None].float() - ref_points[:, None].float()).square().sum(-1)
        for block in self.reference_blocks:
            h = self._run(block, h, ref_memory, relations, mask, distance)
        q = self.query(self.query_norm(h)).reshape(b, j, heads, d // heads).transpose(1, 2)
        normalized = self.anchor_norm(memory)
        keys = self.key(normalized).reshape(b, t, n, heads, d // heads).permute(0, 1, 3, 2, 4)
        values = self.value(normalized).reshape(b, t, n, heads, d // heads).permute(0, 1, 3, 2, 4)
        if self.mode == 'reference_transport':
            keys = reference_slots(keys, reference)[:, None].expand(-1, t, -1, -1, -1)
            distance = distance[:, None].expand(-1, t, -1, -1)
        else:
            distance = (xyz[:, None, :, None].float() - points[:, :, None].float()).square().sum(-1)
        # Only this key selection differs between arms. Values always come from the target pose.
        logits = torch.einsum('bhjd,bthnd->bthjn', q.float(), keys.float()) / math.sqrt(d // heads)
        logits = logits - distance[:, :, None] * F.softplus(self.distance_scale)[None, None, :, None, None]
        association = logits.softmax(-1).to(values.dtype)
        transferred = torch.matmul(association, values).permute(0, 1, 3, 2, 4).reshape(b, t, j, d)
        target = h[:, None] + self.transfer_output(transferred)
        for block in self.target_blocks:
            target = self._run(block, target, memory, relations, hops, mask)
        prediction = xyz[:, None].float() + self.output(target).float()
        is_reference = torch.arange(t, device=features.device)[None] == reference[:, None]
        prediction = torch.where(is_reference[:, :, None, None], xyz[:, None].float(), prediction)
        prediction = prediction.masked_fill(~mask[:, None, :, None], 0.)
        if return_association:
            return prediction, association.masked_fill(~mask[:, None, None, :, None], 0.)
        return prediction


def coordinate_objective(prediction, target, parents, mask, reference, bone_weight=.25):
    if prediction.shape != target.shape or prediction.shape[:1] + prediction.shape[2:3] != parents.shape:
        raise ValueError('complete prediction and target shapes differ')
    valid = mask[:, None] & (torch.arange(target.shape[1], device=target.device)[None] != reference[:, None])[:, :, None]
    if not bool(torch.isfinite(prediction[valid]).all()) or not bool(torch.isfinite(target[valid]).all()):
        raise ValueError('nonfinite valid joint coordinates')
    clean = target.masked_fill(~mask[:, None, :, None], 0.)
    pred = prediction.masked_fill(~mask[:, None, :, None], 0.)
    error = (pred - clean).abs().mean(-1).masked_fill(~valid, 0.)
    xyz = error.sum((1, 2)) / valid.sum((1, 2))
    indices = parents.clamp_min(0)[:, None, :, None].expand(-1, target.shape[1], -1, 3)
    bone_error = ((pred - pred.gather(2, indices)) - (clean - clean.gather(2, indices))).abs().mean(-1)
    bones = valid & (parents[:, None] >= 0)
    bone = bone_error.masked_fill(~bones, 0.).sum((1, 2)) / bones.sum((1, 2)).clamp_min(1)
    return {'xyz_loss': xyz, 'bone_loss': bone, 'loss_per_asset': xyz + bone_weight * bone}


class SharedAnchorJointRig(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, surface, targets, geometry, indices, index_mask, reference, weight, bone_weight):
        features, virtual = self.backbone.encode(surface)
        ar = self.backbone.forward_encoded(features, virtual, targets, auxiliary_weight=0.,
            target_frame_indices=indices, target_frame_mask=index_mask)
        xyz = reference_slots(geometry['frame_target_joints'], reference)
        prediction = self.head(features, surface['query_points'], surface['query_normals'], xyz,
            geometry['target_parents'], geometry['joint_mask'], reference)
        loss = coordinate_objective(prediction, geometry['frame_target_joints'],
            geometry['target_parents'], geometry['joint_mask'], reference, bone_weight)
        return {**loss, 'loss_per_asset': ar['loss_per_asset'] + weight * loss['loss_per_asset'],
            'ar_loss': ar['loss_per_asset'], 'main_accuracy': ar['main_accuracy']}
