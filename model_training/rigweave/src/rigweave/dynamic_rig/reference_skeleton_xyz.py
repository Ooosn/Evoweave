"""Reference-coordinate joint queries over shared, corresponding mesh anchors."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReferenceXYZConfig:
    input_dim: int = 1024
    dim: int = 256
    heads: int = 8
    reference_depth: int = 2
    target_depth: int = 2
    max_joints: int = 512

    def __post_init__(self):
        values = (self.input_dim, self.dim, self.heads, self.reference_depth,
                  self.target_depth, self.max_joints)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError('configuration values must be positive integers')
        if self.dim % self.heads:
            raise ValueError('dim must be divisible by heads')


def validate_reference(xyz, parents, mask, *, max_joints=512):
    if (xyz.ndim != 3 or xyz.shape[-1] != 3 or not xyz.is_floating_point()
            or parents.shape != xyz.shape[:2] or parents.dtype != torch.long
            or mask.shape != parents.shape or mask.dtype != torch.bool):
        raise ValueError('reference requires XYZ[B,J,3], long parents[B,J], bool mask[B,J]')
    if xyz.device != parents.device or xyz.device != mask.device:
        raise ValueError('reference tensors must share a device')
    batch, joints = parents.shape
    if batch < 1 or not 1 <= joints <= max_joints:
        raise ValueError('empty batch or joint count exceeds declared capacity')
    count = mask.sum(1)
    index = torch.arange(joints, device=parents.device)[None]
    if bool((count < 1).any()) or not torch.equal(mask, index < count[:, None]):
        raise ValueError('joint mask must be a nonempty contiguous prefix')
    if bool((parents[:, 0] != -1).any()) or bool((parents[~mask] != -2).any()):
        raise ValueError('root parent must be -1; padded parents must be -2')
    if bool((mask & (index > 0) & ((parents < 0) | (parents >= index))).any()):
        raise ValueError('one complete tree with parent-before-child order is required')
    if not bool(torch.isfinite(xyz[mask]).all()):
        raise ValueError('valid reference coordinates must be finite')
    return count


def joint_relations(parents):
    """Other, self, parent, child, sibling. Indices have no learned semantics."""
    joints = parents.shape[1]
    indices = torch.arange(joints, device=parents.device)
    relation = torch.zeros((*parents.shape, joints), dtype=torch.long, device=parents.device)
    sibling = (parents[:, :, None] == parents[:, None, :]) & (parents[:, :, None] >= 0)
    relation = relation.masked_fill(sibling, 4)
    relation = relation.masked_fill(parents[:, None, :] == indices[None, :, None], 3)
    relation = relation.masked_fill(parents[:, :, None] == indices[None, None, :], 2)
    return relation.masked_fill(torch.eye(joints, device=parents.device, dtype=torch.bool)[None], 1)


def reference_slots(values, slots):
    if (slots.dtype != torch.long or slots.shape != (values.shape[0],)
            or slots.device != values.device or bool(((slots < 0) | (slots >= values.shape[1])).any())):
        raise ValueError('reference slots must be valid long indices[B] on the input device')
    return values[torch.arange(values.shape[0], device=values.device), slots]


class JointMeshBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.dim
        self.heads = config.heads
        self.graph_bias = nn.Embedding(5, config.heads)
        nn.init.zeros_(self.graph_bias.weight)
        self.graph_norm = nn.LayerNorm(dim)
        self.graph = nn.MultiheadAttention(dim, config.heads, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, config.heads, batch_first=True)
        self.distance_scale = nn.Parameter(torch.zeros(config.heads))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, joints, memory, relations, mask, reference_distance2):
        batch, count, _ = joints.shape
        bias = self.graph_bias(relations).permute(0, 3, 1, 2).to(joints.dtype)
        bias = bias.masked_fill(~mask[:, None, None, :], float('-inf'))
        query = self.graph_norm(joints)
        joints = joints + self.graph(query, query, query,
            attn_mask=bias.reshape(batch * self.heads, count, count), need_weights=False)[0]
        # Reference distance is a soft prior over corresponding surface anchors,
        # not a nearest-point binding or a claim that attention equals skinning.
        bias = -reference_distance2[:, None] * F.softplus(self.distance_scale)[None, :, None, None]
        memory = self.memory_norm(memory)
        joints = joints + self.cross(self.query_norm(joints), memory, memory,
            attn_mask=bias.to(joints.dtype).reshape(batch * self.heads, count, memory.shape[1]),
            need_weights=False)[0]
        joints = joints + self.ffn(self.ffn_norm(joints))
        return joints.masked_fill(~mask[:, :, None], 0.)


class ReferenceSkeletonXYZ(nn.Module):
    """An appended readout, not another surface encoder or a new AR vocabulary."""
    def __init__(self, config=ReferenceXYZConfig()):
        super().__init__()
        self.config = config
        dim = config.dim
        self.surface_projection = nn.Linear(config.input_dim, dim)
        self.reference_projection = nn.Linear(config.input_dim, dim)
        self.reference_geometry = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.target_geometry = nn.Sequential(nn.Linear(15, dim), nn.GELU(), nn.Linear(dim, dim))
        self.joint_geometry = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.reference_blocks = nn.ModuleList([JointMeshBlock(config) for _ in range(config.reference_depth)])
        self.target_blocks = nn.ModuleList([JointMeshBlock(config) for _ in range(config.target_depth)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, frame_features, anchor_points, anchor_normals,
                reference_xyz, parents, joint_mask, reference_frame):
        validate_reference(reference_xyz, parents, joint_mask, max_joints=self.config.max_joints)
        if (frame_features.ndim != 4 or frame_features.shape[-1] != self.config.input_dim
                or anchor_points.shape != (*frame_features.shape[:3], 3)
                or anchor_normals.shape != anchor_points.shape
                or frame_features.shape[0] != reference_xyz.shape[0]
                or min(frame_features.shape[:3]) < 1):
            raise ValueError('aligned features[B,T,Q,D] and geometry[B,T,Q,3] are required')
        if any(v.device != frame_features.device for v in
               (anchor_points, anchor_normals, reference_xyz, parents, joint_mask, reference_frame)):
            raise ValueError('all head inputs must share a device')
        batch, frames, anchors, _ = frame_features.shape
        count = reference_xyz.shape[1]
        ref_features = reference_slots(frame_features, reference_frame)
        ref_points = reference_slots(anchor_points, reference_frame)
        ref_normals = reference_slots(anchor_normals, reference_frame)
        xyz = reference_xyz.masked_fill(~joint_mask[:, :, None], 0.)
        parent_xyz = xyz.gather(1, parents.clamp_min(0)[:, :, None].expand(-1, -1, 3))
        root = parents == -1
        offset = (xyz - parent_xyz).masked_fill(root[:, :, None], 0.)
        joints = self.joint_geometry(torch.cat((xyz, offset, root[:, :, None].to(xyz.dtype)), -1))
        relations = joint_relations(parents)
        distance2 = (xyz[:, :, None].float() - ref_points[:, None].float()).square().sum(-1)
        memory = self.surface_projection(ref_features) + self.reference_geometry(torch.cat((ref_points, ref_normals), -1))
        for block in self.reference_blocks:
            joints = block(joints, memory, relations, joint_mask, distance2)
        ref_points_all = ref_points[:, None].expand(-1, frames, -1, -1)
        ref_normals_all = ref_normals[:, None].expand(-1, frames, -1, -1)
        geometry = torch.cat((ref_points_all, anchor_points, anchor_points - ref_points_all,
                              ref_normals_all, anchor_normals), -1)
        memory = (self.surface_projection(frame_features)
                  + self.reference_projection(ref_features)[:, None]
                  + self.target_geometry(geometry)).reshape(batch * frames, anchors, -1)
        joints = joints[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, count, -1)
        relation_all = relations[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, count, count)
        mask_all = joint_mask[:, None].expand(-1, frames, -1).reshape(batch * frames, count)
        distance_all = distance2[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, count, anchors)
        for block in self.target_blocks:
            joints = block(joints, memory, relation_all, mask_all, distance_all)
        delta = self.output(joints).reshape(batch, frames, count, 3).float()
        is_reference = torch.arange(frames, device=reference_frame.device)[None] == reference_frame[:, None]
        prediction = xyz[:, None].float() + delta.masked_fill(is_reference[:, :, None, None], 0.)
        return prediction.masked_fill(~joint_mask[:, None, :, None], 0.)

    def configuration(self):
        return asdict(self.config)


def localization_loss(prediction, targets, joint_mask, reference_frame):
    if prediction.shape != targets.shape or prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError('predicted and target XYZ must have matching [B,T,J,3] shapes')
    if joint_mask.shape != (prediction.shape[0], prediction.shape[2]) or joint_mask.dtype != torch.bool:
        raise ValueError('joint mask must be bool[B,J]')
    reference_slots(targets, reference_frame)
    valid_frames = torch.arange(prediction.shape[1], device=prediction.device)[None] != reference_frame[:, None]
    valid = valid_frames[:, :, None] & joint_mask[:, None]
    counts = valid.sum((1, 2))
    if bool((counts == 0).any()) or not bool(torch.isfinite(targets[valid]).all()):
        raise ValueError('every asset needs finite non-reference target joints')
    difference = (prediction.float() - targets.float()).masked_fill(~valid[:, :, :, None], 0.)
    error = difference.abs().mean(-1)
    return error.sum((1, 2)) / counts


@torch.no_grad()
def local_rigid_transfer(reference_xyz, anchor_points, reference_frame, *, neighbors=32):
    """Weighted local Kabsch baseline using only observed anchor correspondences."""
    if (anchor_points.ndim != 4 or anchor_points.shape[-1] != 3
            or reference_xyz.ndim != 3 or reference_xyz.shape[-1] != 3
            or anchor_points.shape[0] != reference_xyz.shape[0] or neighbors < 3
            or anchor_points.shape[2] < 3):
        raise ValueError('local rigid transfer needs aligned batches and at least three anchors')
    with torch.autocast(device_type=anchor_points.device.type, enabled=False):
        points = anchor_points.float()
        reference = reference_slots(points, reference_frame)
        distance = torch.cdist(reference_xyz.float(), reference)
        distance, index = distance.topk(min(neighbors, points.shape[2]), largest=False, dim=-1)
        rows = torch.arange(points.shape[0], device=points.device)[:, None, None]
        source = reference[rows, index]
        target = points[rows[:, None], torch.arange(points.shape[1], device=points.device)[None, :, None, None], index[:, None]]
        weight = (distance.square() + 1.e-4).reciprocal()
        weight = weight / weight.sum(-1, keepdim=True)
        source_mean = (source * weight[..., None]).sum(-2)
        target_mean = (target * weight[:, None, :, :, None]).sum(-2)
        source_centered = source - source_mean[:, :, None]
        target_centered = target - target_mean[:, :, :, None]
        covariance = (source_centered[:, None] * weight[:, None, :, :, None]).transpose(-1, -2) @ target_centered
        u, singular, vh = torch.linalg.svd(covariance)
        correction = torch.ones_like(singular)
        correction[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1., 1.)
        rotation = (u * correction[..., None, :]) @ vh
        result = ((reference_xyz[:, None].float() - source_mean[:, None])[..., None, :] @ rotation).squeeze(-2) + target_mean
        degenerate = singular[..., 1] < 1.e-7
        translation = reference_xyz[:, None].float() + target_mean - source_mean[:, None]
        result = torch.where(degenerate[..., None], translation, result)
        return result, degenerate
