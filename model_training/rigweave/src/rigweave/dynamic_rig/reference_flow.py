"""Joint continuous flow conditioned on a complete reference and mesh anchors."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .reference_skeleton_xyz import (
    JointMeshBlock, joint_relations, reference_slots, validate_reference,
)


@dataclass(frozen=True)
class ReferenceFlowConfig:
    input_dim: int = 1024
    dim: int = 256
    heads: int = 8
    reference_depth: int = 2
    flow_depth: int = 6
    max_joints: int = 512
    max_hop: int = 8
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for key in ('input_dim', 'dim', 'heads', 'reference_depth', 'flow_depth', 'max_joints', 'max_hop'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.dim % self.heads:
            raise ValueError('dim must be divisible by heads')


def graph_hops(parents, mask, maximum):
    """Parent-before-child trees allow a quadratic, exact distance recurrence."""
    batch, joints = parents.shape
    result = torch.zeros(batch, joints, joints, dtype=torch.long)
    for b, (tree, valid) in enumerate(zip(parents.cpu().tolist(), mask.sum(1).cpu().tolist())):
        for j in range(1, valid):
            distance = result[b, tree[j], :j] + 1
            result[b, j, :j] = distance
            result[b, :j, j] = distance
    return result.clamp_max(maximum).to(parents.device)


class FlowBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.heads = config.heads
        self.relation_bias = nn.Embedding(5, config.heads)
        self.hop_bias = nn.Embedding(config.max_hop + 1, config.heads)
        nn.init.zeros_(self.relation_bias.weight)
        nn.init.zeros_(self.hop_bias.weight)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(5)])
        self.graph = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.poses = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, h, memory, relations, hops, mask):
        b, t, j, d = h.shape
        bias = (self.relation_bias(relations) + self.hop_bias(hops)).permute(0, 3, 1, 2)
        bias = bias.to(h.dtype).masked_fill(~mask[:, None, None, :], float('-inf'))
        bias = bias[:, None].expand(-1, t, -1, -1, -1).reshape(b * t * self.heads, j, j)
        h = h.reshape(b * t, j, d)
        q = self.norms[0](h)
        h = h + self.graph(q, q, q, attn_mask=bias, need_weights=False)[0]
        m = self.norms[1](memory.reshape(b * t, -1, d))
        h = h + self.cross(self.norms[2](h), m, m, need_weights=False)[0]
        # Unordered pose-set attention: no video index or generative-time ordering.
        h = h.reshape(b, t, j, d).transpose(1, 2).reshape(b * j, t, d)
        q = self.norms[3](h)
        h = h + self.poses(q, q, q, need_weights=False)[0]
        h = h + self.ffn(self.norms[4](h))
        return h.reshape(b, j, t, d).transpose(1, 2).masked_fill(~mask[:, None, :, None], 0.)


class ReferenceFlow(nn.Module):
    def __init__(self, config=ReferenceFlowConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.surface = nn.Linear(config.input_dim, d)
        self.reference_surface = nn.Linear(config.input_dim, d)
        self.reference_geometry = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.target_geometry = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d))
        self.joint_geometry = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        self.reference_blocks = nn.ModuleList([JointMeshBlock(config) for _ in range(config.reference_depth)])
        self.state_embedding = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.time_embedding = nn.Sequential(nn.Linear(33, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([FlowBlock(config) for _ in range(config.flow_depth)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer('frequencies', 2. ** torch.arange(16) * torch.pi, persistent=False)

    def _run(self, module, *args):
        if self.training and self.config.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def condition(self, features, points, normals, xyz, parents, mask, reference):
        validate_reference(xyz, parents, mask, max_joints=self.config.max_joints)
        if (features.ndim != 4 or features.shape[-1] != self.config.input_dim
                or features.shape[0] != xyz.shape[0] or features.shape[1] < 2
                or points.shape != (*features.shape[:3], 3) or normals.shape != points.shape):
            raise ValueError('flow needs corresponding mesh features and at least two poses')
        if any(v.device != features.device for v in (points, normals, xyz, parents, mask, reference)):
            raise ValueError('flow inputs must share a device')
        xyz = xyz.masked_fill(~mask[:, :, None], 0.)
        f = reference_slots(features, reference)
        p = reference_slots(points, reference)
        n = reference_slots(normals, reference)
        parent_xyz = xyz.gather(1, parents.clamp_min(0)[:, :, None].expand(-1, -1, 3))
        root = parents == -1
        offset = (xyz - parent_xyz).masked_fill(root[:, :, None], 0.)
        h = self.joint_geometry(torch.cat((xyz, offset, root[:, :, None].to(xyz.dtype)), -1))
        relations = joint_relations(parents)
        hops = graph_hops(parents, mask, self.config.max_hop)
        memory = self.surface(f) + self.reference_geometry(torch.cat((p, n), -1))
        distance = (xyz[:, :, None].float() - p[:, None].float()).square().sum(-1)
        for block in self.reference_blocks:
            h = self._run(block, h, memory, relations, mask, distance)
        t = features.shape[1]
        rp, rn = p[:, None].expand(-1, t, -1, -1), n[:, None].expand(-1, t, -1, -1)
        memory = (self.surface(features) + self.reference_surface(f)[:, None]
                  + self.target_geometry(torch.cat((rp, points, points - rp, rn, normals), -1)))
        return h, memory, relations, hops, mask, reference, xyz

    @staticmethod
    def clamp_reference(state, condition):
        _, _, _, _, mask, reference, xyz = condition
        is_reference = torch.arange(state.shape[1], device=state.device)[None] == reference[:, None]
        return torch.where(is_reference[:, :, None, None], xyz[:, None].float(), state.float()).masked_fill(
            ~mask[:, None, :, None], 0.)

    def velocity(self, state, tau, condition):
        ref, memory, relations, hops, mask, reference, xyz = condition
        b, t, j, _ = state.shape
        if state.shape != (ref.shape[0], memory.shape[1], ref.shape[1], 3) or tau.shape != (b,):
            raise ValueError('flow state/time shape differs from its condition')
        state = self.clamp_reference(state, condition)
        angle = tau.float()[:, None] * self.frequencies[None]
        time = self.time_embedding(torch.cat((tau.float()[:, None], angle.sin(), angle.cos()), -1))
        h = ref[:, None] + self.state_embedding(state) + time[:, None, None]
        for block in self.blocks:
            h = self._run(block, h, memory, relations, hops, mask)
        v = self.output(h).float()
        fixed = torch.arange(t, device=state.device)[None] == reference[:, None]
        return v.masked_fill(fixed[:, :, None, None] | ~mask[:, None, :, None], 0.)

    def forward(self, features, points, normals, xyz, parents, mask, reference, *, targets, noise, tau):
        condition = self.condition(features, points, normals, xyz, parents, mask, reference)
        if targets.shape != noise.shape or targets.shape != (*features.shape[:2], xyz.shape[1], 3):
            raise ValueError('targets/noise must match complete poses and joints')
        valid = mask[:, None, :] & (torch.arange(targets.shape[1], device=targets.device)[None] != reference[:, None])[:, :, None]
        if (not bool(torch.isfinite(targets[valid]).all()) or not bool(torch.isfinite(noise).all())
                or tau.shape != (targets.shape[0],) or not bool(((tau >= 0) & (tau <= 1)).all())):
            raise ValueError('nonfinite flow inputs or invalid time')
        clean = targets.float().masked_fill(~mask[:, None, :, None], 0.)
        state = (1 - tau[:, None, None, None]) * noise + tau[:, None, None, None] * clean
        prediction = self.velocity(state, tau, condition)
        difference = (prediction - (clean - noise)).masked_fill(~valid[:, :, :, None], 0.)
        loss = difference.square().mean(-1).sum((1, 2)) / valid.sum((1, 2))
        return {'loss_per_asset': loss}

    @torch.no_grad()
    def generate(self, features, points, normals, xyz, parents, mask, reference, *, noise, steps=32):
        if type(steps) is not int or steps < 1:
            raise ValueError('positive integer integration steps required')
        condition = self.condition(features, points, normals, xyz, parents, mask, reference)
        if noise.shape != (*features.shape[:2], xyz.shape[1], 3) or not bool(torch.isfinite(noise).all()):
            raise ValueError('finite noise with complete target shape required')
        state = self.clamp_reference(noise, condition)
        for step in range(steps):
            tau = torch.full((state.shape[0],), step / steps, device=state.device)
            state = self.clamp_reference(state + self.velocity(state, tau, condition) / steps, condition)
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError('nonfinite flow generation')
        return state


class SharedReferenceFlowRig(nn.Module):
    def __init__(self, backbone, flow):
        super().__init__()
        self.backbone = backbone
        self.flow = flow

    def forward(self, surface, targets, geometry, indices, index_mask, reference, noise, tau, flow_weight):
        features, virtual = self.backbone.encode(surface)
        ar = self.backbone.forward_encoded(features, virtual, targets, auxiliary_weight=0.,
            target_frame_indices=indices, target_frame_mask=index_mask)
        flow = self.flow(features, surface['query_points'], surface['query_normals'],
            reference_slots(geometry['frame_target_joints'], reference), geometry['target_parents'],
            geometry['joint_mask'], reference, targets=geometry['frame_target_joints'], noise=noise, tau=tau)
        return {'loss_per_asset': ar['loss_per_asset'] + flow_weight * flow['loss_per_asset'],
                'ar_loss': ar['loss_per_asset'], 'flow_loss': flow['loss_per_asset'],
                'main_accuracy': ar['main_accuracy']}
