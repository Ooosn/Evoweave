"""Decoder-free reference-anchor transport on corresponded mesh frames."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .anchor_joint_binding import coordinate_objective
from .motion_encoder import AnchorWiseAlternatingBlock
from .motion_evidence import MotionEvidence, MotionEvidenceInjection, grouped_motion_evidence
from .reference_skeleton_xyz import reference_slots


@dataclass(frozen=True)
class AnchorStage2Config:
    spatial_layers: int
    temporal_layers: int
    dim: int = 1024
    heads: int = 8
    query_tokens: int = 1024
    register_tokens: int = 96
    max_frames: int = 24
    motion_evidence_fusion: str = 'bias'
    motion_evidence_heads: int = 8
    gradient_checkpointing: bool = True
    temporal_chunk_size: int = 4096
    checkpoint_asset_chunk_size: int = 0

    def __post_init__(self):
        if not self.temporal_layers or self.spatial_layers not in (
                self.temporal_layers, 2 * self.temporal_layers):
            raise ValueError('use one or two spatial layers per temporal layer')
        if self.dim % self.heads or not 1 <= self.temporal_chunk_size or self.checkpoint_asset_chunk_size < 0:
            raise ValueError('invalid attention dimensions or chunk size')


class SpatialAnchorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(config.dim, config.heads, 4 * config.dim,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.injection = MotionEvidenceInjection(config.dim, config.heads,
            fusion=config.motion_evidence_fusion, biased_heads=config.motion_evidence_heads)

    def forward(self, tokens, evidence):
        if evidence is not None:
            return self.injection(self.layer, tokens, evidence)
        return self.layer(tokens.flatten(0, 1)).reshape_as(tokens)


class AnchorStage2Encoder(nn.Module):
    def __init__(self, surface_tokenizer, config):
        super().__init__()
        self.config = config
        self.surface_tokenizer = surface_tokenizer
        self.role_token = nn.Parameter(torch.randn(1, 1, 1, config.dim) * .02)
        self.register = nn.Parameter(torch.randn(config.register_tokens, config.dim) * .02)
        self.blocks = nn.ModuleList([
            AnchorWiseAlternatingBlock(config.dim, config.heads, dropout=0.,
                motion_evidence_fusion=config.motion_evidence_fusion,
                motion_evidence_heads=config.motion_evidence_heads,
                register_tokens=config.register_tokens)
            for _ in range(config.temporal_layers)
        ])
        for block in self.blocks:
            block.set_temporal_batch_chunk_size(config.temporal_chunk_size)
        # Construct the common pairs first so both schedules share their initial tensors.
        self.extra_spatial = nn.ModuleList([
            SpatialAnchorLayer(config) for _ in range(config.spatial_layers - config.temporal_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)

    def _group(self, index, tokens, evidence):
        if self.extra_spatial:
            tokens = self.extra_spatial[index](tokens, evidence)
        return self.blocks[index](tokens, evidence)

    def _checkpoint_group(self, index, tokens, evidence):
        size = self.config.checkpoint_asset_chunk_size
        if not size or tokens.shape[0] <= size:
            return checkpoint(self._group, index, tokens, evidence, use_reentrant=False)
        # A group never attends across assets. Recompute one independent slice at a time.
        result = []
        for start in range(0, tokens.shape[0], size):
            stop = start + size
            part = None if evidence is None else MotionEvidence(
                evidence.states[start:stop], evidence.anchor_features[start:stop],
                evidence.valid_groups[start:stop])
            result.append(checkpoint(self._group, index, tokens[start:stop], part,
                                     use_reentrant=False))
        return torch.cat(result, dim=0)

    def encode(self, surface):
        config = self.config
        points = surface['query_points']
        if points.ndim != 4 or points.shape[2:] != (config.query_tokens, 3):
            raise ValueError('expected corresponded [B,T,Q,3] query points')
        batch, frames = points.shape[:2]
        if not 1 <= frames <= config.max_frames:
            raise ValueError('frame count outside declared capacity')
        recompute = config.gradient_checkpointing and self.training and torch.is_grad_enabled()
        tokens = []
        for frame in range(frames):
            args = tuple(surface[key][:, frame] for key in (
                'dense_points', 'dense_normals', 'query_points', 'query_normals'))
            tokens.append(checkpoint(self.surface_tokenizer, *args, use_reentrant=False)
                if recompute else self.surface_tokenizer(*args))
        evidence = None if config.motion_evidence_fusion == 'off' else grouped_motion_evidence(
            surface['dense_points'].detach(), surface['query_indices'])
        real = torch.cat((self.role_token.expand(batch, frames, -1, -1),
                          self.register.expand(batch, frames, -1, -1),
                          torch.stack(tokens, dim=1)), dim=2)
        for index in range(len(self.blocks)):
            real = (self._checkpoint_group(index, real, evidence)
                    if recompute else self._group(index, real, evidence))
        return self.norm(real)[:, :, 1 + config.register_tokens:], None


class AnchorStage2(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def predict(self, surface, xyz, parents, mask, reference):
        features, _ = self.backbone.encode(surface)
        return self.head(features, surface['query_points'], surface['query_normals'],
                         xyz, parents, mask, reference)

    def forward(self, surface, geometry, reference, weight, bone_weight):
        target = geometry['frame_target_joints']
        parents, mask = geometry['target_parents'], geometry['joint_mask']
        xyz = reference_slots(target, reference)
        prediction = self.predict(surface, xyz, parents, mask, reference)
        losses = coordinate_objective(prediction, target, parents, mask, reference, bone_weight)
        losses['loss_per_asset'] = weight * losses['loss_per_asset']
        return losses
