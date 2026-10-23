"""Original real-query-first conditioning with one full AR target per asset."""
from __future__ import annotations

from dataclasses import asdict, replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .motion_encoder import AnchorWiseAlternatingMotionEncoder
from .motion_evidence import grouped_motion_evidence
from .virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig, sequence_ce


class LegacySingleTargetRig(nn.Module):
    def __init__(self, surface_tokenizer, motion_encoder, xyz_decoder, config):
        super().__init__()
        if not isinstance(config, VirtualQueryXYZConfig):
            raise TypeError('config must be the matched VirtualQueryXYZConfig')
        if type(motion_encoder) is not AnchorWiseAlternatingMotionEncoder:
            raise TypeError('single-target conditioning requires the original motion encoder')
        if motion_encoder.dim != config.dim or xyz_decoder.get_input_embeddings().embedding_dim != config.dim:
            raise ValueError('surface condition, motion and AR widths must agree')
        self.config = config
        self.surface_tokenizer = surface_tokenizer
        self.motion_encoder = motion_encoder
        self.xyz_decoder = xyz_decoder

    @classmethod
    def from_virtual_base(cls, base):
        """Adopt the shared surface/AR modules; callers should retire the base."""
        if not isinstance(base, VirtualQueryXYZRig):
            raise TypeError('extraction requires a VirtualQueryXYZRig base')
        config = replace(base.config)
        source = base.motion_encoder
        state = dict(source.state_dict())
        if 'virtual_slots' not in state:
            raise ValueError('base motion state lacks the expected virtual_slots')
        del state['virtual_slots']
        with torch.random.fork_rng(devices=[]):
            motion = AnchorWiseAlternatingMotionEncoder(
                dim=config.dim, depth=config.motion_depth, heads=config.heads,
                register_tokens=config.register_tokens, max_frames=config.max_frames,
                dropout=config.dropout, gradient_checkpointing=config.gradient_checkpointing,
                use_motion_features=source.use_motion_features,
                use_time_embedding=source.use_time_embedding,
                motion_evidence_fusion=config.motion_evidence_fusion,
                motion_evidence_heads=config.motion_evidence_heads,
                register_temporal_mode=source.register_temporal_mode,
            )
        # Assignment retains source device/dtype and exact values without recasting.
        motion.load_state_dict(state, strict=True, assign=True)
        source_parameters = dict(source.named_parameters())
        for name, parameter in motion.named_parameters():
            parameter.requires_grad_(source_parameters[name].requires_grad)
        source_modules = dict(source.named_modules())
        for name, module in motion.named_modules():
            module.training = source_modules[name].training
        result = cls(base.surface_tokenizer, motion, base.xyz_decoder, config)
        result.training = base.training
        return result

    def _surface_shape(self, surface):
        fields = ('dense_points', 'dense_normals', 'query_points', 'query_normals')
        if any(not isinstance(surface.get(key), torch.Tensor) for key in fields):
            raise ValueError('surface needs dense/query points and normals')
        points = surface['query_points']
        if points.ndim != 4 or points.shape[2:] != (self.config.query_tokens, 3):
            raise ValueError('query_points must have shape [B,T,Q,3]')
        batch, frames = points.shape[:2]
        if batch < 1 or not 1 <= frames <= self.config.max_frames:
            raise ValueError('surface batch/frame count outside configured capacity')
        for key in fields:
            value = surface[key]
            if (value.ndim != 4 or value.shape[:2] != (batch, frames)
                    or value.shape[2] < 1 or value.shape[-1] != 3
                    or value.device != points.device or not value.is_floating_point()):
                raise ValueError(f'{key} must align with surface [B,T,N,3] and device')
        if (surface['query_normals'].shape != points.shape
                or surface['dense_normals'].shape != surface['dense_points'].shape):
            raise ValueError('surface normals must align with their points')
        return batch, frames, points.device

    @staticmethod
    def _validate_target_indices(target_indices, batch, frames, device):
        if (not isinstance(target_indices, torch.Tensor) or target_indices.dtype != torch.long
                or target_indices.shape != (batch,) or target_indices.device != device):
            raise ValueError('target_indices must be explicit [B] int64 on the surface device')
        if bool(((target_indices < 0) | (target_indices >= frames)).any()):
            raise ValueError('target_indices outside the input frame range')

    def encode_surface(self, surface):
        _, frames, _ = self._surface_shape(surface)
        tokens = []
        for frame in range(frames):
            args = tuple(surface[key][:, frame] for key in
                         ('dense_points', 'dense_normals', 'query_points', 'query_normals'))
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens.append(checkpoint(self.surface_tokenizer, *args, use_reentrant=False))
            else:
                tokens.append(self.surface_tokenizer(*args))
        result = torch.stack(tokens, dim=1)
        if result.shape != (*surface['query_points'].shape[:3], self.config.dim):
            raise ValueError('surface tokenizer must return [B,Q,D] for every input frame')
        return result

    def condition_from_tokens(self, surface, tokens, target_indices):
        batch, frames, device = self._surface_shape(surface)
        self._validate_target_indices(target_indices, batch, frames, device)
        if (not isinstance(tokens, torch.Tensor)
                or tokens.shape != (batch, frames, self.config.query_tokens, self.config.dim)
                or tokens.device != device or not tokens.is_floating_point()):
            raise ValueError('cached surface tokens must align with [B,T,Q,D] and device')
        frame_ids = torch.arange(frames, device=device).expand(batch, -1)
        remaining = frame_ids[frame_ids != target_indices[:, None]].reshape(batch, frames - 1)
        order = torch.cat((target_indices[:, None], remaining), dim=1)
        rows = torch.arange(batch, device=device)[:, None]
        evidence = None
        if self.config.motion_evidence_fusion != 'off':
            indices = surface.get('query_indices')
            if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
                    or indices.shape != (batch, self.config.query_tokens) or indices.device != device):
                raise ValueError('motion bias requires shared [B,Q] int64 query_indices')
            evidence = grouped_motion_evidence(surface['dense_points'][rows, order].detach(), indices)
        # Normals are already represented in tokens; reorder only consumed geometry.
        return self.motion_encoder(tokens[rows, order],
                                   query_points=surface['query_points'][rows, order],
                                   motion_evidence=evidence)

    def main_logits(self, condition, input_ids, attention_mask):
        return VirtualQueryXYZRig.main_logits(self, condition, input_ids, attention_mask)

    def forward(self, surface, targets, *, target_indices):
        batch, frames, device = self._surface_shape(surface)
        self._validate_target_indices(target_indices, batch, frames, device)
        ids, mask = targets['main_input_ids'], targets['main_attention_mask']
        if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
                or ids.ndim != 3 or ids.shape != mask.shape or ids.shape[:2] != (batch, frames)
                or ids.shape[-1] < 2 or ids.dtype != torch.long
                or ids.device != device or mask.device != device):
            raise ValueError('main target IDs/mask must align with [B,T,L>=2] and device')
        rows = torch.arange(batch, device=device)
        ids, mask = ids[rows, target_indices], mask[rows, target_indices]
        condition = self.condition_from_tokens(surface, self.encode_surface(surface), target_indices)
        labels = ids[:, 1:].clone().masked_fill(~mask[:, 1:].bool(), -100)
        logits = self.main_logits(condition, ids[:, :-1], mask[:, :-1])
        per_asset, accuracy = sequence_ce(logits, labels)
        loss = per_asset.mean()
        return {'loss': loss, 'loss_per_asset': per_asset, 'main_loss': loss,
                'main_accuracy': accuracy.mean(), 'main_loss_per_asset': per_asset,
                'main_accuracy_per_asset': accuracy}

    @torch.no_grad()
    def generate_condition(self, condition, start_tokens, *, eos_token, max_new_tokens, logits_processor=None):
        return VirtualQueryXYZRig.generate_condition(
            self, condition, start_tokens, eos_token=eos_token,
            max_new_tokens=max_new_tokens, logits_processor=logits_processor)

    def configuration(self):
        return {'family': 'legacy_real_query_single_target_v1', 'target_policy': 'explicit_per_asset',
                'supervision': 'single_target_full_ar',
                'register_temporal_mode': self.motion_encoder.register_temporal_mode,
                'matched_construction': asdict(self.config)}
