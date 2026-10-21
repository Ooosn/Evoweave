"""Autoregressive structure and mesh-conditioned, bidirectional XYZ token filling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .aux_skeleton_tokens import AuxSkeletonTokens
from .motion_evidence import grouped_motion_evidence
from .virtual_query_xyz import (
    AuxiliaryIDDecoder,
    VirtualQueryMotionEncoder,
    VirtualQueryXYZConfig,
    sequence_ce,
)


@dataclass(frozen=True)
class StructureXYZFillConfig:
    upstream: VirtualQueryXYZConfig = field(default_factory=VirtualQueryXYZConfig)
    fill_depth: int = 4
    fill_max_tokens: int = 3076
    fill_ffn_dim: int = 4096
    coordinate_bins: int = 256

    def __post_init__(self):
        if not isinstance(self.upstream, VirtualQueryXYZConfig):
            raise ValueError('upstream must be a VirtualQueryXYZConfig')
        for key in ('fill_depth', 'fill_max_tokens', 'fill_ffn_dim', 'coordinate_bins'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.fill_max_tokens < 6 or self.coordinate_bins < 2:
            raise ValueError('fill capacity must cover a root and three coordinate slots')


class CoordinateTokenFillDecoder(nn.Module):
    """Read discrete structure and mesh memory; never read structure-decoder states."""

    def __init__(self, config: StructureXYZFillConfig, codec: AuxSkeletonTokens):
        super().__init__()
        self.config = config
        self.codec = codec
        dim = config.upstream.dim
        self.embedding = nn.Embedding(codec.vocab_size + 3, dim, padding_idx=codec.pad)
        self.position = nn.Embedding(config.fill_max_tokens, dim)
        self.point_embedding = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.memory_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                dim, config.upstream.heads, config.fill_ffn_dim,
                dropout=config.upstream.dropout, activation='gelu',
                batch_first=True, norm_first=True,
            ) for _ in range(config.fill_depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, config.coordinate_bins, bias=False)

    def validate_inputs(self, inputs: dict[str, torch.Tensor], batch: int):
        ids, attention = inputs['fill_input_ids'], inputs['fill_attention_mask']
        owners, positions = inputs['fill_joint_ids'], inputs['fill_positions']
        joints = inputs['fill_joint_mask']
        if (ids.ndim != 2 or ids.shape[0] != batch or ids.dtype != torch.long
                or attention.shape != ids.shape or attention.dtype != torch.bool
                or owners.shape != ids.shape or owners.dtype != torch.long):
            raise ValueError('fill IDs, attention and joint owners must align in [B,L]')
        if not 1 <= ids.shape[1] <= self.config.fill_max_tokens:
            raise ValueError('expanded fill sequence exceeds its position capacity')
        if (positions.ndim != 3 or positions.shape[0] != batch or positions.shape[-1] != 3
                or positions.dtype != torch.long or joints.shape != positions.shape[:2]
                or joints.dtype != torch.bool or not bool(joints.any(dim=1).all())):
            raise ValueError('XYZ gather positions and joint masks must align in [B,J,3]')
        if any(value.device != ids.device for value in (attention, owners, positions, joints)):
            raise ValueError('all fill inputs must be on the same device')
        if (bool((ids < 0).any()) or bool((ids >= self.embedding.num_embeddings).any())
                or bool((owners < -1).any()) or bool((owners >= self.codec.max_joints).any())):
            raise ValueError('fill token or joint owner is outside the declared vocabulary')
        lengths = attention.sum(dim=1)
        expected = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        if not torch.equal(attention, expected) or not bool((lengths >= 6).all()):
            raise ValueError('fill attention must contain a valid nonempty prefix')
        if bool((ids[~attention] != self.codec.pad).any()):
            raise ValueError('fill padding must use the structure pad token')
        slot_mask = ids >= self.codec.vocab_size
        if not torch.equal(slot_mask, owners >= 0) or bool((slot_mask & ~attention).any()):
            raise ValueError('only XYZ mask positions may have a joint owner')
        if (bool((positions < 0).any()) or bool((positions >= ids.shape[1]).any())
                or not torch.equal(joints, torch.arange(joints.shape[1], device=ids.device)[None]
                                   < joints.sum(dim=1)[:, None])):
            raise ValueError('invalid XYZ positions or non-prefix joint mask')
        gather = positions.reshape(batch, -1)
        token_at_slots = ids.gather(1, gather).reshape_as(positions)
        owners_at_slots = owners.gather(1, gather).reshape_as(positions)
        expected_tokens = torch.arange(3, device=ids.device) + self.codec.vocab_size
        expected_owners = torch.arange(joints.shape[1], device=ids.device)[None, :, None]
        valid = joints[..., None].expand_as(positions)
        if (bool((token_at_slots[valid] != expected_tokens.expand_as(positions)[valid]).any())
                or bool((owners_at_slots[valid] != expected_owners.expand_as(positions)[valid]).any())
                or not torch.equal(slot_mask.sum(dim=1), 3 * joints.sum(dim=1))):
            raise ValueError('each valid joint needs exactly one ordered X/Y/Z slot triplet')

    def initial_sequence(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, owners = inputs['fill_input_ids'], inputs['fill_joint_ids']
        positions = torch.arange(ids.shape[1], device=ids.device)
        value = self.embedding(ids) + self.position(positions)[None]
        joint_ids = owners.clamp_min(0) + self.codec.joint_offset
        return value + self.embedding(joint_ids) * (owners >= 0)[..., None]

    def forward_frame(self, sequence, memory, inputs):
        value = sequence
        padding = ~inputs['fill_attention_mask']
        for layer in self.layers:
            kwargs = {'tgt_key_padding_mask': padding, 'tgt_is_causal': False}
            if self.config.upstream.gradient_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(layer, value, memory, use_reentrant=False, **kwargs)
            else:
                value = layer(value, memory, **kwargs)
        batch, joints = inputs['fill_positions'].shape[:2]
        positions = inputs['fill_positions'].reshape(batch, joints * 3)
        selected = self.norm(value).gather(1, positions[..., None].expand(-1, -1, value.shape[-1]))
        return self.head(selected).reshape(batch, joints, 3, self.config.coordinate_bins)

    def forward(self, final_frames, anchor_points, inputs):
        if (final_frames.ndim != 4 or final_frames.shape[-1] != self.config.upstream.dim
                or final_frames.shape[2] != self.config.upstream.query_tokens
                or anchor_points.shape != (*final_frames.shape[:3], 3)
                or not anchor_points.is_floating_point() or anchor_points.device != final_frames.device):
            raise ValueError('final anchor features and input positions must align in [B,T,Q]')
        batch, frames = final_frames.shape[:2]
        if not 1 <= frames <= self.config.upstream.max_frames:
            raise ValueError('frame count is outside the declared upstream capacity')
        self.validate_inputs(inputs, batch)
        if inputs['fill_input_ids'].device != final_frames.device:
            raise ValueError('fill inputs and mesh memory must share a device')
        sequence = self.initial_sequence(inputs)
        # The input-mesh locations identify geometry; GT joint coordinates never enter memory.
        memory = self.memory_norm(final_frames + self.point_embedding(anchor_points))
        return torch.stack([
            self.forward_frame(sequence, memory[:, frame], inputs) for frame in range(frames)
        ], dim=1)


def coordinate_token_loss(logits, labels, joint_mask, frame_mask):
    if (logits.ndim != 5 or logits.shape[-2] != 3 or labels.shape != logits.shape[:-1]
            or labels.dtype != torch.long or joint_mask.shape != (logits.shape[0], logits.shape[2])
            or frame_mask.shape != logits.shape[:2]
            or joint_mask.dtype != torch.bool or frame_mask.dtype != torch.bool):
        raise ValueError('coordinate logits, labels, joint and frame masks do not align')
    valid = frame_mask[:, :, None, None] & joint_mask[:, None, :, None]
    valid = valid.expand_as(labels)
    if (bool((labels[~valid] != -100).any()) or bool((labels[valid] < 0).any())
            or bool((labels[valid] >= logits.shape[-1]).any())
            or not bool(frame_mask.any(dim=1).all()) or not bool(joint_mask.any(dim=1).all())):
        raise ValueError('coordinate targets are invalid or do not use -100 for padding')
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction='none').reshape_as(labels)
    denominator = valid.sum(dim=(2, 3)).clamp_min(1)
    per_frame = loss.sum(dim=(2, 3)) / denominator
    accuracy = ((logits.argmax(-1) == labels) & valid).sum(dim=(2, 3)) / denominator
    per_asset = (per_frame * frame_mask).sum(1) / frame_mask.sum(1)
    asset_accuracy = (accuracy * frame_mask).sum(1) / frame_mask.sum(1)
    return per_asset, asset_accuracy, per_frame, accuracy


def allowed_structure_tokens(prefix, codec: AuxSkeletonTokens):
    """Validate a partial explicit-ID stream and return its grammar-legal next IDs."""
    tokens = list(prefix)
    if not tokens or type(tokens[0]) is not int or tokens[0] != codec.bos:
        raise ValueError('a structure prefix must start with BOS')
    active, count, phase = [], 0, 'joint_or_end'
    for token in tokens[1:]:
        if type(token) is not int:
            raise ValueError('structure prefix tokens must be integers')
        if phase == 'ended':
            raise ValueError('tokens may not follow structure EOS')
        if phase == 'branch_parent':
            parent = token - codec.joint_offset
            if parent not in active[:-1]:
                raise ValueError('branch parent must be a strict active ancestor')
            active = active[:active.index(parent) + 1]
            phase = 'branch_child'
        elif token == codec.eos and phase == 'joint_or_end' and count:
            phase = 'ended'
        elif token == codec.branch and phase == 'joint_or_end' and len(active) > 1 and count < codec.max_joints:
            phase = 'branch_parent'
        elif token == codec.joint_offset + count and count < codec.max_joints:
            active.append(count)
            count += 1
            phase = 'joint_or_end'
        else:
            raise ValueError('invalid or nonsequential joint token in structure prefix')
    if phase == 'ended':
        return []
    if phase == 'branch_parent':
        return [codec.joint_offset + parent for parent in active[:-1]]
    if phase == 'branch_child':
        if count >= codec.max_joints:
            raise ValueError('a branch prefix exceeds joint capacity')
        return [codec.joint_offset + count]
    if count == 0:
        return [codec.joint_offset]
    allowed = [codec.eos]
    if count < codec.max_joints:
        allowed.append(codec.joint_offset + count)
        if len(active) > 1:
            allowed.append(codec.branch)
    return allowed


class StructureXYZFillRig(nn.Module):
    def __init__(self, surface_tokenizer, config, codec, *, motion_encoder=None, structure_decoder=None):
        super().__init__()
        self.config, self.codec = config, codec
        if codec.max_sequence_tokens != config.upstream.auxiliary_max_tokens:
            raise ValueError('structure codec and decoder capacities differ')
        self.surface_tokenizer = surface_tokenizer
        self.motion_encoder = motion_encoder if motion_encoder is not None else VirtualQueryMotionEncoder(config.upstream)
        self.structure_decoder = (structure_decoder if structure_decoder is not None else
                                  AuxiliaryIDDecoder(config.upstream, codec.vocab_size, codec.pad))
        self.fill_decoder = CoordinateTokenFillDecoder(config, codec)

    def encode(self, surface):
        cfg = self.config.upstream
        points = surface['query_points']
        if points.ndim != 4 or points.shape[2:] != (cfg.query_tokens, 3):
            raise ValueError('real anchor geometry must be [B,T,Q,3]')
        tokens = []
        for frame in range(points.shape[1]):
            args = tuple(surface[key][:, frame] for key in
                         ('dense_points', 'dense_normals', 'query_points', 'query_normals'))
            if cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens.append(checkpoint(self.surface_tokenizer, *args, use_reentrant=False))
            else:
                tokens.append(self.surface_tokenizer(*args))
        evidence = (grouped_motion_evidence(surface['dense_points'].detach(), surface['query_indices'])
                    if cfg.motion_evidence_fusion != 'off' else None)
        return self.motion_encoder(torch.stack(tokens, dim=1), motion_evidence=evidence)

    def forward_encoded(self, final_frames, virtual, anchor_points, targets, *, structure_weight=1., return_logits=False):
        if isinstance(structure_weight, bool) or not math.isfinite(structure_weight) or structure_weight <= 0:
            raise ValueError('structure_weight must be finite and positive')
        structure_logits = self.structure_decoder(
            virtual, targets['structure_input_ids'], targets['structure_attention_mask'])
        structure_loss, structure_accuracy = sequence_ce(structure_logits, targets['structure_labels'])
        # This interface intentionally contains only discrete IDs/masks, not the structure hidden states.
        fill_inputs = {key: targets[key] for key in (
            'fill_input_ids', 'fill_attention_mask', 'fill_joint_ids', 'fill_positions', 'fill_joint_mask')}
        xyz_logits = self.fill_decoder(final_frames, anchor_points, fill_inputs)
        xyz_loss, xyz_accuracy, frame_loss, frame_accuracy = coordinate_token_loss(
            xyz_logits, targets['xyz_labels'], targets['fill_joint_mask'], targets['frame_mask'])
        per_asset = xyz_loss + structure_weight * structure_loss
        result = {'loss': per_asset.mean(), 'loss_per_asset': per_asset,
                  'xyz_loss': xyz_loss.mean(), 'structure_loss': structure_loss.mean(),
                  'xyz_accuracy': xyz_accuracy.mean(), 'structure_accuracy': structure_accuracy.mean(),
                  'xyz_loss_per_frame': frame_loss, 'xyz_accuracy_per_frame': frame_accuracy}
        if return_logits:
            result.update(xyz_logits=xyz_logits, structure_logits=structure_logits)
        return result

    def forward(self, surface, targets, *, structure_weight=1., return_logits=False):
        final_frames, virtual = self.encode(surface)
        return self.forward_encoded(final_frames, virtual, surface['query_points'], targets,
                                    structure_weight=structure_weight, return_logits=return_logits)

    @torch.no_grad()
    def generate_structure(self, virtual, *, max_new_tokens=None):
        if self.training or self.structure_decoder.training:
            raise ValueError('structure generation requires eval mode')
        cfg, codec = self.config.upstream, self.codec
        if (virtual.shape != (1, cfg.query_tokens, cfg.dim) or not virtual.is_floating_point()
                or not bool(torch.isfinite(virtual).all())):
            raise ValueError('structure generation needs one finite complete virtual sequence')
        limit = codec.max_sequence_tokens - 1 if max_new_tokens is None else max_new_tokens
        if type(limit) is not int or not 1 <= limit < codec.max_sequence_tokens:
            raise ValueError('invalid structure generation length limit')
        tokens, constrained = [codec.bos], 0
        for _ in range(limit):
            ids = torch.tensor([tokens], dtype=torch.long, device=virtual.device)
            logits = self.structure_decoder(virtual, ids, torch.ones_like(ids, dtype=torch.bool))[0, -1]
            if not bool(torch.isfinite(logits).all()):
                return {'success': False, 'status': 'nonfinite_logits', 'tokens': tokens}
            allowed = allowed_structure_tokens(tokens, codec)
            raw = int(logits.argmax())
            if allowed == [codec.eos] and raw != codec.eos:
                return {'success': False, 'status': 'max_joints', 'tokens': tokens,
                        'grammar_constrained_steps': constrained}
            constrained += int(raw not in allowed)
            selected = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            token = allowed[int(logits[selected].argmax())]
            tokens.append(token)
            if token == codec.eos:
                parents = codec.decode(tokens).tolist()
                return {'success': True, 'status': 'eos', 'tokens': tokens, 'parents': parents,
                        'grammar_constrained_steps': constrained}
        return {'success': False, 'status': 'max_tokens', 'tokens': tokens,
                'grammar_constrained_steps': constrained}

    def configuration(self) -> dict[str, Any]:
        return {'family': 'structure_xyz_token_fill_v1', **asdict(self.config), 'codec': asdict(self.codec)}


@torch.no_grad()
def initialize_fill_embeddings(model, *, structure_embedding=None, position_embedding=None,
                               position_row_offset=0, source_name='random'):
    """Copy explicitly selected embedding rows; never copy attention, FFN, or output weights."""
    report = {'source': source_name, 'structure_rows': 0, 'position_rows': 0,
              'position_row_offset': position_row_offset, 'attention_ffn_head': 'fresh'}
    fill = model.fill_decoder
    if structure_embedding is not None:
        expected = (model.codec.vocab_size, model.config.upstream.dim)
        if tuple(structure_embedding.shape) != expected or not bool(torch.isfinite(structure_embedding).all()):
            raise ValueError('source structure embedding is incompatible or nonfinite')
        fill.embedding.weight[:model.codec.vocab_size].copy_(structure_embedding)
        report['structure_rows'] = model.codec.vocab_size
    if position_embedding is not None:
        if (position_embedding.ndim != 2 or position_embedding.shape[1] != model.config.upstream.dim
                or type(position_row_offset) is not int or not 0 <= position_row_offset < position_embedding.shape[0]
                or not bool(torch.isfinite(position_embedding).all())):
            raise ValueError('source position embedding is incompatible or nonfinite')
        count = min(fill.position.num_embeddings, position_embedding.shape[0] - position_row_offset)
        fill.position.weight[:count].copy_(position_embedding[position_row_offset:position_row_offset + count])
        report['position_rows'] = count
    return report


__all__ = ['StructureXYZFillConfig', 'CoordinateTokenFillDecoder', 'StructureXYZFillRig',
           'coordinate_token_loss', 'allowed_structure_tokens', 'initialize_fill_embeddings']
