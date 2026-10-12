"""1024 virtual anchors, per-frame original XYZ AR, auxiliary IDs only as loss."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .motion_encoder import AnchorWiseAlternatingMotionEncoder
from .motion_evidence import grouped_motion_evidence


@dataclass(frozen=True)
class VirtualQueryXYZConfig:
    dim: int = 1024
    heads: int = 8
    query_tokens: int = 1024
    register_tokens: int = 96
    motion_depth: int = 12
    fusion_depth: int = 1
    frame_residual_scale: float = 0.
    auxiliary_depth: int = 2
    auxiliary_max_tokens: int = 1024
    max_frames: int = 24
    dropout: float = 0.
    motion_evidence_fusion: str = 'bias'
    motion_evidence_heads: int = 8
    gradient_checkpointing: bool = True
    temporal_chunk_size: int = 4096

    def __post_init__(self):
        for key in ('dim', 'heads', 'query_tokens', 'motion_depth', 'fusion_depth',
                    'auxiliary_depth', 'auxiliary_max_tokens', 'max_frames', 'temporal_chunk_size'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if isinstance(self.register_tokens, bool) or not isinstance(self.register_tokens, int) or self.register_tokens < 0:
            raise ValueError('register_tokens must be a nonnegative integer')
        if self.dim % self.heads or not 0 <= self.dropout < 1:
            raise ValueError('invalid attention dimension/dropout')
        if isinstance(self.frame_residual_scale, bool) or not math.isfinite(self.frame_residual_scale) or self.frame_residual_scale < 0:
            raise ValueError('frame_residual_scale must be finite and nonnegative')
        if self.motion_evidence_fusion not in {'off', 'bias'}:
            raise ValueError('only established off/bias motion evidence is supported')


class VirtualQueryMotionEncoder(AnchorWiseAlternatingMotionEncoder):
    """Every real anchor participates with its corresponding virtual slot."""

    def __init__(self, config: VirtualQueryXYZConfig):
        super().__init__(dim=config.dim, depth=config.motion_depth, heads=config.heads,
                         register_tokens=config.register_tokens, max_frames=config.max_frames,
                         dropout=config.dropout, gradient_checkpointing=config.gradient_checkpointing,
                         motion_evidence_fusion=config.motion_evidence_fusion,
                         motion_evidence_heads=config.motion_evidence_heads,
                         register_temporal_mode='aligned')
        self.query_tokens = config.query_tokens
        self.temporal_chunk_size = config.temporal_chunk_size
        self.virtual_slots = nn.Parameter(torch.randn(1, config.query_tokens, config.dim) * .02)

    def _block(self, block, real, virtual, evidence):
        batch, frames, slots, dim = real.shape
        if block.motion_injection.fusion == 'off':
            real = block.pose_inner(real.reshape(batch*frames, slots, dim)).reshape(batch, frames, slots, dim)
        else:
            real = block.motion_injection(block.pose_inner, real, evidence)
        virtual = block.pose_inner(virtual)
        # Role, registers and ALL 1024 anchors include the virtual frame in time.
        joined = torch.cat((real, virtual[:, None]), dim=1)
        sequence = joined.permute(0, 2, 1, 3).reshape(batch*slots, frames+1, dim)
        sequence = torch.cat([block.anchor_temporal(part) for part in
                              sequence.split(self.temporal_chunk_size, dim=0)], dim=0)
        joined = sequence.reshape(batch, slots, frames+1, dim).permute(0, 2, 1, 3)
        return joined[:, :frames], joined[:, -1]

    def forward(self, frame_tokens, *, motion_evidence=None):
        if frame_tokens.ndim != 4 or frame_tokens.shape[2:] != (self.query_tokens, self.dim):
            raise ValueError('surface tokens must be [B,T,Q,D] with configured Q,D')
        batch, frames, _, dim = frame_tokens.shape
        if not 1 <= frames <= self.max_frames:
            raise ValueError('frame count outside declared capacity')
        if (self.motion_evidence_fusion != 'off') != (motion_evidence is not None):
            raise ValueError('motion evidence presence differs from configured fusion')
        registers = (self.register[None].expand(batch, -1, -1) if self.register is not None
                     else frame_tokens.new_empty(batch, 0, dim))
        real = torch.cat((self.role_token[:, 1:2].expand(batch, frames, -1, -1),
                          registers[:, None].expand(-1, frames, -1, -1), frame_tokens), dim=2)
        virtual = torch.cat((self.role_token[:, 0].expand(batch, -1, -1), registers,
                             self.virtual_slots.expand(batch, -1, -1)), dim=1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                real, virtual = checkpoint(self._block, block, real, virtual, motion_evidence, use_reentrant=False)
            else:
                real, virtual = self._block(block, real, virtual, motion_evidence)
        offset = 1 + self.register_tokens
        return self.norm(real)[:, :, offset:], self.norm(virtual)[:, offset:]


class FrameConditionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_norm = nn.LayerNorm(config.dim)
        self.memory_norm = nn.LayerNorm(config.dim)
        self.cross = nn.MultiheadAttention(config.dim, config.heads, dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(nn.Linear(config.dim, 4*config.dim), nn.GELU(),
                                 nn.Linear(4*config.dim, config.dim))

    def forward(self, query, final_frame):
        memory = self.memory_norm(final_frame)
        query = query + self.cross(self.query_norm(query), memory, memory, need_weights=False)[0]
        return query + self.ffn(self.ffn_norm(query))


class AuxiliaryIDDecoder(nn.Module):
    """Independent lightweight decoder; its outputs never enter the XYZ route."""

    def __init__(self, config, vocab_size, pad_token):
        super().__init__()
        self.max_tokens = config.auxiliary_max_tokens
        self.embedding = nn.Embedding(vocab_size, config.dim, padding_idx=pad_token)
        self.position = nn.Embedding(self.max_tokens, config.dim)
        layer = nn.TransformerDecoderLayer(config.dim, config.heads, 2*config.dim,
                                           dropout=config.dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.auxiliary_depth, norm=nn.LayerNorm(config.dim))
        self.head = nn.Linear(config.dim, vocab_size)

    def forward(self, virtual, inputs, attention):
        if inputs.ndim != 2 or inputs.shape != attention.shape or inputs.shape[1] > self.max_tokens:
            raise ValueError('invalid auxiliary prefix shape/capacity')
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        values = self.embedding(inputs) + self.position(positions)[None]
        causal = torch.ones(inputs.shape[1], inputs.shape[1], dtype=torch.bool, device=inputs.device).triu(1)
        hidden = self.decoder(values, virtual, tgt_mask=causal, tgt_key_padding_mask=~attention.bool())
        return self.head(hidden)


def sequence_ce(logits, labels):
    if logits.shape[:-1] != labels.shape:
        raise ValueError('next-token logits and labels differ')
    valid = labels != -100
    counts = valid.sum(dim=-1)
    if not bool((counts > 0).all()):
        raise ValueError('each sequence needs valid next-token targets')
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none', ignore_index=-100)
    per_sequence = (losses * valid).sum(-1) / counts
    accuracy = ((logits.argmax(-1) == labels) & valid).sum(-1) / counts
    return per_sequence, accuracy


def original_xyz_vocabulary_processor(tokenizer, start_tokens):
    """Apply the original tokenizer grammar without importing its asset loader.

    Equivalent to upstream VocabSwitchingLogitsProcessor: prepend the fixed
    BOS/class prefix, ask next_posible_token, and mask all other vocabulary IDs.
    """
    def process(generated_ids, scores):
        output = scores.clone()
        for batch, sequence in enumerate(generated_ids):
            full = torch.cat((start_tokens, sequence))
            allowed = tokenizer.next_posible_token(ids=full.detach().cpu().numpy())
            if not len(allowed):
                raise ValueError('original XYZ tokenizer returned no legal next token')
            mask = torch.full_like(scores[batch], float('-inf'))
            mask[allowed] = 0
            output[batch] = scores[batch] + mask
        return output
    return process


class VirtualQueryXYZRig(nn.Module):
    def __init__(self, surface_tokenizer, xyz_decoder, config, *, auxiliary_vocab_size, auxiliary_pad=0):
        super().__init__()
        self.config = config
        self.surface_tokenizer = surface_tokenizer
        self.motion_encoder = VirtualQueryMotionEncoder(config)
        self.frame_fusion = nn.ModuleList([FrameConditionLayer(config) for _ in range(config.fusion_depth)])
        self.condition_norm = nn.LayerNorm(config.dim)
        self.xyz_decoder = xyz_decoder
        self.auxiliary = AuxiliaryIDDecoder(config, auxiliary_vocab_size, auxiliary_pad)
        if xyz_decoder.get_input_embeddings().embedding_dim != config.dim:
            raise ValueError('original XYZ decoder width differs from condition width')

    def encode(self, surface):
        points = surface['query_points']
        if points.ndim != 4 or points.shape[2:] != (self.config.query_tokens, 3):
            raise ValueError('real anchor geometry must be [B,T,Q,3]')
        tokens = []
        for frame in range(points.shape[1]):
            args = tuple(surface[key][:, frame] for key in
                         ('dense_points', 'dense_normals', 'query_points', 'query_normals'))
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens.append(checkpoint(self.surface_tokenizer, *args, use_reentrant=False))
            else:
                tokens.append(self.surface_tokenizer(*args))
        evidence = None
        if self.config.motion_evidence_fusion != 'off':
            evidence = grouped_motion_evidence(surface['dense_points'].detach(), surface['query_indices'])
        return self.motion_encoder(torch.stack(tokens, dim=1), motion_evidence=evidence)

    def conditions(self, final_frames, virtual):
        if final_frames.ndim != 4 or virtual.shape != (final_frames.shape[0], self.config.query_tokens, self.config.dim):
            raise ValueError('final frame/virtual shape violates the query contract')
        if final_frames.shape[2:] != virtual.shape[1:]:
            raise ValueError('per-frame conditions must retain all virtual anchor slots')
        batch, frames, queries, dim = final_frames.shape
        query = virtual[:, None].expand(-1, frames, -1, -1).reshape(batch*frames, queries, dim)
        memory = final_frames.reshape(batch*frames, queries, dim)
        for layer in self.frame_fusion:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                query = checkpoint(layer, query, memory, use_reentrant=False)
            else:
                query = layer(query, memory)
        # Preserve aligned final-frame detail without replacing virtual queries.
        if self.config.frame_residual_scale:
            query = query + self.config.frame_residual_scale * memory
        return self.condition_norm(query).reshape(batch, frames, queries, dim)

    def main_logits(self, condition, input_ids, attention_mask):
        embedding = self.xyz_decoder.get_input_embeddings()(input_ids)
        prefix = torch.cat((condition.to(embedding.dtype), embedding), dim=1)
        mask = torch.cat((attention_mask.new_ones(condition.shape[:2]), attention_mask), dim=1)
        result = self.xyz_decoder(inputs_embeds=prefix, attention_mask=mask, use_cache=False, return_dict=True)
        return result.logits[:, condition.shape[1]:]

    def forward_encoded(self, final_frames, virtual, targets, *, auxiliary_weight):
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError('auxiliary_weight must be finite and nonnegative')
        condition = self.conditions(final_frames, virtual)
        ids, mask = targets['main_input_ids'], targets['main_attention_mask']
        if ids.ndim != 3 or ids.shape != mask.shape or ids.shape[:2] != condition.shape[:2]:
            raise ValueError('each real frame needs its own full XYZ target sequence')
        losses, accuracies = [], []
        for frame in range(condition.shape[1]):
            labels = ids[:, frame, 1:].clone().masked_fill(~mask[:, frame, 1:].bool(), -100)
            logits = self.main_logits(condition[:, frame], ids[:, frame, :-1], mask[:, frame, :-1])
            loss, accuracy = sequence_ce(logits, labels)
            losses.append(loss)
            accuracies.append(accuracy)
        main = torch.stack(losses, dim=1).mean(dim=1)
        auxiliary = main.new_zeros(main.shape)
        auxiliary_accuracy = auxiliary
        if auxiliary_weight > 0:
            aux_logits = self.auxiliary(virtual, targets['aux_input_ids'], targets['aux_attention_mask'])
            auxiliary, auxiliary_accuracy = sequence_ce(aux_logits, targets['aux_labels'])
        per_asset = main + auxiliary_weight * auxiliary
        return {'loss': per_asset.mean(), 'loss_per_asset': per_asset, 'main_loss': main.mean(),
                'auxiliary_loss': auxiliary.mean(), 'main_accuracy': torch.stack(accuracies, dim=1).mean(),
                'auxiliary_accuracy': auxiliary_accuracy.mean()}

    def forward(self, surface, targets, *, auxiliary_weight):
        final_frames, virtual = self.encode(surface)
        return self.forward_encoded(final_frames, virtual, targets, auxiliary_weight=auxiliary_weight)

    @torch.no_grad()
    def generate_condition(self, condition, start_tokens, *, eos_token, max_new_tokens, logits_processor=None):
        """Original XYZ next-token decoding; no auxiliary grammar/count guidance."""
        if self.training or condition.shape[0] != 1 or start_tokens.ndim != 1:
            raise ValueError('generation requires eval mode and one condition sequence')
        ids = start_tokens[None].to(device=condition.device, dtype=torch.long)
        embedding = self.xyz_decoder.get_input_embeddings()(ids)
        mask = torch.ones((1, condition.shape[1]+ids.shape[1]), dtype=torch.long, device=condition.device)
        output = self.xyz_decoder(inputs_embeds=torch.cat((condition.to(embedding.dtype), embedding), dim=1),
                                  attention_mask=mask, use_cache=True, return_dict=True)
        generated = []
        for _ in range(max_new_tokens):
            logits = output.logits[:, -1]
            if logits_processor is not None:
                logits = logits_processor(ids[:, start_tokens.numel():], logits)
            token = int(logits.argmax(-1)[0])
            generated.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
            if token == eos_token:
                return {'success': True, 'status': 'eos', 'tokens': ids[0].cpu().tolist()}
            if output.past_key_values is None:
                raise RuntimeError('original decoder did not return the requested KV cache')
            mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
            output = self.xyz_decoder(input_ids=ids[:, -1:], attention_mask=mask,
                                      past_key_values=output.past_key_values, use_cache=True, return_dict=True)
        return {'success': False, 'status': 'max_tokens', 'tokens': ids[0].cpu().tolist()}

    def configuration(self):
        return asdict(self.config)
