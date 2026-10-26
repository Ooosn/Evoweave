"""Matched virtual-path ablation wrapper for the legacy real-query B route."""
from __future__ import annotations

from typing import Any

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .motion_evidence import grouped_motion_evidence
from .virtual_query_xyz import VirtualQueryXYZRig, sequence_ce


class VirtualPathAblationRig(nn.Module):
    """Expose matched V+/V- paths without changing ``VirtualQueryXYZRig``.

    V+ delegates every computation to the initialized B model. V- reuses the
    same surface tokenizer, motion blocks and decoder, but omits the virtual
    frame and the frame-fusion cross-attention route.
    """

    def __init__(self, base: VirtualQueryXYZRig, virtual_enabled: bool):
        super().__init__()
        if not isinstance(base, VirtualQueryXYZRig):
            raise TypeError('base must be a VirtualQueryXYZRig')
        if not isinstance(virtual_enabled, bool):
            raise TypeError('virtual_enabled must be bool')
        if base.config.fusion_mode != 'real_queries_virtual':
            raise ValueError('VirtualPathAblationRig requires the real_queries_virtual B route')
        self.base = base
        self.virtual_enabled = virtual_enabled

        # Auxiliary IDs are outside this causal comparison in both arms.
        self._freeze_module(self.base.auxiliary)
        if not virtual_enabled:
            self._freeze_no_virtual_parameters()

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_no_virtual_parameters(self) -> None:
        self.base.motion_encoder.virtual_slots.requires_grad_(False)
        for layer in self.base.frame_fusion:
            self._freeze_module(layer.query_norm)
            self._freeze_module(layer.memory_norm)
            self._freeze_module(layer.cross)

    @property
    def config(self):
        return self.base.config

    @property
    def surface_tokenizer(self):
        return self.base.surface_tokenizer

    @property
    def motion_encoder(self):
        return self.base.motion_encoder

    @property
    def frame_fusion(self):
        return self.base.frame_fusion

    @property
    def condition_norm(self):
        return self.base.condition_norm

    @property
    def xyz_decoder(self):
        return self.base.xyz_decoder

    @property
    def auxiliary(self):
        return self.base.auxiliary

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    @property
    def disabled_parameter_names(self) -> tuple[str, ...]:
        names = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
        if not self.virtual_enabled:
            # role_token is a shared parameter; only its virtual row is dormant.
            names.append('base.motion_encoder.role_token[virtual_row_0]')
        return tuple(sorted(names))

    def _surface_tokens_and_evidence(self, surface: dict[str, torch.Tensor]):
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
        if self.motion_encoder.motion_evidence_fusion != 'off':
            evidence = grouped_motion_evidence(surface['dense_points'].detach(), surface['query_indices'])
        return torch.stack(tokens, dim=1), evidence

    def _real_block(self, block: nn.Module, real: torch.Tensor, evidence: Any) -> torch.Tensor:
        batch, frames, slots, dim = real.shape
        if block.motion_injection.fusion == 'off':
            real = block.pose_inner(real.reshape(batch * frames, slots, dim))
            real = real.reshape(batch, frames, slots, dim)
        else:
            real = block.motion_injection(block.pose_inner, real, evidence)

        # The same real-frame anchor_temporal module and chunk size as B, with
        # no virtual sequence appended to the temporal dimension.
        sequence = real.permute(0, 2, 1, 3).reshape(batch * slots, frames, dim)
        chunk_size = self.motion_encoder.temporal_chunk_size
        sequence = torch.cat([
            block.anchor_temporal(part)
            for part in sequence.split(chunk_size, dim=0)
        ], dim=0)
        return sequence.reshape(batch, slots, frames, dim).permute(0, 2, 1, 3)

    def _encode_real_only(self, frame_tokens: torch.Tensor, evidence: Any):
        if frame_tokens.ndim != 4 or frame_tokens.shape[2:] != (
                self.config.query_tokens, self.config.dim):
            raise ValueError('surface tokens must be [B,T,Q,D] with configured Q,D')
        batch, frames, _, dim = frame_tokens.shape
        if not 1 <= frames <= self.motion_encoder.max_frames:
            raise ValueError('frame count outside declared capacity')
        if (self.motion_encoder.motion_evidence_fusion != 'off') != (evidence is not None):
            raise ValueError('motion evidence presence differs from configured fusion')

        registers = (self.motion_encoder.register[None].expand(batch, -1, -1)
                     if self.motion_encoder.register is not None
                     else frame_tokens.new_empty(batch, 0, dim))
        real = torch.cat((
            self.motion_encoder.role_token[:, 1:2].expand(batch, frames, -1, -1),
            registers[:, None].expand(-1, frames, -1, -1),
            frame_tokens,
        ), dim=2)
        for block in self.motion_encoder.blocks:
            if (self.motion_encoder.gradient_checkpointing and self.training
                    and torch.is_grad_enabled()):
                real = checkpoint(self._real_block, block, real, evidence, use_reentrant=False)
            else:
                real = self._real_block(block, real, evidence)
        offset = 1 + self.motion_encoder.register_tokens
        return self.motion_encoder.norm(real)[:, :, offset:], None

    def encode(self, surface):
        if self.virtual_enabled:
            return self.base.encode(surface)
        tokens, evidence = self._surface_tokens_and_evidence(surface)
        return self._encode_real_only(tokens, evidence)

    @staticmethod
    def _real_condition_layer(layer: nn.Module, query: torch.Tensor) -> torch.Tensor:
        return query + layer.ffn(layer.ffn_norm(query))

    def conditions(self, final_frames, virtual, *, query_points=None):
        if self.virtual_enabled:
            return self.base.conditions(final_frames, virtual, query_points=query_points)
        if virtual is not None:
            raise ValueError('V- conditions require virtual=None')
        if not isinstance(final_frames, torch.Tensor) or final_frames.ndim != 4:
            raise ValueError('final frame conditions must be [B,T,Q,D]')
        batch, frames, queries, dim = final_frames.shape
        if (batch < 1 or not 1 <= frames <= self.config.max_frames
                or (queries, dim) != (self.config.query_tokens, self.config.dim)):
            raise ValueError('condition batch/frame/slot count outside configured capacity')
        query = final_frames.reshape(batch * frames, queries, dim)
        for layer in self.frame_fusion:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                query = checkpoint(self._real_condition_layer, layer, query, use_reentrant=False)
            else:
                query = self._real_condition_layer(layer, query)
        return self.condition_norm(query).reshape(batch, frames, queries, dim)

    def main_logits(self, condition, input_ids, attention_mask):
        return self.base.main_logits(condition, input_ids, attention_mask)

    @staticmethod
    def _validate_auxiliary_weight(auxiliary_weight: float) -> None:
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError('auxiliary_weight must be finite and nonnegative')
        if auxiliary_weight > 0:
            raise ValueError('auxiliary_weight must be exactly 0 for virtual path ablation')

    def forward_encoded(self, final_frames, virtual, targets, *, auxiliary_weight, query_points=None,
                        target_frame_indices=None, target_frame_mask=None):
        self._validate_auxiliary_weight(auxiliary_weight)
        if self.virtual_enabled:
            if target_frame_indices is not None or target_frame_mask is not None:
                raise ValueError('selected-frame supervision requires virtual_enabled=False')
            return self.base.forward_encoded(final_frames, virtual, targets,
                                             auxiliary_weight=0., query_points=query_points)

        condition = self.conditions(final_frames, virtual, query_points=query_points)
        ids, mask = targets['main_input_ids'], targets['main_attention_mask']
        if ids.ndim != 3 or ids.shape != mask.shape or ids.shape[:2] != condition.shape[:2]:
            raise ValueError('each real frame needs its own full XYZ target sequence')
        valid = None
        if target_frame_mask is not None and target_frame_indices is None:
            raise ValueError('target_frame_mask requires target_frame_indices')
        if target_frame_indices is not None:
            batch, frames = condition.shape[:2]
            if (not isinstance(target_frame_indices, torch.Tensor)
                    or target_frame_indices.dtype != torch.int64
                    or target_frame_indices.ndim not in (1, 2)
                    or target_frame_indices.shape[0] != batch):
                raise ValueError('target_frame_indices must be an int64 tensor with shape [B] or [B,K]')
            if target_frame_indices.device.type == 'meta':
                raise ValueError('target_frame_indices must be on a device with concrete values')
            indices = target_frame_indices[:, None] if target_frame_indices.ndim == 1 else target_frame_indices
            if not 1 <= indices.shape[1] <= frames:
                raise ValueError('target_frame_indices must select between 1 and T slots per asset')
            if target_frame_mask is not None:
                if (not isinstance(target_frame_mask, torch.Tensor)
                        or target_frame_mask.dtype != torch.bool
                        or target_frame_mask.shape != indices.shape
                        or target_frame_mask.device.type == 'meta'):
                    raise ValueError('target_frame_mask must be a concrete bool tensor with shape [B,K]')
                valid = target_frame_mask.to(device=indices.device)
                if (not bool(valid[:, 0].all())
                        or bool((valid[:, 1:] & ~valid[:, :-1]).any())
                        or not bool(valid[:, -1].any())):
                    raise ValueError('target_frame_mask requires nonempty left-aligned selections without empty columns')
            if bool(((target_frame_indices < 0) | (target_frame_indices >= frames)).any()):
                raise ValueError('target_frame_indices outside the input frame range')
            ordered = (indices if valid is None else indices.masked_fill(~valid, frames)).sort(dim=1).values
            if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < frames)).any()):
                raise ValueError('target_frame_indices must contain distinct slots per asset')
            indices = indices.to(device=condition.device)
            rows = torch.arange(batch, device=condition.device)[:, None]
            condition = condition[rows, indices]
            ids, mask = ids[rows, indices], mask[rows, indices]
            if valid is not None:
                valid = valid.to(device=condition.device)
        losses, accuracies = [], []
        for frame in range(condition.shape[1]):
            rows = slice(None) if valid is None else valid[:, frame].nonzero(as_tuple=True)[0]
            labels = ids[rows, frame, 1:].clone().masked_fill(~mask[rows, frame, 1:].bool(), -100)
            logits = self.main_logits(condition[rows, frame], ids[rows, frame, :-1], mask[rows, frame, :-1])
            loss, accuracy = sequence_ce(logits, labels)
            if valid is not None:
                loss = loss.new_zeros(condition.shape[0]).index_copy(0, rows, loss)
                accuracy = accuracy.new_zeros(condition.shape[0]).index_copy(0, rows, accuracy)
            losses.append(loss)
            accuracies.append(accuracy)
        main_per_frame = torch.stack(losses, dim=1)
        accuracy_per_frame = torch.stack(accuracies, dim=1)
        main = (main_per_frame.mean(dim=1) if valid is None
                else main_per_frame.sum(dim=1) / valid.sum(dim=1))
        accuracy = (accuracy_per_frame.mean() if valid is None
                    else (accuracy_per_frame.sum(dim=1) / valid.sum(dim=1)).mean())
        auxiliary = main.new_zeros(main.shape)
        return {
            'loss': main.mean(),
            'loss_per_asset': main,
            'main_loss': main.mean(),
            'main_loss_per_frame': main_per_frame,
            'main_accuracy_per_frame': accuracy_per_frame,
            'auxiliary_loss': auxiliary.mean(),
            'main_accuracy': accuracy,
            'auxiliary_accuracy': auxiliary.mean(),
        }

    def forward(self, surface, targets, *, auxiliary_weight, target_frame_indices=None, target_frame_mask=None):
        self._validate_auxiliary_weight(auxiliary_weight)
        if self.virtual_enabled:
            if target_frame_indices is not None or target_frame_mask is not None:
                raise ValueError('selected-frame supervision requires virtual_enabled=False')
            return self.base.forward(surface, targets, auxiliary_weight=0.)
        final_frames, virtual = self.encode(surface)
        arguments = {'target_frame_indices': target_frame_indices}
        if target_frame_mask is not None:
            arguments['target_frame_mask'] = target_frame_mask
        return self.forward_encoded(final_frames, virtual, targets, auxiliary_weight=0., **arguments)

    def generate_condition(self, condition, start_tokens, *, eos_token, max_new_tokens,
                           logits_processor=None):
        # ``nn.Module.train/eval`` normally propagates this; synchronize again
        # for callers that changed the wrapper mode immediately before decode.
        self.base.train(self.training)
        return self.base.generate_condition(condition, start_tokens, eos_token=eos_token,
                                            max_new_tokens=max_new_tokens,
                                            logits_processor=logits_processor)

    def configuration(self) -> dict[str, Any]:
        return {
            **self.base.configuration(),
            'virtual_enabled': self.virtual_enabled,
            'evaluate_auxiliary': False,
            'auxiliary_weight': 0.,
            'motion_route': 'base_virtual_and_real' if self.virtual_enabled else 'real_only',
            'condition_route': 'base' if self.virtual_enabled else 'real_ffn_only',
            'trainable_parameter_names': list(self.trainable_parameter_names),
            'disabled_parameter_names': list(self.disabled_parameter_names),
        }
