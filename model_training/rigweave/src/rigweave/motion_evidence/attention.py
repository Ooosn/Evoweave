"""Decoder cross-attention with static addressing and motion-only values."""

from __future__ import annotations

import torch
from torch import nn

from .coverage import PrefixSurfaceSupportHead


class MotionEvidenceCrossAttention(nn.Module):
    """Attend to motion values using aligned static query tokens as keys."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        *,
        residual_scale: float = 0.1,
        detach_static_keys: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or heads <= 0 or hidden_size % heads != 0:
            raise ValueError("hidden_size must be positive and divisible by heads")
        if not 0.0 <= residual_scale < 1.0:
            raise ValueError("residual_scale must be in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.residual_scale = float(residual_scale)
        self.detach_static_keys = bool(detach_static_keys)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            batch_first=True,
            bias=False,
        )

    def _attend(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
        motion_values: torch.Tensor,
        *,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tensors = (prefix_states, static_keys, motion_values)
        if any(value.ndim != 3 for value in tensors):
            raise ValueError("prefix_states, static_keys, and motion_values must be rank-3")
        if static_keys.shape != motion_values.shape:
            raise ValueError(
                "static_keys and motion_values must have identical aligned shape, "
                f"got {tuple(static_keys.shape)} and {tuple(motion_values.shape)}"
            )
        if prefix_states.shape[0] != static_keys.shape[0]:
            raise ValueError("prefix and evidence batch sizes differ")
        if any(value.shape[-1] != self.hidden_size for value in tensors):
            raise ValueError(f"all hidden widths must equal {self.hidden_size}")

        keys = static_keys.detach() if self.detach_static_keys else static_keys
        compute_dtype = self.query_norm.weight.dtype
        # Teacher forcing uses all prefix positions while cached generation uses
        # one query. Autocast MHA selects different bf16 kernels for those two
        # shapes and can move final vocabulary logits by 0.1. Keep this small
        # evidence read in parameter dtype so both routes implement one contract.
        with torch.autocast(device_type=prefix_states.device.type, enabled=False):
            query = self.query_norm(prefix_states.to(dtype=compute_dtype))
            keys = self.key_norm(keys.to(dtype=compute_dtype))
            values = motion_values.to(dtype=compute_dtype)
            # A token-independent value mean is not articulation evidence. If it
            # reaches the residual path, the adapter can learn one global logit
            # correction while ignoring which mesh region each joint queried.
            values = values - values.mean(dim=1, keepdim=True)
            update, weights = self.cross_attention(
                query,
                keys,
                values,
                need_weights=need_weights,
                average_attn_weights=False,
            )
        return update, weights

    def attention_weights(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
        motion_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-head prefix-to-anchor probabilities used by the adapter."""

        _, weights = self._attend(
            prefix_states,
            static_keys,
            motion_values,
            need_weights=True,
        )
        if weights is None:
            raise RuntimeError("cross-attention did not return attention weights")
        return weights

    def forward(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
        motion_values: torch.Tensor,
    ) -> torch.Tensor:
        update, _ = self._attend(
            prefix_states,
            static_keys,
            motion_values,
            need_weights=False,
        )
        return prefix_states + (self.residual_scale * update).to(dtype=prefix_states.dtype)


class CoverageAwareMotionEvidenceCrossAttention(nn.Module):
    """Use a soft surface prior and explicit null gate without hard masking."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        *,
        residual_scale: float = 0.1,
        coverage_bias_strength: float = 0.5,
        support_projection_size: int = 128,
        detach_static_keys: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or heads <= 0 or hidden_size % heads != 0:
            raise ValueError("hidden_size must be positive and divisible by heads")
        if not 0.0 <= residual_scale < 1.0:
            raise ValueError("residual_scale must be in [0, 1)")
        if not 0.0 <= coverage_bias_strength < 1.0:
            raise ValueError("coverage_bias_strength must be in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.heads = int(heads)
        self.head_size = self.hidden_size // self.heads
        self.residual_scale = float(residual_scale)
        self.coverage_bias_strength = float(coverage_bias_strength)
        self.detach_static_keys = bool(detach_static_keys)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.support_head = PrefixSurfaceSupportHead(
            hidden_size,
            projection_size=support_projection_size,
            detach_static_keys=detach_static_keys,
        )
        self.scale = float(self.head_size) ** -0.5

    def load_attention_state(self, source: MotionEvidenceCrossAttention) -> None:
        """Initialize projections from the existing ungated evidence adapter."""

        source_weight = source.cross_attention.in_proj_weight
        if source_weight.shape != (3 * self.hidden_size, self.hidden_size):
            raise ValueError("source attention projection shape is incompatible")
        query, key, value = source_weight.detach().chunk(3, dim=0)
        with torch.no_grad():
            self.query_norm.load_state_dict(source.query_norm.state_dict())
            self.key_norm.load_state_dict(source.key_norm.state_dict())
            self.query_projection.weight.copy_(query)
            self.key_projection.weight.copy_(key)
            self.value_projection.weight.copy_(value)
            self.output_projection.weight.copy_(
                source.cross_attention.out_proj.weight.detach()
            )

    def support_logits(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
    ) -> torch.Tensor:
        return self.support_head(prefix_states, static_keys)

    def _attend(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
        motion_values: torch.Tensor,
        *,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        tensors = (prefix_states, static_keys, motion_values)
        if any(value.ndim != 3 for value in tensors):
            raise ValueError("prefix_states, static_keys, and motion_values must be rank-3")
        if static_keys.shape != motion_values.shape:
            raise ValueError("static keys and motion values must be query aligned")
        if prefix_states.shape[0] != static_keys.shape[0]:
            raise ValueError("prefix and evidence batch sizes differ")
        if any(value.shape[-1] != self.hidden_size for value in tensors):
            raise ValueError(f"all hidden widths must equal {self.hidden_size}")

        keys = static_keys.detach() if self.detach_static_keys else static_keys
        compute_dtype = self.query_norm.weight.dtype
        with torch.autocast(device_type=prefix_states.device.type, enabled=False):
            prefix = prefix_states.to(compute_dtype)
            keys = keys.to(compute_dtype)
            values = motion_values.to(compute_dtype)
            values = values - values.mean(dim=1, keepdim=True)
            query = self.query_projection(self.query_norm(prefix))
            key = self.key_projection(self.key_norm(keys))
            value = self.value_projection(values)

            batch_size, prefix_length, _ = query.shape
            query_count = key.shape[1]
            query = query.view(
                batch_size, prefix_length, self.heads, self.head_size
            ).transpose(1, 2)
            key = key.view(
                batch_size, query_count, self.heads, self.head_size
            ).transpose(1, 2)
            value = value.view(
                batch_size, query_count, self.heads, self.head_size
            ).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            support_logits = self.support_head(prefix, keys)
            support_probabilities = nn.functional.softmax(
                support_logits.float(), dim=-1
            )
            anchor_probability = support_probabilities[..., :-1]
            null_probability = support_probabilities[..., -1]
            conditional_anchor = anchor_probability / (
                1.0 - null_probability
            )[..., None].clamp_min(1.0e-8)
            uniform = conditional_anchor.new_full(
                conditional_anchor.shape,
                1.0 / float(query_count),
            )
            mixed_prior = (
                (1.0 - self.coverage_bias_strength) * uniform
                + self.coverage_bias_strength * conditional_anchor
            )
            log_prior_bias = torch.log(
                (mixed_prior * float(query_count)).clamp_min(1.0e-8)
            )
            weights = nn.functional.softmax(
                scores + log_prior_bias[:, None],
                dim=-1,
            )
            context = torch.matmul(weights, value)
            context = context.transpose(1, 2).reshape(
                batch_size, prefix_length, self.hidden_size
            )
            update = self.output_projection(context)
            update = update * (1.0 - null_probability)[..., None]
        return update, weights if need_weights else None, support_logits

    def attention_weights(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
        motion_values: torch.Tensor,
    ) -> torch.Tensor:
        _, weights, _ = self._attend(
            prefix_states,
            static_keys,
            motion_values,
            need_weights=True,
        )
        if weights is None:
            raise RuntimeError("coverage attention did not return weights")
        return weights

    def forward(
        self,
        prefix_states: torch.Tensor,
        static_keys: torch.Tensor,
        motion_values: torch.Tensor,
    ) -> torch.Tensor:
        update, _, _ = self._attend(
            prefix_states,
            static_keys,
            motion_values,
            need_weights=False,
        )
        return prefix_states + (self.residual_scale * update).to(
            dtype=prefix_states.dtype
        )
