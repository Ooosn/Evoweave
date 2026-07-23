"""Decoder cross-attention with static addressing and motion-only values."""

from __future__ import annotations

import torch
from torch import nn


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
