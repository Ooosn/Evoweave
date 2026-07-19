from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rigweave.stack_close.model import StackCloseDynamicRigAR


class BottleneckConditionRefresh(nn.Module):
    """Zero-gated cross-attention from decoder state to mesh condition."""

    def __init__(
        self,
        hidden_size: int,
        *,
        refresh_dim: int = 256,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if refresh_dim <= 0 or refresh_dim % heads != 0:
            raise ValueError(
                f"refresh_dim={refresh_dim} must be positive and divisible by heads={heads}"
            )
        self.hidden_size = int(hidden_size)
        self.refresh_dim = int(refresh_dim)
        self.heads = int(heads)
        self.head_dim = self.refresh_dim // self.heads
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.condition_norm = nn.LayerNorm(self.hidden_size)
        self.query_proj = nn.Linear(self.hidden_size, self.refresh_dim)
        self.key_proj = nn.Linear(self.hidden_size, self.refresh_dim)
        self.value_proj = nn.Linear(self.hidden_size, self.refresh_dim)
        self.output_proj = nn.Linear(self.refresh_dim, self.hidden_size)
        self.gate = nn.Parameter(torch.zeros((self.hidden_size,)))

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        return (
            values.reshape(batch, tokens, self.heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or condition.ndim != 3:
            raise ValueError("query and condition must both have shape (B,T,D)")
        if query.shape[0] != condition.shape[0]:
            raise ValueError("query and condition batch sizes differ")
        if query.shape[-1] != self.hidden_size or condition.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, "
                f"got query={query.shape[-1]} condition={condition.shape[-1]}"
            )

        query_f = self.query_norm(query.float()).to(dtype=query.dtype)
        condition_f = self.condition_norm(condition.float()).to(
            dtype=condition.dtype
        )
        q = self._heads(self.query_proj(query_f))
        k = self._heads(self.key_proj(condition_f))
        v = self._heads(self.value_proj(condition_f))
        update = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        update = (
            update.transpose(1, 2)
            .contiguous()
            .reshape(query.shape[0], query.shape[1], self.refresh_dim)
        )
        update = self.output_proj(update).to(dtype=query.dtype)
        gate = torch.tanh(self.gate).to(dtype=query.dtype)
        return query + gate.view(1, 1, -1) * update


class ConditionRefreshStackCloseDynamicRigAR(StackCloseDynamicRigAR):
    """Stack-close route with explicit condition cross-attention refresh."""

    def __init__(
        self,
        *args: Any,
        refresh_layer_indices: Sequence[int] = (7, 15, 23),
        refresh_dim: int = 256,
        refresh_heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        indices = tuple(int(value) for value in refresh_layer_indices)
        if not indices:
            raise ValueError("condition refresh requires at least one decoder layer")
        if len(set(indices)) != len(indices) or tuple(sorted(indices)) != indices:
            raise ValueError(
                f"refresh layer indices must be unique and sorted, got {indices}"
            )
        decoder_layers = self.transformer.model.decoder.layers
        layer_count = len(decoder_layers)
        if indices[0] < 0 or indices[-1] >= layer_count:
            raise ValueError(
                f"refresh layer indices {indices} exceed decoder depth {layer_count}"
            )

        self.refresh_layer_indices = indices
        self.refresh_dim = int(refresh_dim)
        self.refresh_heads = int(refresh_heads)
        self.condition_refresh_adapters = nn.ModuleList(
            [
                BottleneckConditionRefresh(
                    self.unirig_ar.hidden_size,
                    refresh_dim=self.refresh_dim,
                    heads=self.refresh_heads,
                )
                for _ in indices
            ]
        )
        self._active_refresh_condition: torch.Tensor | None = None
        self._refresh_hook_handles = [
            decoder_layers[layer_index].register_forward_hook(
                self._make_refresh_hook(adapter_index),
            )
            for adapter_index, layer_index in enumerate(indices)
        ]

    def _make_refresh_hook(self, adapter_index: int) -> Any:
        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            condition = self._active_refresh_condition
            if condition is None:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                raise TypeError(
                    f"unsupported decoder-layer output type {type(output)!r}"
                )

            condition_length = int(condition.shape[1])
            query_start = (
                condition_length
                if hidden.shape[1] > condition_length
                else 0
            )
            if query_start >= hidden.shape[1]:
                return output
            query = hidden[:, query_start:]
            refreshed = self.condition_refresh_adapters[adapter_index](
                query,
                condition.to(device=hidden.device, dtype=hidden.dtype),
            )
            if query_start == 0:
                updated = refreshed
            else:
                updated = torch.cat(
                    [hidden[:, :query_start], refreshed],
                    dim=1,
                )
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        return hook

    @contextmanager
    def condition_refresh(
        self,
        condition: torch.Tensor,
    ) -> Iterator[None]:
        if self._active_refresh_condition is not None:
            raise RuntimeError("condition refresh context is already active")
        self._active_refresh_condition = condition
        try:
            yield
        finally:
            self._active_refresh_condition = None

    def refresh_gate_max_abs(self) -> torch.Tensor:
        return torch.stack(
            [adapter.gate.abs().max() for adapter in self.condition_refresh_adapters]
        ).max()

    def _stack_close_losses(
        self,
        cond: torch.Tensor,
        batch: dict[str, Any],
        decoder_input_ids: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        with self.condition_refresh(cond):
            output = super()._stack_close_losses(
                cond,
                batch,
                decoder_input_ids,
            )
        output["condition_refresh_gate_max_abs"] = self.refresh_gate_max_abs()
        return output

    @torch.no_grad()
    def generate_from_condition(
        self,
        cond: torch.Tensor,
        *,
        cls: str | None,
        max_new_tokens: int = 1_100,
    ) -> Any:
        with self.condition_refresh(cond):
            return super().generate_from_condition(
                cond,
                cls=cls,
                max_new_tokens=max_new_tokens,
            )
