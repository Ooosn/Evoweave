"""Lightweight pair-aware value residual for anchor tokens."""
from __future__ import annotations

import torch
from torch import nn


class MotionRelationResidual(nn.Module):
    """Mix unnormalized pair evidence with each anchor's own features.

    Inputs exclude role/register tokens. Evidence channels are ordered as
    unknown, co-moving/undecided, and relative-change. The caller owns evidence
    construction and controls; this module never changes or normalizes it.
    Optional reference subtraction anchors the correction to all-unknown input.
    """

    def __init__(self, dim: int, bottleneck_dim: int = 64, *, reference_subtraction: bool = False) -> None:
        super().__init__()
        for name, value in (("dim", dim), ("bottleneck_dim", bottleneck_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(reference_subtraction, bool):
            raise ValueError("reference_subtraction must be a bool")
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.reference_subtraction = reference_subtraction
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, bottleneck_dim)
        self.mix_proj = nn.Linear(4 * bottleneck_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, anchor_tokens: torch.Tensor, evidence_states: torch.Tensor) -> torch.Tensor:
        """Return updated (B,T,Q,D) anchors from read-only (B,Q,Q,3) evidence."""
        if anchor_tokens.ndim != 4:
            raise ValueError("anchor_tokens must have shape (B,T,Q,D)")
        batch, frames, anchors, dim = anchor_tokens.shape
        if min(batch, frames, anchors) < 1 or dim != self.dim:
            raise ValueError("anchor_tokens require positive B,T,Q and the configured D")
        if evidence_states.shape != (batch, anchors, anchors, 3):
            raise ValueError("evidence_states must have shape (B,Q,Q,3) matching anchor_tokens")
        if evidence_states.device != anchor_tokens.device:
            raise ValueError("evidence_states must be on the anchor_tokens device")
        if not anchor_tokens.is_floating_point() or not evidence_states.is_floating_point():
            raise ValueError("anchor_tokens and evidence_states must be floating-point tensors")

        hidden = self.in_proj(self.norm(anchor_tokens))
        # Fold time into value columns, never expand pair matrices across frames.
        values = hidden.permute(0, 2, 1, 3).reshape(batch, anchors, frames * self.bottleneck_dim)
        evidence = evidence_states.to(dtype=hidden.dtype)
        messages = []
        for channel in range(3):
            weights = evidence[..., channel]
            if self.reference_subtraction:
                weights = weights.contiguous()
            message = torch.bmm(weights, values) / anchors
            messages.append(message.reshape(batch, anchors, frames, self.bottleneck_dim).permute(0, 2, 1, 3))
        mixed = self.activation(self.mix_proj(torch.cat((hidden, *messages), dim=-1)))
        residual = self.out_proj(mixed).to(dtype=anchor_tokens.dtype)
        if self.reference_subtraction:
            # Match the actual BMM's dtype/layout; a mean can round differently.
            unknown = torch.bmm(values.new_ones(batch, anchors, anchors), values) / anchors
            unknown = unknown.reshape(batch, anchors, frames, self.bottleneck_dim).permute(0, 2, 1, 3)
            zero = torch.zeros_like(unknown)
            reference = self.activation(self.mix_proj(torch.cat((hidden, unknown, zero, zero), dim=-1)))
            residual = residual - self.out_proj(reference).to(dtype=anchor_tokens.dtype)
        return anchor_tokens + residual
