from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CenteredMotionCondition:
    static: torch.Tensor
    dynamic: torch.Tensor
    zero_motion_dynamic: torch.Tensor
    motion_delta: torch.Tensor
    fused: torch.Tensor


def build_centered_motion_condition(
    frame_tokens: torch.Tensor,
    query_points: torch.Tensor,
    motion_encoder: nn.Module,
    *,
    alpha: float,
) -> CenteredMotionCondition:
    """Separate query-pose identity from the encoder's motion response.

    Frame zero is the query pose by the data contract. Repeating it across the
    evidence slots preserves the encoder's role/register-token baseline while
    removing all observed motion. Subtracting that response prevents the large
    zero-motion rewrite from entering the final condition.
    """

    if frame_tokens.ndim != 4:
        raise ValueError(f"frame_tokens must be (B,T,Q,D), got {tuple(frame_tokens.shape)}")
    if query_points.ndim != 4 or query_points.shape[-1] != 3:
        raise ValueError(f"query_points must be (B,T,Q,3), got {tuple(query_points.shape)}")
    if query_points.shape[:3] != frame_tokens.shape[:3]:
        raise ValueError(
            "query_points must align with frame_tokens in (B,T,Q), "
            f"got {tuple(query_points.shape)} vs {tuple(frame_tokens.shape)}"
        )

    static = frame_tokens[:, 0]
    dynamic = motion_encoder(frame_tokens, query_points=query_points)
    zero_frame_tokens = frame_tokens[:, :1].expand_as(frame_tokens)
    zero_query_points = query_points[:, :1].expand_as(query_points)
    zero_motion_dynamic = motion_encoder(zero_frame_tokens, query_points=zero_query_points)
    motion_delta = dynamic - zero_motion_dynamic
    fused = static + float(alpha) * motion_delta
    return CenteredMotionCondition(
        static=static,
        dynamic=dynamic,
        zero_motion_dynamic=zero_motion_dynamic,
        motion_delta=motion_delta,
        fused=fused,
    )
