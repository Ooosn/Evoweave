from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_TRAINING_ROOT))

from analysis.centered_motion_condition import build_centered_motion_condition  # noqa: E402


class _MotionEncoder(nn.Module):
    def forward(self, frame_tokens: torch.Tensor, *, query_points: torch.Tensor) -> torch.Tensor:
        del query_points
        baseline = frame_tokens[:, 0] * 2.0 + 3.0
        motion = (frame_tokens - frame_tokens[:, :1]).mean(dim=1)
        return baseline + motion


def test_centered_motion_condition_cancels_static_encoder_rewrite() -> None:
    frame_tokens = torch.tensor(
        [[[[1.0, 2.0]], [[2.0, 4.0]], [[4.0, 8.0]]]],
        dtype=torch.float32,
    )
    query_points = torch.zeros((1, 3, 1, 3), dtype=torch.float32)

    result = build_centered_motion_condition(
        frame_tokens,
        query_points,
        _MotionEncoder(),
        alpha=0.5,
    )

    expected_motion = (frame_tokens - frame_tokens[:, :1]).mean(dim=1)
    assert torch.equal(result.static, frame_tokens[:, 0])
    assert torch.allclose(result.motion_delta, expected_motion)
    assert torch.allclose(result.fused, frame_tokens[:, 0] + 0.5 * expected_motion)


def test_centered_motion_condition_is_exact_static_for_zero_motion() -> None:
    query = torch.randn((2, 1, 5, 8), generator=torch.Generator().manual_seed(7))
    frame_tokens = query.expand(-1, 4, -1, -1).clone()
    query_points = torch.randn((2, 1, 5, 3), generator=torch.Generator().manual_seed(8)).expand(-1, 4, -1, -1).clone()

    result = build_centered_motion_condition(
        frame_tokens,
        query_points,
        _MotionEncoder(),
        alpha=3.0,
    )

    assert torch.equal(result.motion_delta, torch.zeros_like(result.motion_delta))
    assert torch.equal(result.fused, result.static)


def test_centered_motion_condition_rejects_misaligned_points() -> None:
    frame_tokens = torch.zeros((1, 4, 5, 8))
    query_points = torch.zeros((1, 4, 6, 3))

    with pytest.raises(ValueError, match="must align"):
        build_centered_motion_condition(
            frame_tokens,
            query_points,
            _MotionEncoder(),
            alpha=1.0,
        )
