from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from analyze_flat_motion_reliability import (  # noqa: E402
    _best_rigid_residual,
    _effective_rank,
    _motion_evidence_metrics,
)


def test_best_rigid_residual_removes_rotation_and_translation() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    posed = rest @ rotation + torch.tensor([2.0, -3.0, 4.0])
    assert float(_best_rigid_residual(rest, posed).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_motion_metrics_separate_rigid_and_articulated_change() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rigid = rest + torch.tensor([0.2, 0.1, -0.3])
    articulated = rigid.clone()
    articulated[-1, 2] += 0.5
    metrics = _motion_evidence_metrics(torch.stack([rest, rigid, articulated]).unsqueeze(0))
    assert metrics["motion_rms"] > 0
    assert metrics["articulated_rms"] > 0
    assert 0 < metrics["moving_fraction_0p05"] <= 1
    assert metrics["motion_mode_effective_rank"] > 0


def test_effective_rank_handles_zero_and_orthogonal_modes() -> None:
    assert _effective_rank(torch.zeros(3)) == 0.0
    assert _effective_rank(torch.ones(3)) == pytest.approx(3.0)
