from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from analyze_flat_motion_reliability import (  # noqa: E402
    _best_rigid_residual,
    _evidence_subset_batch,
    _effective_rank,
    _motion_evidence_metrics,
    _subset_consistency_metrics,
)


def test_best_rigid_residual_removes_rotation_and_translation() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    posed = rest @ rotation + torch.tensor([2.0, -3.0, 4.0])
    assert float(_best_rigid_residual(rest, posed).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_motion_metrics_force_fp32_inside_bfloat16_autocast() -> None:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    posed = rest + torch.tensor([0.2, -0.1, 0.3])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        residual = _best_rigid_residual(rest, posed)
        metrics = _motion_evidence_metrics(torch.stack([rest, posed]).unsqueeze(0))
    assert residual.dtype == torch.float32
    assert float(residual.abs().max()) == pytest.approx(0.0, abs=1e-5)
    assert metrics["motion_rms"] > 0


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


def test_evidence_subsets_are_disjoint_and_preserve_frame_zero() -> None:
    sequence = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)
    batch = {"frame_vertices": sequence}
    odd = _evidence_subset_batch(batch, keep_even_evidence=False)["frame_vertices"]
    even = _evidence_subset_batch(batch, keep_even_evidence=True)["frame_vertices"]
    assert odd[:, 0].item() == even[:, 0].item() == sequence[:, 0].item()
    assert odd.flatten().tolist() == [0.0, 1.0, 0.0, 3.0, 0.0, 5.0]
    assert even.flatten().tolist() == [0.0, 0.0, 2.0, 0.0, 4.0, 0.0]


def test_subset_consistency_is_exact_for_matching_evidence() -> None:
    zero = torch.zeros(1, 2, 3)
    normal = torch.ones(1, 2, 3)
    metrics = _subset_consistency_metrics(normal, normal, normal, zero)
    assert metrics["evidence_subset_delta_cosine"] == pytest.approx(1.0)
    assert metrics["evidence_subset_disagreement_to_full"] == pytest.approx(0.0)
    assert metrics["evidence_subset_mean_to_full"] == pytest.approx(0.0)
