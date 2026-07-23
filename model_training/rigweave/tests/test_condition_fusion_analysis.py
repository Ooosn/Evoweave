from __future__ import annotations

from pathlib import Path
import sys

import torch


ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from analyze_condition_fusion import (  # noqa: E402
    CONDITION_LABELS,
    _condition_stats,
    _coordinate_prediction_stats,
)


def test_coordinate_prediction_stats_use_root_and_first_child_xyz_positions() -> None:
    labels = torch.tensor([[266, 10, 20, 30, 40, 50, 60]]).repeat(len(CONDITION_LABELS), 1)
    logits = torch.full((len(CONDITION_LABELS), 7, 256), -20.0)
    for condition_index in range(len(CONDITION_LABELS)):
        for position, target in enumerate(labels[condition_index].tolist()):
            if target < 256:
                logits[condition_index, position, target] = 20.0

    # Only the fused condition is intentionally wrong at root x and joint1 z.
    logits[1, 1, 10] = -20.0
    logits[1, 1, 13] = 20.0
    logits[1, 6, 60] = -20.0
    logits[1, 6, 64] = 20.0

    result = _coordinate_prediction_stats(logits, labels, num_discrete=256)

    assert result["static"]["root_bin_l2"] == 0.0
    assert result["static"]["joint1_bin_l2"] == 0.0
    assert result["fused"]["root_bin_l2"] == 3.0
    assert result["fused"]["joint1_bin_l2"] == 4.0


def test_condition_stats_detect_token_shared_residual() -> None:
    static = torch.ones((1, 4, 2))
    residual = torch.full_like(static, 0.5)
    fused = static + residual
    dynamic = static * 2.0

    result = _condition_stats(static, dynamic, dynamic, fused, fused)

    assert result["residual_rms_ratio"] == 0.5
    assert result["residual_shared_energy_fraction"] == 1.0
    assert result["motion_effect_fused_shared_energy_fraction"] == 0.0


def test_condition_stats_detect_anchor_specific_motion_effect() -> None:
    static = torch.ones((1, 4, 2))
    zero_fused = static.clone()
    fused = static.clone()
    fused[:, 0, 0] += 1.0
    fused[:, 1, 0] -= 1.0
    dynamic = static * 2.0

    result = _condition_stats(static, dynamic, static, fused, zero_fused)

    assert result["motion_effect_fused_shared_energy_fraction"] == 0.0
