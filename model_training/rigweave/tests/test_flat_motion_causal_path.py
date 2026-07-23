from __future__ import annotations

import sys
from pathlib import Path

import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from analyze_flat_motion_causal_path import (  # noqa: E402
    _motion_summary,
    _pair_summary,
    _target_role,
)


class _Tokenizer:
    eos = 300
    token_id_branch = 301
    num_discrete = 256


def test_target_roles_cover_early_coordinates_and_actions() -> None:
    tokenizer = _Tokenizer()
    assert _target_role(tokenizer, 10, 0) == "root_coordinate"
    assert _target_role(tokenizer, 10, 2) == "root_coordinate"
    assert _target_role(tokenizer, 10, 3) == "first_child_coordinate"
    assert _target_role(tokenizer, 10, 5) == "first_child_coordinate"
    assert _target_role(tokenizer, 10, 6) == "later_coordinate"
    assert _target_role(tokenizer, tokenizer.eos, 6) == "eos"
    assert _target_role(tokenizer, tokenizer.token_id_branch, 6) == "branch"


def test_motion_summary_ignores_padded_vertices() -> None:
    frames = torch.zeros(1, 2, 3, 3)
    frames[0, 1, 0, 0] = 3.0
    frames[0, 1, 1, 0] = 4.0
    frames[0, 1, 2, 0] = 1000.0
    summary = _motion_summary(frames, vertex_count=2)
    assert summary["motion_mean"] == 3.5
    assert abs(summary["motion_rms"] - (12.5**0.5)) < 1.0e-6
    assert summary["motion_max"] == 4.0


def test_pair_summary_reports_identity_and_relative_change() -> None:
    left = torch.tensor([[1.0, 2.0]])
    identity = _pair_summary(left, left.clone())
    assert identity["max_abs_diff"] == 0.0
    assert identity["relative_delta_rms"] == 0.0
    assert abs(identity["cosine"] - 1.0) < 1.0e-6

    changed = _pair_summary(left, torch.zeros_like(left))
    assert changed["max_abs_diff"] == 2.0
    assert changed["relative_delta_rms"] > 0.0
