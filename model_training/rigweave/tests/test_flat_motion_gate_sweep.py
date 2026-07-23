from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from evaluate_flat_motion_gate_sweep import (  # noqa: E402
    _alpha_name,
    _blend_conditions,
    _parse_alphas,
)


def test_alpha_parser_and_names_are_stable() -> None:
    assert _parse_alphas("0,0.25,1") == (0.0, 0.25, 1.0)
    assert _alpha_name(0.25) == "alpha_0p25"
    with pytest.raises(ValueError):
        _parse_alphas("-0.1,1")
    with pytest.raises(ValueError):
        _parse_alphas("0.5,0.5")


def test_blend_preserves_exact_endpoints() -> None:
    normal = torch.tensor([[1.0, 3.0]])
    zero = torch.tensor([[5.0, 7.0]])
    assert _blend_conditions(normal, zero, 1.0) is normal
    assert _blend_conditions(normal, zero, 0.0) is zero
    assert torch.equal(
        _blend_conditions(normal, zero, 0.25),
        torch.tensor([[4.0, 6.0]]),
    )
