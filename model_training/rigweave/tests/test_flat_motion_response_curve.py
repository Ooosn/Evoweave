from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from analyze_flat_motion_response_curve import (  # noqa: E402
    _parse_betas,
    _scale_evidence_sequence,
    _shared_energy_fraction,
)


def test_beta_parser_requires_endpoints() -> None:
    assert _parse_betas("0,0.1,1") == (0.0, 0.1, 1.0)
    with pytest.raises(ValueError):
        _parse_betas("0.1,1")
    with pytest.raises(ValueError):
        _parse_betas("0,1.1")


def test_sequence_scaling_has_exact_endpoints() -> None:
    sequence = torch.tensor([[[[1.0]], [[3.0]], [[5.0]]]])
    zero = _scale_evidence_sequence(sequence, 0.0, normalize_vectors=False)
    half = _scale_evidence_sequence(sequence, 0.5, normalize_vectors=False)
    full = _scale_evidence_sequence(sequence, 1.0, normalize_vectors=False)
    assert zero.flatten().tolist() == [1.0, 1.0, 1.0]
    assert half.flatten().tolist() == [1.0, 2.0, 3.0]
    assert torch.equal(full, sequence)


def test_shared_energy_fraction_detects_token_shared_delta() -> None:
    shared = torch.ones(1, 4, 3)
    assert _shared_energy_fraction(shared, token_dim=1) == pytest.approx(1.0)
    local = torch.tensor([[[1.0], [-1.0]]])
    assert _shared_energy_fraction(local, token_dim=1) == pytest.approx(0.0)
