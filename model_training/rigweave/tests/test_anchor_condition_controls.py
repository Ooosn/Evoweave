from __future__ import annotations

import sys
from pathlib import Path

import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from evaluate_anchor_condition_controls import (  # noqa: E402
    _anchor_condition,
    _first_divergence,
    _install_condition_control,
    _restore_condition,
)


class _FakeConditioner:
    def __init__(self, frame_tokens: torch.Tensor) -> None:
        self.frame_tokens = frame_tokens

    def tokenize_frames(self, *args, **kwargs):
        del args, kwargs
        query_points = torch.zeros(self.frame_tokens.shape[:-1] + (3,))
        return self.frame_tokens, query_points

    def motion_encoder(self, frame_tokens: torch.Tensor, *, query_points: torch.Tensor):
        del query_points
        return frame_tokens.mean(dim=1)


class _FakeAnchorModel:
    condition_fusion = "anchor_motion_residual_zero"
    branch_prior = None

    def __init__(self, frame_tokens: torch.Tensor) -> None:
        self.conditioner = _FakeConditioner(frame_tokens)

    def sample_references(self, batch):
        del batch
        return object()


class _FakeControlModel:
    def __init__(self) -> None:
        self.controls: list[str] = []

    def build_condition(self, batch, control="normal", refs=None, return_branch_prior=False):
        del batch, refs, return_branch_prior
        self.controls.append(control)
        return control


def test_anchor_condition_keeps_query_anchor_identity() -> None:
    frame_tokens = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    model = _FakeAnchorModel(frame_tokens)
    batch = {
        "frame_vertices": torch.empty(1),
        "faces": torch.empty(1),
        "vertex_normals": torch.empty(1),
        "face_normals": torch.empty(1),
    }

    static = _anchor_condition(
        model,
        batch,
        mode="static_bypass",
        refs=None,
        return_branch_prior=False,
    )
    dynamic, branch_prior = _anchor_condition(
        model,
        batch,
        mode="dynamic_only",
        refs=None,
        return_branch_prior=True,
    )

    assert torch.equal(static, frame_tokens[:, 0])
    assert torch.equal(dynamic, frame_tokens.mean(dim=1))
    assert branch_prior is None


def test_zero_motion_control_forces_zero_sequence_mode_and_restores() -> None:
    model = _FakeControlModel()
    original = _install_condition_control(model, "zero_motion")
    assert model.build_condition({}) == "zero"
    _restore_condition(model, original)
    assert model.build_condition({}) == "normal"
    assert model.controls == ["zero", "normal"]


def test_first_divergence_covers_value_and_length_changes() -> None:
    assert _first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert _first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert _first_divergence([1, 2], [1, 2, 3]) == 2
