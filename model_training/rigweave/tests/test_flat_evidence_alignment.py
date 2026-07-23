from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from evaluate_flat_evidence_alignment import _align_evidence_batch, _parse_modes  # noqa: E402


def _fixture_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    query = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    posed = query @ rotation + torch.tensor([3.0, -2.0, 0.5])
    vertex_normals = torch.nn.functional.normalize(query + 0.25, dim=-1)
    posed_normals = vertex_normals @ rotation
    batch = {
        "frame_vertices": torch.stack([query, posed]).unsqueeze(0),
        "vertex_normals": torch.stack([vertex_normals, posed_normals]).unsqueeze(0),
        "face_normals": torch.stack([vertex_normals[:2], posed_normals[:2]]).unsqueeze(0),
        "vertex_count": torch.tensor([4]),
        "face_count": torch.tensor([2]),
    }
    return batch, query


def test_rigid_alignment_recovers_query_geometry_and_normals() -> None:
    batch, query = _fixture_batch()
    aligned = _align_evidence_batch(batch, "rigid")
    assert torch.allclose(aligned["frame_vertices"][0, 0], query)
    assert torch.allclose(aligned["frame_vertices"][0, 1], query, atol=1.0e-5)
    assert torch.allclose(
        aligned["vertex_normals"][0, 1], aligned["vertex_normals"][0, 0], atol=1.0e-5
    )
    assert torch.allclose(
        aligned["face_normals"][0, 1], aligned["face_normals"][0, 0], atol=1.0e-5
    )


def test_center_alignment_removes_translation_but_not_rotation() -> None:
    batch, query = _fixture_batch()
    aligned = _align_evidence_batch(batch, "center")
    assert torch.allclose(
        aligned["frame_vertices"][0, 1].mean(dim=0), query.mean(dim=0), atol=1.0e-6
    )
    assert not torch.allclose(aligned["frame_vertices"][0, 1], query)
    assert torch.equal(aligned["vertex_normals"], batch["vertex_normals"])


def test_normal_mode_is_identity_and_unknown_mode_fails() -> None:
    batch, _ = _fixture_batch()
    assert _align_evidence_batch(batch, "normal") is batch
    with pytest.raises(ValueError, match="unknown alignment mode"):
        _align_evidence_batch(batch, "scale")


def test_parse_modes_requires_known_unique_modes() -> None:
    assert _parse_modes("normal,rigid") == ("normal", "rigid")
    with pytest.raises(ValueError, match="unknown alignment modes"):
        _parse_modes("rigid,scale")
    with pytest.raises(ValueError, match="must be unique"):
        _parse_modes("rigid,rigid")
