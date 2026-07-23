from __future__ import annotations

import torch

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences
from rigweave.motion_evidence import query_aligned_skin_boundary_targets


def _vertex_references() -> TrackableSurfaceReferences:
    return TrackableSurfaceReferences(
        vertex_indices=torch.tensor([[0, 1, 2, 3]]),
        face_indices=torch.empty((1, 0), dtype=torch.long),
        barycentric=torch.empty((1, 0, 3)),
        query_indices=torch.tensor([[0, 1, 2, 3]]),
    )


def test_query_skin_boundary_targets_follow_real_mesh_edges() -> None:
    faces = torch.tensor([[[0, 1, 2], [0, 2, 3]]])
    skin = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ]
    )
    targets = query_aligned_skin_boundary_targets(
        skin,
        faces,
        _vertex_references(),
    )
    expected_mean = torch.tensor([[2.0 / 3.0, 0.5, 2.0 / 3.0, 0.5]])
    torch.testing.assert_close(targets.values[..., 0], expected_mean)
    torch.testing.assert_close(targets.values[..., 1], torch.ones((1, 4)))
    assert torch.equal(targets.valid_mask, torch.ones((1, 4), dtype=torch.bool))
    assert targets.valid_edge_counts.tolist() == [5]


def test_unskinned_vertices_are_not_auxiliary_targets() -> None:
    faces = torch.tensor([[[0, 1, 2], [0, 2, 3]]])
    skin = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ]
    )
    targets = query_aligned_skin_boundary_targets(
        skin,
        faces,
        _vertex_references(),
    )
    assert targets.valid_edge_counts.tolist() == [1]
    assert targets.valid_mask.tolist() == [[True, True, False, False]]
    assert torch.count_nonzero(targets.values) == 0
