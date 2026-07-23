from __future__ import annotations

import numpy as np

from rigweave.dynamic_rig.motion_evidence import (
    FEATURE_NAMES,
    extract_local_motion_evidence,
    relative_edge_length_trajectories,
    soft_skin_overlap,
    unique_mesh_edges,
)


def _square_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_unique_mesh_edges_are_sorted_and_deduplicated() -> None:
    vertices, faces = _square_mesh()
    edges = unique_mesh_edges(faces, vertex_count=vertices.shape[0])
    assert edges.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]


def test_relative_edge_lengths_ignore_global_rigid_motion() -> None:
    query, faces = _square_mesh()
    theta = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    moved = query @ rotation.T + np.asarray([4.0, -2.0, 0.5], dtype=np.float32)
    edges = unique_mesh_edges(faces, vertex_count=query.shape[0])
    relative, _, _, dropped = relative_edge_length_trajectories(
        np.stack([query, moved], axis=0),
        edges,
    )
    assert dropped == 0
    np.testing.assert_allclose(relative, 0.0, atol=2.0e-6)


def test_soft_skin_overlap_is_joint_order_independent() -> None:
    weights = np.asarray(
        [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    expected = np.asarray([0.75, 0.0, 0.25], dtype=np.float32)
    np.testing.assert_allclose(soft_skin_overlap(weights, edges), expected)
    np.testing.assert_allclose(soft_skin_overlap(weights[:, ::-1], edges), expected)


def test_zero_motion_produces_zero_features() -> None:
    query, faces = _square_mesh()
    frames = np.repeat(query[None], 5, axis=0)
    weights = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    evidence = extract_local_motion_evidence(frames, faces, weights)
    assert evidence.features.shape[1] == len(FEATURE_NAMES)
    np.testing.assert_allclose(evidence.features, 0.0)
    np.testing.assert_allclose(evidence.observability, 0.0)


def test_cross_segment_pair_can_be_more_observable_than_rigid_edges() -> None:
    query = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    moved = query.copy()
    moved[2:] = np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    weights = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    evidence = extract_local_motion_evidence(
        np.stack([query, moved], axis=0),
        faces,
        weights,
    )
    edge_to_index = {tuple(edge): i for i, edge in enumerate(evidence.edges.tolist())}
    within_left = evidence.observability[edge_to_index[(0, 1)]]
    # The edge (1,2) is incident on the exact rotation center and therefore
    # remains length-preserving. The wider local pair (0,2) crosses the same
    # articulation and carries observable relative motion.
    across_segments = evidence.observability[edge_to_index[(0, 2)]]
    assert across_segments > within_left + 0.1
