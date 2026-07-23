from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


RIGWEAVE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(RIGWEAVE_SRC))

from rigweave.dynamic_rig.data import _align_frames_to_query_rigid, _select_query_sequence


def _sequence() -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    rigid = query @ rotation + np.asarray([3.0, -2.0, 0.5], dtype=np.float32)
    articulated = rigid.copy()
    articulated[-1, 2] += 0.75
    return np.stack([query, rigid, articulated]), query


def test_query_rigid_alignment_removes_global_transform() -> None:
    frames, query = _sequence()
    aligned = _align_frames_to_query_rigid(
        frames,
        query_index=0,
        fit_vertex_indices=np.arange(4, dtype=np.int64),
    )
    np.testing.assert_array_equal(aligned[0], query)
    np.testing.assert_allclose(aligned[1], query, atol=1.0e-5)
    assert float(np.sqrt(np.mean((aligned[2] - query) ** 2))) > 0.05


def test_query_rigid_alignment_supports_nonzero_query_index() -> None:
    frames, _ = _sequence()
    reordered = frames[[1, 0, 2]]
    aligned = _align_frames_to_query_rigid(
        reordered,
        query_index=1,
        fit_vertex_indices=np.arange(4, dtype=np.int64),
    )
    np.testing.assert_array_equal(aligned[1], reordered[1])
    np.testing.assert_allclose(aligned[0], reordered[1], atol=1.0e-5)


def test_query_rigid_sequence_keeps_query_target_and_removes_rigid_evidence(tmp_path: Path) -> None:
    frames, query = _sequence()
    translated = query + np.asarray([-1.25, 0.75, 2.0], dtype=np.float32)
    frame_vertices = np.concatenate([frames[:2], translated[None]], axis=0)
    posed_joints = np.stack(
        [
            np.asarray([[float(frame_index), 0.0, 0.0]], dtype=np.float32)
            for frame_index in range(frame_vertices.shape[0])
        ],
        axis=0,
    )

    selected_frames, target_joints, _, selected = _select_query_sequence(
        frame_vertices,
        posed_joints,
        None,
        frame_count=3,
        path=tmp_path / "fixture.npz",
        index=0,
        random_query=False,
        seed=37,
        motion_fps_ratio=1.0,
        motion_vertex_samples=4,
        motion_alignment_policy="query_rigid",
    )

    np.testing.assert_array_equal(selected_frames[0], frame_vertices[selected[0]])
    np.testing.assert_array_equal(target_joints, posed_joints[selected[0]])
    for evidence in selected_frames[1:]:
        np.testing.assert_allclose(evidence, selected_frames[0], atol=1.0e-5)


def test_none_policy_preserves_legacy_sequence_exactly(tmp_path: Path) -> None:
    frames, _ = _sequence()
    posed_joints = frames[:, :2].copy()
    common = dict(
        frame_count=2,
        path=tmp_path / "legacy.npz",
        index=0,
        random_query=False,
        seed=73,
        motion_fps_ratio=1.0,
        motion_vertex_samples=4,
    )
    legacy = _select_query_sequence(frames, posed_joints, None, **common)
    explicit_none = _select_query_sequence(
        frames,
        posed_joints,
        None,
        motion_alignment_policy="none",
        **common,
    )
    for legacy_value, explicit_value in zip(legacy, explicit_none):
        if legacy_value is None:
            assert explicit_value is None
        else:
            np.testing.assert_array_equal(legacy_value, explicit_value)
