from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rigweave.dynamic_rig.data import DynamicRigSample
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.shared_structure_data import (
    SharedStructureDataset,
    shared_structure_collate,
)


def _normalized(raw_targets: np.ndarray, selected: list[int], center: np.ndarray, scale: float) -> np.ndarray:
    return ((raw_targets[np.asarray(selected)] - center) / scale).astype(np.float32)


def _base_sample(
    path: Path,
    raw_targets: np.ndarray,
    parents: np.ndarray,
    selected: list[int],
    center: np.ndarray,
    scale: float,
    *,
    target_joints: np.ndarray | None = None,
    active_skin_joint_count: int | None = None,
) -> DynamicRigSample:
    joints = _normalized(raw_targets, selected, center, scale)
    joint_count = int(raw_targets.shape[1])
    frame_count = len(selected)
    if target_joints is None:
        target_joints = joints[0]
    return DynamicRigSample(
        path=str(path),
        cls="test",
        frame_vertices=torch.zeros((frame_count, 2, 3), dtype=torch.float32),
        faces=torch.zeros((1, 3), dtype=torch.long),
        vertex_normals=torch.zeros((frame_count, 2, 3), dtype=torch.float32),
        face_normals=torch.zeros((frame_count, 1, 3), dtype=torch.float32),
        input_ids=torch.tensor([4, 5], dtype=torch.long),
        attention_mask=torch.ones((2,), dtype=torch.float32),
        target_joints=torch.tensor(target_joints, dtype=torch.float32),
        target_parents=torch.tensor(parents, dtype=torch.long),
        branch_prior_roots=torch.zeros((0, 3), dtype=torch.float32),
        branch_prior_children=torch.zeros((0, 3), dtype=torch.float32),
        branch_prior_mask=torch.zeros((0,), dtype=torch.bool),
        selected_frames=torch.tensor(selected, dtype=torch.long),
        query_center=torch.tensor(center, dtype=torch.float32),
        query_scale=torch.tensor(scale, dtype=torch.float32),
        joint_count=joint_count,
        source_joint_count=joint_count,
        active_skin_joint_count=joint_count if active_skin_joint_count is None else active_skin_joint_count,
        target_start_index=0,
        alignment_root_index=0,
        vertex_count=2,
        face_count=1,
    )


@dataclass
class _FakeBaseDataset:
    sample: DynamicRigSample
    random_query: bool = False
    frame_count: int = 4

    def __post_init__(self) -> None:
        self.paths = [Path(self.sample.path)]
        self.seen_indices: list[object] = []

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int | FrameSampleRequest) -> DynamicRigSample:
        self.seen_indices.append(index)
        return self.sample


class SharedStructureDataTest(unittest.TestCase):
    def write_npz(self, directory: Path, name: str, targets: np.ndarray, parents: np.ndarray) -> Path:
        path = directory / name
        np.savez(path, target_joints_rootspace=targets, target_parents=parents)
        return path

    def make_dataset(
        self,
        directory: Path,
        name: str,
        targets: np.ndarray,
        parents: np.ndarray,
        selected: list[int],
        center: np.ndarray,
        scale: float,
        *,
        target_joints: np.ndarray | None = None,
        active_skin_joint_count: int | None = None,
    ) -> tuple[SharedStructureDataset, _FakeBaseDataset, Path, DynamicRigSample]:
        path = self.write_npz(directory, name, targets, parents)
        sample = _base_sample(
            path,
            targets,
            parents,
            selected,
            center,
            scale,
            target_joints=target_joints,
            active_skin_joint_count=active_skin_joint_count,
        )
        base = _FakeBaseDataset(sample)
        return SharedStructureDataset(base), base, path, sample

    def test_nonchronological_selection_uses_exact_shared_affine_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = np.asarray(
                [
                    [[10.0, -2.0, 4.0], [12.0, -2.0, 4.0]],
                    [[11.0, -1.0, 6.0], [13.0, -1.0, 6.0]],
                    [[14.0, 0.0, 8.0], [16.0, 0.0, 8.0]],
                    [[15.0, 1.0, 10.0], [17.0, 1.0, 10.0]],
                ],
                dtype=np.float32,
            )
            selected = [2, 0, 3]
            center = np.asarray([10.0, -2.0, 4.0], dtype=np.float32)
            dataset, _, _, sample = self.make_dataset(
                root, "asset.npz", targets, np.asarray([-1, 0]), selected, center, 2.0
            )

            result = dataset[0]

            expected = _normalized(targets, selected, center, 2.0)
            torch.testing.assert_close(result.frame_target_joints, torch.from_numpy(expected))
            torch.testing.assert_close(result.frame_target_joints[0], result.base.target_joints)
            self.assertEqual(result.base.selected_frames.tolist(), selected)
            self.assertEqual(sample.selected_frames.tolist(), selected)

    def test_fixed_identity_preserves_full_inactive_joint_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = np.asarray(
                [
                    [[3.0, 4.0, 5.0], [4.0, 4.0, 5.0], [5.0, 4.0, 5.0]],
                ],
                dtype=np.float32,
            )
            targets = np.repeat(targets, 3, axis=0)
            parents = np.asarray([-1, 0, 1], dtype=np.int64)
            dataset, _, _, _ = self.make_dataset(
                root,
                "identity.npz",
                targets,
                parents,
                [2, 0, 1],
                np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
                1.0,
                active_skin_joint_count=1,
            )

            result = dataset[0]

            self.assertEqual(tuple(result.frame_target_joints.shape), (3, 3, 3))
            self.assertEqual(result.base.joint_count, 3)
            self.assertEqual(result.base.active_skin_joint_count, 1)
            torch.testing.assert_close(
                result.frame_target_joints,
                result.frame_target_joints[0].expand_as(result.frame_target_joints),
            )

    def test_collate_pads_variable_joint_counts_and_retains_base_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            center = np.zeros((3,), dtype=np.float32)
            targets_a = np.asarray(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            targets_b = np.asarray(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                    [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            path_a = self.write_npz(root, "a.npz", targets_a, np.asarray([-1, 0], dtype=np.int64))
            path_b = self.write_npz(root, "b.npz", targets_b, np.asarray([-1, 0, 1, 2], dtype=np.int64))
            sample_a = _base_sample(path_a, targets_a, np.asarray([-1, 0]), [2, 0, 1], center, 1.0)
            sample_b = _base_sample(path_b, targets_b, np.asarray([-1, 0, 1, 2]), [2, 0, 1], center, 1.0)
            first = SharedStructureDataset(_FakeBaseDataset(sample_a))[0]
            second = SharedStructureDataset(_FakeBaseDataset(sample_b))[0]

            collated = shared_structure_collate([first, second], pad_token=99)

            self.assertEqual(tuple(collated["frame_target_joints"].shape), (2, 3, 4, 3))
            self.assertEqual(collated["joint_mask"].dtype, torch.bool)
            self.assertEqual(collated["frame_mask"].dtype, torch.bool)
            self.assertEqual(collated["joint_mask"].tolist(), [[True, True, False, False], [True, True, True, True]])
            self.assertEqual(collated["frame_mask"].tolist(), [[True, True, True], [True, True, True]])
            torch.testing.assert_close(
                collated["frame_target_joints"][0, :, :2], first.frame_target_joints
            )
            self.assertTrue(torch.equal(collated["frame_target_joints"][0, :, 2:], torch.zeros((3, 2, 3))))
            self.assertEqual(collated["target_parents"].tolist(), [[-1, 0, -2, -2], [-1, 0, 1, 2]])
            self.assertEqual(collated["path"], [str(path_a), str(path_b)])
            self.assertEqual(collated["pad_token"], 99)

    def test_frame_sample_request_is_forwarded_and_properties_delegate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = np.asarray(
                [
                    [[0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0]],
                    [[2.0, 0.0, 0.0]],
                    [[3.0, 0.0, 0.0]],
                ],
                dtype=np.float32,
            )
            dataset, base, path, _ = self.make_dataset(
                root, "request.npz", targets, np.asarray([-1]), [3, 1], np.zeros((3,), dtype=np.float32), 1.0
            )
            request = FrameSampleRequest(item_index=0, frames=2, sampling_seed=73)

            result = dataset[request]

            self.assertEqual(base.seen_indices, [request])
            self.assertEqual(len(dataset), len(base))
            self.assertIs(dataset.paths, base.paths)
            self.assertEqual(dataset.frame_count, base.frame_count)
            self.assertEqual(dataset.random_query, base.random_query)
            self.assertEqual(result.base.path, str(path))

    def test_source_npz_and_base_sample_are_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = np.asarray(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
                ],
                dtype=np.float32,
            )
            parents = np.asarray([-1, 0], dtype=np.int64)
            dataset, _, path, sample = self.make_dataset(
                root, "immutable.npz", targets, parents, [1, 0], np.zeros((3,), dtype=np.float32), 2.0
            )
            base_before = {
                "target_joints": sample.target_joints.clone(),
                "target_parents": sample.target_parents.clone(),
                "selected_frames": sample.selected_frames.clone(),
                "query_center": sample.query_center.clone(),
                "query_scale": sample.query_scale.clone(),
            }
            with np.load(path) as raw:
                source_before = {name: raw[name].copy() for name in raw.files}

            _ = dataset[0]

            for name, expected in source_before.items():
                with np.load(path) as raw:
                    np.testing.assert_array_equal(raw[name], expected)
            for name, expected in base_before.items():
                torch.testing.assert_close(getattr(sample, name), expected)

    def test_malformed_source_or_selection_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = np.zeros((3, 3, 3), dtype=np.float32)
            targets[:, :, 0] = np.arange(3, dtype=np.float32)
            parents = np.asarray([-1, 0, 1], dtype=np.int64)

            cases = [
                ("bad_shape.npz", targets[:, :, :2], parents, [0, 1], "shape"),
                ("bad_nan.npz", targets.copy(), parents, [0, 1], "non-finite"),
                ("bad_selection.npz", targets, parents, [0, 4], "outside target frame range"),
                ("bad_parent_root.npz", targets, np.asarray([0, 0, 1]), [0, 1], "exactly one root"),
                ("bad_parent_order.npz", targets, np.asarray([-1, 2, 0]), [0, 1], "parent_index < child_index"),
                ("bad_parent_mismatch.npz", targets, np.asarray([-1, 0, 0]), [0, 1], "does not exactly match"),
            ]
            cases[1][1][1, 1, 1] = np.nan
            for name, case_targets, case_parents, selected, message in cases:
                with self.subTest(name=name):
                    path = self.write_npz(root, name, case_targets, case_parents)
                    base_selected = selected if max(selected) < targets.shape[0] else [0, 1]
                    sample = _base_sample(
                        path,
                        targets,
                        parents,
                        base_selected,
                        np.zeros((3,), dtype=np.float32),
                        1.0,
                    )
                    if base_selected != selected:
                        sample.selected_frames.copy_(torch.tensor(selected, dtype=torch.long))
                    with self.assertRaisesRegex(ValueError, message):
                        SharedStructureDataset(_FakeBaseDataset(sample))[0]

            mismatch_path = self.write_npz(root, "bad_base_mismatch.npz", targets, parents)
            mismatch_sample = _base_sample(
                mismatch_path,
                targets,
                parents,
                [0, 1],
                np.zeros((3,), dtype=np.float32),
                1.0,
                target_joints=np.ones((3, 3), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "first selected target"):
                SharedStructureDataset(_FakeBaseDataset(mismatch_sample))[0]


if __name__ == "__main__":
    unittest.main()
