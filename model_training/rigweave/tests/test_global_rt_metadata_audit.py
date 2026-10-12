from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_global_rt_metadata as audit  # noqa: E402


class GlobalRtMetadataAuditTest(unittest.TestCase):
    def write_npz(self, root: Path, name: str, **payload: np.ndarray) -> Path:
        path = root / name
        np.savez(path, **payload)
        return path

    def recorded_payload(self, roots: int = 1, frames: int = 2) -> dict[str, np.ndarray]:
        positions = np.zeros((frames, roots, 3), dtype=np.float32)
        positions[..., 0] = np.arange(roots, dtype=np.float32)[None, :]
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (frames, roots, 3, 3),
        ).copy()
        return {
            "frame_vertices_rootspace": np.zeros((frames, 1, 3), dtype=np.float32),
            "recorded_root_indices": np.arange(roots, dtype=np.int64),
            "recorded_root_positions": positions,
            "recorded_root_rotations_world_to_root": rotations,
            "recorded_root_rotations_root_to_world": rotations.copy(),
        }

    def skeleton_payload(
        self,
        frames: int = 2,
        parents: tuple[int, ...] = (-1, 0, 1),
        target_raw_indices: tuple[int, ...] | None = None,
        target_parents: tuple[int, ...] | None = None,
        target_entry_raw_indices: tuple[int, ...] = (0,),
    ) -> dict[str, np.ndarray]:
        joint_count = len(parents)
        if target_raw_indices is None:
            target_raw_indices = tuple(range(joint_count))
        if target_parents is None:
            target_parents = parents
        eye = np.eye(4, dtype=np.float32)
        return {
            "parents": np.asarray(parents, dtype=np.int64),
            "rest_joints": np.zeros((joint_count, 3), dtype=np.float32),
            "rest_bone_mats": np.broadcast_to(eye, (joint_count, 4, 4)).copy(),
            "bone_transforms": np.broadcast_to(
                eye,
                (frames, joint_count, 4, 4),
            ).copy(),
            "raw_root_index": np.asarray(0, dtype=np.int64),
            "raw_root_indices": np.asarray(
                [index for index, parent in enumerate(parents) if parent < 0],
                dtype=np.int64,
            ),
            "target_parents": np.asarray(target_parents, dtype=np.int64),
            "target_raw_indices": np.asarray(target_raw_indices, dtype=np.int64),
            "target_entry_raw_indices": np.asarray(target_entry_raw_indices, dtype=np.int64),
        }

    def audit_skeleton_file(
        self,
        path: Path,
    ) -> dict:
        with np.load(path, allow_pickle=False) as data:
            recorded = audit._audit_recorded_root(data)
            return audit._audit_skeleton_metadata(data, recorded)

    def audit_file(self, path: Path) -> dict:
        with np.load(path, allow_pickle=False) as data:
            return audit._audit_recorded_root(data)

    def test_absent_metadata_is_distinguished_from_empty_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            absent = self.write_npz(root, "absent.npz", unrelated=np.asarray([1], dtype=np.int64))
            empty = self.write_npz(
                root,
                "empty.npz",
                recorded_root_indices=np.zeros((0,), dtype=np.int64),
                recorded_root_positions=np.zeros((3, 0, 3), dtype=np.float32),
                recorded_root_rotations_world_to_root=np.zeros((3, 0, 3, 3), dtype=np.float32),
                recorded_root_rotations_root_to_world=np.zeros((3, 0, 3, 3), dtype=np.float32),
                frame_vertices_rootspace=np.zeros((3, 1, 3), dtype=np.float32),
            )
            self.assertEqual(self.audit_file(absent)["status"], audit.ABSENT)
            self.assertEqual(self.audit_file(empty)["status"], audit.EMPTY)
            self.assertEqual(self.audit_file(empty)["root_count"], 0)

    def test_malformed_shape_is_not_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = self.write_npz(
                root,
                "malformed.npz",
                recorded_root_indices=np.asarray([0], dtype=np.int64),
                recorded_root_positions=np.zeros((2, 3), dtype=np.float32),
                recorded_root_rotations_world_to_root=np.broadcast_to(
                    np.eye(3, dtype=np.float32), (2, 1, 3, 3)
                ),
            )
            result = self.audit_file(malformed)
            self.assertEqual(result["status"], audit.MALFORMED)
            self.assertFalse(result["shape_valid"])

    def test_recorded_frame_count_must_match_geometry_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload(frames=2)
            payload["frame_vertices_rootspace"] = np.zeros((3, 1, 3), dtype=np.float32)
            path = self.write_npz(Path(directory), "wrong_frame_count.npz", **payload)
            result = self.audit_file(path)
            self.assertEqual(result["status"], audit.MALFORMED)
            self.assertTrue(any("recorded_frame_count_mismatch" in detail for detail in result["details"]))

    def test_both_rotation_directions_must_be_transposes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload["recorded_root_rotations_root_to_world"] = np.broadcast_to(
                np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
                (2, 1, 3, 3),
            ).copy()
            path = self.write_npz(Path(directory), "inverse_mismatch.npz", **payload)
            result = self.audit_file(path)
            self.assertEqual(result["status"], audit.INVERSE_MISMATCH)
            self.assertTrue(result["rotation_valid"])
            self.assertGreater(result["rotation_inverse_max_error"], audit.ROTATION_INVERSE_ATOL)

    def test_missing_recorded_root_index_cannot_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            del payload["recorded_root_indices"]
            path = self.write_npz(Path(directory), "missing_index.npz", **payload)
            result = self.audit_file(path)
            self.assertEqual(result["status"], audit.MALFORMED)
            self.assertFalse(result["shape_valid"])
            self.assertIn("recorded_root_indices", result["details"][0])

    def test_multiple_roots_are_ambiguous_after_numeric_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_npz(Path(directory), "multi.npz", **self.recorded_payload(roots=2))
            result = self.audit_file(path)
            self.assertEqual(result["status"], audit.AMBIGUOUS)
            self.assertEqual(result["root_count"], 2)
            self.assertTrue(result["shape_valid"])
            self.assertTrue(result["finite"])
            self.assertTrue(result["rotation_valid"])

    def test_nonfinite_metadata_is_reported_before_rotation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload["recorded_root_positions"][0, 0, 1] = np.nan
            path = self.write_npz(Path(directory), "nonfinite.npz", **payload)
            result = self.audit_file(path)
            self.assertEqual(result["status"], audit.NONFINITE)
            self.assertTrue(result["shape_valid"])
            self.assertFalse(result["finite"])

    def test_non_rotation_matrix_is_rejected_after_finite_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload["recorded_root_rotations_world_to_root"][0, 0, 0, 0] = 2.0
            path = self.write_npz(Path(directory), "invalid_rotation.npz", **payload)
            result = self.audit_file(path)
            self.assertEqual(result["status"], audit.INVALID_ROTATION)
            self.assertTrue(result["shape_valid"])
            self.assertTrue(result["finite"])
            self.assertFalse(result["rotation_valid"])

    def test_good_metadata_is_unique_candidate_but_training_fields_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.recorded_payload()
            payload.update(
                {
                    "root_positions": np.zeros((2, 3), dtype=np.float32),
                    "root_rotations_world_to_root": np.broadcast_to(
                        np.eye(3, dtype=np.float32), (2, 3, 3)
                    ),
                    "root_rotations_root_to_world": np.broadcast_to(
                        np.eye(3, dtype=np.float32), (2, 3, 3)
                    ),
                }
            )
            path = self.write_npz(root, "good.npz", **payload)
            with np.load(path, allow_pickle=False) as data:
                recorded = audit._audit_recorded_root(data)
                training = audit._audit_training_root_compatibility(data)
            self.assertEqual(recorded["status"], audit.UNIQUE)
            self.assertTrue(recorded["shape_valid"])
            self.assertTrue(recorded["finite"])
            self.assertTrue(recorded["rotation_valid"])
            self.assertEqual(training["status"], audit.IDENTITY_COMPATIBILITY)
            self.assertEqual(audit._candidate_status(recorded, audit._new_family_result()), "recorded_root_unique_candidate")

    def test_explicit_source_object_transform_is_separate_from_recorded_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix = np.broadcast_to(
                np.eye(4, dtype=np.float32),
                (2, 4, 4),
            ).copy()
            path = self.write_npz(Path(directory), "object.npz", source_object_to_world=matrix)
            with np.load(path, allow_pickle=False) as data:
                explicit = audit._audit_explicit_object_transform(data)
            self.assertEqual(explicit["status"], audit.UNIQUE)
            self.assertEqual(explicit["root_count"], 1)
            self.assertEqual(explicit["frame_count"], 2)

    def test_skeleton_metadata_reports_valid_single_root_gauge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload.update(self.skeleton_payload())
            path = self.write_npz(Path(directory), "skeleton_good.npz", **payload)
            result = self.audit_skeleton_file(path)
            self.assertEqual(result["status"], audit.SKELETON_VALID)
            self.assertEqual(result["raw_root_mapping"]["status"], audit.SKELETON_VALID)
            self.assertEqual(result["target_mapping"]["status"], audit.SKELETON_VALID)
            self.assertEqual(result["raw_root_transform_rigid"]["status"], audit.SKELETON_VALID)
            self.assertEqual(result["recorded_root_relationship"]["class"], audit.RELATIONSHIP_SINGLE)
            self.assertIn("not authoritative object/world RT", result["interpretation"])
            self.assertEqual(result["fields"]["bone_transforms"]["shape"], [2, 3, 4, 4])

    def test_skeleton_frame_count_matches_geometry_header_and_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mismatch_payload = self.recorded_payload(frames=2)
            mismatch_payload.update(self.skeleton_payload(frames=2))
            mismatch_payload["frame_vertices_rootspace"] = np.zeros((3, 1, 3), dtype=np.float32)
            mismatch_path = self.write_npz(Path(directory), "skeleton_frame_mismatch.npz", **mismatch_payload)
            mismatch = self.audit_skeleton_file(mismatch_path)
            self.assertEqual(mismatch["status"], audit.MALFORMED)
            self.assertEqual(mismatch["bone_transforms_frame_count"], 2)
            self.assertEqual(mismatch["frame_vertices_rootspace_header_shape"], [3, 1, 3])
            self.assertIn("bone_transforms_frame_count_mismatch:2!=3", mismatch["details"])

            empty_payload = self.recorded_payload(frames=0)
            empty_payload.update(self.skeleton_payload(frames=0))
            empty_path = self.write_npz(Path(directory), "skeleton_zero_frames.npz", **empty_payload)
            empty = self.audit_skeleton_file(empty_path)
            self.assertEqual(empty["status"], audit.MALFORMED)
            self.assertIn("bone_transforms_frame_count_must_be_positive", empty["details"])
            self.assertIn("frame_vertices_rootspace_frame_count_must_be_positive", empty["details"])

    def test_recorded_roots_are_classified_as_ancestor_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload(roots=3)
            payload.update(self.skeleton_payload())
            path = self.write_npz(Path(directory), "ancestor_chain.npz", **payload)
            result = self.audit_skeleton_file(path)
            self.assertEqual(result["status"], audit.SKELETON_VALID)
            self.assertEqual(
                result["recorded_root_relationship"]["class"],
                audit.RELATIONSHIP_ANCESTOR_CHAIN,
            )
            self.assertEqual(
                result["recorded_root_relationship"]["parent_edges"],
                [
                    {"child": 0, "parent": -1},
                    {"child": 1, "parent": 0},
                    {"child": 2, "parent": 1},
                ],
            )

    def test_recorded_roots_are_classified_as_disconnected_or_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload(roots=2)
            payload.update(
                self.skeleton_payload(
                    parents=(-1, -1, 0),
                    target_raw_indices=(0, 2),
                    target_parents=(-1, 0),
                )
            )
            path = self.write_npz(Path(directory), "disconnected_roots.npz", **payload)
            result = self.audit_skeleton_file(path)
            self.assertEqual(result["status"], audit.SKELETON_VALID)
            self.assertEqual(
                result["recorded_root_relationship"]["class"],
                audit.RELATIONSHIP_DISCONNECTED_OR_MIXED,
            )

    def test_recorded_root_relationship_is_invalid_for_out_of_range_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload["recorded_root_indices"] = np.asarray([9], dtype=np.int64)
            payload.update(self.skeleton_payload())
            path = self.write_npz(Path(directory), "invalid_recorded_relationship.npz", **payload)
            result = self.audit_skeleton_file(path)
            self.assertEqual(result["status"], audit.SKELETON_VALID)
            self.assertEqual(
                result["recorded_root_relationship"]["class"],
                audit.RELATIONSHIP_INVALID,
            )

    def test_skeleton_nonfinite_and_invalid_rigid_rows_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nonfinite_payload = self.recorded_payload()
            nonfinite_payload.update(self.skeleton_payload())
            nonfinite_payload["bone_transforms"][0, 0, 0, 0] = np.nan
            nonfinite_path = self.write_npz(Path(directory), "skeleton_nonfinite.npz", **nonfinite_payload)
            nonfinite = self.audit_skeleton_file(nonfinite_path)
            self.assertEqual(nonfinite["status"], audit.NONFINITE)
            self.assertEqual(nonfinite["raw_root_transform_rigid"]["status"], audit.NONFINITE)

            invalid_payload = self.recorded_payload()
            invalid_payload.update(self.skeleton_payload())
            invalid_payload["bone_transforms"][0, 0, 0, 0] = 2.0
            invalid_path = self.write_npz(Path(directory), "skeleton_invalid_rotation.npz", **invalid_payload)
            invalid = self.audit_skeleton_file(invalid_path)
            self.assertEqual(invalid["status"], audit.INVALID_ROTATION)
            self.assertEqual(invalid["raw_root_transform_rigid"]["status"], audit.INVALID_ROTATION)
            self.assertTrue(invalid["fields"]["bone_transforms"]["finite"])
            self.assertGreater(invalid["raw_root_transform_rigid"]["orthogonality_max_error"], audit.ROTATION_ORTHOGONALITY_ATOL)

            affine_payload = self.recorded_payload()
            affine_payload.update(self.skeleton_payload())
            affine_payload["bone_transforms"][0, 0, 3, 3] = 0.0
            affine_path = self.write_npz(Path(directory), "skeleton_invalid_affine.npz", **affine_payload)
            affine = self.audit_skeleton_file(affine_path)
            self.assertEqual(affine["status"], audit.INVALID_AFFINE)
            self.assertEqual(affine["raw_root_transform_rigid"]["status"], audit.INVALID_AFFINE)

    def test_skeleton_target_entry_mapping_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload.update(self.skeleton_payload(target_entry_raw_indices=(2,)))
            path = self.write_npz(Path(directory), "bad_target_entry.npz", **payload)
            result = self.audit_skeleton_file(path)
            self.assertEqual(result["status"], audit.MALFORMED)
            self.assertEqual(result["target_mapping"]["status"], audit.MALFORMED)
            self.assertIn("target_entry_does_not_match_target_root", result["target_mapping"]["details"])

    def test_skeleton_missing_core_field_is_not_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.recorded_payload()
            payload.update(self.skeleton_payload())
            del payload["target_raw_indices"]
            path = self.write_npz(Path(directory), "missing_target_raw_indices.npz", **payload)
            result = self.audit_skeleton_file(path)
            self.assertEqual(result["status"], audit.MALFORMED)
            self.assertEqual(result["fields"]["target_raw_indices"]["status"], audit.ABSENT)
            self.assertIn("skeleton_core_fields_missing:target_raw_indices", result["details"])

    def test_streaming_census_preserves_split_source_denominators_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good_payload = self.recorded_payload()
            good_payload.update(self.skeleton_payload())
            good = self.write_npz(root, "good.npz", **good_payload)
            empty = self.write_npz(
                root,
                "empty.npz",
                recorded_root_indices=np.zeros((0,), dtype=np.int64),
                recorded_root_positions=np.zeros((2, 0, 3), dtype=np.float32),
                recorded_root_rotations_world_to_root=np.zeros((2, 0, 3, 3), dtype=np.float32),
                recorded_root_rotations_root_to_world=np.zeros((2, 0, 3, 3), dtype=np.float32),
                frame_vertices_rootspace=np.zeros((2, 1, 3), dtype=np.float32),
            )
            train_manifest = root / "train_manifest.jsonl"
            train_manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"asset_id": "good", "sequence_id": "0", "path": str(good), "split": "train", "dataset_source": "base"}),
                        json.dumps({"asset_id": "empty", "sequence_id": "0", "path": str(empty), "split": "train", "dataset_source": "unimate"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            valid_manifest = root / "valid_manifest.jsonl"
            valid_manifest.write_text(
                json.dumps(
                    {"asset_id": "missing", "sequence_id": "0", "path": str(root / "missing.npz"), "split": "valid", "dataset_source": "base"}
                )
                + "\n",
                encoding="utf-8",
            )
            rows_path = root / "rows.jsonl"
            summary_path = root / "summary.json"
            summary = audit.run_census(
                [train_manifest, valid_manifest],
                rows_path,
                summary_path,
                representative_limit=1,
                progress_every=0,
            )
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["denominators"]["split"], {"train": 2, "valid": 1})
            self.assertEqual(summary["denominators"]["source"], {"base": 2, "unimate": 1})
            self.assertEqual(
                summary["denominators"]["split_source"],
                {"train": {"base": 1, "unimate": 1}, "valid": {"base": 1}},
            )
            self.assertEqual(summary["families"]["recorded_root"]["status_counts"][audit.UNIQUE], 1)
            self.assertEqual(summary["families"]["recorded_root"]["status_counts"][audit.EMPTY], 1)
            self.assertEqual(summary["families"]["recorded_root"]["status_counts"].get(audit.NOT_AUDITED), 1)
            self.assertEqual(summary["families"]["skeleton_metadata"]["status_counts"][audit.SKELETON_VALID], 1)
            self.assertEqual(summary["families"]["skeleton_metadata"]["status_counts"][audit.ABSENT], 1)
            self.assertEqual(
                summary["recorded_root_relationship"]["status_counts"][audit.RELATIONSHIP_SINGLE],
                1,
            )
            self.assertEqual(
                summary["recorded_root_relationship"]["status_counts"][audit.RELATIONSHIP_NONE],
                1,
            )
            self.assertEqual(
                summary["recorded_root_relationship"]["status_counts"][audit.NOT_AUDITED],
                1,
            )
            self.assertEqual(summary["failure_categories"]["npz_missing"], 1)
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["candidate_global_rt"]["status"], "recorded_root_unique_candidate")
            self.assertEqual(rows[1]["candidate_global_rt"]["status"], audit.EMPTY)
            self.assertEqual(rows[2]["row_status"], "npz_missing")


if __name__ == "__main__":
    unittest.main()
