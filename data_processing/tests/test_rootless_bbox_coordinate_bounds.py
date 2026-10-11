from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_rootless_bbox_consistency_distributions as bbox_stage  # noqa: E402


class RootlessBBoxCoordinateBoundsTest(unittest.TestCase):
    @staticmethod
    def screening_args(*, min_target_joint_mesh_bbox_consistency: float = 0.0) -> argparse.Namespace:
        return argparse.Namespace(
            min_target_joint_mesh_bbox_consistency=min_target_joint_mesh_bbox_consistency,
            min_active_joint_mesh_bbox_consistency=0.0,
            target_mesh_padding=bbox_stage.DEFAULT_TARGET_MESH_PADDING,
        )

    @staticmethod
    def base_vertices(frame_count: int = 1) -> np.ndarray:
        frame = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        return np.repeat(frame[None, :, :], frame_count, axis=0)

    @staticmethod
    def compute_metrics(
        vertices: np.ndarray,
        joints: np.ndarray,
        *,
        skin: np.ndarray | None = None,
        frame_index: int = 0,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.npz"
            if skin is None:
                skin = np.ones((vertices.shape[1], joints.shape[1]), dtype=np.float32)
            np.savez(
                path,
                frame_vertices_rootspace=vertices,
                target_joints_rootspace=joints,
                target_skin_weights=skin,
            )
            return bbox_stage.compute_bbox_consistency_metrics(
                path,
                active_skin_threshold=1.0e-4,
                frame_index=frame_index,
            )

    def screened_row(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
        *,
        skin: np.ndarray | None = None,
        frame_index: int = 0,
        min_target_joint_mesh_bbox_consistency: float = 0.0,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "status": "ok",
            **self.compute_metrics(vertices, joints, skin=skin, frame_index=frame_index),
        }
        bbox_stage.apply_screening(
            [row],
            self.screening_args(
                min_target_joint_mesh_bbox_consistency=min_target_joint_mesh_bbox_consistency
            ),
        )
        return row

    def test_exact_positive_and_negative_one_boundary_is_accepted(self) -> None:
        joints = np.asarray([[[-1.1, 0.0, 0.0], [1.1, 0.0, 0.0]]], dtype=np.float32)
        row = self.screened_row(self.base_vertices(), joints)

        self.assertEqual(row["target_joint_mesh_normalized_coordinate_maxabs"], 1.0)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_violating_joint_frame_count"], 0)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_violating_frame_count"], 0)
        self.assertEqual(row["screening_status"], "accept")
        self.assertEqual(row["screening_reasons"], [])

    def test_coordinate_strictly_above_one_is_rejected(self) -> None:
        joints = np.asarray([[[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        row = self.screened_row(self.base_vertices(), joints)

        self.assertGreater(row["target_joint_mesh_normalized_coordinate_maxabs"], 1.0)
        self.assertEqual(
            row["screening_reasons"],
            ["target_joint_mesh_normalized_coordinate_out_of_bounds"],
        )
        self.assertEqual(row["screening_status"], "reject")

    def test_41st_frame_is_checked_and_reported(self) -> None:
        vertices = self.base_vertices(41)
        joints = np.zeros((41, 2, 3), dtype=np.float32)
        joints[40, 0, 0] = 1.2
        row = self.screened_row(vertices, joints)

        self.assertEqual(row["target_joint_mesh_normalized_coordinate_worst_frame"], 40)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_worst_joint"], 0)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_worst_axis"], 0)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_violating_joint_frame_count"], 1)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_violating_frame_count"], 1)
        self.assertIn("target_joint_mesh_normalized_coordinate_out_of_bounds", row["screening_reasons"])

    def test_inactive_skin_joint_is_still_checked(self) -> None:
        joints = np.asarray([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]], dtype=np.float32)
        skin = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        row = self.screened_row(self.base_vertices(), joints, skin=skin)

        self.assertEqual(row["rootless_active_joint_count"], 1)
        self.assertEqual(row["target_joint_mesh_normalized_coordinate_worst_joint"], 1)
        self.assertIn("target_joint_mesh_normalized_coordinate_out_of_bounds", row["screening_reasons"])

    def test_symmetric_negative_coordinate_is_rejected_with_signed_worst_value(self) -> None:
        joints = np.asarray([[[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        row = self.screened_row(self.base_vertices(), joints)

        self.assertGreater(row["target_joint_mesh_normalized_coordinate_maxabs"], 1.0)
        self.assertLess(row["target_joint_mesh_normalized_coordinate_worst_value"], -1.0)
        self.assertIn("target_joint_mesh_normalized_coordinate_out_of_bounds", row["screening_reasons"])

    def test_bounds_are_computed_per_frame_not_from_reference_frame(self) -> None:
        vertices = np.asarray(
            [
                [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        joints = np.zeros((2, 2, 3), dtype=np.float32)
        joints[1, 0, 0] = 0.6
        row = self.screened_row(vertices, joints)

        self.assertEqual(row["target_joint_mesh_normalized_coordinate_worst_frame"], 1)
        self.assertGreater(row["target_joint_mesh_normalized_coordinate_maxabs"], 1.0)
        self.assertIn("target_joint_mesh_normalized_coordinate_out_of_bounds", row["screening_reasons"])

    def test_nonfinite_malformed_empty_and_degenerate_inputs_fail_closed(self) -> None:
        good_vertices = self.base_vertices()
        good_joints = np.zeros((1, 2, 3), dtype=np.float32)
        cases = {
            "nan_vertices": (good_vertices.copy(), good_joints),
            "inf_joints": (good_vertices, good_joints.copy()),
            "degenerate_vertices": (np.zeros((1, 2, 3), dtype=np.float32), good_joints),
            "empty_frames": (np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.float32)),
            "empty_joints": (good_vertices, np.empty((1, 0, 3), dtype=np.float32)),
            "malformed_vertices": (np.zeros((1, 2, 2), dtype=np.float32), good_joints),
            "malformed_joints": (good_vertices, np.zeros((1, 2), dtype=np.float32)),
        }
        cases["nan_vertices"][0][0, 0, 0] = np.nan
        cases["inf_joints"][1][0, 0, 0] = np.inf

        for name, (vertices, joints) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    bbox_stage.compute_target_joint_mesh_normalized_coordinate_metrics(vertices, joints)

    def test_original_single_frame_bbox_filter_coexists(self) -> None:
        joints = np.zeros((1, 2, 3), dtype=np.float32)
        row = self.screened_row(
            self.base_vertices(),
            joints,
            min_target_joint_mesh_bbox_consistency=0.4,
        )

        self.assertEqual(row["target_joint_mesh_normalized_coordinate_maxabs"], 0.0)
        self.assertEqual(
            row["screening_reasons"],
            ["target_joint_mesh_bbox_diag_consistency_below_threshold"],
        )
        self.assertEqual(row["screening_status"], "reject")

    def test_invalid_asset_becomes_metric_error_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.npz"
            np.savez(
                path,
                frame_vertices_rootspace=np.zeros((1, 2, 3), dtype=np.float32),
                target_joints_rootspace=np.zeros((1, 2, 3), dtype=np.float32),
                target_skin_weights=np.ones((2, 2), dtype=np.float32),
            )
            result = bbox_stage.analyze_job(
                {
                    "path": str(path),
                    "active_skin_threshold": 1.0e-4,
                    "frame_index": 0,
                }
            )

        self.assertEqual(result["status"], "error")
        row = dict(result)
        bbox_stage.apply_screening([row], self.screening_args())
        self.assertEqual(row["screening_status"], "reject")
        self.assertEqual(row["screening_reasons"], ["bbox_consistency_metric_error"])

    def test_final_manifests_preserve_accept_reject_split_provenance(self) -> None:
        accepted_metrics = self.compute_metrics(
            self.base_vertices(),
            np.zeros((1, 2, 3), dtype=np.float32),
        )
        rejected_metrics = self.compute_metrics(
            self.base_vertices(),
            np.asarray([[[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
        )
        rows: list[dict[str, object]] = [
            {
                "status": "ok",
                "split": "test",
                "asset_id": "accepted-test",
                "manifest_row": {"asset_id": "accepted-test", "path": "accepted.npz", "split": "test"},
                **accepted_metrics,
            },
            {
                "status": "ok",
                "split": "val",
                "asset_id": "rejected-val",
                "manifest_row": {"asset_id": "rejected-val", "path": "rejected.npz", "split": "val"},
                **rejected_metrics,
            },
        ]
        bbox_stage.apply_screening(rows, self.screening_args())

        with tempfile.TemporaryDirectory() as tmp:
            bbox_stage.write_screened_manifests(rows, Path(tmp))
            accepted = json.loads((Path(tmp) / "final_manifests" / "accepted_manifest.jsonl").read_text())
            rejected = json.loads((Path(tmp) / "final_manifests" / "rejected_manifest.jsonl").read_text())

        self.assertEqual(accepted["split"], "train")
        self.assertEqual(accepted["source_split"], "test")
        self.assertEqual(accepted["final_bbox_consistency_screening"]["status"], "accept")
        self.assertEqual(accepted["final_bbox_consistency_screening"]["metrics"]["coordinate_abs_limit"], 1.0)
        self.assertEqual(rejected["split"], "valid")
        self.assertEqual(rejected["source_split"], "val")
        self.assertEqual(rejected["final_bbox_consistency_screening"]["status"], "reject")
        self.assertEqual(
            rejected["final_bbox_consistency_screening"]["reasons"],
            ["target_joint_mesh_normalized_coordinate_out_of_bounds"],
        )

    def test_rescreening_preserves_existing_source_and_integration_provenance(self) -> None:
        metrics = self.compute_metrics(
            self.base_vertices(),
            np.zeros((1, 2, 3), dtype=np.float32),
        )
        row: dict[str, object] = {
            "status": "ok",
            "split": "train",
            "asset_id": "rescreened-test",
            "source_split": "train",
            "manifest_row": {
                "asset_id": "rescreened-test",
                "path": "rescreened.npz",
                "split": "train",
                "source_split": "test",
                "license": "CC-BY-4.0",
                "integration_source": "prior-final-manifest",
            },
            **metrics,
        }
        bbox_stage.apply_screening([row], self.screening_args())

        with tempfile.TemporaryDirectory() as tmp:
            bbox_stage.write_screened_manifests([row], Path(tmp))
            rescreened = json.loads(
                (Path(tmp) / "final_manifests" / "accepted_manifest.jsonl").read_text()
            )

        self.assertEqual(rescreened["split"], "train")
        self.assertEqual(rescreened["source_split"], "test")
        self.assertEqual(rescreened["license"], "CC-BY-4.0")
        self.assertEqual(rescreened["integration_source"], "prior-final-manifest")


if __name__ == "__main__":
    unittest.main()
