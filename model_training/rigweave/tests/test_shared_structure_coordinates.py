from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.shared_structure_coordinates import (  # noqa: E402
    normalize_target_meshes,
    restore_target_coordinates,
)


class SharedStructureCoordinatesTest(unittest.TestCase):
    def test_precise_inverse_changes_only_affine_metadata(self) -> None:
        batch = self.make_batch()
        batch["query_center"] = batch["query_center"] + 1e8
        default = normalize_target_meshes(batch)
        precise = normalize_target_meshes(batch, precise_inverse=True)
        for key in ("frame_vertices", "frame_target_joints", "target_joints", "target_mesh_centers", "target_mesh_scales"):
            self.assertTrue(torch.equal(default[key], precise[key]), key)
            self.assertEqual(default[key].dtype, precise[key].dtype)
        self.assertEqual(precise["target_source_centers"].dtype, torch.float64)
        self.assertEqual(precise["target_source_scales"].dtype, torch.float64)
        restored = restore_target_coordinates(precise["frame_target_joints"],
            precise["target_source_centers"], precise["target_source_scales"])
        expected = (batch["frame_target_joints"].double() * batch["query_scale"].double()[:, None, None, None]
                    + batch["query_center"].double()[:, None, None])
        valid = batch["joint_mask"][:, None].expand(*restored.shape[:3])
        delta = ((restored - expected) / precise["target_source_scales"][:, :, None, None])[valid]
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=2e-6, rtol=0)
        with self.assertRaises(TypeError):
            normalize_target_meshes(batch, precise_inverse=1)

    def test_default_preserves_native_mixed_affine_dtype(self) -> None:
        batch = self.make_batch()
        batch["query_center"] = batch["query_center"].double()
        normalized = normalize_target_meshes(batch)
        self.assertEqual(normalized["target_source_centers"].dtype, torch.float64)
        expected = (batch["query_center"][:, None] + batch["query_scale"][:, None, None]
                    * normalized["target_mesh_centers"])
        self.assertTrue(torch.equal(normalized["target_source_centers"], expected))

    def make_batch(self) -> dict[str, torch.Tensor | object]:
        query_center = torch.tensor([[100.0, -50.0, 20.0], [-30.0, 80.0, 12.0]])
        query_scale = torch.tensor([2.0, 4.0])
        source_meshes = torch.tensor(
            [
                [
                    [[98.0, -54.0, 20.0], [102.0, -46.0, 28.0], [100.0, -50.0, 24.0], [99.0, -48.0, 21.0], [9999.0, 9999.0, 9999.0]],
                    [[118.0, -30.0, 20.0], [126.0, -22.0, 36.0], [122.0, -26.0, 28.0], [120.0, -24.0, 24.0], [9999.0, 9999.0, 9999.0]],
                ],
                [
                    [[-38.0, 72.0, 4.0], [-22.0, 88.0, 36.0], [-30.0, 80.0, 20.0], [-26.0, 82.0, 28.0], [9999.0, 9999.0, 9999.0]],
                    [[-50.0, 60.0, 12.0], [-10.0, 100.0, 12.0], [-30.0, 80.0, 12.0], [9999.0, 9999.0, 9999.0], [9999.0, 9999.0, 9999.0]],
                ],
            ],
            dtype=torch.float32,
        )
        canonical_vertices = (source_meshes - query_center[:, None, None, :]) / query_scale[:, None, None, None]
        source_targets = torch.tensor(
            [
                [
                    [[104.0, -48.0, 22.0], [101.0, -51.0, 23.0], [500.0, 500.0, 500.0]],
                    [[140.0, -10.0, 20.0], [126.0, -24.0, 30.0], [700.0, 700.0, 700.0]],
                ],
                [
                    [[-26.0, 84.0, 16.0], [-34.0, 76.0, 24.0], [1000.0, 1000.0, 1000.0]],
                    [[-5.0, 95.0, 12.0], [-45.0, 65.0, 12.0], [1000.0, 1000.0, 1000.0]],
                ],
            ],
            dtype=torch.float32,
        )
        canonical_targets = (source_targets - query_center[:, None, None, :]) / query_scale[:, None, None, None]
        return {
            "frame_vertices": canonical_vertices,
            "frame_target_joints": canonical_targets,
            "target_joints": canonical_targets[:, 0].clone(),
            "vertex_count": torch.tensor([4, 3], dtype=torch.long),
            "joint_mask": torch.tensor([[True, True, False], [True, True, False]]),
            "query_center": query_center,
            "query_scale": query_scale,
            "faces": object(),
            "vertex_normals": object(),
            "topology": {"parents": "preserve"},
        }

    def test_per_mesh_frame_affine_roundtrips_source_coordinates(self) -> None:
        batch = self.make_batch()
        source_targets = torch.tensor(
            [
                [[[104.0, -48.0, 22.0], [101.0, -51.0, 23.0], [500.0, 500.0, 500.0]],
                 [[140.0, -10.0, 20.0], [126.0, -24.0, 30.0], [700.0, 700.0, 700.0]]],
                [[[-26.0, 84.0, 16.0], [-34.0, 76.0, 24.0], [1000.0, 1000.0, 1000.0]],
                 [[-5.0, 95.0, 12.0], [-45.0, 65.0, 12.0], [1000.0, 1000.0, 1000.0]]],
            ],
            dtype=torch.float32,
        )
        before = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        normalized = normalize_target_meshes(batch, padding=1.1)
        restored = restore_target_coordinates(
            normalized["frame_target_joints"],
            normalized["target_source_centers"],
            normalized["target_source_scales"],
        )

        valid = normalized["joint_mask"][:, None, :].expand_as(normalized["frame_target_joints"][..., 0])
        torch.testing.assert_close(restored[valid], source_targets[valid], atol=1e-5, rtol=0)
        self.assertEqual(tuple(normalized["target_mesh_centers"].shape), (2, 2, 3))
        self.assertEqual(tuple(normalized["target_mesh_scales"].shape), (2, 2))
        self.assertFalse(torch.equal(normalized["frame_target_joints"][0, 0], normalized["frame_target_joints"][1, 0]))
        torch.testing.assert_close(
            normalized["target_joints"], normalized["frame_target_joints"][:, 0], atol=0, rtol=0
        )
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
            else:
                self.assertIs(value, before[key])

    def test_padding_excludes_padded_vertices_and_large_targets_are_not_clipped(self) -> None:
        batch = self.make_batch()
        normalized = normalize_target_meshes(batch, padding=1.1)
        self.assertTrue(torch.equal(normalized["frame_vertices"][0, :, 4], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(normalized["frame_vertices"][1, 0, 3:], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(normalized["frame_target_joints"][:, :, 2], torch.zeros(2, 2, 3)))
        self.assertGreater(float(normalized["frame_target_joints"][0, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(float(normalized["target_mesh_scales"][0, 0]), 2.2, places=6)
        self.assertAlmostEqual(float(normalized["target_mesh_scales"][0, 1]), 4.4, places=6)

    def test_metadata_topology_and_joint_order_are_preserved(self) -> None:
        batch = self.make_batch()
        normalized = normalize_target_meshes(batch)
        self.assertIs(normalized["faces"], batch["faces"])
        self.assertIs(normalized["vertex_normals"], batch["vertex_normals"])
        self.assertIs(normalized["topology"], batch["topology"])
        self.assertEqual(tuple(normalized["frame_target_joints"].shape), (2, 2, 3, 3))
        self.assertTrue(torch.equal(normalized["joint_mask"], batch["joint_mask"]))

    def test_invalid_padding_meshes_and_affines_fail(self) -> None:
        for padding in (0.99, float("nan"), float("inf"), True):
            with self.subTest(padding=padding), self.assertRaises(ValueError):
                normalize_target_meshes(self.make_batch(), padding=padding)

        nonfinite = self.make_batch()
        nonfinite["frame_vertices"][0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            normalize_target_meshes(nonfinite)

        degenerate = self.make_batch()
        degenerate["frame_vertices"][0, 0, :4] = 3.0
        with self.assertRaises(ValueError):
            normalize_target_meshes(degenerate)

        bad_count = self.make_batch()
        bad_count["vertex_count"][0] = 0
        with self.assertRaises(ValueError):
            normalize_target_meshes(bad_count)

        bad_scale = self.make_batch()
        bad_scale["query_scale"][0] = 0.0
        with self.assertRaises(ValueError):
            normalize_target_meshes(bad_scale)

    def test_restore_rejects_shape_nonfinite_and_nonpositive_scale(self) -> None:
        local = torch.zeros(1, 2, 3, 3)
        centers = torch.zeros(1, 2, 3)
        scales = torch.ones(1, 2)
        with self.assertRaises(ValueError):
            restore_target_coordinates(local[:, :, :, :2], centers, scales)
        with self.assertRaises(ValueError):
            restore_target_coordinates(local, centers, torch.zeros_like(scales))
        nonfinite = local.clone()
        nonfinite[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            restore_target_coordinates(nonfinite, centers, scales)

    def test_inference_mesh_normalization_needs_no_skeleton_or_labels(self) -> None:
        batch = self.make_batch()
        expected = normalize_target_meshes(batch)
        mesh_only = {key: batch[key] for key in ("frame_vertices", "vertex_count", "query_center", "query_scale")}
        result = normalize_target_meshes(mesh_only)
        for name in ("frame_vertices", "target_mesh_centers", "target_mesh_scales",
                     "target_source_centers", "target_source_scales"):
            torch.testing.assert_close(result[name], expected[name], atol=0, rtol=0)
        self.assertNotIn("frame_target_joints", result)
        self.assertNotIn("target_joints", result)
        self.assertNotIn("joint_mask", result)
        self.assertNotIn("target_parents", result)

    def test_raw_mesh_inference_defaults_to_identity_input_affine(self) -> None:
        batch = self.make_batch()
        mesh_only = {key: batch[key] for key in ("frame_vertices", "vertex_count")}
        result = normalize_target_meshes(mesh_only)
        restored = restore_target_coordinates(result["frame_vertices"], result["target_source_centers"],
                                               result["target_source_scales"])
        for index, count in enumerate(batch["vertex_count"].tolist()):
            torch.testing.assert_close(restored[index, :, :count], batch["frame_vertices"][index, :, :count])
        with self.assertRaisesRegex(ValueError, "provided together"):
            normalize_target_meshes({**mesh_only, "joint_mask": batch["joint_mask"]})
        with self.assertRaisesRegex(ValueError, "provided together"):
            normalize_target_meshes({**mesh_only, "query_scale": batch["query_scale"]})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
