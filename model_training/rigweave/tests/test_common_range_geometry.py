import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common_range_geometry import coordinate_summary, fit_points, mesh_gauge, root_gauge, transform_sequence


class CommonRangeGeometryTests(unittest.TestCase):
    def test_rigid_motion_removed_and_roundtrip(self):
        reference = np.random.default_rng(12).normal(size=(101, 3))
        rotation = Rotation.from_euler("xyz", [31, 17, -24], degrees=True).as_matrix()
        moving = reference @ rotation.T + [10, -5, 2]
        vertices = np.stack((reference, moving))
        matrices, offsets, report = mesh_gauge(vertices, "rigid")
        result = transform_sequence(vertices, matrices, offsets)
        np.testing.assert_array_equal(result[0], reference)
        np.testing.assert_allclose(result[1], reference, atol=2e-14)
        self.assertLess(report["frames"][1]["inverse_max_error_bbox"], 1e-13)
        self.assertLess(report["frames"][1]["split_rotation_degrees"], 1e-10)
        self.assertLess(report["frames"][1]["split_alignment_rms_bbox"], 1e-13)

    def test_translation_does_not_rotate(self):
        ref = np.random.default_rng(22).normal(size=(12, 3))
        vertices = np.stack((ref, ref + [3, 4, 5]))
        rotation, translation, _ = mesh_gauge(vertices, "translation")
        np.testing.assert_array_equal(rotation, np.repeat(np.eye(3)[None], 2, 0))
        np.testing.assert_allclose(transform_sequence(vertices, rotation, translation)[1], ref, atol=1e-14)

    def test_robust_fit_keeps_proper_rotation(self):
        ref = np.random.default_rng(22).normal(size=(30, 3))
        moving = ref + [4, 0, 0]
        moving[-2:] += [0, 12, 5]
        rotation, translation, report = fit_points(moving, ref, robust=True)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        self.assertLess(report["minimum_weight"], 1.)
        self.assertLess(np.linalg.norm(moving[:-2] @ rotation.T + translation - ref[:-2], axis=1).mean(), .1)

    def test_normals_only_rotate(self):
        points = np.ones((2, 6, 3))
        rotations = np.repeat(np.eye(3)[None], 2, 0)
        np.testing.assert_array_equal(transform_sequence(points, rotations, np.ones((2, 3)), normals=True), points)

    def test_translation_stability_is_measured(self):
        ref = np.random.default_rng(7).normal(size=(12, 3))
        moving = ref.copy()
        moving[::2] += [2, 0, 0]
        _, _, report = mesh_gauge(np.stack((ref, moving)), "translation")
        self.assertGreater(report["frames"][1]["split_alignment_rms_bbox"], 0.)

    def test_root_gauge_matches_world_to_common_coordinates(self):
        deltas = np.repeat(np.eye(4)[None, None], 3, 0)
        deltas[0, 0, :3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
        deltas[0, 0, :3, 3] = [1, 2, 3]
        deltas[2, 0, :3, :3] = Rotation.from_euler("x", -30, degrees=True).as_matrix()
        deltas[2, 0, :3, 3] = [-2, 8, 5]
        reference = np.random.default_rng(3).normal(size=(8, 3))
        selected, center, scale = [0, 2], np.array([2, 1, -3]), 4.
        posed = np.stack([reference @ deltas[i, 0, :3, :3].T + deltas[i, 0, :3, 3] for i in selected])
        common = (posed - center) / scale
        rotations, translations = root_gauge(deltas, 0, selected, center, scale)
        aligned = transform_sequence(common, rotations, translations)
        np.testing.assert_allclose(aligned[1], common[0], atol=1e-14)

    def test_nonrigid_root_is_not_silently_projected(self):
        deltas = np.repeat(np.eye(4)[None, None], 2, 0)
        deltas[1, 0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "nonrigid"):
            root_gauge(deltas, 0, [0, 1], [0, 0, 0], 1)

    def test_range_uses_points_not_individual_axis_denominator(self):
        points = np.array([[[2., 2., 0.], [0., 0., 0.]]])
        self.assertEqual(coordinate_summary(points)["point_fraction_outside_1"], .5)


if __name__ == "__main__":
    unittest.main()
