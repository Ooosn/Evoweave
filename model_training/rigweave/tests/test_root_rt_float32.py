from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from rigweave.dynamic_rig import root_rt


class RootRTFloat32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for parent in Path(__file__).resolve().parents:
            path = parent / "data_processing/scripts/build_root_rt_overlay.py"
            if path.is_file():
                break
        else:
            raise FileNotFoundError("production build_root_rt_overlay.py is required for equivalence tests")
        # Execute the producer's definitions without its CLI or bytecode writes.
        namespace = {"__name__": "_root_rt_builder_reference", "__file__": str(path)}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        cls.builder_bounds = staticmethod(namespace["pair_window_bounds"])

    def assert_pair_equivalent(self, vertices, targets):
        lo = vertices.min(axis=(0, 1), keepdims=False)[None]
        hi = vertices.max(axis=(0, 1), keepdims=False)[None]
        report = self.builder_bounds(lo, hi, targets.min(axis=(0, 1))[None],
                                     targets.max(axis=(0, 1))[None])
        # Keep the scale expression vectorized exactly like pair_window_bounds;
        # Python-float times np.float32-scalar differs on NumPy 1.x.
        center = ((lo + hi) * .5).astype(np.float32)[0]
        scale = (1.1 * np.max((hi - lo) * .5, axis=-1)).astype(np.float32)[0]
        expected_vertices = (vertices - center) / scale
        expected_targets = (targets - center) / scale
        maximum = float(np.max(np.abs(expected_targets)))
        self.assertEqual(report["pair_window_normalized_joint_maxabs"], maximum)
        self.assertEqual(report["violating_frame_pairs"], int(maximum > 1))
        if maximum > 1:
            with self.assertRaisesRegex(ValueError, "GT outside"):
                root_rt.normalize_inputs(vertices, targets)
        else:
            vertices_n, targets_n, actual_center, actual_scale, _ = root_rt.normalize_inputs(vertices, targets)
            for actual, expected in ((vertices_n, expected_vertices), (targets_n, expected_targets),
                                     (actual_center, center)):
                self.assertEqual(actual.dtype, np.dtype("float32"))
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_scale, float(scale))
        return maximum

    def test_one_ulp_boundary_matches_builder(self):
        lo, hi, edge = np.asarray([0xbfff3b95, 0x4043638e, 0x40538a06], dtype=np.uint32).view(np.float32)
        base_vertices = np.asarray([[[lo, -1, -.5], [hi, 1, .5]]] * 2, dtype=np.float32)
        base_targets = np.asarray([[[edge, 0, 0]]] * 2, dtype=np.float32)
        for sign in (np.float32(1), np.float32(-1)):
            vertices, targets = sign * base_vertices, sign * base_targets
            with self.subTest(sign=sign):
                self.assertEqual(self.assert_pair_equivalent(vertices, targets), 1.)
                lo64, hi64 = vertices.astype(np.float64).min(axis=(0, 1)), vertices.astype(np.float64).max(axis=(0, 1))
                old_center = ((lo64 + hi64) * .5).astype(np.float32)
                old_scale = np.float32(1.1 * np.max((hi64 - lo64) * .5))
                self.assertGreater(np.max(np.abs((targets.astype(np.float64) - old_center) / old_scale)), 1.)
                inside, outside = targets.copy(), targets.copy()
                inside[..., 0] = np.nextafter(inside[..., 0], -sign * np.float32(np.inf))
                outside[..., 0] = np.nextafter(outside[..., 0], sign * np.float32(np.inf))
                self.assertLessEqual(self.assert_pair_equivalent(vertices, inside), 1.)
                self.assertGreater(self.assert_pair_equivalent(vertices, outside), 1.)

    def test_scale_matches_vector_multiply_not_promoted_scalar(self):
        half = np.float32(.05)
        vector_scale = (1.1 * np.asarray([half], dtype=np.float32)).astype(np.float32)[0]
        promoted_scalar_scale = np.float32(1.1 * float(half))
        self.assertNotEqual(vector_scale, promoted_scalar_scale)
        vertices = np.asarray([[[-half, -half, -half], [half, half, half]]], dtype=np.float32)
        targets = np.asarray([[[vector_scale, 0, 0]]], dtype=np.float32)
        self.assertEqual(self.assert_pair_equivalent(vertices, targets), 1.)

    def test_random_pairs_match_production_builder(self):
        rng = np.random.default_rng(20261013)
        accepted = rejected = 0
        for trial in range(24):
            amplitude = np.float32(10. ** rng.uniform(-6, 6))
            offsets = rng.uniform(-4, 4, (8, 1, 3)).astype(np.float32) * amplitude
            vertices = rng.uniform(-2, 2, (8, 13, 3)).astype(np.float32) * amplitude + offsets
            targets = rng.uniform(-2.8, 2.8, (8, 5, 3)).astype(np.float32) * amplitude + offsets
            full_report = self.builder_bounds(vertices.min(axis=1), vertices.max(axis=1),
                                              targets.min(axis=1), targets.max(axis=1))
            maxima = []
            for first in range(8):
                for second in range(first, 8):
                    ids = [first] if first == second else [first, second]
                    with self.subTest(trial=trial, pair=ids):
                        maximum = self.assert_pair_equivalent(vertices[ids], targets[ids])
                    maxima.append(maximum)
                    accepted += int(maximum <= 1)
                    rejected += int(maximum > 1)
            self.assertEqual(full_report["pair_window_normalized_joint_maxabs"], max(maxima))
            self.assertEqual(full_report["violating_frame_pairs"], sum(x > 1 for x in maxima))
        self.assertEqual(accepted + rejected, 864)
        self.assertGreater(accepted, 100)
        self.assertGreater(rejected, 100)

    def test_augmentation_casts_to_float32_before_bbox(self):
        rng = np.random.default_rng(117)
        vertices = rng.uniform(-2, 2, (3, 31, 3)).astype(np.float32)
        targets = rng.uniform(-.5, .5, (3, 7, 3)).astype(np.float32)
        for _ in range(128):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            q[:, 0] *= np.linalg.det(q)
            q = q.astype(np.float32)
            rotated_vertices = (vertices.astype(np.float64) @ q.astype(np.float64).T).astype(np.float32)
            rotated_targets = (targets.astype(np.float64) @ q.astype(np.float64).T).astype(np.float32)
            expected = root_rt.normalize_inputs(rotated_vertices, rotated_targets)
            actual = root_rt.normalize_inputs(vertices, targets, q)
            for value, reference in zip(actual[:3], expected[:3]):
                np.testing.assert_array_equal(value, reference)
            self.assertEqual(actual[3], expected[3])
            np.testing.assert_array_equal(actual[4], q.astype(np.float64))
        identity = root_rt.normalize_inputs(vertices, targets, np.eye(3, dtype=np.float32))
        plain = root_rt.normalize_inputs(vertices, targets)
        for actual, expected in zip(identity, plain):
            np.testing.assert_array_equal(actual, expected)

    def test_shapes_checked_before_normalization(self):
        vertices = np.ones((2, 4, 3), dtype=np.float32)
        targets = np.zeros((2, 2, 3), dtype=np.float32)
        bad_shapes = ((), (3,), (4, 3), (2, 4, 2), (2, 4, 3, 1), (0, 4, 3), (2, 0, 3))
        for shape in bad_shapes:
            malformed = np.zeros(shape, dtype=np.float32)
            with self.subTest(vertices=shape), self.assertRaisesRegex(ValueError, r"shape \[T,V,3\]"):
                root_rt.normalize_inputs(malformed, targets)
            with self.subTest(targets=shape), self.assertRaisesRegex(ValueError, r"shape \[T,J,3\]"):
                root_rt.normalize_inputs(vertices, malformed)
        with self.assertRaisesRegex(ValueError, "matching frame counts"):
            root_rt.normalize_inputs(vertices, targets[:1])

    def test_positive_small_scales_match_builder(self):
        for half in np.asarray([1e-20, 1e-9, 1e-4, 1, 1e8], dtype=np.float32):
            vertices = np.asarray([[[-half, -half, -half], [half, half, half]]], dtype=np.float32)
            with self.subTest(half=half):
                self.assertLess(self.assert_pair_equivalent(vertices, vertices * np.float32(.5)), 1.)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            root_rt.normalize_inputs(np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 1, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
