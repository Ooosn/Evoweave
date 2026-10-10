from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np


def helpers():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_motion_bias.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"gradient_summary", "grouped_gradient_summaries", "gradient_repeat_summary", "summarize"}]
    scope = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope


class BiasDiagnosticTest(unittest.TestCase):
    def test_aligned_and_opposite_gradients(self):
        analyze = helpers()["gradient_summary"]
        aligned = analyze([[1, 0], [2, 0]], [0.5, 0])
        self.assertEqual(aligned["norm_sum_over_sum_norms"], 1)
        self.assertEqual(aligned["mean_pairwise_cosine"], 1)
        self.assertEqual(aligned["mean_scale_derivative_at_one"], 0.75)
        cancelled = analyze([[1, 0], [-1, 0]], [0.5, 0])
        self.assertEqual(cancelled["norm_sum_over_sum_norms"], 0)
        self.assertEqual(cancelled["negative_pairwise_cosine_fraction"], 1)
        self.assertEqual(cancelled["resultant_over_independent_direction_rms"], 0)
        self.assertEqual(cancelled["norm_weighted_pairwise_cosine"], -1)
        orthogonal = analyze([[1, 0], [0, 1]], [1, 1])
        self.assertAlmostEqual(orthogonal["resultant_over_independent_direction_rms"], 1)
        self.assertEqual(orthogonal["norm_weighted_pairwise_cosine"], 0)

    def test_per_channel_gradient_groups(self):
        vectors = np.ones((2, 12, 8, 3))
        vectors[1, ..., 1] = -1
        vectors[..., 2] = 0
        result = helpers()["grouped_gradient_summaries"](vectors.reshape(2, -1), np.ones(288))
        self.assertEqual(result["u"]["norm_sum_over_sum_norms"], 1)
        self.assertEqual(result["c"]["norm_sum_over_sum_norms"], 0)
        self.assertTrue(result["d"]["all_gradients_zero"])

    def test_gradient_repeat_floor(self):
        result = helpers()["gradient_repeat_summary"]([[1, 0], [0, 1]], [[1.01, 0], [0, 1.01]])
        self.assertAlmostEqual(result["median_sample_relative_l2"], .01)
        self.assertAlmostEqual(result["mean_gradient_relative_difference"], .01)
        with self.assertRaises(ValueError):
            helpers()["gradient_repeat_summary"]([[1]], [[float("nan")]])

    def test_invalid_and_zero_gradients(self):
        analyze = helpers()["gradient_summary"]
        for vectors, weights in (([[0, 0]], [1, 1]), ([[1]], [1, 2]), ([[float("nan")]], [1])):
            with self.assertRaises(ValueError):
                analyze(vectors, weights)

    def test_pairing_uses_asset_and_frame(self):
        rows = [
            {"index": 0, "frames": 8, "control": "normal", "scale": 1, "ce": 2},
            {"index": 1, "frames": 8, "control": "normal", "scale": 1, "ce": 4},
            {"index": 1, "frames": 8, "control": "normal", "scale": 3, "ce": 3.5},
            {"index": 0, "frames": 8, "control": "normal", "scale": 3, "ce": 1.5},
        ]
        result = helpers()["summarize"](rows)["normal_scale3_t8"]
        self.assertEqual(result["paired_ce_delta_vs_normal_scale1"], -0.5)
        self.assertEqual(result["paired_asset_bootstrap_95pct_ce_delta"], [-0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
