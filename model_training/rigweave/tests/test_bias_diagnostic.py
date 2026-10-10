from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np


def helpers():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_motion_bias.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"gradient_summary", "summarize"}]
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
