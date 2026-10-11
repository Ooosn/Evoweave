from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "analysis/audit_inactive_joint_predictions.py"
SPEC = importlib.util.spec_from_file_location("inactive_saved_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SavedInactiveAuditTest(unittest.TestCase):
    def test_coverage_and_topology_are_distinct(self):
        target = np.array([[0.0, 0, 0], [0.0, 0.2, 0], [0.0, 0.4, 0]])
        result = MODULE.joint_matches(target, [-1, 0, 0], target, [-1, 0, 1])
        self.assertEqual(result["covered"], [True, True, True])
        self.assertEqual(result["edge_found"], [False, True, False])

    def test_failed_generation_does_not_disappear(self):
        result = MODULE.joint_matches(None, [], [[0, 0, 0], [0, 1, 0]], [-1, 0])
        self.assertEqual(result["covered"], [False, False])
        score = MODULE.score_mask({"joint_matches": result, "success": False}, [False, True])
        self.assertEqual(score["joints"], 1)
        self.assertEqual(score["covered"], 0)
        self.assertIsNone(score["mean_distance"])

    def test_comparison_uses_identical_joint_ids(self):
        rows = []
        for view, quiet in (("normal8", [False, True, False]), ("weak2", [False, True, True]),
                            ("static8", [False, True, True])):
            rows.append({"index": 1, "asset_id": "case", "view": view, "success": True,
                         "observation": {"strata": {"quiet_strict": quiet,
                             "hidden_demonstrated_strict": [False, False, True],
                             "observed_active": [True, False, False] if view != "static8" else [False] * 3}},
                         "joint_matches": {"covered": [True] * 3, "edge_found": [False, True, True],
                                           "distance": [0.0] * 3}})
        masks = MODULE.summarize_same_joints(rows)[0]["matched_masks"]
        self.assertEqual(masks["quiet_in_both_inputs"]["joint_indices"], [1])
        self.assertEqual(masks["hidden_in_weak2"]["joint_indices"], [2])
        self.assertEqual([x["joints"] for x in masks["quiet_in_both_inputs"]["views"].values()], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
