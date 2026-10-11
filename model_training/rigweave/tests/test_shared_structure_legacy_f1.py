import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[2] / "analysis/shared_structure_legacy_f1.py"
SPEC = importlib.util.spec_from_file_location("shared_structure_legacy_f1", PATH)
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


class LegacyF1Tests(unittest.TestCase):
    def setUp(self):
        self.metric = scoring.load_legacy_metric()
        self.points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=np.float32)
        self.parents = [-1, 0, 1]

    def test_exact_and_reindexed_tree(self):
        result = self.metric(self.points, self.parents, self.points, self.parents)
        self.assertEqual(result["edge_f1"], 1.)
        result = self.metric(self.points[[2, 0, 1]], [2, -1, 1], self.points, self.parents)
        self.assertEqual(result["edge_f1"], 1.)

    def test_partial_tree_gets_partial_credit(self):
        result = self.metric(self.points, [-1, -1, 1], self.points, self.parents)
        self.assertEqual(result["edge_precision"], 1.)
        self.assertEqual(result["edge_recall"], .5)
        self.assertAlmostEqual(result["edge_f1"], 2 / 3)

    def test_no_geometric_threshold_and_common_affine_invariance(self):
        moved = self.points + np.array([0., 10., 0.], dtype=np.float32)
        result = self.metric(moved, self.parents, self.points, self.parents)
        self.assertEqual(result["edge_f1"], 1.)
        transformed = self.metric(moved * 3 + 7, self.parents, self.points * 3 + 7, self.parents)
        self.assertEqual(result, transformed)

    def saved(self, success=True):
        return {"selection": {"target_raw_frames": list(range(9)), "query_raw_frame": 0,
                              "heldout_target_positions": list(range(1, 9))},
                "gt_parents": np.array([self.parents]),
                "gt_joints_mesh_local": np.tile(self.points, (1, 9, 1, 1)),
                "pred_joints_mesh_local": np.tile(self.points, (1, 9, 1, 1)) if success else None,
                "generation": {"success": success, "parents": self.parents if success else []}}

    def test_failed_generation_is_zero_and_success_scores_each_frame(self):
        for success in (True, False):
            result = scoring.score_view(self.saved(success), self.metric)
            self.assertEqual(result["query"]["edge_f1"], float(success))
            self.assertEqual(result["heldout_mean"]["edge_f1"], float(success))
            self.assertEqual(len(result["heldout"]), 8)

    def test_mean_f1_is_not_f1_of_mean_precision_recall(self):
        rows = []
        for p, r in ((1., .5), (.5, 1.)):
            scores = {"edge_precision": p, "edge_recall": r, "edge_f1": 2 / 3}
            rows.append({"label": "trained", "split": "valid", "view": "T8", "success": True,
                         "query": scores, "heldout_mean": scores})
        result = scoring.summarize(rows)["trained/valid/T8"]["heldout_mean"]
        self.assertEqual(result["edge_precision"], .75)
        self.assertAlmostEqual(result["edge_f1"], 2 / 3)

    def test_invalid_frame_partition_rejected(self):
        saved = self.saved()
        saved["selection"]["heldout_target_positions"][-1] = 1
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            scoring.score_view(saved, self.metric)


if __name__ == "__main__":
    unittest.main()
