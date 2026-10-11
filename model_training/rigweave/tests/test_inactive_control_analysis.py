from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from compare_inactive_motion_controls import paired_score


class InactiveControlAnalysisTest(unittest.TestCase):
    def test_identical_controls_have_zero_delta(self):
        rows = [{"index": i, "base": {"n": i + 1, "m": i}, "control": {"n": i + 1, "m": i}}
                for i in range(4)]
        result = paired_score(rows, "m", "n", 100)
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["delta_95_interval"], [0, 0])

    def test_same_asset_views_are_clustered(self):
        rows = [{"index": i, "base": {"n": 2, "m": 1}, "control": {"n": 2, "m": 2}}
                for i in range(4) for _ in range(2)]
        result = paired_score(rows, "m", "n", 100)
        self.assertEqual(result["assets"], 4)
        self.assertEqual(result["count"], 16)
        self.assertEqual(result["delta"], 0.5)

    def test_empty_and_small_strata_do_not_get_confidence_intervals(self):
        self.assertIsNone(paired_score([], "m", "n")["delta"])
        result = paired_score([{"index": 1, "base": {"n": 3, "m": 1}, "control": {"n": 3, "m": 2}}], "m", "n")
        self.assertIsNone(result["delta_95_interval"])


if __name__ == "__main__":
    unittest.main()
