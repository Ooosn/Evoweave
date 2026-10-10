from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_relation_probe import paired_measurement


class PairedAnalysisTest(unittest.TestCase):
    def rows(self):
        return [{"asset_id": str(i), "view": view, "ce": float(i + 1)}
                for i in range(4) for view in ("weak2", "static8", "normal8")]

    def test_clusters_and_constant_effect(self):
        a = self.rows()
        b = [{**row, "ce": row["ce"] + 0.2} for row in a]
        result = paired_measurement(a, b, "ce")
        self.assertEqual(result["assets"], 4)
        self.assertEqual(result["weight"], 12)
        self.assertAlmostEqual(result["delta"], 0.2)
        for endpoint in result["bootstrap95"]:
            self.assertAlmostEqual(endpoint, 0.2)

    def test_pair_mismatch_and_nonfinite_rejected(self):
        a = self.rows()
        with self.assertRaises(ValueError):
            paired_measurement(a, a[:-1], "ce")
        with self.assertRaises(ValueError):
            paired_measurement(a + [a[0]], a, "ce")
        b = [{**row, "ce": float("nan")} for row in a]
        with self.assertRaises(ValueError):
            paired_measurement(a, b, "ce")

    def test_sparse_strata_do_not_invent_confidence(self):
        rows = self.rows()
        for row in rows:
            row["generation"] = {"strata": {"quiet": {"joints": 0, "edges": 0,
                "joint_coverage_005": None, "edge_recall": None}}}
        empty = paired_measurement(rows, rows, "quiet:edge_recall")
        self.assertIsNone(empty["delta"])
        self.assertIsNone(empty["bootstrap95"])
        rows[0]["generation"]["strata"]["quiet"].update(edges=2, edge_recall=0.5)
        tiny = paired_measurement(rows, rows, "quiet:edge_recall")
        self.assertEqual(tiny["contributing_assets"], 1)
        self.assertEqual(tiny["delta"], 0)
        self.assertIsNone(tiny["bootstrap95"])


if __name__ == "__main__":
    unittest.main()
