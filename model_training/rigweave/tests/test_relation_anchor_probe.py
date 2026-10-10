import json
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_motion_experiment import build_plan
from run_relation_anchor import check_parity, extremes


class AnchoredProbeTest(unittest.TestCase):
    def test_extremes_keep_all_three_views(self):
        entries = [{"index": i, "joint_count": count, "view": view} for i, count in enumerate((20, 3, 9, 150))
                   for view in ("normal8", "weak2", "static8")]
        result = extremes(entries)
        self.assertEqual(len(result), 6)
        self.assertEqual({row["index"] for row in result}, {1, 3})

    def test_parity_requires_condition_prefix_and_ce(self):
        good = {"zero_ce_difference": 0., "zero_condition": {"max_abs": 0.}, "same_prefix": True}
        self.assertTrue(check_parity([good]))
        self.assertFalse(check_parity([]))
        for values in ({"zero_ce_difference": 3e-5}, {"zero_condition": {"max_abs": 1e-8}}, {"same_prefix": False}):
            self.assertFalse(check_parity([{**good, **values}]))

    def test_recorded_stages_reuse_inputs_on_one_gpu(self):
        config = json.loads((SCRIPTS.parents[1] / "experiments/motion_evidence_base_compare_20261010.json").read_text())
        for stage in ("relation_anchor_preflight", "relation_anchor_screen"):
            plan = build_plan(config, stage, None)
            self.assertEqual(plan["devices"], [5])
            self.assertIsNone(plan["expected"])
            self.assertTrue(plan["output"].startswith(config["relation_anchor_probe"]["output_root"]))
            self.assertTrue(plan["command"][2].endswith("run_relation_anchor.py"))
        prior, current = config["relation_probe"], config["relation_anchor_probe"]
        for key in ("seed", "steps", "accumulation", "lr", "weight_decay", "bottleneck_dim", "views"):
            self.assertEqual(prior[key], current[key])
        audit = build_plan(config, "relation_static_identity", None)
        self.assertEqual(audit["devices"], [5])
        self.assertIsNone(audit["expected"])
        self.assertIsNone(audit["output"])
        self.assertTrue(audit["command"][2].endswith("audit_static_relation_identity.py"))


if __name__ == "__main__":
    unittest.main()
