from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from relation_probe_metrics import local_motion, select_rows, structure_by_stratum
from run_motion_experiment import build_plan
from run_relation_probe import summarize, tree_checks, train_arm


class RelationProbeTest(unittest.TestCase):
    def test_parent_global_motion_is_cancelled(self):
        matrices = np.tile(np.eye(4), (3, 3, 1, 1))
        matrices[1, :, :3, 3] = [2, 3, 4]
        matrices[2, :, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        angle, shift, valid = local_motion(matrices, [0, 1, 2], [-1, 0, 1], np.ones((3, 3)), 0, 2)
        self.assertEqual(valid.tolist(), [False, True, True])
        np.testing.assert_allclose(angle, 0, atol=1e-6)
        np.testing.assert_allclose(shift, 0, atol=1e-6)

    def test_local_rotation_and_prismatic_shift(self):
        matrices = np.tile(np.eye(4), (3, 2, 1, 1))
        matrices[1, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        matrices[2, 1, :3, 3] = [0.2, 0, 0]
        angle, shift, _ = local_motion(matrices, [0, 1], [-1, 0], np.zeros((2, 3)), 0, 2)
        self.assertAlmostEqual(angle[1, 1], 90)
        self.assertAlmostEqual(shift[1, 1], 0)
        self.assertAlmostEqual(shift[2, 1], 0.1)

    def test_nonrigid_and_nonfinite_are_unknown(self):
        matrices = np.tile(np.eye(4), (3, 3, 1, 1))
        matrices[1, 1, 0, 0] = 2
        matrices[1, 2, 0, 0] = np.nan
        _, _, valid = local_motion(matrices, [0, 1, 2], [-1, 0, 0], np.ones((3, 3)), 0, 1)
        self.assertFalse(valid.any())

    def test_geometric_correspondence_not_joint_index(self):
        joints = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        result = structure_by_stratum(joints[[2, 1, 0]], [1, 2, None], joints, [-1, 0, 1],
                                     {"strata": {"quiet": [False, True, True]}})["quiet"]
        self.assertEqual(result["joint_coverage_005"], 1)
        self.assertEqual(result["edge_recall"], 1)
        failed = structure_by_stratum(None, [], joints, [-1, 0, 1],
                                     {"strata": {"quiet": [False, True, True]}})["quiet"]
        self.assertEqual(failed["edge_recall"], 0)
        self.assertEqual(failed["joint_coverage_005"], 0)
        self.assertIsNone(failed["mean_gt_to_pred_distance"])

    def test_selection_is_seeded_and_spans_bands(self):
        rows = [{"path": str(i), "asset_id": str(i), "dataset_source": "s",
                 "canonical_metrics": {"target_joint_count": [8, 20, 52, 80][i % 4]}} for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            a, b = select_rows(path, 16, 5), select_rows(path, 16, 5)
        self.assertEqual(a, b)
        self.assertEqual(len({row["index"] for row in a}), 16)
        self.assertEqual([sum(row["joint_band"] == band for row in a) for band in range(4)], [4] * 4)

    def test_tree_and_summary_keep_failed_rows(self):
        self.assertEqual(tree_checks(np.zeros((2, 3)), [1, 0])["nodes_reaching_cycle"], 2)
        rows = [{"view": "static8", "ce": 1., "token_weight": 2., "generation": {
            "success": True, "hit_max_without_eos": False, "topology_f1_or_zero": 1}},
            {"view": "static8", "ce": 2., "token_weight": 6., "generation": {
            "success": False, "hit_max_without_eos": True, "topology_f1_or_zero": 0}}]
        actual = summarize(rows)["static8"]
        self.assertEqual(actual["ce_macro"], 1.5)
        self.assertEqual(actual["ce_token_weighted"], 1.75)
        self.assertEqual(actual["topology_f1_all_rows"], 0.5)

    def test_controller_plan_is_single_gpu_adapter_only(self):
        config = json.loads((SCRIPTS.parents[1] / "experiments/motion_evidence_base_compare_20261010.json").read_text())
        original = deepcopy(config)
        for stage in ("relation_preflight", "relation_screen"):
            plan = build_plan(config, stage, None)
            self.assertEqual(plan["devices"], [5])
            self.assertIsNone(plan["expected"])
            self.assertIn("run_relation_probe.py", plan["command"][2])
            self.assertTrue(plan["output"].endswith(stage))
            with self.assertRaises(ValueError):
                build_plan(config, stage, "bias_h8")
        self.assertEqual(config, original)

    def test_two_step_optimizer_and_token_accounting(self):
        sys.path.insert(0, str(SCRIPTS.parent / "src"))
        from rigweave.dynamic_rig.motion_relation_residual import MotionRelationResidual
        torch.manual_seed(11)
        adapters = torch.nn.ModuleList([MotionRelationResidual(4, 2)])
        entries = [{"input": {"tokens": torch.randn(1, frames, 5, 4)}, "token_weight": weight}
                   for frames, weight in ((2, 4), (8, 8))]
        def forward(model, entry, arm):
            states = torch.zeros(1, 5, 5, 3)
            states[..., 0] = 1
            cond = adapters[0](entry["input"]["tokens"], states)
            return cond, {"ce_loss": cond.square().mean()}, {}
        report = {"training": {}}
        plan = {"lr": 1e-4, "weight_decay": 0.04, "clip_grad_norm": 1., "seed": 11, "accumulation": 2}
        with tempfile.TemporaryDirectory() as directory, patch("run_relation_probe.cached_forward", forward), \
                patch("torch.cuda.max_memory_allocated", return_value=0):
            train_arm(None, adapters, entries, "unknown", plan, report, Path(directory) / "report.json", 2)
        rows = report["training"]["unknown"]
        self.assertEqual([row["samples"] for row in rows], [2, 4])
        self.assertEqual([row["frames"] for row in rows], [10, 20])
        self.assertEqual([row["target_tokens"] for row in rows], [12, 24])
        self.assertGreater(rows[0]["parameter_group_gradient_norms"]["out_proj"], 0)
        self.assertGreater(rows[1]["parameter_group_gradient_norms"]["in_proj"], 0)


if __name__ == "__main__":
    unittest.main()
