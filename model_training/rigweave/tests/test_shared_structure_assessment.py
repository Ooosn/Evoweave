from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assess_shared_structure as assessment
import run_shared_structure_experiment as controller
import record_shared_structure_operation as recorder
import test_shared_structure_controller as controller_fixtures


class AssessmentTest(unittest.TestCase):
    def config(self):
        path = Path(__file__).resolve().parents[2] / "experiments/shared_structure_review_20261011.json"
        return json.loads(path.read_text())

    def test_two_shards_cover_exact_original_evaluation_budget(self):
        config = self.config()
        before = deepcopy(config)
        entries = {split: [{"index": index} for index in range(count)]
                   for split, count in (("train", 256), ("valid", 64))}
        for label, count in (("trained", 224), ("initialized", 96)):
            all_keys = assessment.expected_row_keys(entries, config, label)
            self.assertEqual(len(all_keys), count)
            shards = []
            for rank in (0, 1):
                subset, view_config = assessment.worker_entries(entries, config, rank)
                shards.append(assessment.expected_row_keys(subset, view_config, label))
                self.assertEqual(view_config["runtime"]["host_memory_budget_gib"], 32)
            self.assertFalse(shards[0] & shards[1])
            self.assertEqual(shards[0] | shards[1], all_keys)
        self.assertEqual(config, before)
        with self.assertRaises(ValueError):
            assessment.worker_entries(entries, config, 2)

    def test_recorded_architecture_values_and_types_are_unchanged(self):
        config = self.config()
        path = Path(__file__).resolve().parents[2] / "experiments/shared_structure_medium_20261011.json"
        original = json.loads(path.read_text())
        self.assertEqual(config["architecture"], original["architecture"])
        for key, value in original["architecture"].items():
            self.assertIs(type(config["architecture"][key]), type(value), key)

    def test_candidate_contract_rejects_architecture_sampling_and_holdout_drift(self):
        config = self.config()
        payload = {"step": 1024, "source_checkpoint": controller._checkpoint_identity(config),
                   "configuration": deepcopy(config), "feature_encoder_frozen": True,
                   "training_accounting": {"assets": 4096, "heldout_frame_exposures": 0},
                   "optimizer": {}, "scheduler": {}, "sampler": {}}
        self.assertEqual(assessment.validate_candidate(payload, config)["epochs"], 16)
        for key in ("architecture", "data", "features"):
            changed = deepcopy(payload)
            changed["configuration"][key]["unexpected"] = True
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                assessment.validate_candidate(changed, config)
        changed = deepcopy(payload)
        changed["training_accounting"]["heldout_frame_exposures"] = 1
        with self.assertRaises(RuntimeError):
            assessment.validate_candidate(changed, config)

    def test_assessment_resources_gate_and_reused_preflight(self):
        fixture = controller_fixtures.SharedStructureControllerTest()
        fixture.setUp()
        try:
            config, _ = fixture._online_fixture()
            config["runtime"].update(probe_script="assess_shared_structure.py", physical_gpus=[5, 7])
            config["assessment"] = {"checkpoint": {"path": "candidate.pt", "sha256": "a" * 64, "step": 1024},
                                    "acceptance": "quality and throughput only"}
            plan = controller.build_stage_plan(config, "assessment", fixture.config_path)
            self.assertEqual(plan["resources"]["physical_gpus"], [5, 7])
            self.assertEqual(plan["guard_operations"], ["matched_eval"])
            controller.validate_launch_environment(config, {"CUDA_VISIBLE_DEVICES": "5,7"}, linux=True)
            with self.assertRaises(RuntimeError):
                controller.validate_launch_environment(config, {"CUDA_VISIBLE_DEVICES": "5"}, linux=True)
            self.assertEqual(controller.require_online_preflight(config)["case_count"], 1)
            prepared = recorder.prepare_state({"active_operation": {"status": "failed"}}, config, plan, fixture.source)
            self.assertIn("matched_eval", prepared["allowed_operations"])
            self.assertIn("train", prepared["blocked_operations"])
            for stage in ("probe", "preflight", "cache"):
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    controller.build_stage_plan(config, stage, fixture.config_path)
        finally:
            fixture.tearDown()

    def test_merge_rejects_missing_or_duplicate_rows_before_summarizing(self):
        config = self.config()
        entries = {"train": [{"index": 0}], "valid": []}
        with self.assertRaises(RuntimeError):
            assessment.merge_evaluations([{"rows": []}], entries, config, "initialized")
        row = {"split": "train", "index": 0, "view": "T8"}
        with self.assertRaises(RuntimeError):
            assessment.merge_evaluations([{"rows": [row, row]}], entries, config, "initialized")


if __name__ == "__main__":
    unittest.main()
