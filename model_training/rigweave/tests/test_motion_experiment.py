from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = SCRIPTS.parents[2]
sys.path.insert(0, str(SCRIPTS))
from run_motion_experiment import build_plan, training_environment


class ExperimentPlanTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "model_training/experiments/motion_evidence_base_compare_20261010.json").read_text())

    def test_fresh_initialization_and_full_schedule(self):
        for candidate in self.config["screening"]["candidates"]:
            plan = build_plan(self.config, "screen", candidate["name"])
            expected = plan["expected"]
            self.assertIsNone(expected["resume_checkpoint"])
            self.assertIsNone(expected["init_checkpoint"])
            self.assertEqual(expected["max_steps"], 1667)
            self.assertEqual(expected["stop_after_steps"], 120)
            self.assertEqual(expected["frames_min"], 2)
            self.assertEqual(expected["minimum_random_frames"], 1)
            self.assertEqual(expected["motion_evidence_heads"], candidate["heads"])
            self.assertNotIn("\\", plan["output"])
        with self.assertRaises(ValueError):
            build_plan(self.config, "full", "bias_h2")
        self.config["full_training"]["selected_candidate"] = self.config["screening"]["candidates"][1]
        full = build_plan(self.config, "full", "bias_h2")["expected"]
        self.assertEqual(full["stop_after_steps"], 0)
        self.assertFalse(full["no_save_optimizer"])
        self.assertEqual(full["sample_milestones"], "5000,10000,20000,30000,50000,80000")

    def test_bias_diagnosis_is_not_training(self):
        plan = build_plan(self.config, "bias_diagnostic", None)
        self.assertIsNone(plan["expected"])
        self.assertIsNone(plan["output"])
        self.assertEqual(plan["devices"], [5])
        self.assertIn("diagnose_motion_bias.py", " ".join(plan["command"]))
        self.assertEqual(self.config["bias_diagnostic"]["scales"], [0, 1, 3, 10])
        retry = build_plan(self.config, "bias_diagnostic_paired", None)
        self.assertNotEqual(retry["result"], plan["result"])
        self.assertNotEqual(retry["evaluation_path"], plan["evaluation_path"])

    def test_frame_profile_is_bounded_single_gpu(self):
        plan = build_plan(self.config, "frame_profile", None)
        self.assertIsNone(plan["expected"])
        self.assertEqual(plan["devices"], [5])
        self.assertIn("profile_frame_budget.py", " ".join(plan["command"]))
        for frames, batch in self.config["frame_budget_profile"]["cases"]:
            self.assertLessEqual(frames * batch, 72)

    @unittest.skipUnless(os.name == "posix", "requires the actual Linux Bash launcher")
    def test_actual_launcher_args_match_recorded_recipe_without_running_trainer(self):
        # Execute only the actual argparse construction, never trainer setup or CUDA.
        tree = ast.parse((SCRIPTS / "train_dynamic_rig.py").read_text())
        main = next(value for value in tree.body if isinstance(value, ast.FunctionDef) and value.name == "main")
        body = []
        for node in main.body:
            body.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(value, ast.Name) and value.id == "args" for value in node.targets):
                break
        parser_code = compile(ast.Module(body=body, type_ignores=[]), "trainer_argument_parser", "exec")
        config = deepcopy(self.config)
        config["runtime"]["repo"] = str(ROOT)
        config["full_training"]["selected_candidate"] = config["screening"]["candidates"][1]
        with tempfile.TemporaryDirectory(prefix="motion_cli_contract_") as directory:
            stub = Path(directory) / "torchrun"
            capture = Path(directory) / "argv.json"
            stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ['ARGV_CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n")
            stub.chmod(0o700)
            cases = [("screen", row["name"]) for row in config["screening"]["candidates"]] + [("full", "bias_h2")]
            for stage, name in cases:
                plan = build_plan(config, stage, name)
                with patch("run_motion_experiment.runtime_environment", return_value=dict(os.environ)):
                    env = training_environment(config, plan)
                env.update(EVOWEAVE_ROOT=str(ROOT / "model_training"),
                           EVOWEAVE_UNIRIG_ROOT=config["runtime"]["unirig_root"],
                           RIGWEAVE_SKIP_PREFLIGHT="1", RIGWEAVE_SKIP_DATALOADER_CHECK="1",
                           ARGV_CAPTURE=str(capture), PATH=directory + os.pathsep + env["PATH"])
                subprocess.run(plan["command"], env=env, stdout=subprocess.DEVNULL, check=True)
                argv = json.loads(capture.read_text())
                script_index = argv.index("rigweave/scripts/train_dynamic_rig.py")
                namespace = {"argparse": argparse, "Path": Path, "os": os}
                with patch.object(sys, "argv", ["train_dynamic_rig.py", *argv[script_index + 1:]]), patch.dict(os.environ, env):
                    exec(parser_code, namespace)
                actual = json.loads(json.dumps(vars(namespace["args"]), default=str))
                self.assertEqual(actual, plan["expected"], msg=f"{stage} {name}")


if __name__ == "__main__":
    unittest.main()
