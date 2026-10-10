from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = SCRIPTS.parents[2]
sys.path.insert(0, str(SCRIPTS.parent / "src"))
sys.path.insert(0, str(SCRIPTS))
import run_frame_budget_smoke as smoke


def settings():
    # Provisional arithmetic example only, not a GPU-safety approval.
    return {"frame_budget": 72, "frame_batch_cap": 6, "max_steps": 4,
            "first_stop_after_steps": 1, "target_stop_step": 3, "sample_overshoot": 1}


def bash_path():
    found = shutil.which("bash")
    if found:
        return found
    git = shutil.which("git")
    if git:
        candidate = Path(git).parents[1] / "bin/bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def log_rows(plan, phase):
    count = phase["expected_final_step"]
    train_rows = plan["prediction"]["train_rows"]
    rows = [{"event": "run_config", "args": smoke.recorded_args(plan["phases"][0], train_rows),
             "world_size": 2, "batch_mode": "frame_budget", "effective_batch": None,
             "loss_normalization": "global_target_token_weight", "train_rows": train_rows,
             "sample_milestones": plan["sample_milestones"]}]
    token_total = 0.0
    for predicted in plan["prediction"]["steps"][:count]:
        step = predicted["step"]
        token_total += 10000.0 + step
        rows.append({**smoke.expected_counter_fields(predicted, train_rows), "step": step,
                     "epoch": predicted["epoch"], "frames_in_step": predicted["frames_in_step"],
                     "micro_batch_sizes_in_step": predicted["micro_batch_sizes_in_step"],
                     "grad_accum": 8, "micro_batch_per_gpu": None, "loss": 1.5, "ce": 1.5,
                     "gradient_norm_before_clip": 0.7, "motion_adapter_gradient_norm_after_clip": 0.002,
                     "motion_adapter_parameter_norm": 0.005, "seconds": 1.0, "gpu_peak_gb": 50.0,
                     "lrs": plan["optimizer_schedule"][step]["lrs"],
                     "target_token_weight_in_step": 10000.0 + step, "target_token_weight_seen": token_total})
        if count > 1 and step == 1:
            rows.append({**smoke.expected_counter_fields(predicted, train_rows), "event": "resume", "step": 1,
                         "resume_checkpoint": phase["expected"]["resume_checkpoint"],
                         "epoch": predicted["next_epoch"], "skipped_batches_in_epoch": predicted["next_batch_in_epoch"],
                         "target_token_weight_in_step": 10001.0, "target_token_weight_seen": token_total,
                         "archived_interrupted_log": str(Path(plan["output"]) / "logs/archived.log")})
    if count == 3:
        rows.append({**smoke.expected_counter_fields(plan["prediction"]["steps"][2], train_rows),
                     "event": "sample_milestone_checkpoint", "step": 3, "include_optimizer": True,
                     "sample_milestone": plan["max_samples"],
                     "path": str(smoke.PurePosixPath(plan["output"]) / f"checkpoint_sample_{plan['max_samples']}.pt")})
    return rows


def checkpoint_payload(plan, phase, logs):
    step = phase["expected_final_step"]
    prediction = plan["prediction"]["steps"][step - 1]
    train_rows = plan["prediction"]["train_rows"]
    progress = smoke.FrameBudgetProgress(
        samples_seen=prediction["sample_seen"], input_frames_seen=prediction["input_frames_seen"],
        target_token_weight_seen=logs["target_token_weight_seen"], microbatches_seen=prediction["consumed_microbatches"],
        next_epoch=prediction["next_epoch"], next_batch_in_epoch=prediction["next_batch_in_epoch"],
        last_samples=prediction["samples_in_step"], last_frames=prediction["input_frames_in_step"],
        last_token_weight=logs["last_token_weight"], samples_by_frame_count=prediction["samples_by_frame_count"],
    )
    schedule = plan["optimizer_schedule"][step]
    groups, offset = [], 0
    for index, (name, count) in enumerate(zip(("motion", "ar", "surface"), (304, 388, 196))):
        groups.append({"name": name, "params": list(range(offset, offset + count)), "lr": schedule["lrs"][name],
                       "weight_decay": phase["expected"]["weight_decay"], "betas": tuple(schedule["betas"][index])})
        offset += count
    return {
        "args": smoke.recorded_args(phase, train_rows), "step": step,
        "sample_seen": prediction["sample_seen"], "input_frames_seen": prediction["input_frames_seen"],
        "effective_batch": None, "train_rows": train_rows, "epoch_equivalent": prediction["sample_seen"] / train_rows,
        "batch_accounting": progress.checkpoint(),
        "model": {f"conditioner.motion_encoder.blocks.{index}.motion_injection.bias_weights": torch.full((8, 3), 0.001 * step)
                  for index in range(12)},
        "optimizer": {"param_groups": groups, "state": {index: {"step": torch.tensor(float(step)),
                      "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)} for index in range(888)}},
        "scheduler": {"total_steps": 4, "last_epoch": step, "_step_count": step + 1,
                      "_last_lr": [group["lr"] for group in groups]},
    }


class FrameBudgetSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "model_training/experiments/motion_evidence_base_compare_20261010.json").read_text(encoding="utf-8"))
        self.config["frame_budget_smoke"] = settings()
        self.plan = smoke.build_smoke_plan(self.config, "/reports/frame_budget_smoke.json")

    def local_plan(self, directory):
        config = deepcopy(self.config)
        config["runtime"].update(repo=(Path(directory) / "repo").as_posix(), job_root=(Path(directory) / "job").as_posix(),
                                 output_root=(Path(directory) / "outputs").as_posix())
        plan = smoke.build_smoke_plan(config, (Path(directory) / "report.json").as_posix())
        return config, plan

    def test_prediction_matches_per_rank_seed_stream_and_actual_global_counts(self):
        for cap in (1, 2, 3, 6, 12):
            expected = {**self.plan["phases"][0]["expected"], "frame_batch_cap": cap}
            prediction = smoke.predict_schedule(expected, 15920)
            rng = np.random.default_rng(np.random.SeedSequence([expected["seed"], 0, 99011]))
            frames = rng.integers(2, 25, size=32).tolist()
            batches = [min(cap, 72 // count) for count in frames]
            samples = input_frames = 0
            histogram = {}
            for index, row in enumerate(prediction["steps"]):
                window_t, window_b = frames[8 * index:8 * (index + 1)], batches[8 * index:8 * (index + 1)]
                samples += 2 * sum(window_b)
                input_frames += 2 * sum(batch * count for batch, count in zip(window_b, window_t))
                for batch, count in zip(window_b, window_t):
                    histogram[str(count)] = histogram.get(str(count), 0) + 2 * batch
                self.assertEqual(row["frames_in_step"], window_t)
                self.assertEqual(row["micro_batch_sizes_in_step"], window_b)
                self.assertEqual(row["sample_seen"], samples)
                self.assertEqual(row["input_frames_seen"], input_frames)
                self.assertEqual(row["samples_by_frame_count"], histogram)
                self.assertEqual(row["rank_samples_seen"], [samples // 2] * 2)
                self.assertEqual(row["rank_input_frames_seen"], [input_frames // 2] * 2)
                self.assertEqual((row["next_epoch"], row["next_batch_in_epoch"]), (0, 8 * (index + 1)))

    def test_prediction_handles_epoch_boundaries_and_distributed_padding(self):
        expected = self.plan["phases"][0]["expected"]
        prediction = smoke.predict_schedule(expected, 3)
        self.assertEqual(prediction["rows_per_rank"], 2)
        for index, row in enumerate(prediction["steps"], 1):
            self.assertEqual(row["samples_in_step"], 32)
            self.assertEqual(row["micro_batch_sizes_in_step"], [2] * 8)
            self.assertEqual((row["next_epoch"], row["next_batch_in_epoch"]), (index * 8, 0))
            self.assertEqual(row["consumed_microbatches"], index * 8)
            self.assertEqual(sum(row["samples_by_frame_count"].values()), row["sample_seen"])
            self.assertEqual(sum(int(key) * value for key, value in row["samples_by_frame_count"].items()), row["input_frames_seen"])

    def test_two_phase_plan_preserves_recipe_and_minimizes_checkpoints(self):
        before = deepcopy(self.config)
        plan = smoke.build_smoke_plan(self.config, "/reports/another.json")
        self.assertEqual(self.config, before)
        fresh, resume = plan["phases"]
        self.assertEqual(fresh["command"], resume["command"])
        self.assertEqual(fresh["devices"], [5, 7])
        self.assertTrue(plan["output"].endswith("/frame_budget_smoke/bias_h8"))
        self.assertNotEqual(fresh["expected_path"], resume["expected_path"])
        changed = {key for key in fresh["expected"] if fresh["expected"][key] != resume["expected"][key]}
        self.assertEqual(changed, {"stop_after_steps", "resume_checkpoint", "expected_args"})
        self.assertEqual(fresh["expected"]["stop_after_steps"], 1)
        self.assertEqual(resume["expected"]["stop_after_steps"], 0)
        self.assertIsNone(fresh["expected"]["resume_checkpoint"])
        self.assertEqual(resume["expected"]["resume_checkpoint"], plan["output"] + "/checkpoint_last.pt")
        threshold = plan["prediction"]["steps"][2]["sample_seen"] - 1
        self.assertGreater(threshold, plan["prediction"]["steps"][1]["sample_seen"])
        for phase in plan["phases"]:
            expected = phase["expected"]
            self.assertEqual(expected["max_samples"], threshold)
            self.assertEqual(expected["sample_milestones"], str(threshold))
            self.assertEqual(expected["max_steps"], 4)
            self.assertEqual(expected["grad_accum_steps"], 8)
            self.assertEqual(expected["limit_train"], 0)
            self.assertEqual(expected["val_every"], 0)
            self.assertEqual(expected["save_every"], 0)
            self.assertFalse(expected["no_save_optimizer"])
            self.assertIsNone(expected["init_checkpoint"])
            self.assertIsNone(expected["init_dynamic_encoder_checkpoint"])
            self.assertEqual(expected["unirig_checkpoint"], self.config["baseline"]["args"]["unirig_checkpoint"])

    def test_settings_and_incompatible_full_recipe_fail_closed(self):
        for key, value in (("frame_budget", 23), ("frame_batch_cap", 0), ("frame_batch_cap", True),
                           ("max_steps", 1667), ("first_stop_after_steps", 2), ("target_stop_step", 4), ("sample_overshoot", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_settings({**settings(), key: value})
        with self.assertRaises(ValueError):
            smoke.validate_settings({**settings(), "resume_checkpoint": "old.pt"})
        for key, value in (("limit_train", 10), ("grad_accum_steps", 4), ("freeze_ar", True), ("train_surface_tokenizer", False)):
            config = deepcopy(self.config)
            config["baseline"]["args"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.build_smoke_plan(config, "/reports/fail.json")

    def test_environments_set_real_launcher_resume_variable_and_clear_old_initializers(self):
        initial = {**os.environ, "EVOWEAVE_RESUME_CHECKPOINT": "/finished/checkpoint_sample_80000.pt",
                   "EVOWEAVE_INIT_CHECKPOINT": "/finished/old_init.pt"}
        with patch("run_motion_experiment.runtime_environment", return_value=initial.copy()):
            fresh = smoke.smoke_environment(self.config, self.plan["phases"][0])
        with patch("run_motion_experiment.runtime_environment", return_value=initial.copy()):
            resume = smoke.smoke_environment(self.config, self.plan["phases"][1])
        self.assertEqual(fresh["EVOWEAVE_INIT_CHECKPOINT"], "")
        self.assertEqual(resume["EVOWEAVE_INIT_CHECKPOINT"], "")
        self.assertEqual(fresh["EVOWEAVE_RESUME_CHECKPOINT"], "")
        self.assertEqual(resume["EVOWEAVE_RESUME_CHECKPOINT"], self.plan["output"] + "/checkpoint_last.pt")
        self.assertEqual(fresh["RIGWEAVE_MAX_SAMPLES"], resume["RIGWEAVE_MAX_SAMPLES"])
        self.assertEqual(fresh["RIGWEAVE_NPROC"], "2")

    def test_expected_arguments_cover_the_entire_actual_parser(self):
        for phase in self.plan["phases"]:
            parser, _ = smoke.trainer_contract(SCRIPTS / "train_dynamic_rig.py", {})
            self.assertEqual(set(vars(parser.parse_args([]))), set(phase["expected"]))
            recorded = smoke.recorded_args(phase, 15920)
            self.assertEqual(set(recorded) - set(phase["expected"]), {"effective_batch", "train_rows", "sample_milestones_parsed"})

    @unittest.skipUnless(bash_path(), "actual Bash launcher unavailable")
    def test_actual_linux_launcher_and_argparse_match_both_fresh_and_resume(self):
        config = deepcopy(self.config)
        config["runtime"]["repo"] = ROOT.as_posix()
        plan = smoke.build_smoke_plan(config, "/reports/capture.json")
        for phase in plan["phases"]:
            phase["command"][0] = bash_path()
            with self.subTest(phase=phase["phase"]):
                with patch("run_motion_experiment.runtime_environment", return_value=dict(os.environ)):
                    environment = smoke.smoke_environment(config, phase)
                environment.update(EVOWEAVE_ROOT=(ROOT / "model_training").as_posix(),
                                   EVOWEAVE_UNIRIG_ROOT=config["runtime"]["unirig_root"], CUDA_VISIBLE_DEVICES="")
                result = smoke.verify_launcher_arguments(config, phase, environment)
                self.assertTrue(result["complete_exact_match"])
                self.assertTrue(result["two_rank_launcher"])
                self.assertEqual(result["argument_count"], len(phase["expected"]))

    def test_prepared_state_rejects_current_other_stages_even_when_train_is_allowed(self):
        config_path = ROOT / "model_training/experiments/motion_evidence_base_compare_20261010.json"
        state = {"state_id": "test", "allowed_operations": ["train"], "blocked_operations": [],
                 "active_operation": {"experiment": self.config["experiment"], "stage": "frame_budget_smoke",
                                      "candidate": "bias_h8", "status": "prepared", "config": str(config_path)}}
        self.assertEqual(smoke.validate_prepared_state(state, self.config, config_path)["stage"], "frame_budget_smoke")
        for stage in ("frame_profile", "frame_confirm", "bias_replay_audit", "full"):
            changed = deepcopy(state)
            changed["active_operation"]["stage"] = stage
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                smoke.validate_prepared_state(changed, self.config, config_path)
        state["blocked_operations"] = ["train"]
        with self.assertRaises(ValueError):
            smoke.validate_prepared_state(state, self.config, config_path)

    def test_collisions_existing_output_logs_expected_report_and_internal_aliases(self):
        with tempfile.TemporaryDirectory(prefix="frame_smoke_paths_") as directory:
            config, plan = self.local_plan(directory)
            config_path = Path(config["runtime"]["repo"]) / "config.json"
            smoke.reject_collisions(plan, config_path, config["runtime"]["repo"])
            for target in (plan["result"], *(phase[name] for phase in plan["phases"] for name in ("log", "expected_path"))):
                with self.subTest(target=target), patch.object(Path, "exists", lambda path, target=target: path.resolve() == Path(target).resolve()):
                    with self.assertRaises(FileExistsError):
                        smoke.reject_collisions(plan, config_path, config["runtime"]["repo"])
            altered = deepcopy(plan)
            altered["result"] = altered["phases"][0]["expected_path"]
            with self.assertRaises(ValueError):
                smoke.reject_collisions(altered, config_path, config["runtime"]["repo"])
            for report in (str(Path(plan["output"]) / "report.json"), str(config_path), str(Path(directory) / "wrong.pt")):
                with self.subTest(report=report), self.assertRaises(ValueError):
                    smoke.reject_collisions({**plan, "result": report}, config_path, config["runtime"]["repo"])
            Path(plan["output"]).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                smoke.reject_collisions(plan, config_path, config["runtime"]["repo"])

    def test_logged_counts_histogram_schedule_resume_and_threshold(self):
        for phase in self.plan["phases"]:
            rows = log_rows(self.plan, phase)
            evidence = smoke.validate_log_rows(rows, self.plan, phase)
            self.assertEqual(len(evidence["training_steps"]), phase["expected_final_step"])
            self.assertEqual(evidence["resume_event"] is not None, phase["phase"] == "resume")
            self.assertEqual(evidence["milestone_event"] is not None, phase["phase"] == "resume")

    def test_logged_bad_finite_metrics_counters_cursor_or_milestone_fail(self):
        phase = self.plan["phases"][1]
        original = log_rows(self.plan, phase)
        for key, value in (("loss", float("nan")), ("gradient_norm_before_clip", float("inf")),
                           ("sample_seen", 48), ("input_frames_seen", 1), ("target_token_weight_seen", 1),
                           ("frames_in_step", [2] * 8), ("micro_batch_sizes_in_step", [1] * 8)):
            rows = deepcopy(original)
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_log_rows(rows, self.plan, phase)
        for event, key, value in (("resume", "skipped_batches_in_epoch", 0),
                                   ("sample_milestone_checkpoint", "sample_milestone", 48),
                                   ("sample_milestone_checkpoint", "include_optimizer", False)):
            rows = deepcopy(original)
            next(row for row in rows if row.get("event") == event)[key] = value
            with self.subTest(event=event, key=key), self.assertRaises(ValueError):
                smoke.validate_log_rows(rows, self.plan, phase)
        rows = deepcopy(original)
        rows.insert(2, deepcopy(rows[1]))
        with self.assertRaises(ValueError):
            smoke.validate_log_rows(rows, self.plan, phase)

    def test_cpu_mmap_headers_groups_scheduler_and_strict_resume(self):
        with tempfile.TemporaryDirectory(prefix="frame_smoke_header_") as directory:
            _, plan = self.local_plan(directory)
            Path(plan["output"]).mkdir(parents=True)
            for phase in plan["phases"]:
                logs = smoke.validate_log_rows(log_rows(plan, phase), plan, phase)
                payload = checkpoint_payload(plan, phase, logs)
                checkpoint = Path(plan["output"]) / "checkpoint_last.pt"
                torch.save(payload, checkpoint)
                resume = plan["phases"][1] if phase["phase"] == "fresh" else None
                evidence = smoke.inspect_checkpoint(checkpoint, plan, phase, logs, resume)
                self.assertEqual(evidence["optimizer_group_counts"], [304, 388, 196])
                self.assertEqual(evidence["optimizer_states"], 888)
                self.assertEqual(evidence["all_adam_steps"], phase["expected_final_step"])
                self.assertEqual(evidence["scheduler"]["total_steps"], 4)
                if resume is not None:
                    self.assertTrue(evidence["strict_resume_contract_passed"])
                    bad_resume = deepcopy(resume)
                    bad_resume["expected"]["max_samples"] += 1
                    with self.assertRaises(ValueError):
                        smoke.inspect_checkpoint(checkpoint, plan, phase, logs, bad_resume)
            with patch.object(smoke.torch, "load") as load:
                with self.assertRaises(ValueError):
                    smoke.inspect_checkpoint(Path(directory) / "checkpoint_sample_80000.pt", plan, phase, logs)
                load.assert_not_called()

    def test_bad_checkpoint_headers_or_optimizer_state_do_not_pass(self):
        with tempfile.TemporaryDirectory(prefix="frame_smoke_bad_header_") as directory:
            _, plan = self.local_plan(directory)
            phase = plan["phases"][0]
            logs = smoke.validate_log_rows(log_rows(plan, phase), plan, phase)
            original = checkpoint_payload(plan, phase, logs)
            checkpoint = Path(plan["output"]) / "checkpoint_last.pt"
            checkpoint.parent.mkdir(parents=True)
            changes = [lambda value: value.update(sample_seen=48),
                       lambda value: value["batch_accounting"].update(next_batch_in_epoch=7),
                       lambda value: value["batch_accounting"].update(samples_by_frame_count={"2": 10}),
                       lambda value: value["optimizer"]["state"].pop(0),
                       lambda value: value["optimizer"]["state"][0].update(step=torch.tensor(2.0)),
                       lambda value: value["scheduler"].update(total_steps=1667),
                       lambda value: value["args"].update(frame_batch_cap=12)]
            for change in changes:
                payload = deepcopy(original)
                change(payload)
                torch.save(payload, checkpoint)
                with self.assertRaises(ValueError):
                    smoke.inspect_checkpoint(checkpoint, plan, phase, logs)

    def test_no_second_launch_after_execution_or_inspection_failure_and_no_retries(self):
        with self.assertRaises(ValueError):
            smoke.run_two_phases(self.plan["phases"] * 2, Mock(), Mock())
        for location in ("execute", "inspect"):
            execute = Mock(return_value={"child_exit": 0})
            inspect = Mock(return_value={"passed": True})
            failing = execute if location == "execute" else inspect
            failing.side_effect = RuntimeError("intentional failure")
            with self.subTest(location=location), self.assertRaises(RuntimeError):
                smoke.run_two_phases(self.plan["phases"], execute, inspect)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(inspect.call_count, int(location == "inspect"))
        execute = Mock(side_effect=[{"child_exit": 0}, RuntimeError("resume failed")])
        inspect = Mock(return_value={"passed": True})
        with self.assertRaises(RuntimeError):
            smoke.run_two_phases(self.plan["phases"], execute, inspect)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(inspect.call_count, 1)

    def test_child_nonzero_exit_retains_log_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory(prefix="frame_smoke_fail_") as directory:
            config, plan = self.local_plan(directory)
            phase = plan["phases"][0]
            with patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 9)) as run:
                with self.assertRaisesRegex(RuntimeError, "exited 9"):
                    smoke.execute_phase(config, phase, {})
                self.assertTrue(Path(phase["log"]).is_file())
                with self.assertRaises(FileExistsError):
                    smoke.execute_phase(config, phase, {})
                self.assertEqual(run.call_count, 1)

    def test_child_lock_is_distinct_and_orchestrator_never_saves_a_checkpoint(self):
        tree = ast.parse((SCRIPTS / "run_frame_budget_smoke.py").read_text(encoding="utf-8"))
        strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn(".frame_budget_smoke.lock", strings)
        self.assertNotIn(".operation.lock", strings)
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertNotIn("torch.save", calls)
        self.assertNotIn("torch.cuda.set_device", calls)
        self.assertNotIn("torch.compile", calls)


if __name__ == "__main__":
    unittest.main()
