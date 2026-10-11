from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import record_shared_structure_operation as recorder  # noqa: E402
import run_shared_structure_experiment as controller  # noqa: E402


class _FakeChild:
    def __init__(self, code: int = 0) -> None:
        self.pid = 731
        self.code = code

    def wait(self) -> int:
        return self.code


class SharedStructureControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "model_training/rigweave/scripts").mkdir(parents=True)
        (self.repo / "model_training/state").mkdir(parents=True)
        (self.repo / "model_training/rigweave/scripts/probe_shared_structure.py").write_text("# fixture\n")
        self.job = self.root / "job"
        self.outputs = self.root / "outputs"
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.manifest_train = self.root / "train.jsonl"
        self.manifest_valid = self.root / "valid.jsonl"
        self.manifest_train.write_text("train\n", encoding="utf-8")
        self.manifest_valid.write_text("valid\n", encoding="utf-8")
        self.config_path = self.root / "shared_structure.json"
        self.config = self._config()
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.plan = controller.build_stage_plan(self.config, "preflight", self.config_path)
        self.source = {"repo": str(self.repo), "git_head": "a" * 40, "git_status": ""}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _config(self) -> dict:
        return {
            "schema_version": 1,
            "experiment": "shared_structure_test",
            "baseline": {
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                "manifests": [
                    {
                        "argument": "train_manifest",
                        "path": str(self.manifest_train),
                        "sha256": hashlib.sha256(self.manifest_train.read_bytes()).hexdigest(),
                        "rows": 1,
                    },
                    {
                        "argument": "valid_manifest",
                        "path": str(self.manifest_valid),
                        "sha256": hashlib.sha256(self.manifest_valid.read_bytes()).hexdigest(),
                        "rows": 1,
                    },
                ],
            },
            "runtime": {
                "python": "/opt/venv/bin/python",
                "conda_bin": "/opt/venv/bin",
                "repo": str(self.repo),
                "job_root": str(self.job),
                "output_root": str(self.outputs),
                "artifact_prefix": "shared_structure",
                "unirig_root": "/opt/unirig",
                "opt_config_root": "/opt/opt",
                "extra_pythonpath": "/opt/extra",
                "allocation": 99,
                "host": "compute-test",
                "physical_gpus": [5],
                "memory_fraction": 0.9,
            },
            "data": {"train_assets": 8, "valid_assets": 16},
            "preflight": {"acceptance": "preflight acceptance"},
            "probe": {"acceptance": "probe acceptance"},
        }

    def _prepared_state(self, stage: str = "preflight") -> dict:
        plan = controller.build_stage_plan(self.config, stage, self.config_path)
        state = {
            "schema_version": 1,
            "unrelated": {"keep": [1, 2, 3]},
            "allowed_operations": ["inspect", "report", *plan["guard_operations"]],
            "blocked_operations": [
                operation
                for operation in ["submit", "train", "preflight", "matched_eval"]
                if operation not in plan["guard_operations"]
            ],
        }
        return recorder.prepare_state(state, self.config, plan, self.source, now="2026-10-11T00:00:00+00:00")

    def _child_report(self, stage: str, *, complete: bool = True, accepted: bool = True) -> dict:
        return {
            "experiment": self.config["experiment"],
            "stage": stage,
            "complete": complete,
            "accepted": accepted,
            "source": deepcopy(self.source),
            "source_checkpoint": controller._checkpoint_identity(self.config),
            "started_at": "2026-10-11T00:01:00+00:00",
            "finished_at": "2026-10-11T00:02:00+00:00",
        }

    def _controller_report(self, stage: str, plan: dict, *, status: str = "completed", child_exit: int = 0) -> dict:
        return {
            "experiment": self.config["experiment"],
            "stage": stage,
            "artifact_prefix": plan["artifact_prefix"],
            "config": plan["config"],
            "command": plan["command"],
            "output_dir": plan["output_dir"],
            "evaluation_path": plan["evaluation_path"],
            "result_path": plan["result_path"],
            "log_path": plan["log_path"],
            "status": status,
            "complete": status == "completed",
            "controller_exit": 0 if status == "completed" else 1,
            "child_exit": child_exit,
        }

    def _write_reports(self, stage: str, *, cache: bool = False) -> tuple[dict, dict, dict]:
        plan = controller.build_stage_plan(self.config, stage, self.config_path)
        evaluation = self._child_report(stage)
        if cache:
            cache_path = self.outputs / "cache/surface_cache.pt"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(b"cache")
            evaluation["cache"] = {
                "path": str(cache_path),
                "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "train_assets": 8,
                "valid_assets": 16,
            }
        Path(plan["evaluation_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(plan["result_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(plan["evaluation_path"]).write_text(json.dumps(evaluation), encoding="utf-8")
        controller_report = self._controller_report(stage, plan)
        Path(plan["result_path"]).write_text(json.dumps(controller_report), encoding="utf-8")
        return plan, controller_report, evaluation

    def test_plan_has_exact_child_argv_and_collision_check_does_not_create_stage(self) -> None:
        command = self.plan["command"]
        self.assertEqual(
            command,
            [
                self.config["runtime"]["python"],
                "-B",
                str(self.repo / "model_training/rigweave/scripts/probe_shared_structure.py"),
                "--config",
                str(self.config_path.resolve()),
                "--stage",
                "preflight",
                "--output",
                str(self.job / "results/shared_structure_preflight_evaluation.json"),
            ],
        )
        controller.check_collisions(self.plan)
        self.assertFalse(Path(self.plan["output_dir"]).exists())
        Path(self.plan["result_path"]).parent.mkdir(parents=True)
        Path(self.plan["result_path"]).write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            controller.check_collisions(self.plan)

    def test_artifact_prefix_is_safe_and_job_artifacts_stay_under_job_root(self) -> None:
        self.assertEqual(Path(self.plan["result_path"]).name, "shared_structure_preflight.json")
        self.assertEqual(
            Path(self.plan["evaluation_path"]).name,
            "shared_structure_preflight_evaluation.json",
        )
        self.assertEqual(Path(self.plan["log_path"]).name, "shared_structure_preflight.log")
        for prefix in ("", "bad/name", "bad.name", "bad space", ".."):
            broken = deepcopy(self.config)
            broken["runtime"]["artifact_prefix"] = prefix
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                controller.build_stage_plan(broken, "preflight", self.config_path)

        broken_plan = deepcopy(self.plan)
        broken_plan["result_path"] = str(self.root / "outside.json")
        with self.assertRaises(ValueError):
            controller.validate_artifact_paths(broken_plan)

        future = deepcopy(self.config)
        future["runtime"]["artifact_prefix"] = "shared_structure_attempt2"
        future["runtime"]["output_root"] = str(self.root / "future_outputs")
        future_plan = controller.build_stage_plan(future, "preflight", self.config_path)
        self.assertEqual(Path(future_plan["result_path"]).name, "shared_structure_attempt2_preflight.json")
        self.assertNotEqual(future_plan["output_dir"], self.plan["output_dir"])

    def test_state_receipt_commits_preserve_source_but_code_changes_do_not(self) -> None:
        import subprocess

        repo = self.root / "source_identity_repo"
        repo.mkdir()

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()

        git("init", "-q")
        git("config", "user.name", "SharedStructure Test")
        git("config", "user.email", "test@example.invalid")
        state = repo / "model_training/state/current.json"
        state.parent.mkdir(parents=True)
        state.write_text('{"status":"completed"}\n')
        code = repo / "model.py"
        code.write_text("value = 1\n")
        git("add", ".")
        git("commit", "-qm", "code")
        original = {"repo": str(repo), "git_head": git("rev-parse", "HEAD"), "git_status": ""}
        state.write_text('{"status":"prepared"}\n')
        git("add", ".")
        git("commit", "-qm", "state receipt")
        current = {**original, "git_head": git("rev-parse", "HEAD")}
        controller.validate_source_revision(original, current)
        code.write_text("value = 2\n")
        git("add", ".")
        git("commit", "-qm", "changed code")
        changed = {**current, "git_head": git("rev-parse", "HEAD")}
        with self.assertRaisesRegex(RuntimeError, "source/config changed"):
            controller.validate_source_revision(original, changed)
        with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            controller.validate_source_revision(changed, original)

    def test_prepared_state_rejects_mismatches_and_resource_drift(self) -> None:
        state = self._prepared_state()
        controller.validate_prepared_state(state, self.config, self.plan, self.source)
        for mutation in (
            {"experiment": "wrong"},
            {"stage": "cache"},
            {"status": "completed"},
            {"command": ["wrong"]},
            {"resources": {**self.plan["resources"], "allocation": 100}},
        ):
            broken = deepcopy(state)
            broken["active_operation"].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                controller.validate_prepared_state(broken, self.config, self.plan, self.source)
        dirty = deepcopy(self.source)
        dirty["git_status"] = " M file.py"
        with self.assertRaises(RuntimeError):
            controller.validate_prepared_state(state, self.config, self.plan, dirty)

    def test_recorder_preserves_unrelated_state_and_refuses_unfinished_replacement(self) -> None:
        state = {"unrelated": {"keep": True}, "active_operation": {"status": "prepared"}}
        with self.assertRaises(RuntimeError):
            recorder.prepare_state(state, self.config, self.plan, self.source)
        old = {"status": "completed", "stage": "old", "keep": "history"}
        state = {"unrelated": {"keep": True}, "active_operation": old}
        prepared = recorder.prepare_state(state, self.config, self.plan, self.source, now="now")
        self.assertEqual(prepared["unrelated"], {"keep": True})
        self.assertEqual(prepared["operation_history"], [old])
        self.assertEqual(prepared["active_operation"]["command"], self.plan["command"])
        self.assertEqual(prepared["active_operation"]["resources"], self.plan["resources"])
        self.assertEqual(prepared["active_operation"]["guard_after_record"], "begin")

    def test_recorder_completion_updates_only_operation_fields_and_records_failure(self) -> None:
        state_path = self.root / "state.json"
        prepared = self._prepared_state()
        state_path.write_text(json.dumps(prepared), encoding="utf-8")
        plan, _, _ = self._write_reports("preflight")
        with mock.patch.object(recorder, "source_metadata", return_value=self.source), mock.patch.object(
            recorder, "verify_manifests", return_value=self.config["baseline"]["manifests"]
        ), mock.patch.object(recorder, "begin_context_guard"):
            completed = recorder.record_completion(
                self.config_path,
                "preflight",
                "completed",
                state_path=state_path,
                note="accepted execution",
            )
        self.assertEqual(completed["unrelated"], {"keep": [1, 2, 3]})
        self.assertEqual(completed["active_operation"]["status"], "completed")
        self.assertTrue(completed["active_operation"]["accepted"])
        self.assertEqual(completed["allowed_operations"], ["inspect", "report"])

        failed_state = self._prepared_state()
        state_path.write_text(json.dumps(failed_state), encoding="utf-8")
        failed_controller = self._controller_report("preflight", plan, status="failed", child_exit=9)
        failed_evaluation = self._child_report("preflight", complete=False, accepted=False)
        Path(plan["result_path"]).write_text(json.dumps(failed_controller), encoding="utf-8")
        Path(plan["evaluation_path"]).write_text(json.dumps(failed_evaluation), encoding="utf-8")
        with mock.patch.object(recorder, "source_metadata", return_value=self.source), mock.patch.object(
            recorder, "verify_manifests", return_value=self.config["baseline"]["manifests"]
        ), mock.patch.object(recorder, "begin_context_guard"):
            failed = recorder.record_completion(
                self.config_path,
                "preflight",
                "failed",
                state_path=state_path,
                note="child failed",
            )
        self.assertEqual(failed["active_operation"]["status"], "failed")
        self.assertFalse(failed["active_operation"]["accepted"])
        self.assertEqual(failed["allowed_operations"], ["inspect", "report"])

    def test_prerequisites_require_exact_completed_reports_and_cache_hash(self) -> None:
        pre_plan, _, pre_eval = self._write_reports("preflight")
        controller.require_prior_stage(self.config, self.config_path, "preflight", self.source)
        cache_plan = controller.build_stage_plan(self.config, "cache", self.config_path)
        with self.assertRaises(RuntimeError):
            controller.require_prior_stage(self.config, self.config_path, "cache", self.source)
        _, _, cache_eval = self._write_reports("cache", cache=True)
        controller.require_prior_stage(self.config, self.config_path, "cache", self.source)
        cache_eval["cache"]["sha256"] = "0" * 64
        Path(cache_plan["evaluation_path"]).write_text(json.dumps(cache_eval), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            controller.require_prior_stage(self.config, self.config_path, "cache", self.source)

    def test_completion_status_requires_matching_reports(self) -> None:
        plan, _, _ = self._write_reports("preflight")
        controller_report, evaluation = controller.validate_completed_reports(
            self.config, plan, self.source
        )
        self.assertEqual(controller_report["child_exit"], 0)
        self.assertTrue(evaluation["accepted"])
        broken = self._child_report("preflight", accepted=False)
        Path(plan["evaluation_path"]).write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            controller.validate_completed_reports(self.config, plan, self.source)
        broken_controller = self._controller_report("preflight", plan)
        broken_controller["command"] = ["wrong"]
        Path(plan["evaluation_path"]).write_text(json.dumps(self._child_report("preflight")), encoding="utf-8")
        Path(plan["result_path"]).write_text(json.dumps(broken_controller), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            controller.validate_completed_reports(self.config, plan, self.source)

    def test_cache_and_probe_reports_enforce_schema_and_source_checkpoint(self) -> None:
        cache_plan, _, cache_eval = self._write_reports("cache", cache=True)
        controller.validate_completed_reports(self.config, cache_plan, self.source)
        cache_eval["source_checkpoint"]["sha256"] = "f" * 64
        Path(cache_plan["evaluation_path"]).write_text(json.dumps(cache_eval), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            controller.validate_completed_reports(self.config, cache_plan, self.source)
        probe_plan = controller.build_stage_plan(self.config, "probe", self.config_path)
        probe_eval = self._child_report("probe")
        Path(probe_plan["evaluation_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(probe_plan["evaluation_path"]).write_text(json.dumps(probe_eval), encoding="utf-8")
        Path(probe_plan["result_path"]).write_text(
            json.dumps(self._controller_report("probe", probe_plan)), encoding="utf-8"
        )
        controller.validate_completed_reports(self.config, probe_plan, self.source)

    def test_execute_captures_zero_exits_without_shell_and_preserves_failure(self) -> None:
        source = self.source
        manifests = self.config["baseline"]["manifests"]
        environment = {"CUDA_VISIBLE_DEVICES": "5"}
        evaluation = self._child_report("preflight")
        Path(self.plan["evaluation_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(self.plan["evaluation_path"]).write_text(json.dumps(evaluation), encoding="utf-8")

        with mock.patch.object(controller.subprocess, "Popen", return_value=_FakeChild(0)) as popen:
            result = controller.execute_stage(
                self.config, self.plan, source, manifests, environment
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["controller_exit"], 0)
        self.assertEqual(result["child_exit"], 0)
        self.assertEqual(popen.call_args.args[0], self.plan["command"])
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertFalse(Path(self.plan["output_dir"]).exists())
        log = Path(self.plan["log_path"])
        self.assertTrue(log.is_file())
        self.assertIn("COMMAND", log.read_text(encoding="utf-8"))

        failed_root = self.root / "failed"
        failed_config = deepcopy(self.config)
        failed_config["runtime"] = deepcopy(self.config["runtime"])
        failed_config["runtime"]["job_root"] = str(failed_root / "job")
        failed_config["runtime"]["output_root"] = str(failed_root / "outputs")
        failed_config_path = self.root / "failed.json"
        failed_config_path.write_text(json.dumps(failed_config), encoding="utf-8")
        failed_plan = controller.build_stage_plan(failed_config, "preflight", failed_config_path)
        with mock.patch.object(controller.subprocess, "Popen", return_value=_FakeChild(7)):
            with self.assertRaises(RuntimeError):
                controller.execute_stage(
                    failed_config, failed_plan, source, manifests, environment
                )
        failed_result = json.loads(Path(failed_plan["result_path"]).read_text(encoding="utf-8"))
        self.assertEqual(failed_result["child_exit"], 7)
        self.assertEqual(failed_result["status"], "failed")
        self.assertTrue(Path(failed_plan["log_path"]).is_file())

    def test_linux_environment_requires_recorded_device_and_unlimited_memory(self) -> None:
        with self.assertRaises(RuntimeError):
            controller.validate_launch_environment(self.config, {"CUDA_VISIBLE_DEVICES": "7"}, linux=True)
        controller.validate_launch_environment(self.config, {"CUDA_VISIBLE_DEVICES": "5"}, linux=True)

    def test_recording_requires_post_record_context_begin(self) -> None:
        with mock.patch.object(recorder.subprocess, "run") as run:
            recorder.begin_context_guard(self.config)
        run.assert_called_once_with(
            [
                self.config["runtime"]["python"],
                "-B",
                str(self.repo / "model_training/tools/agent_work_guard.py"),
                "begin",
            ],
            cwd=self.repo,
            check=True,
            shell=False,
        )


if __name__ == "__main__":
    unittest.main()
