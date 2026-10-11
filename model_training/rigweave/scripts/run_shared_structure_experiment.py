"""Run one strictly recorded shared-structure probe stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from motion_experiment_runtime import (
    guard,
    load_config,
    runtime_environment,
    source_metadata,
    verify_manifests,
    write_json,
)


STAGES = ("preflight", "cache", "probe")
TERMINAL_STATUSES = frozenset(("completed", "failed"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_PREFIX_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PROBE_SCRIPTS = frozenset(("probe_shared_structure.py", "probe_shared_structure_medium.py"))
_ONLINE_PREFLIGHT_DATA_FIELDS = (
    "seed", "model_seed", "frame_seed", "train_assets", "valid_assets", "pool_frames",
    "train_frames", "heldout_frames", "source_banks", "frames_min", "frames_max",
    "target_mesh_padding", "selection_audit_sha256",
)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _path_string(path: str | Path) -> str:
    return str(Path(path).resolve())


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA256")
    return value


def _checkpoint_identity(config: dict[str, Any]) -> dict[str, str]:
    baseline = _require_dict(config.get("baseline"), "baseline")
    path = baseline.get("checkpoint")
    if not isinstance(path, str) or not path:
        raise ValueError("baseline.checkpoint must be a non-empty path")
    digest = _require_sha256(baseline.get("checkpoint_sha256"), "baseline.checkpoint_sha256")
    return {"path": path, "sha256": digest}


def _expected_resources(config: dict[str, Any]) -> dict[str, Any]:
    runtime = _require_dict(config.get("runtime"), "runtime")
    devices = runtime.get("physical_gpus")
    if devices != [5]:
        raise ValueError(f"shared-structure operation requires physical_gpus=[5], got {devices!r}")
    if not isinstance(runtime.get("allocation"), int) or isinstance(runtime.get("allocation"), bool):
        raise ValueError("runtime.allocation must be an integer")
    if not isinstance(runtime.get("host"), str) or not runtime["host"]:
        raise ValueError("runtime.host must be a non-empty string")
    return {
        "allocation": runtime["allocation"],
        "host": runtime["host"],
        "physical_gpus": [5],
        "virtual_memory": "unlimited",
        "new_allocation": False,
    }


def validate_config(config: dict[str, Any], stage: str) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported shared-structure configuration schema")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if not isinstance(config.get("experiment"), str) or not config["experiment"]:
        raise ValueError("experiment must be a non-empty string")
    runtime = _require_dict(config.get("runtime"), "runtime")
    for key in ("python", "repo", "job_root", "output_root", "artifact_prefix"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError(f"runtime.{key} must be a non-empty string")
    if _ARTIFACT_PREFIX_RE.fullmatch(runtime["artifact_prefix"]) is None:
        raise ValueError(
            "runtime.artifact_prefix must contain only letters, numbers, '_' or '-'")
    if runtime.get("probe_script", "probe_shared_structure.py") not in _PROBE_SCRIPTS:
        raise ValueError("runtime.probe_script is not an approved shared-structure entry point")
    feature_mode = runtime.get("feature_mode", "cached")
    if feature_mode not in ("cached", "online"):
        raise ValueError("runtime.feature_mode must be 'cached' or 'online'")
    if feature_mode == "online":
        if runtime.get("probe_script") != "probe_shared_structure_medium.py" or stage != "probe":
            raise ValueError("online features require the medium probe entry point and stage='probe'")
        reference = _require_dict(runtime.get("preflight_reference"), "runtime.preflight_reference")
        if not isinstance(reference.get("path"), str) or not reference["path"]:
            raise ValueError("runtime.preflight_reference.path must be a non-empty path")
        _require_sha256(reference.get("sha256"), "runtime.preflight_reference.sha256")
    _expected_resources(config)
    _checkpoint_identity(config)
    baseline = _require_dict(config["baseline"], "baseline")
    manifests = baseline.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("baseline.manifests must be a non-empty list")
    for index, manifest in enumerate(manifests):
        manifest = _require_dict(manifest, f"baseline.manifests[{index}]")
        if not isinstance(manifest.get("argument"), str) or not manifest["argument"]:
            raise ValueError(f"baseline.manifests[{index}].argument is required")
        if not isinstance(manifest.get("path"), str) or not manifest["path"]:
            raise ValueError(f"baseline.manifests[{index}].path is required")
        _require_sha256(manifest.get("sha256"), f"baseline.manifests[{index}].sha256")
        if not isinstance(manifest.get("rows"), int) or manifest["rows"] < 0:
            raise ValueError(f"baseline.manifests[{index}].rows must be a non-negative integer")
    for section in ("preflight", "probe"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} section is required")


def stage_acceptance(config: dict[str, Any], stage: str) -> str:
    validate_config(config, stage)
    if stage == "preflight":
        return str(config["preflight"]["acceptance"])
    if stage == "probe":
        return str(config["probe"]["acceptance"])
    data = config["data"]
    return (
        "Complete the immutable per-frame surface cache with exact source, checkpoint, "
        f"manifest, cache hash, and {data['train_assets']} train/{data['valid_assets']} validation assets."
    )


def guard_operations(stage: str) -> list[str]:
    if stage == "preflight":
        return ["preflight", "train"]
    if stage == "cache":
        return ["matched_eval"]
    if stage == "probe":
        return ["train"]
    raise ValueError(f"unsupported stage {stage!r}")


def build_stage_plan(config: dict[str, Any], stage: str, config_path: str | Path) -> dict[str, Any]:
    validate_config(config, stage)
    config_path = Path(config_path).resolve()
    runtime = config["runtime"]
    artifact_prefix = runtime["artifact_prefix"]
    job_root = Path(runtime["job_root"]).resolve()
    output_dir = Path(runtime["output_root"]).resolve() / stage
    evaluation_path = job_root / "results" / f"{artifact_prefix}_{stage}_evaluation.json"
    result_path = job_root / "results" / f"{artifact_prefix}_{stage}.json"
    log_path = job_root / "logs" / f"{artifact_prefix}_{stage}.log"
    script_name = runtime.get("probe_script", "probe_shared_structure.py")
    child_script = Path(runtime["repo"]).resolve() / "model_training/rigweave/scripts" / script_name
    command = [
        runtime["python"],
        "-B",
        str(child_script),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--output",
        str(evaluation_path),
    ]
    plan = {
        "experiment": config["experiment"],
        "stage": stage,
        "artifact_prefix": artifact_prefix,
        "config": str(config_path),
        "command": command,
        "script": str(child_script),
        "job_root": str(job_root),
        "output_dir": str(output_dir),
        "evaluation_path": str(evaluation_path),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "guard_operations": guard_operations(stage),
        "resources": _expected_resources(config),
        "acceptance": stage_acceptance(config, stage),
    }
    validate_artifact_paths(plan)
    return plan


def validate_artifact_paths(plan: dict[str, Any]) -> None:
    job_root = Path(plan["job_root"]).resolve()
    for key in ("result_path", "evaluation_path", "log_path"):
        path = Path(plan[key]).resolve()
        try:
            path.relative_to(job_root)
        except ValueError as error:
            raise ValueError(f"{key} must stay under job_root: {path} is outside {job_root}") from error


def validate_launch_paths(plan: dict[str, Any]) -> None:
    validate_artifact_paths(plan)
    if not Path(plan["config"]).is_file():
        raise FileNotFoundError(f"configuration path does not exist: {plan['config']}")
    if not Path(plan["script"]).is_file():
        raise FileNotFoundError(f"child script path does not exist: {plan['script']}")
    if plan["command"][0] == "" or plan["command"][2] != plan["script"]:
        raise RuntimeError("child command does not match the recorded probe script")
    if plan["command"][4] != plan["config"] or plan["command"][6] != plan["stage"]:
        raise RuntimeError("child command does not match the recorded configuration or stage")
    if plan["command"][8] != plan["evaluation_path"]:
        raise RuntimeError("child command does not match the recorded evaluation path")


def check_collisions(plan: dict[str, Any]) -> None:
    for key in ("result_path", "evaluation_path", "log_path"):
        path = Path(plan[key])
        if path.exists():
            raise FileExistsError(f"refusing to overwrite prior shared-structure artifact: {path}")
    output_dir = Path(plan["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse shared-structure output directory: {output_dir}")


def _active_path_matches(active: dict[str, Any], key: str, expected: str) -> bool:
    value = active.get(key)
    return isinstance(value, str) and _same_path(value, expected)


def validate_source_revision(recorded: dict[str, Any], current: dict[str, Any]) -> None:
    if recorded.get("repo") != current.get("repo"):
        raise RuntimeError("source metadata repo does not match the prepared source")
    for metadata in (recorded, current):
        if not isinstance(metadata.get("git_status"), str) or metadata["git_status"].strip():
            raise RuntimeError("source metadata is dirty")
        if not isinstance(metadata.get("git_head"), str) or re.fullmatch(r"[0-9a-f]{40}", metadata["git_head"]) is None:
            raise RuntimeError("source metadata has an invalid git HEAD")
    before, after = recorded["git_head"], current["git_head"]
    if before == after:
        return
    # State receipts must be committed, but are not a change to executed code/config.
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", before, after],
                              cwd=current["repo"], capture_output=True, text=True, check=False)
    if ancestor.returncode:
        raise RuntimeError("recorded source is not an ancestor of the current checkout")
    delta = subprocess.run(
        ["git", "diff", "--name-only", before, after, "--", ".",
         ":(exclude)model_training/state/current.json"],
        cwd=current["repo"], capture_output=True, text=True, check=True,
    )
    if delta.stdout.strip():
        raise RuntimeError(f"source/config changed since the recorded stage: {delta.stdout[:1000]}")


def validate_prepared_state(
    state: dict[str, Any],
    config: dict[str, Any],
    plan: dict[str, Any],
    source: dict[str, Any],
) -> None:
    active = state.get("active_operation")
    if not isinstance(active, dict):
        raise RuntimeError("current state has no active_operation")
    expected = {
        "experiment": plan["experiment"],
        "stage": plan["stage"],
        "artifact_prefix": plan["artifact_prefix"],
        "status": "prepared",
        "command": plan["command"],
        "manifests": config["baseline"]["manifests"],
        "output_root": str(Path(config["runtime"]["output_root"]).resolve()),
        "output_path": plan["output_dir"],
        "evaluation_artifact": plan["evaluation_path"],
        "runtime_result": plan["result_path"],
        "job_root": plan["job_root"],
        "resources": plan["resources"],
        "acceptance": plan["acceptance"],
        "source_checkpoint": _checkpoint_identity(config),
    }
    for key, wanted in expected.items():
        if key in {"output_root", "output_path", "evaluation_artifact", "runtime_result", "job_root"}:
            if not _active_path_matches(active, key, wanted):
                raise RuntimeError(f"prepared state {key} does not match the requested stage")
        elif active.get(key) != wanted:
            raise RuntimeError(f"prepared state {key} does not match the requested stage")
    for key in ("reference_checkpoint", "initialization"):
        if not _active_path_matches(active, key, _checkpoint_identity(config)["path"]):
            raise RuntimeError(f"prepared state {key} does not match the configured checkpoint")
    validate_source_revision({"repo": source.get("repo"), "git_head": active.get("source_code_commit"),
                              "git_status": ""}, source)
    if not isinstance(source.get("git_status"), str) or source["git_status"].strip():
        raise RuntimeError("source worktree is not clean")
    if not isinstance(source.get("git_head"), str) or not source["git_head"]:
        raise RuntimeError("source metadata has no git HEAD")
    allowed = state.get("allowed_operations", [])
    blocked = state.get("blocked_operations", [])
    if not isinstance(allowed, list) or not isinstance(blocked, list):
        raise RuntimeError("current state operation gates are malformed")
    for operation in plan["guard_operations"]:
        if operation not in allowed or operation in blocked:
            raise RuntimeError(f"current state does not allow recorded operation {operation!r}")


def validate_launch_environment(
    config: dict[str, Any], environment: dict[str, str], *, linux: bool | None = None
) -> None:
    resources = _expected_resources(config)
    if linux is None:
        linux = sys.platform.startswith("linux")
    if linux:
        if environment.get("CUDA_VISIBLE_DEVICES") != "5":
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly recorded physical GPU 5")
        if resources["virtual_memory"] != "unlimited":
            raise RuntimeError("Linux compute-node launch requires unlimited virtual memory")


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"unable to parse {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_timestamp(report: dict[str, Any], key: str, label: str) -> None:
    if not isinstance(report.get(key), str) or not report[key]:
        raise RuntimeError(f"{label} is missing {key}")


def validate_child_report(
    report: dict[str, Any],
    config: dict[str, Any],
    stage: str,
    source: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    label = f"{stage} child evaluation"
    if report.get("experiment") != config["experiment"] or report.get("stage") != stage:
        raise RuntimeError(f"{label} experiment/stage does not match the requested operation")
    if not isinstance(report.get("complete"), bool) or not isinstance(report.get("accepted"), bool):
        raise RuntimeError(f"{label} must contain boolean complete and accepted fields")
    if require_complete and (report["complete"] is not True or report["accepted"] is not True):
        raise RuntimeError(f"{label} is not complete and accepted")
    report_source = report.get("source")
    if not isinstance(report_source, dict):
        raise RuntimeError(f"{label} is missing source metadata")
    validate_source_revision(report_source, source)
    if not isinstance(report_source.get("git_status"), str) or report_source["git_status"].strip():
        raise RuntimeError(f"{label} source metadata is dirty")
    if report_source["git_status"] != source.get("git_status"):
        raise RuntimeError(f"{label} source metadata git_status does not match the prepared source")
    if report.get("source_checkpoint") != _checkpoint_identity(config):
        raise RuntimeError(f"{label} source checkpoint identity does not match the config")
    _validate_timestamp(report, "started_at", label)
    _validate_timestamp(report, "finished_at", label)
    return report


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache_report(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cache = report.get("cache")
    if not isinstance(cache, dict):
        raise RuntimeError("cache evaluation is missing cache metadata")
    expected_path = Path(config["runtime"]["output_root"]).resolve() / "cache" / "surface_cache.pt"
    if not isinstance(cache.get("path"), str) or not _same_path(cache["path"], expected_path):
        raise RuntimeError("cache path does not match runtime.output_root/cache/surface_cache.pt")
    cache_sha = _require_sha256(cache.get("sha256"), "cache.sha256")
    data = config["data"]
    if cache.get("train_assets") != data["train_assets"] or cache.get("valid_assets") != data["valid_assets"]:
        raise RuntimeError("cache asset counts do not match the configuration")
    cache_path = Path(cache["path"])
    if not cache_path.is_file():
        raise RuntimeError(f"cache artifact is missing: {cache_path}")
    if sha256_file(cache_path) != cache_sha:
        raise RuntimeError("cache artifact checksum does not match the child report")
    return cache


def _validate_controller_identity(
    controller: dict[str, Any], plan: dict[str, Any], result_path: str | Path
) -> None:
    if controller.get("experiment") != plan["experiment"] or controller.get("stage") != plan["stage"]:
        raise RuntimeError("controller result experiment/stage does not match the requested operation")
    if controller.get("artifact_prefix") != plan["artifact_prefix"]:
        raise RuntimeError("controller result artifact prefix does not match the requested operation")
    if not isinstance(controller.get("config"), str) or not _same_path(controller["config"], plan["config"]):
        raise RuntimeError("controller result config path does not match the requested operation")
    if controller.get("command") != plan["command"]:
        raise RuntimeError("controller result command does not match the requested operation")
    if not _same_path(controller.get("result_path", ""), result_path):
        raise RuntimeError("controller result path does not match the requested artifact")
    if not _same_path(controller.get("evaluation_path", ""), plan["evaluation_path"]):
        raise RuntimeError("controller evaluation path does not match the requested artifact")
    if not _same_path(controller.get("log_path", ""), plan["log_path"]):
        raise RuntimeError("controller log path does not match the requested artifact")
    if not _same_path(controller.get("output_dir", ""), plan["output_dir"]):
        raise RuntimeError("controller output directory does not match the requested stage")


def validate_completed_reports(
    config: dict[str, Any],
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    result_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = plan["result_path"] if result_path is None else str(Path(result_path).resolve())
    controller = _load_json_object(result_path, "controller result")
    _validate_controller_identity(controller, plan, result_path)
    if controller.get("status") != "completed" or controller.get("complete") is not True:
        raise RuntimeError("controller result is not complete")
    if controller.get("controller_exit") != 0 or controller.get("child_exit") != 0:
        raise RuntimeError("completed controller result does not record two zero exits")
    evaluation = _load_json_object(plan["evaluation_path"], "child evaluation")
    validate_child_report(evaluation, config, plan["stage"], source)
    if plan["stage"] == "cache":
        validate_cache_report(evaluation, config)
    return controller, evaluation


def require_prior_stage(config: dict[str, Any], config_path: str | Path, stage: str, source: dict[str, Any]) -> None:
    prior_plan = build_stage_plan(config, stage, config_path)
    controller, evaluation = validate_completed_reports(config, prior_plan, source)
    if stage == "preflight" and evaluation.get("accepted") is not True:
        raise RuntimeError("preflight execution was not accepted")
    if stage == "cache":
        validate_cache_report(evaluation, config)
    if controller.get("child_exit") != 0:
        raise RuntimeError(f"prior {stage} child did not exit zero")


def require_online_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Validate an explicitly pinned medium preflight for online feature delivery."""
    validate_config(config, "probe")
    if config["runtime"].get("feature_mode", "cached") != "online":
        raise ValueError("require_online_preflight requires runtime.feature_mode='online'")
    reference = config["runtime"]["preflight_reference"]
    path = Path(reference["path"])
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"unable to read pinned online preflight: {path}: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if digest != reference["sha256"]:
        raise RuntimeError("online preflight SHA256 does not match the pinned reference")
    try:
        report = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(f"unable to parse pinned online preflight: {path}") from error
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("stage") != "preflight":
        raise RuntimeError("online preflight reference must be a preflight child report with schema_version=1")
    if report.get("complete") is not True or report.get("accepted") is not True:
        raise RuntimeError("online preflight reference is not complete and accepted")
    if report.get("source_checkpoint") != _checkpoint_identity(config):
        raise RuntimeError("online preflight source checkpoint does not match the current baseline")
    if report.get("source_checkpoint_unchanged") is not True:
        raise RuntimeError("online preflight did not verify an unchanged source checkpoint")
    cases, updates = report.get("preflight_cases"), report.get("update_checks")
    if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):
        raise RuntimeError("online preflight must contain non-empty preflight_cases")
    if not isinstance(updates, dict) or not updates or not all(value is True for value in updates.values()):
        raise RuntimeError("online preflight update_checks must be non-empty and all true")
    previous = _require_dict(report.get("configuration"), "online preflight configuration")
    previous_runtime = _require_dict(previous.get("runtime"), "online preflight runtime")
    if previous_runtime.get("probe_script") != "probe_shared_structure_medium.py":
        raise RuntimeError("online preflight must come from the medium probe entry point")
    architecture = _require_dict(config.get("architecture"), "architecture")
    previous_architecture = _require_dict(previous.get("architecture"), "online preflight architecture")
    if not architecture or previous_architecture != architecture or any(
        type(previous_architecture[key]) is not type(value) for key, value in architecture.items()
    ):
        raise RuntimeError("online preflight architecture differs from the current configuration")
    previous_baseline = _require_dict(previous.get("baseline"), "online preflight baseline")
    if previous_baseline.get("manifests") != config["baseline"]["manifests"]:
        raise RuntimeError("online preflight manifests differ from the current configuration")
    for section, fields in (
        ("data", _ONLINE_PREFLIGHT_DATA_FIELDS),
        ("features", ("freeze_for_feasibility", "register_temporal_mode")),
    ):
        before = _require_dict(previous.get(section), f"online preflight {section}")
        after = _require_dict(config.get(section), section)
        for key in fields:
            if key not in before or key not in after or type(before[key]) is not type(after[key]) or before[key] != after[key]:
                raise RuntimeError(f"online preflight {section}.{key} differs from the current configuration")
    # The pinned compatible receipt replaces cross-stage code ancestry only;
    # the new operation still goes through the exact prepared-state guard.
    return {
        "path": str(path), "sha256": digest, "experiment": report.get("experiment"),
        "source": report.get("source"), "source_checkpoint": report["source_checkpoint"],
        "case_count": len(cases), "update_checks": updates, "finished_at": report.get("finished_at"),
    }


def execute_stage(
    config: dict[str, Any],
    plan: dict[str, Any],
    source: dict[str, Any],
    manifests: list[dict[str, Any]],
    environment: dict[str, str],
) -> dict[str, Any]:
    Path(plan["job_root"]).mkdir(parents=True, exist_ok=True)
    Path(plan["evaluation_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(plan["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "stage": plan["stage"],
        "artifact_prefix": plan["artifact_prefix"],
        "config": plan["config"],
        "command": plan["command"],
        "output_dir": plan["output_dir"],
        "evaluation_path": plan["evaluation_path"],
        "result_path": plan["result_path"],
        "log_path": plan["log_path"],
        "source": source,
        "source_checkpoint": _checkpoint_identity(config),
        "manifests": manifests,
        "resources": plan["resources"],
        "status": "running",
        "complete": False,
        "accepted": False,
        "controller_pid": os.getpid(),
        "child_pid": None,
        "controller_exit": None,
        "child_exit": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if "online_preflight" in plan:
        result["online_preflight"] = plan["online_preflight"]
    write_json(Path(plan["result_path"]), result)
    try:
        with Path(plan["log_path"]).open("x", encoding="utf-8", buffering=1) as handle:
            handle.write("COMMAND " + json.dumps(plan["command"]) + "\n")
            child = subprocess.Popen(
                plan["command"],
                cwd=config["runtime"]["repo"],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            result["child_pid"] = child.pid
            write_json(Path(plan["result_path"]), result)
            child_exit = child.wait()
        result["child_exit"] = child_exit
        if child_exit != 0:
            raise RuntimeError(f"shared-structure child failed with exit {child_exit}; see {plan['log_path']}")
        evaluation = _load_json_object(plan["evaluation_path"], "child evaluation")
        validate_child_report(evaluation, config, plan["stage"], source)
        if plan["stage"] == "cache":
            validate_cache_report(evaluation, config)
        result.update(
            status="completed",
            complete=True,
            accepted=True,
            evaluation=evaluation,
            controller_exit=0,
        )
        return result
    except BaseException as error:
        result.update(
            status="failed",
            complete=False,
            accepted=False,
            controller_exit=1,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(Path(plan["result_path"]), result)


def run_stage(config_path: str | Path, stage: str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    plan = build_stage_plan(config, stage, config_path)
    validate_launch_paths(plan)
    source = source_metadata(config)
    manifests = verify_manifests(config)
    state_path = Path(config["runtime"]["repo"]) / "model_training/state/current.json"
    state = _load_json_object(state_path, "current state")
    validate_prepared_state(state, config, plan, source)
    if config["runtime"].get("feature_mode", "cached") == "online":
        plan["online_preflight"] = require_online_preflight(config)
    elif stage == "cache":
        require_prior_stage(config, config_path, "preflight", source)
    elif stage == "probe":
        require_prior_stage(config, config_path, "preflight", source)
        require_prior_stage(config, config_path, "cache", source)
    check_collisions(plan)
    environment = runtime_environment(config, plan["resources"]["physical_gpus"])
    validate_launch_environment(config, environment)
    for operation in plan["guard_operations"]:
        guard(config, operation)
    Path(config["runtime"]["output_root"]).mkdir(parents=True, exist_ok=True)
    return execute_stage(config, plan, source, manifests, environment)


build_plan = build_stage_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    run_stage(args.config, args.stage)


if __name__ == "__main__":
    main()
