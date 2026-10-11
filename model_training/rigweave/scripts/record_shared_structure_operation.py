"""Prepare or close one shared-structure operation in current.json."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from motion_experiment_runtime import load_config, runtime_environment, source_metadata, verify_manifests, write_json
from run_shared_structure_experiment import (
    STAGES,
    TERMINAL_STATUSES,
    _checkpoint_identity,
    _load_json_object,
    build_stage_plan,
    guard_operations,
    _validate_controller_identity,
    validate_config,
    validate_child_report,
    validate_prepared_state,
    validate_completed_reports,
)


def _allowed_operations(stage: str, *, terminal: bool = False) -> tuple[list[str], list[str]]:
    all_operations = ["submit", "train", "preflight", "matched_eval"]
    if terminal:
        allowed = ["inspect", "report"]
    elif stage == "preflight":
        allowed = ["inspect", "report", "preflight", "train"]
    elif stage in {"cache", "assessment"}:
        allowed = ["inspect", "report", "matched_eval"]
    elif stage in {"probe", "full"}:
        allowed = ["inspect", "report", "train"]
    else:
        raise ValueError(f"unsupported stage {stage!r}")
    blocked = [operation for operation in all_operations if operation not in allowed]
    return allowed, blocked


def build_prepared_operation(
    config: dict[str, Any],
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    note: str = "",
    now: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat() if now is None else now
    checkpoint = _checkpoint_identity(config)
    return {
        "experiment": plan["experiment"],
        "stage": plan["stage"],
        "artifact_prefix": plan["artifact_prefix"],
        "authorization": config.get("authorization", ""),
        "status": "prepared",
        "prepared_at": now,
        "config": plan["config"],
        "command": deepcopy(plan["command"]),
        "source_code_commit": source["git_head"],
        "source_metadata": deepcopy(source),
        "manifests": deepcopy(config["baseline"]["manifests"]),
        "checkpoint": checkpoint["path"],
        "source_checkpoint": checkpoint,
        "initialization": checkpoint["path"],
        "reference_checkpoint": checkpoint["path"],
        **({"warmstart": deepcopy(config["full"]["warmstart"]),
            "initialization": config["full"]["warmstart"]["path"],
            "training_configuration": deepcopy(config["full"])} if plan["stage"] == "full" else {}),
        "output_root": str(Path(config["runtime"]["output_root"]).resolve()),
        "output_path": plan["output_dir"],
        "evaluation_artifact": plan["evaluation_path"],
        "runtime_result": plan["result_path"],
        "job_root": plan["job_root"],
        "resources": deepcopy(plan["resources"]),
        "guard_operations": guard_operations(plan["stage"]),
        "guard_after_record": "begin",
        "acceptance": plan["acceptance"],
        "note": note,
    }


def prepare_state(
    state: dict[str, Any],
    config: dict[str, Any],
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    note: str = "",
    now: str | None = None,
) -> dict[str, Any]:
    old = state.get("active_operation")
    if old is not None and not isinstance(old, dict):
        raise RuntimeError("current state active_operation is malformed")
    if old is not None and old.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("an unfinished active operation must be completed or failed before preparing another stage")
    updated = deepcopy(state)
    if old is not None:
        updated.setdefault("operation_history", []).append(deepcopy(old))
    active = build_prepared_operation(config, plan, source, note=note, now=now)
    allowed, blocked = _allowed_operations(plan["stage"])
    updated["active_operation"] = active
    updated["allowed_operations"] = allowed
    updated["blocked_operations"] = blocked
    updated["next_required_result"] = plan["acceptance"]
    updated["updated_at"] = active["prepared_at"]
    return updated


def _validate_active_for_completion(
    state: dict[str, Any], config: dict[str, Any], plan: dict[str, Any], source: dict[str, Any]
) -> None:
    active = state.get("active_operation")
    if not isinstance(active, dict) or active.get("status") != "prepared":
        raise RuntimeError("completion requires the matching operation to remain prepared")
    validate_prepared_state(state, config, plan, source)


def begin_context_guard(config: dict[str, Any]) -> None:
    root = Path(config["runtime"]["repo"])
    command = [
        config["runtime"]["python"],
        "-B",
        str(root / "model_training/tools/agent_work_guard.py"),
        "begin",
    ]
    subprocess.run(command, cwd=root, check=True, shell=False)


def complete_state(
    state: dict[str, Any],
    config: dict[str, Any],
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    status: str,
    controller: dict[str, Any],
    evaluation: dict[str, Any] | None,
    result_path: str,
    note: str = "",
    now: str | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError(f"completion status must be completed or failed, got {status!r}")
    _validate_active_for_completion(state, config, plan, source)
    now = datetime.now(timezone.utc).isoformat() if now is None else now
    updated = deepcopy(state)
    active = updated["active_operation"]
    active.update(
        status=status,
        result_artifact=result_path,
        result_note=note,
        controller_exit=controller.get("controller_exit"),
        child_exit=controller.get("child_exit"),
        complete=status == "completed",
        accepted=status == "completed",
        updated_at=now,
    )
    if evaluation is not None:
        active["child_report"] = {
            "complete": evaluation.get("complete"),
            "accepted": evaluation.get("accepted"),
            "source_checkpoint": evaluation.get("source_checkpoint"),
        }
    allowed, blocked = _allowed_operations(plan["stage"], terminal=True)
    updated["allowed_operations"] = allowed
    updated["blocked_operations"] = blocked
    updated["next_required_result"] = (
        "No automatic retry; inspect the failed shared-structure stage and its preserved log."
        if status == "failed"
        else {
            "preflight": "Prepare cache only after the accepted preflight evaluation is reviewed.",
            "cache": "Prepare probe only after the complete cache evaluation and checksum are reviewed.",
            "probe": "Inspect the completed execution; this is not model promotion or quality approval.",
            "assessment": "Review held-out generation/reuse and physical-batch throughput before authorizing any training continuation.",
        }[plan["stage"]]
    )
    updated["updated_at"] = now
    return updated


def record_prepared(
    config_path: str | Path, stage: str, *, note: str = "", state_path: str | Path | None = None
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_config(config, stage)
    plan = build_stage_plan(config, stage, config_path)
    source = source_metadata(config)
    verify_manifests(config)
    runtime_environment(config, plan["resources"]["physical_gpus"])
    if source.get("git_status", "").strip():
        raise RuntimeError("source worktree is not clean")
    if state_path is None:
        state_path = Path(config["runtime"]["repo"]) / "model_training/state/current.json"
    state = _load_json_object(state_path, "current state")
    updated = prepare_state(state, config, plan, source, note=note)
    write_json(Path(state_path), updated)
    begin_context_guard(config)
    return updated


def record_completion(
    config_path: str | Path,
    stage: str,
    status: str,
    *,
    result_path: str | Path | None = None,
    note: str = "",
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_config(config, stage)
    plan = build_stage_plan(config, stage, config_path)
    source = source_metadata(config)
    verify_manifests(config)
    if state_path is None:
        state_path = Path(config["runtime"]["repo"]) / "model_training/state/current.json"
    state = _load_json_object(state_path, "current state")
    _validate_active_for_completion(state, config, plan, source)
    result_path = plan["result_path"] if result_path is None else str(Path(result_path).resolve())
    if status == "completed":
        controller, evaluation = validate_completed_reports(
            config, plan, source, result_path=result_path
        )
    elif status == "failed":
        controller = _load_json_object(result_path, "controller result")
        _validate_controller_identity(controller, plan, result_path)
        if controller.get("status") != "failed" or controller.get("complete") is not False:
            raise RuntimeError("failed recording requires a failed, incomplete controller result")
        if controller.get("child_exit") == 0 and controller.get("controller_exit") == 0:
            raise RuntimeError("failed controller result does not record a failing exit or validation error")
        evaluation = None
        evaluation_path = Path(plan["evaluation_path"])
        if evaluation_path.is_file():
            evaluation = _load_json_object(evaluation_path, "failed child evaluation")
            validate_child_report(
                evaluation,
                config,
                plan["stage"],
                source,
                require_complete=False,
            )
    else:
        raise ValueError(f"unsupported completion status {status!r}")
    updated = complete_state(
        state,
        config,
        plan,
        source,
        status=status,
        controller=controller,
        evaluation=evaluation,
        result_path=result_path,
        note=note,
    )
    write_json(Path(state_path), updated)
    begin_context_guard(config)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--status", choices=["prepared", "completed", "failed"], required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if args.status == "prepared":
        state = record_prepared(args.config, args.stage, note=args.note)
    else:
        state = record_completion(
            args.config,
            args.stage,
            args.status,
            result_path=args.result,
            note=args.note,
        )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": state["active_operation"]["status"],
                "guard_after_record": "begin",
                "guard_operations": guard_operations(args.stage),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
