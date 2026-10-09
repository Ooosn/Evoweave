"""Prepare or close one comparison operation before committing the state gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess

from motion_experiment_runtime import load_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "reference", "screen", "full"])
    parser.add_argument("--candidate")
    parser.add_argument("--result", choices=["accepted", "rejected", "stable_running", "completed"])
    parser.add_argument("--artifact")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    args.config = args.config.resolve()
    root = Path(__file__).resolve().parents[3]
    config = load_config(args.config)
    path = root / "model_training/state/current.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    if args.result:
        active = state["active_operation"]
        if active.get("experiment") != config["experiment"]:
            raise RuntimeError("active operation is not this comparison")
        if not args.artifact:
            raise ValueError("a result needs an evidence artifact")
        active.update(status=args.result, result_artifact=args.artifact,
                      result_note=args.note, updated_at=now)
        state["next_required_result"] = (
            "Continue the recorded full run; inspect generated skeleton quality at matched checkpoints."
            if args.result == "stable_running" else "Prepare and commit the next operation only after inspecting this result."
        )
        state["allowed_operations"] = ["inspect", "report"]
        state["blocked_operations"] = ["submit", "train", "preflight", "matched_eval"]
    else:
        if not args.stage:
            raise ValueError("prepare needs --stage")
        old = state.get("active_operation")
        if old and old.get("experiment") == config["experiment"] and old.get("status") not in {"accepted", "rejected", "completed"}:
            raise RuntimeError("close the previous operation before preparing another")
        if old:
            state.setdefault("operation_history", []).append(old)
        candidate = None
        if args.stage in {"screen", "full"}:
            candidates = list(config["screening"]["candidates"])
            selected = config["full_training"]["selected_candidate"]
            if selected:
                candidates.append(selected)
            candidate = next((value for value in candidates if value["name"] == args.candidate), None)
            if candidate is None:
                raise ValueError("candidate must be explicitly recorded in the config")
            if args.stage == "full" and selected != candidate:
                raise ValueError("full training requires the selected candidate")
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        remote_config = str(PurePosixPath(config["runtime"]["repo"]) / args.config.relative_to(root).as_posix())
        command = [config["runtime"]["python"], "-B", config["runtime"]["repo"] + "/model_training/rigweave/scripts/run_motion_experiment.py",
                   "--config", remote_config, "--stage", args.stage]
        if candidate:
            command.extend(["--candidate", candidate["name"]])
        acceptance = {
            "preflight": "Zero-init parity and actual cached-generation/teacher-forcing same-prefix checks pass; B3 T24 and T2 finite optimizer steps with all adapter gradients and memory recorded.",
            "reference": "Re-evaluate the accepted sample80000 baseline on the exact paired query/frame/surface/GT protocol; complete metrics for every planned row.",
            "screen": "120 finite optimizer steps at the original 1667-step schedule, exact expected args, complete paired CE/static/generation metrics, adapter updates and resource costs recorded.",
            "full": config["full_training"]["stability_acceptance"],
        }
        operation = "train" if args.stage in {"screen", "full"} else "matched_eval" if args.stage == "reference" else "preflight"
        allowed = ["inspect", "report", operation]
        if args.stage == "preflight":
            allowed.append("train")  # Includes the two recorded in-memory optimizer checks.
        state.update(
            state_id="motion-evidence-base-compare-20261010",
            human_context="model_training/docs/CURRENT_MODEL_CONTEXT.md",
            required_context=["PROJECT_MAP.md", "DATASET_SOURCE_OF_TRUTH.md", "model_training/docs/CURRENT_MODEL_CONTEXT.md",
                              "model_training/docs/MOTION_EVIDENCE_COMPARISON_20261010.md"],
            allowed_operations=allowed,
            blocked_operations=[value for value in ["submit", "train", "preflight", "matched_eval"] if value not in allowed],
            active_operation={"experiment": config["experiment"], "stage": args.stage, "candidate": args.candidate,
                              "candidate_config": candidate, "status": "prepared", "prepared_at": now,
                              "authorization": config["approval"], "source_code_commit": source,
                              "config": remote_config, "command": command,
                              "manifests": config["baseline"]["manifests"],
                              "initialization": config["baseline"]["args"]["unirig_checkpoint"],
                              "reference_checkpoint": config["baseline"]["checkpoint"],
                              "resume_checkpoint": None, "changes": config["changes"],
                              "output_root": config["runtime"]["output_root"],
                              "job_root": config["runtime"]["job_root"], "resources": {
                                  "allocation": config["runtime"]["allocation"], "host": config["runtime"]["host"],
                                  "physical_gpus": config["runtime"]["physical_gpus"] if operation == "train" else config["runtime"]["physical_gpus"][:1],
                                  "virtual_memory": "unlimited", "new_allocation": False},
                              "acceptance": acceptance[args.stage], "note": args.note},
            next_required_result=acceptance[args.stage],
            unknowns=["Best practical head count and fusion mode await paired short-run measurements.",
                      "Short screening is not proof of the final full-training quality or global optimum.",
                      "Historical motion-encoder initialization RNG was not saved; only initialization recipe is reproducible."],
        )
    state["updated_at"] = now
    write_json(path, state)
    print(json.dumps({"state": str(path), "stage": state["active_operation"]["stage"],
                      "status": state["active_operation"]["status"]}))


if __name__ == "__main__":
    main()
