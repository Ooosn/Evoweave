"""Prepare or close one comparison operation before committing the state gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess

from motion_experiment_runtime import load_config, write_json
from run_motion_experiment import build_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "reference", "screen", "full", "bias_diagnostic", "bias_diagnostic_paired", "bias_diagnostic_cached", "bias_replay_audit", "frame_profile", "frame_confirm", "frame_budget_smoke", "relation_preflight", "relation_screen", "relation_anchor_preflight", "relation_anchor_screen"])
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
        if args.stage in {"screen", "full", "frame_budget_smoke"}:
            candidates = list(config["screening"]["candidates"])
            selected = config["full_training"]["selected_candidate"]
            if selected:
                candidates.append(selected)
            candidate = next((value for value in candidates if value["name"] == args.candidate), None)
            if candidate is None:
                raise ValueError("candidate must be explicitly recorded in the config")
            if args.stage in {"full", "frame_budget_smoke"} and selected != candidate:
                raise ValueError("full training requires the selected candidate")
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        plan = build_plan(config, args.stage, args.candidate)
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
            "bias_diagnostic": "Fixed final checkpoint, identical per-asset inputs/references/complete GT, scales 0/1/3/10, static and misaligned-evidence controls, per-asset gradients, natural generation and exact parameter restoration. No optimizer or checkpoint mutation.",
            "bias_diagnostic_paired": "Same fixed-checkpoint diagnostic after aligning all CE forwards to the gradient-enabled path and disabling Transformer fastpath. T8 replay must pass the unchanged 2e-5 threshold before accepting scale/gradient evidence. Original first-attempt artifacts are preserved.",
            "frame_profile": "Bounded single-H100 calibration with complete GT, resident Adam states and accumulated gradients. Record finite updates and measured peak memory; stop after first OOM. Completion of profiling does not approve unsafe/unmeasured batches or a new full training.",
            "frame_confirm": "Confirm F72/cap6 for every integer T2..24 with complete stress targets, resident Adam and accumulated gradients. All measured cases must be finite and below90pct memory; no DDP or dataset-wide guarantee. Stop firstOOM; no checkpoint writes or full training.",
            "bias_replay_audit": "First two T8 validation assets only. Localize full-forward noise with captured feature/evidence tensors and CPU/CUDA RNG traces; cached path and repeat must match the actual full first forward within2e-5 CE. Record training flags and unchanged parameter versions. No optimizer, generation, or checkpoint writes.",
            "bias_diagnostic_cached": "Validated cached-motion-boundary protocol on32 assets atT8/2/24, scales0/1/3/10, static and misaligned controls,32 repeated sample gradients and64 natural generations. Every captured normal/static input must match its actual full-forward CE within2e-5; repeated baseline CE must pass same threshold. Quantify backward numerical repeat floor and restore coefficients exactly. No optimizer or checkpoint mutation.",
            "frame_budget_smoke": "Two-rank F72/cap6 smoke from fresh official initialization: step1 save, strict resume, stop after actual samples cross the predicted step3 threshold with overshoot1 under a4-step scheduler. Exact launcher args, T/B schedules, global sample/frame/token counters, cursor, milestone and optimizer/scheduler metadata must agree. Complete GT, no full training, no historical checkpoint mutation, no retries.",
            "relation_preflight": "Frozen final80k base, 2 train/2 valid assets including selected target-count extremes, normal8/weak2/static8 inputs, all complete GT. Exact zero-init condition/prefix and <=2e-5 captured CE parity; both paired adapter arms finite for 2 steps, only adapter updates, <=90pct single-H100 allocation. No full run; review before relation_screen.",
            "relation_screen": "Both identically initialized r64 adapter arms complete 120 token-weighted steps on the same 32 assets/3 views/480 exposures. Evaluate 16 disjoint validation asset IDs, complete natural generations and offline local-motion strata; compare frozen base, actual versus unknown-only adapter, and actual adapter evidence ablation. Full u/c/d and GT retained; original model/checkpoint immutable. Completion does not establish quality gain or authorize full training.",
            "relation_anchor_preflight": "Reuse hash-verified original cache for min/max-count2 train/2valid assets, complete GT, all3views. F(x,E)-F(x,U) reference-subtracted adapter must match cached base exactly at zero init and after2finiteupdates under allunknown new-branch evidence, including64token greedy prefixes. Old parameters unchanged,90pctmemory cap, no retry or fulltraining; primary reviews before anchored screen.",
            "relation_anchor_screen": "Same frozenbase, initialseed,32train/16valid cachedassets,3views and120steps/480exposures asplain probe; only change is allunknown reference subtraction. Complete48natural generations and48each unknown/permuted-E CEcontrols; trainedunknown condition must match base exactly. OriginalGT/E andacceptedcheckpoint immutable. This is exploratory same-validation follow-up, not independent confirmation or fulltraining approval.",
        }
        operation = "train" if args.stage in {"screen", "full", "frame_budget_smoke", "relation_preflight", "relation_screen", "relation_anchor_preflight", "relation_anchor_screen"} else "matched_eval" if args.stage in {"reference", "bias_diagnostic", "bias_diagnostic_paired", "bias_diagnostic_cached", "bias_replay_audit"} else "preflight"
        allowed = ["inspect", "report", operation]
        if args.stage in {"preflight", "frame_profile", "frame_confirm"}:
            allowed.append("train")  # Includes the two recorded in-memory optimizer checks.
        stage_config = config[{"frame_profile": "frame_budget_profile", "frame_confirm": "frame_budget_confirm",
                               "bias_replay_audit": "bias_replay_audit", "frame_budget_smoke": "frame_budget_smoke",
                               "relation_preflight": "relation_probe", "relation_screen": "relation_probe",
                               "relation_anchor_preflight": "relation_anchor_probe", "relation_anchor_screen": "relation_anchor_probe"}.get(args.stage, "bias_diagnostic"
                               if args.stage.startswith("bias_diagnostic") else "changes")]
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
                              "reference_checkpoint": None if args.stage == "frame_budget_smoke" else stage_config.get("checkpoint", config["baseline"]["checkpoint"]),
                              "resume_checkpoint": None,
                              "changes": stage_config,
                              "output_root": config["runtime"]["output_root"],
                              "output_path": plan["output"], "evaluation_artifact": plan["evaluation_path"],
                              "runtime_result": plan["result"],
                              "job_root": config["runtime"]["job_root"], "resources": {
                                  "allocation": config["runtime"]["allocation"], "host": config["runtime"]["host"],
                                  "physical_gpus": plan["devices"],
                                  "virtual_memory": "unlimited", "new_allocation": False},
                              "acceptance": acceptance[args.stage], "note": args.note},
            next_required_result=acceptance[args.stage],
            unknowns=[("Final-checkpoint scale/gradient diagnostics do not establish the effect of retraining with a larger bias LR."
                       if args.stage.startswith("bias_diagnostic") else "A practical head/fusion configuration was selected; full-budget generation superiority remains unverified."),
                      "Short screening is not proof of the final full-training quality or global optimum.",
                      "Historical motion-encoder initialization RNG was not saved; only initialization recipe is reproducible."],
        )
        if args.stage.startswith("relation_"):
            state["required_context"].append("model_training/docs/MOTION_RELATION_PROBE_20261011.md")
    state["updated_at"] = now
    write_json(path, state)
    print(json.dumps({"state": str(path), "stage": state["active_operation"]["stage"],
                      "status": state["active_operation"]["status"]}))


if __name__ == "__main__":
    main()
