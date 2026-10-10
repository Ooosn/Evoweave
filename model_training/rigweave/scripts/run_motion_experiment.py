"""Run exactly one committed stage of the checkpoint-matched HGC comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

from motion_experiment_runtime import guard, load_config, runtime_environment, source_metadata, verify_manifests, write_json


DERIVED_ARGS = {"effective_batch", "train_rows", "sample_milestones_parsed"}


def build_plan(config, stage, candidate_name):
    runtime = config["runtime"]
    root = PurePosixPath(runtime["repo"])
    job = PurePosixPath(runtime["job_root"])
    key = stage + ("_" + candidate_name if candidate_name else "")
    candidate = None
    training = stage in {"screen", "full"}
    smoke = stage == "frame_budget_smoke"
    relation = stage in {"relation_preflight", "relation_screen", "relation_anchor_preflight", "relation_anchor_screen"}
    if training or smoke:
        options = config["screening"]["candidates"]
        if stage in {"full", "frame_budget_smoke"}:
            selected = config["full_training"]["selected_candidate"]
            options = [selected] if selected else []
        candidate = next((value for value in options if value["name"] == candidate_name), None)
        if candidate is None:
            raise ValueError("candidate must be explicitly recorded for this stage")
    elif candidate_name is not None:
        raise ValueError("preflight/reference do not take a candidate")
    scripts = root / "model_training/rigweave/scripts"
    output = PurePosixPath(runtime["output_root"]) / stage / str(candidate_name)
    expected_path = job / "results" / (key + "_expected_args.json")
    evaluation_path = job / "results" / (key + "_evaluation.json")
    command = [runtime["python"], "-B"]
    config_path = root / "model_training/experiments/motion_evidence_base_compare_20261010.json"
    expected = None
    if stage == "preflight":
        evaluation_path = job / "results/preflight_checks.json"
        command += [str(scripts / "preflight_motion_evidence.py"), "--config", str(config_path), "--output", str(evaluation_path)]
    elif stage == "reference":
        command += [str(scripts / "evaluate_motion_candidate.py"), "--config", str(config_path),
                    "--checkpoint", config["baseline"]["checkpoint"], "--output", str(evaluation_path)]
    elif stage in {"bias_diagnostic", "bias_diagnostic_paired", "bias_diagnostic_cached"}:
        command += [str(scripts / "diagnose_motion_bias.py"), "--config", str(config_path),
                    "--output", str(evaluation_path)]
    elif stage == "bias_replay_audit":
        command += [str(scripts / "audit_bias_replay.py"), "--config", str(config_path),
                    "--output", str(evaluation_path)]
    elif stage in {"frame_profile", "frame_confirm"}:
        command += [str(scripts / "profile_frame_budget.py"), "--config", str(config_path),
                    "--output", str(evaluation_path), "--plan-key",
                    "frame_budget_confirm" if stage == "frame_confirm" else "frame_budget_profile"]
    elif smoke:
        command += [str(scripts / "run_frame_budget_smoke.py"), "--config", str(config_path),
                    "--output", str(evaluation_path)]
    elif relation:
        anchored = stage.startswith("relation_anchor_")
        output = PurePosixPath(config["relation_anchor_probe" if anchored else "relation_probe"]["output_root"]) / stage
        command += [str(scripts / ("run_relation_anchor.py" if anchored else "run_relation_probe.py")), "--config", str(config_path),
                    "--stage", stage, "--output", str(evaluation_path)]
    else:
        expected = {key: value for key, value in config["baseline"]["args"].items() if key not in DERIVED_ARGS}
        expected.update(frame_budget=0, frame_batch_cap=0, max_samples=0)
        expected.update(output_dir=str(output), resume_checkpoint=None, init_checkpoint=None,
                        frames=config["changes"]["frames_max"], frames_min=config["changes"]["frames_min"],
                        motion_fps_ratio=config["changes"]["motion_fps_ratio"],
                        minimum_random_frames=config["changes"]["minimum_random_frames"],
                        initialization_seed=config["changes"]["initialization_seed"],
                        motion_evidence_fusion=candidate["fusion"], motion_evidence_heads=candidate["heads"],
                        stop_after_steps=config["screening"]["optimizer_steps"] if stage == "screen" else 0,
                        no_save_optimizer=stage == "screen", expected_args=str(expected_path))
        if stage == "screen":
            expected["sample_milestones"] = "80000"
        command = ["bash", str(scripts / "run_dynamic_ar_train.sh")]
    return {"stage": stage, "candidate": candidate_name, "key": key, "command": command,
            "output": str(output) if training or smoke or relation else None, "expected": expected,
            "expected_path": str(expected_path), "evaluation_path": str(evaluation_path),
            "log": str(job / "logs" / (key + ".log")),
            "result": str(job / "results" / (key + ".json")),
            "devices": runtime["physical_gpus"] if training or smoke else runtime["physical_gpus"][:1]}


def training_environment(config, plan):
    environment = runtime_environment(config, plan["devices"])
    expected = plan["expected"]
    if expected is None:
        return environment
    for key, value in expected.items():
        if value is not None:
            environment["RIGWEAVE_" + key.upper()] = str(int(value)) if isinstance(value, bool) else str(value)
    environment.update({
        "EVOWEAVE_TRAIN_MANIFEST": expected["train_manifest"],
        "EVOWEAVE_VAL_MANIFEST": expected["val_manifest"],
        "EVOWEAVE_TRAIN_ROWS": str(config["baseline"]["args"]["train_rows"]),
        "EVOWEAVE_UNIRIG_CKPT": expected["unirig_checkpoint"],
        "EVOWEAVE_MODEL_CONFIG": expected["model_config"],
        "EVOWEAVE_TOKENIZER_CONFIG": expected["tokenizer_config"],
        "EVOWEAVE_OUTPUT_DIR": expected["output_dir"],
        "RIGWEAVE_GRAD_ACCUM": str(expected["grad_accum_steps"]),
        "RIGWEAVE_NPROC": str(config["full_training"]["world_size"]),
        "RIGWEAVE_PREFLIGHT_ONLY": "0", "RIGWEAVE_SKIP_PREFLIGHT": "0",
        "RIGWEAVE_SKIP_DATALOADER_CHECK": "1", "RIGWEAVE_REQUIRE_CUDA": "1",
    })
    return environment


def require_report(path, field):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get(field) is not True:
        raise RuntimeError(f"required report did not pass: {path}")
    return report


def execute(command, config, environment, log_path, result):
    with Path(log_path).open("a", buffering=1) as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        child = subprocess.Popen(command, cwd=config["runtime"]["repo"], env=environment,
                                 stdout=handle, stderr=subprocess.STDOUT)
        result["child_pid"] = child.pid
        write_json(Path(result["result"]), result)
        code = child.wait()
    result["child_exit"] = code
    if code:
        raise RuntimeError(f"stage command failed with exit {code}; see {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "reference", "screen", "full", "bias_diagnostic", "bias_diagnostic_paired", "bias_diagnostic_cached", "bias_replay_audit", "frame_profile", "frame_confirm", "frame_budget_smoke", "relation_preflight", "relation_screen", "relation_anchor_preflight", "relation_anchor_screen"], required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    plan = build_plan(config, args.stage, args.candidate)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    import fcntl
    job = Path(config["runtime"]["job_root"])
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / ".operation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((Path(config["runtime"]["repo"]) / "model_training/state/current.json").read_text())
    active = state["active_operation"]
    wanted = {"experiment": config["experiment"], "stage": args.stage,
              "candidate": args.candidate, "status": "prepared"}
    if any(active.get(key) != value for key, value in wanted.items()):
        raise RuntimeError("requested stage does not match the committed prepared operation")
    if Path(active["config"]).resolve() != args.config.resolve():
        raise RuntimeError("requested configuration does not match the recorded path")
    source = source_metadata(config)
    manifests = verify_manifests(config)
    for path in (plan["result"], plan["log"], plan["evaluation_path"], plan["expected_path"]):
        if Path(path).exists():
            raise FileExistsError(f"refusing to overwrite prior operation artifact: {path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    if plan["output"] and Path(plan["output"]).exists():
        raise FileExistsError(f"refusing to reuse training directory: {plan['output']}")
    if args.stage != "preflight":
        require_report(job / "results/preflight_checks.json", "passed")
    if args.stage in {"screen", "full"}:
        require_report(job / "results/reference_evaluation.json", "complete")
    if args.stage == "full":
        require_report(job / f"results/screen_{args.candidate}_evaluation.json", "complete")
    if args.stage in {"bias_diagnostic", "bias_diagnostic_paired", "bias_diagnostic_cached", "bias_replay_audit", "frame_profile", "frame_confirm", "relation_preflight", "relation_screen", "relation_anchor_preflight", "relation_anchor_screen"}:
        require_report(job / "results/full_bias_h8_completion.json", "passed")
    if args.stage == "relation_screen":
        checked = require_report(job / "results/relation_preflight_evaluation.json", "complete")
        if checked["plan"] != config["relation_probe"] or checked["changed_base_parameters"]:
            raise RuntimeError("relation probe preflight does not match the recorded plan")
    if args.stage == "relation_anchor_preflight":
        require_report(job / "results/relation_screen_evaluation.json", "complete")
    if args.stage == "relation_anchor_screen":
        checked = require_report(job / "results/relation_anchor_preflight_evaluation.json", "complete")
        if checked["plan"] != config["relation_anchor_probe"] or checked["changed_base_parameters"]:
            raise RuntimeError("anchored preflight does not match the recorded plan")
    if args.stage == "bias_diagnostic_cached":
        require_report(job / "results/bias_replay_audit_evaluation.json", "complete")
    if args.stage == "frame_budget_smoke":
        confirmation = require_report(job / "results/frame_confirm_evaluation.json", "complete")
        approved_cap = confirmation["summary"].get("full_t2_t24_global_cap")
        if (confirmation["plan"]["frame_budget"] != config["frame_budget_smoke"]["frame_budget"]
                or approved_cap != config["frame_budget_smoke"]["frame_batch_cap"]):
            raise RuntimeError("frame-budget smoke exceeds the all-frame calibrated schedule")
    query = ["nvidia-smi", "-i", ",".join(map(str, plan["devices"])),
             "--query-compute-apps=pid", "--format=csv,noheader"]
    if subprocess.check_output(query, text=True).strip():
        raise RuntimeError("an allocated GPU already has a compute process; inspect before retrying")
    operation = "train" if plan["expected"] or args.stage in {"frame_budget_smoke", "relation_preflight", "relation_screen", "relation_anchor_preflight", "relation_anchor_screen"} else "preflight" if args.stage in {"preflight", "frame_profile", "frame_confirm"} else "matched_eval"
    guard(config, operation)
    environment = training_environment(config, plan)
    if plan["expected"]:
        write_json(Path(plan["expected_path"]), plan["expected"])
    result = {**plan, "source": source, "manifests": manifests, "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "controller_pid": os.getpid(),
              "contract_note": "All 16776 manifest paths were audited. The dedicated motion preflight replaces repeated dataloader smoke checks; original launch path/CUDA preflight stays enabled."}
    write_json(Path(plan["result"]), result)
    print(json.dumps({"status": "running", "key": plan["key"], "log": plan["log"]}), flush=True)
    start = time.monotonic()
    try:
        execute(plan["command"], config, environment, plan["log"], result)
        if plan["expected"]:
            actual = json.loads((Path(plan["output"]) / "args.json").read_text())
            delta = {key: [value, actual.get(key)] for key, value in plan["expected"].items() if actual.get(key) != value}
            if delta:
                raise RuntimeError(f"saved training args differ: {delta}")
        if args.stage == "screen":
            checkpoint = Path(plan["output"]) / "checkpoint_last.pt"
            command = [config["runtime"]["python"], "-B", str(Path(__file__).with_name("evaluate_motion_candidate.py")),
                       "--config", str(args.config), "--checkpoint", str(checkpoint), "--output", plan["evaluation_path"]]
            execute(command, config, runtime_environment(config, plan["devices"][:1]), plan["log"], result)
        if args.stage != "full":
            require_report(plan["evaluation_path"], "passed" if args.stage == "preflight" else "complete")
        result["status"] = "completed"
        result["complete"] = True
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(Path(plan["result"]), result)
    print(json.dumps({"status": "completed", "key": plan["key"], "result": plan["result"]}), flush=True)


if __name__ == "__main__":
    main()
