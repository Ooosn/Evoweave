#!/usr/bin/env python3
"""Plan, record and launch one reviewed virtual-frame experiment stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model_training/rigweave/src"))
from motion_experiment_runtime import load_config, source_metadata, write_json
from virtual_frame_runtime import file_sha256, virtual_runtime_environment


def build_plan(config, config_name, *, mode, plan_name, candidate, label, devices, resume=None):
    if not re.fullmatch(r"[a-z0-9_-]+", label):
        raise ValueError("stage label must be a simple lowercase identifier")
    if not devices or not set(devices).issubset(config["runtime"]["physical_gpus"]) or len(set(devices)) != len(devices):
        raise ValueError("use only the explicitly granted GPU subset")
    if candidate not in config["candidates"]:
        raise ValueError("unknown architecture candidate")
    plan = config[plan_name]
    if mode == "profile":
        if not plan.get("cases") or any(not 2 <= frames <= 24 or batch < 1 for frames, batch in plan["cases"]):
            raise ValueError("capacity plan needs explicit valid T/B cases")
    elif mode == "balance":
        if len(devices) != 1 or plan["frame_budget"] < 24 or plan["batch_cap"] < 1:
            raise ValueError("balance measurements need one GPU and a measured frame budget")
    elif mode == "train":
        if not plan.get("approved") or config["optimization"]["xyz_weight_status"] != "selected_after_measurement":
            raise ValueError("training requires an approved capacity/quality plan and measured loss balance")
        if plan["frame_budget"] < 24 or plan["batch_cap"] < 1 or plan["accumulation"] < 1:
            raise ValueError("training frame-budget settings are invalid")
        endpoints = plan["evaluation_samples"] + [plan["target_samples"]]
        if any(x < 1 or x > plan["target_samples"] or x % len(devices) for x in endpoints):
            raise ValueError("evaluation/exposure endpoints must fit the run and distribute evenly across ranks")
    else:
        raise ValueError("unknown experiment mode")
    runtime = config["runtime"]
    repo = PurePosixPath(runtime["repo"])
    config_path = repo / "model_training/experiments" / config_name
    output = PurePosixPath(runtime["output_root"]) / label
    command = [runtime["python"], "-B"]
    if len(devices) > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(devices)}"]
    command += [str(repo / "model_training/rigweave/scripts/train_virtual_frame.py"),
                "--config", str(config_path), "--mode", mode, "--plan", plan_name,
                "--candidate", candidate, "--output", str(output)]
    if resume is not None:
        if mode != "train":
            raise ValueError("only virtual-frame training can resume its own checkpoint")
        command += ["--resume", str(resume)]
    return {"experiment": config["experiment"], "label": label, "mode": mode, "plan": plan_name,
            "candidate": candidate, "config": str(config_path), "output": str(output), "command": command,
            "operation": "train" if mode == "train" else "preflight",
            "resources": {"allocation": runtime["allocation"], "host": runtime["host"], "physical_gpus": devices},
            "selected_settings": plan}


def record_prepared(config, plan, config_name):
    state_path = ROOT / "model_training/state/current.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous = state.get("active_operation")
    if previous:
        state["last_operation"] = previous
    stamp = datetime.now(timezone.utc).isoformat()
    state["state_id"] = config["experiment"]
    state["updated_at"] = stamp
    state["active_operation"] = {**plan, "status": "prepared", "prepared_at": stamp,
                                 "authorization": config["authorization"], "accepted": False}
    state["allowed_operations"] = ["inspect", "report", plan["operation"]]
    state["blocked_operations"] = [x for x in ("preflight", "train", "submit", "matched_eval")
                                   if x != plan["operation"]]
    state["next_required_result"] = (
        f"Execute and inspect {plan['label']} only with its recorded source/config/data/resources. "
        "Capacity measurements are not learning-quality acceptance. Choose attention/depth using matched early exposure curves; "
        "formal training starts fresh from the official upstream and targets 80016 actual asset exposures. "
        "Never initialize from the old trained method or its descendants. Preserve the existing qlogin allocation."
    )
    contract = state.setdefault("comparison_contract", {})
    contract["current_action"] = f"User authorized the new implementation, capacity/architecture tests and subsequent formal run; current stage {plan['label']}."
    contract["localization_implementation_status"] = "virtual_frame_path_cpu_verified_pending_gpu_verification"
    contract["common_upstream_initialization"] = {
        "checkpoint": config["initialization"]["checkpoint"], "sha256": config["initialization"]["sha256"],
        "new_motion_and_head_seed": config["initialization"]["seed"], "end_to_end": True,
        "old_method_trained_weights": False,
    }
    relative = f"model_training/experiments/{config_name}"
    if relative not in state["required_context"]:
        state["required_context"].append(relative)
    write_json(state_path, state)
    return {"state": str(state_path), "prepared": plan, "updated_at": stamp}


def live_resources(config, plan):
    expected_host = config["runtime"]["host"]
    if socket.gethostname().split(".")[0] not in {expected_host, expected_host.removesuffix("i")}:
        raise RuntimeError("launch is not running on the recorded granted compute host")
    job = subprocess.check_output(["qstat", "-j", str(config["runtime"]["allocation"])], text=True)
    granted_lines = [line for line in job.splitlines() if "granted_devices" in line]
    granted = {int(x) for line in granted_lines for x in re.findall(r"/dev/nvidia(\d+)", line)}
    if not set(plan["resources"]["physical_gpus"]).issubset(granted):
        raise RuntimeError("live scheduler grant does not contain the requested physical GPUs")
    selected = ",".join(map(str, plan["resources"]["physical_gpus"]))
    status = subprocess.check_output([
        "nvidia-smi", "-i", selected, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    for line in status.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if int(fields[3]) > 256 or int(fields[4]) > 5:
            raise RuntimeError(f"granted GPU is already occupied: {line}")
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "granted_devices": granted_lines, "gpu_status": status}


def run(config, plan):
    import fcntl
    import resource

    root = Path(config["runtime"]["repo"])
    state = json.loads((root / "model_training/state/current.json").read_text())
    active = state.get("active_operation", {})
    if any(active.get(key) != plan[key] for key in ("experiment", "label", "mode", "candidate", "command", "resources")):
        raise RuntimeError("launch differs from the prepared operation")
    source = source_metadata(config)
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    job_root = Path(config["runtime"]["job_root"])
    (job_root / "logs").mkdir(parents=True, exist_ok=True)
    (job_root / "results").mkdir(exist_ok=True)
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()) and "--resume" not in plan["command"]:
        raise FileExistsError(f"refusing to overwrite a previous stage: {output}")
    with (job_root / ".gpu_work.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        live = live_resources(config, plan)
        report_path = job_root / "results" / f"{plan['label']}.json"
        report = {"status": "running", "complete": False, "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                  "plan": plan, "source": source, "resources": live, "config_sha256": file_sha256(plan["config"])}
        write_json(report_path, report)
        log_path = job_root / "logs" / f"{plan['label']}.log"
        with log_path.open("a", buffering=1) as log:
            child = subprocess.Popen(plan["command"], cwd=root, env=virtual_runtime_environment(config, plan["resources"]["physical_gpus"]),
                                     stdout=log, stderr=subprocess.STDOUT)
            report["child_pid"] = child.pid
            write_json(report_path, report)
            code = child.wait()
        report.update(status="completed" if code == 0 else "failed", complete=code == 0, exit_code=code,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(report_path, report)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "prepare", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "balance", "train"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = build_plan(config, args.config.name, mode=args.mode, plan_name=args.plan, candidate=args.candidate,
                      label=args.label, devices=args.devices, resume=args.resume)
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    if args.action == "prepare":
        print(json.dumps(record_prepared(config, plan, args.config.name), indent=2))
        return 0
    return run(config, plan)


if __name__ == "__main__":
    raise SystemExit(main())
