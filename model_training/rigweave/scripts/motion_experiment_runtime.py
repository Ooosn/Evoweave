"""Recorded runtime for the checkpoint-matched HGC motion comparison."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported experiment configuration")
    return config


def verify_manifests(config: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    for expected in config["baseline"]["manifests"]:
        path = Path(expected["path"])
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        rows = len(blob.splitlines())
        if digest != expected["sha256"] or rows != expected["rows"]:
            raise RuntimeError(f"historical manifest changed: {path}")
        verified.append({"path": str(path), "sha256": digest, "rows": rows})
    return verified


def runtime_environment(config: dict[str, Any], devices: list[int]) -> dict[str, str]:
    runtime = config["runtime"]
    environment = dict(os.environ)
    saved = Path(config["baseline"]["checkpoint"]).parent / "resolved_env.txt"
    for line in saved.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.startswith(("EVOWEAVE_", "RIGWEAVE_")):
            environment[key] = value
    model_root = str(Path(runtime["repo"]) / "model_training")
    environment.update({
        "PATH": runtime["conda_bin"] + os.pathsep + environment.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join((model_root + "/rigweave/src", runtime["unirig_root"], runtime["extra_pythonpath"])),
        "PYTHONDONTWRITEBYTECODE": "1",
        "EVOWEAVE_ROOT": model_root,
        "MODEL_ROOT": model_root,
        "EVOWEAVE_UNIRIG_ROOT": runtime["unirig_root"],
        "EVOWEAVE_OPT_CONFIG_ROOT": runtime["opt_config_root"],
        "EVOWEAVE_INIT_CHECKPOINT": "",
        "EVOWEAVE_RESUME_CHECKPOINT": "",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "RIGWEAVE_DISABLE_OPEN3D": "1",
        "RIGWEAVE_DISABLE_LIGHTNING_IMPORT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_MODULE_LOADING": "LAZY",
    })
    return environment


def guard(config: dict[str, Any], operation: str) -> None:
    root = Path(config["runtime"]["repo"])
    command = [config["runtime"]["python"], "-B", str(root / "model_training/tools/agent_work_guard.py")]
    log_root = Path(config["runtime"]["job_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / f"guard_{operation}.log").open("w") as handle:
        subprocess.run(command + ["begin"], cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
        subprocess.run(command + ["check", "--operation", operation], cwd=root,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def source_metadata(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["runtime"]["repo"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if status.strip():
        raise RuntimeError(f"training source worktree is not clean: {status[:2000]}")
    return {"repo": str(root), "git_head": head, "git_status": status}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
