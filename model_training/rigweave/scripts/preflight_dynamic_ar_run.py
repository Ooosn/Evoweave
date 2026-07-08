#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def fail(message: str) -> None:
    raise SystemExit(f"[preflight] ERROR: {message}")


def env_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(path))).expanduser()


def existing_file(path: str | os.PathLike[str], label: str) -> Path:
    p = env_path(path)
    if not p.is_file():
        fail(f"{label} does not exist or is not a file: {p}")
    return p


def existing_dir(path: str | os.PathLike[str], label: str) -> Path:
    p = env_path(path)
    if not p.is_dir():
        fail(f"{label} does not exist or is not a directory: {p}")
    return p


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        fail(f"missing required environment variable {name}")
    return value


def check_cuda(require_cuda: bool, min_cuda_devices: int) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment check
        fail(f"cannot import torch: {exc}")

    print(f"[preflight] torch={getattr(torch, '__version__', 'unknown')}")
    if not require_cuda:
        return
    if not torch.cuda.is_available():
        fail("CUDA is required but torch.cuda.is_available() is false")
    count = int(torch.cuda.device_count())
    print(f"[preflight] cuda_devices={count}")
    if count < int(min_cuda_devices):
        fail(f"need at least {min_cuda_devices} CUDA device(s), got {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"[preflight] cuda:{i} name={props.name} memory_gb={props.total_memory / 1024**3:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for dynamic AR training.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-cuda-devices", type=int, default=1)
    args = parser.parse_args()

    existing_file(REPO / "rigweave" / "scripts" / "train_dynamic_rig.py", "training entry")
    existing_file(REPO / "rigweave" / "scripts" / "check_dynamic_dataloader.py", "dataloader check entry")
    existing_file(env_required("EVOWEAVE_TRAIN_MANIFEST"), "EVOWEAVE_TRAIN_MANIFEST")
    existing_file(env_required("EVOWEAVE_VAL_MANIFEST"), "EVOWEAVE_VAL_MANIFEST")
    existing_file(env_required("EVOWEAVE_UNIRIG_CKPT"), "EVOWEAVE_UNIRIG_CKPT")
    unirig_root = existing_dir(env_required("EVOWEAVE_UNIRIG_ROOT"), "EVOWEAVE_UNIRIG_ROOT")

    model_config = env_path(os.environ.get(
        "EVOWEAVE_MODEL_CONFIG",
        str(unirig_root / "configs" / "model" / "unirig_ar_350m_1024_81920_float32.yaml"),
    ))
    tokenizer_config = env_path(os.environ.get(
        "EVOWEAVE_TOKENIZER_CONFIG",
        str(unirig_root / "configs" / "tokenizer" / "tokenizer_parts_articulationxl_256.yaml"),
    ))
    existing_file(model_config, "EVOWEAVE_MODEL_CONFIG")
    existing_file(tokenizer_config, "EVOWEAVE_TOKENIZER_CONFIG")
    if os.environ.get("EVOWEAVE_INIT_CHECKPOINT"):
        existing_file(os.environ["EVOWEAVE_INIT_CHECKPOINT"], "EVOWEAVE_INIT_CHECKPOINT")

    train_rows = env_required("EVOWEAVE_TRAIN_ROWS")
    try:
        if int(train_rows) <= 0:
            fail(f"EVOWEAVE_TRAIN_ROWS must be positive, got {train_rows}")
    except ValueError:
        fail(f"EVOWEAVE_TRAIN_ROWS must be an integer, got {train_rows!r}")

    check_cuda(args.require_cuda, args.min_cuda_devices)
    print("[preflight] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
