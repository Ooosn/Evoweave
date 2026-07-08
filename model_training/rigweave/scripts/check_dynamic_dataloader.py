#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import DataLoader, Subset


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO / "rigweave" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "rigweave" / "src"))


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        try:
            parsed = shlex.split(value, posix=True)
            os.environ[key] = parsed[0] if parsed else ""
        except ValueError:
            os.environ[key] = value.strip("'\"")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() not in {"0", "false", "no", "off"}


def env_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def choose_indices(count: int, mode: str, limit: int, seed: int) -> list[int]:
    if count <= 0:
        raise ValueError("manifest has no rows")
    n = max(1, min(int(limit), count))
    if mode == "first":
        return list(range(n))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return sorted(int(x) for x in rng.choice(count, size=n, replace=False))
    if n == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, count - 1, n)})


def check_sample_contract(sample, *, root_policy: str, start_policy: str) -> None:
    parents = sample.target_parents.detach().cpu().numpy()
    root_positions = np.flatnonzero(parents < 0)
    if len(root_positions) != 1:
        raise ValueError(f"{sample.path} target has {len(root_positions)} roots after dataset mapping")
    if root_policy == "legacy" and start_policy == "joint0" and int(root_positions[0]) != 0:
        raise ValueError(f"{sample.path} rootless target root is not joint0: {int(root_positions[0])}")
    joint_count = int(sample.joint_count)
    if joint_count <= 0:
        raise ValueError(f"{sample.path} has no target joints")
    if sample.input_ids.numel() <= 0:
        raise ValueError(f"{sample.path} produced empty skeleton tokens")
    if sample.frame_vertices.shape[0] <= 1:
        raise ValueError(f"{sample.path} has too few selected frames: {sample.frame_vertices.shape}")


def summarize_batch(batch: dict) -> str:
    keys = [
        "frame_vertices",
        "input_ids",
        "target_joints",
        "target_parents",
        "selected_frames",
        "joint_count",
    ]
    parts = []
    for key in keys:
        value = batch[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else "?"
        parts.append(f"{key}={shape}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DynamicRigManifestDataset construction and collation.")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-mode", choices=["first", "linspace", "random"], default="linspace")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    if args.env_file is not None:
        load_env_file(args.env_file)

    from train_dynamic_rig import build_tokenizer
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate

    unirig_root = env_path(os.environ["EVOWEAVE_UNIRIG_ROOT"])
    tokenizer_config = env_path(os.environ.get(
        "EVOWEAVE_TOKENIZER_CONFIG",
        str(unirig_root / "configs" / "tokenizer" / "tokenizer_parts_articulationxl_256.yaml"),
    ))
    tokenizer = build_tokenizer(tokenizer_config)

    target_root_policy = os.environ.get("RIGWEAVE_TARGET_ROOT_POLICY", "legacy")
    target_start_policy = os.environ.get("RIGWEAVE_TARGET_START_POLICY", "joint0")
    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=args.frames,
        random_query=env_bool("RIGWEAVE_TRAIN_RANDOM_QUERY", True),
        seed=args.seed,
        motion_fps_ratio=env_float("RIGWEAVE_MOTION_FPS_RATIO", 0.7),
        motion_vertex_samples=env_int("RIGWEAVE_MOTION_VERTEX_SAMPLES", 512),
        target_active_skin_only=env_bool("RIGWEAVE_TARGET_ACTIVE_SKIN_ONLY", False),
        active_skin_threshold=env_float("RIGWEAVE_ACTIVE_SKIN_THRESHOLD", 1.0e-4),
        target_start_policy=target_start_policy,
        target_root_policy=target_root_policy,
    )
    indices = choose_indices(len(dataset), args.sample_mode, args.limit, args.seed)
    print(
        "[dataloader-check] "
        f"manifest={args.manifest} rows={len(dataset)} sampled={indices} "
        f"target_root_policy={target_root_policy} target_start_policy={target_start_policy}"
    )

    samples = []
    for idx in indices:
        sample = dataset[idx]
        check_sample_contract(sample, root_policy=target_root_policy, start_policy=target_start_policy)
        samples.append(sample)
        print(
            "[dataloader-check] sample "
            f"idx={idx} joints={sample.joint_count} source_joints={sample.source_joint_count} "
            f"active_joints={sample.active_skin_joint_count} vertices={sample.vertex_count} "
            f"faces={sample.face_count} selected={sample.selected_frames.tolist()} path={sample.path}"
        )

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=max(1, min(args.batch_size, len(indices))),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: dynamic_rig_collate(batch, pad_token=tokenizer.pad),
    )
    batch = next(iter(loader))
    print(f"[dataloader-check] batch {summarize_batch(batch)}")
    print("[dataloader-check] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
