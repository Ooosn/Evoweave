#!/usr/bin/env python3
"""Measure how motion amplitude propagates through the flat condition encoder."""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
sys.path.insert(0, str(RIGWEAVE_SCRIPTS))
if sys.platform == "win32" and "resource" not in sys.modules:
    resource_stub = types.ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.getrusage = lambda _: types.SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = resource_stub

from eval_dynamic_rig_ce import (  # noqa: E402
    CHECKPOINT_DEFAULTS,
    _build_dynamic_model,
    apply_checkpoint_eval_defaults,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402


def _parse_betas(value: str) -> tuple[float, ...]:
    betas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not betas or any(beta < 0.0 or beta > 1.0 for beta in betas):
        raise ValueError(f"betas must be a non-empty subset of [0,1], got {betas}")
    if 0.0 not in betas or 1.0 not in betas:
        raise ValueError("response curve requires beta=0 and beta=1")
    if len(set(betas)) != len(betas):
        raise ValueError(f"betas must be unique, got {betas}")
    return betas


def _scale_evidence_sequence(
    sequence: torch.Tensor,
    beta: float,
    *,
    normalize_vectors: bool,
) -> torch.Tensor:
    if sequence.ndim < 2:
        raise ValueError(f"sequence must have a frame dimension, got {sequence.shape}")
    scaled = sequence[:, :1] + float(beta) * (sequence - sequence[:, :1])
    if normalize_vectors:
        scaled = F.normalize(scaled, dim=-1, eps=1.0e-12)
    return scaled


def _scale_evidence_batch(batch: dict[str, Any], beta: float) -> dict[str, Any]:
    scaled = dict(batch)
    scaled["frame_vertices"] = _scale_evidence_sequence(
        batch["frame_vertices"], beta, normalize_vectors=False
    )
    for key in ("vertex_normals", "face_normals"):
        value = batch.get(key)
        if value is not None:
            scaled[key] = _scale_evidence_sequence(
                value, beta, normalize_vectors=True
            )
    return scaled


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_f = left.float().reshape(1, -1)
    right_f = right.float().reshape(1, -1)
    if float(left_f.norm().item()) <= 1.0e-12 or float(right_f.norm().item()) <= 1.0e-12:
        return None
    return float(F.cosine_similarity(left_f, right_f).item())


def _shared_energy_fraction(delta: torch.Tensor, token_dim: int) -> float:
    if delta.shape[token_dim] <= 0:
        raise ValueError("token dimension must be non-empty")
    shared = delta.mean(dim=token_dim, keepdim=True).expand_as(delta)
    return float(
        (shared.float().square().sum() / delta.float().square().sum().clamp_min(1.0e-20)).item()
    )


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _aggregate(rows: list[dict[str, Any]], betas: tuple[float, ...]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "betas": {
            str(beta): {
                key: _mean([row["response"][str(beta)][key] for row in rows])
                for key in (
                    "input_motion_ratio",
                    "frame_token_response_ratio",
                    "condition_response_ratio",
                    "frame_token_direction_cosine",
                    "condition_direction_cosine",
                    "condition_shared_energy_fraction",
                )
            }
            for beta in betas
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--betas", default="0,0.05,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    betas = _parse_betas(args.betas)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (
        args.manifest,
        args.checkpoint,
        args.tokenizer_config,
        args.model_config,
        args.unirig_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)

    model_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        unirig_checkpoint=args.unirig_checkpoint,
    )
    for name in CHECKPOINT_DEFAULTS:
        setattr(model_args, name, None)
    train_args = apply_checkpoint_eval_defaults(model_args)
    if model_args.condition_fusion != "dynamic":
        raise ValueError("response curve requires the accepted dynamic condition route")

    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate

    tokenizer = build_tokenizer(args.tokenizer_config)
    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=model_args.frames,
        limit=args.limit,
        random_query=False,
        seed=args.seed,
        motion_fps_ratio=model_args.motion_fps_ratio,
        motion_vertex_samples=model_args.motion_vertex_samples,
        target_active_skin_only=model_args.target_active_skin_only,
        active_skin_threshold=model_args.active_skin_threshold,
        target_start_policy=model_args.target_start_policy,
        target_root_policy=model_args.target_root_policy,
        input_space_policy=model_args.input_space_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = _build_dynamic_model(model_args, tokenizer, device)
    model.eval()

    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        for index, batch in enumerate(loader):
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            batch = move_batch(batch, device)
            refs = model.sample_references(batch)
            encoded: dict[float, dict[str, torch.Tensor]] = {}
            for beta in betas:
                scaled = _scale_evidence_batch(batch, beta)
                frame_tokens, query_points = model.conditioner.tokenize_frames(
                    scaled["frame_vertices"],
                    scaled["faces"],
                    refs,
                    vertex_normals=scaled.get("vertex_normals"),
                    face_normals=scaled.get("face_normals"),
                )
                condition = model.conditioner.motion_encoder(
                    frame_tokens, query_points=query_points
                )
                encoded[beta] = {
                    "input": query_points,
                    "frame_tokens": frame_tokens,
                    "condition": condition,
                }

            zero = encoded[0.0]
            full = encoded[1.0]
            full_input_delta = full["input"] - zero["input"]
            full_frame_delta = full["frame_tokens"] - zero["frame_tokens"]
            full_condition_delta = full["condition"] - zero["condition"]
            denominators = {
                "input": max(_rms(full_input_delta), 1.0e-12),
                "frame_tokens": max(_rms(full_frame_delta), 1.0e-12),
                "condition": max(_rms(full_condition_delta), 1.0e-12),
            }
            response = {}
            for beta in betas:
                current = encoded[beta]
                input_delta = current["input"] - zero["input"]
                frame_delta = current["frame_tokens"] - zero["frame_tokens"]
                condition_delta = current["condition"] - zero["condition"]
                response[str(beta)] = {
                    "input_motion_ratio": _rms(input_delta) / denominators["input"],
                    "frame_token_response_ratio": _rms(frame_delta) / denominators["frame_tokens"],
                    "condition_response_ratio": _rms(condition_delta) / denominators["condition"],
                    "frame_token_direction_cosine": _cosine(frame_delta, full_frame_delta),
                    "condition_direction_cosine": _cosine(condition_delta, full_condition_delta),
                    "condition_shared_energy_fraction": _shared_energy_fraction(
                        condition_delta, token_dim=1
                    ),
                }
            row = {
                "index": index,
                "path": str(batch["path"][0]),
                "target_joint_count": int(batch["joint_count"][0].item()),
                "selected_frames": batch["selected_frames"][0].detach().cpu().tolist(),
                "response": response,
            }
            rows.append(row)
            print(json.dumps({
                "row": f"{index + 1}/{len(dataset)}",
                "path": row["path"],
                "beta_0p10_condition_ratio": response.get("0.1", {}).get("condition_response_ratio"),
                "beta_0p25_condition_ratio": response.get("0.25", {}).get("condition_response_ratio"),
            }, sort_keys=True), flush=True)

    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    if len(rows) == 18:
        groups.update({"hard6": rows[:6], "middle6": rows[6:12], "control6": rows[12:18]})
    report = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sample_exposures": train_args.get("sample_exposures"),
            "seed": args.seed,
            "rows": len(rows),
            "betas": list(betas),
        },
        "aggregate": {name: _aggregate(group_rows, betas) for name, group_rows in groups.items()},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
