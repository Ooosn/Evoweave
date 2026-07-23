#!/usr/bin/env python3
"""Measure whether the trained evidence adapter performs localized reads."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_ROOT = SCRIPT_DIR.parent
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
if str(RIGWEAVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from eval_dynamic_rig_ce import (  # noqa: E402
    CHECKPOINT_DEFAULTS,
    _build_dynamic_model,
    apply_checkpoint_eval_defaults,
)
from preflight_motion_evidence_learnability import (  # noqa: E402
    build_dataset,
    make_loader,
    set_probe_modes,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402

from rigweave.motion_evidence import TopologyMotionEvidenceUniRigAR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@torch.no_grad()
def one_usage(
    model: TopologyMotionEvidenceUniRigAR,
    batch: dict[str, Any],
    *,
    seed: int,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    device = batch["frame_vertices"].device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    refs = model.sample_references(batch)
    generator = torch.Generator(device=device).manual_seed(seed + 100000)
    with torch.autocast("cuda", dtype=amp_dtype):
        memory = model.build_memory(batch, refs=refs)
        teacher = model.teacher_forcing(batch, memory=memory)
        boundary = model.boundary_auxiliary_loss(batch, refs, memory)
        positions = torch.arange(
            teacher.token_hidden.shape[1],
            device=teacher.token_hidden.device,
        )
        normal_refined = model.evidence_adapter.refine_hidden(
            teacher.token_hidden,
            memory,
            positions,
        )
        corrupted = memory.controlled(
            "corrupt_correspondence",
            generator=generator,
        )
        corrupt_refined = model.evidence_adapter.refine_hidden(
            teacher.token_hidden,
            corrupted,
            positions,
        )

    attention = model.evidence_adapter.attention
    compute_dtype = attention.query_norm.weight.dtype
    with torch.autocast(device_type="cuda", enabled=False):
        query = attention.query_norm(teacher.token_hidden.to(dtype=compute_dtype))
        keys = attention.key_norm(memory.static_tokens.to(dtype=compute_dtype))
        values = attention.value_norm(memory.motion_values.to(dtype=compute_dtype))
        _, weights = attention.cross_attention(
            query,
            keys,
            values,
            need_weights=True,
            average_attn_weights=False,
        )

    static_steps = model.evidence_adapter.static_prefix_steps
    later_weights = weights[:, :, static_steps:].float()
    entropy = -(later_weights * later_weights.clamp_min(1.0e-12).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(later_weights.shape[-1])
    top_values = torch.topk(later_weights, k=16, dim=-1).values

    motion_values = memory.motion_values.float()
    value_centered = motion_values - motion_values.mean(dim=1, keepdim=True)
    value_spatial_rms = torch.sqrt(value_centered.square().mean())
    value_total_rms = torch.sqrt(motion_values.square().mean()).clamp_min(1.0e-12)
    boundary_logits = memory.boundary_logits.float()
    boundary_spatial_rms = torch.sqrt(
        (boundary_logits - boundary_logits.mean(dim=1, keepdim=True)).square().mean()
    )

    later_delta = normal_refined[:, static_steps:].float() - teacher.token_hidden[:, static_steps:].float()
    corrupt_delta = corrupt_refined[:, static_steps:].float() - teacher.token_hidden[:, static_steps:].float()
    delta_rms = torch.sqrt(later_delta.square().mean()).clamp_min(1.0e-12)
    shared = later_delta.mean(dim=1, keepdim=True)
    shared_energy_ratio = shared.square().mean() / later_delta.square().mean().clamp_min(1.0e-24)
    correspondence_delta_ratio = torch.sqrt(
        (later_delta - corrupt_delta).square().mean()
    ) / delta_rms
    root_delta = (
        normal_refined[:, :static_steps].float()
        - teacher.token_hidden[:, :static_steps].float()
    ).abs().max()
    return {
        "confidence": float(memory.confidence[0]),
        "motion_q90_rms": float(memory.raw_evidence.example_motion_q90_rms[0]),
        "boundary_loss": float(boundary["boundary_loss"]),
        "boundary_mae": float(boundary["boundary_mae"]),
        "value_spatial_rms": float(value_spatial_rms),
        "value_spatial_to_total_rms": float(value_spatial_rms / value_total_rms),
        "boundary_logit_spatial_rms": float(boundary_spatial_rms),
        "attention_normalized_entropy": float(normalized_entropy.mean()),
        "attention_effective_keys": float(torch.exp(entropy).mean()),
        "attention_top1_mass": float(top_values[..., 0].mean()),
        "attention_top16_mass": float(top_values.sum(dim=-1).mean()),
        "later_update_rms": float(delta_rms),
        "token_shared_update_energy_ratio": float(shared_energy_ratio),
        "correct_corrupt_update_delta_ratio": float(correspondence_delta_ratio),
        "root_hidden_max_abs_delta": float(root_delta),
    }


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    apply_checkpoint_eval_defaults(args)
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    tokenizer = build_tokenizer(args.tokenizer_config)
    dataset = build_dataset(
        args,
        args.valid_manifest,
        tokenizer,
        limit=args.limit,
        seed=args.seed,
    )
    loader = make_loader(dataset, tokenizer, shuffle=False, seed=args.seed)
    baseline = _build_dynamic_model(args, tokenizer, device)
    model = TopologyMotionEvidenceUniRigAR(
        baseline.unirig_ar,
        baseline.conditioner.surface_tokenizer,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        evidence_heads=8,
        evidence_static_prefix_steps=4,
    )
    model.conditioner.value_encoder.to(device)
    model.evidence_adapter.to(device)
    del baseline
    gc.collect()
    payload = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
    model.conditioner.value_encoder.load_state_dict(payload["value_encoder"], strict=True)
    model.evidence_adapter.load_state_dict(payload["evidence_adapter"], strict=True)
    set_probe_modes(model, training=False)

    rows: list[dict[str, Any]] = []
    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        usage = one_usage(
            model,
            batch,
            seed=args.seed + row_index,
            amp_dtype=amp_dtype,
        )
        usage["index"] = row_index
        usage["path"] = raw_batch["path"][0]
        rows.append(usage)
        print(json.dumps(usage), flush=True)

    metric_names = [
        key
        for key in rows[0]
        if key not in {"index", "path"}
    ]
    summary = {
        "rows": len(rows),
        "checkpoint": str(args.checkpoint),
        "probe_checkpoint": str(args.probe_checkpoint),
        "metrics": {
            name: summarize([float(row[name]) for row in rows])
            for name in metric_names
        },
        "root_exact": all(row["root_hidden_max_abs_delta"] == 0.0 for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "details": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
