#!/usr/bin/env python3
"""Test an oracle prefix-coverage mask without changing trained weights.

Ground-truth skin support is used only to remove anchors already explained by
the ground-truth prefix.  A zero key/value slot lets the evidence read abstain
when no uncovered support remains.  This is a causal diagnostic, not an
inference implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_ROOT = SCRIPT_DIR.parent
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
if str(RIGWEAVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from analyze_prefix_motion_coverage import parse_prefix_decisions  # noqa: E402
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

from rigweave.motion_evidence import (  # noqa: E402
    MotionEvidenceMemory,
    TopologyMotionEvidenceUniRigAR,
    query_aligned_skin_weights,
)


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
    parser.add_argument("--reference-seed-offset", type=int, default=200000)
    parser.add_argument("--residual-scale", type=float, default=0.01)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def token_nll(logits: torch.Tensor, label: torch.Tensor) -> float:
    return float(nn.functional.cross_entropy(logits.float(), label).detach())


def oracle_coverage_logits(
    model: TopologyMotionEvidenceUniRigAR,
    hidden: torch.Tensor,
    memory: MotionEvidenceMemory,
    *,
    active_anchors: torch.BoolTensor,
) -> torch.Tensor:
    """Read only uncovered anchors and retain an exact zero-value null slot."""

    if hidden.shape[0] != 1 or hidden.shape[1] != 1:
        raise ValueError("oracle intervention expects one row and one prefix position")
    if active_anchors.shape != (memory.motion_values.shape[1],):
        raise ValueError("active_anchors must identify every evidence anchor")

    attention = model.evidence_adapter.attention
    compute_dtype = attention.query_norm.weight.dtype
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        query = attention.query_norm(hidden.to(dtype=compute_dtype))
        keys = attention.key_norm(
            memory.static_tokens.detach().to(dtype=compute_dtype)
        )
        values = memory.motion_values.to(dtype=compute_dtype)
        if bool(active_anchors.any()):
            keys = keys[:, active_anchors]
            values = values[:, active_anchors]
            values = values - values.mean(dim=1, keepdim=True)
        else:
            keys = keys[:, :0]
            values = values[:, :0]

        null_key = keys.new_zeros((1, 1, keys.shape[-1]))
        null_value = values.new_zeros((1, 1, values.shape[-1]))
        keys = torch.cat((keys, null_key), dim=1)
        values = torch.cat((values, null_value), dim=1)
        update, _ = attention.cross_attention(
            query,
            keys,
            values,
            need_weights=False,
        )
        refined = hidden + (
            attention.residual_scale * update
        ).to(dtype=hidden.dtype)
    return model.evidence_adapter.project_hidden(model.transformer, refined)[:, 0]


def value_mask_logits(
    model: TopologyMotionEvidenceUniRigAR,
    hidden: torch.Tensor,
    memory: MotionEvidenceMemory,
    *,
    position: int,
    value_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep the learned attention addresses while attenuating covered values."""

    if value_mask.shape != (memory.motion_values.shape[1],):
        raise ValueError("value_mask must identify every evidence anchor")
    masked = replace(
        memory,
        motion_values=memory.motion_values * value_mask[None, :, None].to(
            dtype=memory.motion_values.dtype
        ),
    )
    logits, _ = model.evidence_adapter.logits_from_hidden(
        model.transformer,
        hidden,
        masked,
        torch.tensor([position], device=hidden.device),
    )
    return logits[:, 0]


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def summarize_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "current_normal_nll",
        "zero_nll",
        "current_corrupt_nll",
        "coverage_normal_nll",
        "coverage_corrupt_nll",
        "hard_value_normal_nll",
        "hard_value_corrupt_nll",
        "soft_value_normal_nll",
        "soft_value_corrupt_nll",
    )
    result: dict[str, Any] = {
        name: summarize([float(row[name]) for row in rows]) for name in metrics
    }

    def gap(left: str, right: str) -> list[float]:
        return [float(row[left]) - float(row[right]) for row in rows]

    result["gaps"] = {}
    variants = {
        "current": ("current_normal_nll", "current_corrupt_nll"),
        "keymask_null": ("coverage_normal_nll", "coverage_corrupt_nll"),
        "hard_value": ("hard_value_normal_nll", "hard_value_corrupt_nll"),
        "soft_value": ("soft_value_normal_nll", "soft_value_corrupt_nll"),
    }
    for name, (normal_name, corrupt_name) in variants.items():
        zero_gap = gap("zero_nll", normal_name)
        corrupt_gap = gap(corrupt_name, normal_name)
        current_gap = gap("current_normal_nll", normal_name)
        result["gaps"][name] = {
            "zero_minus_normal": summarize(zero_gap),
            "corrupt_minus_normal": summarize(corrupt_gap),
            "current_minus_variant": summarize(current_gap),
            "win_rate_vs_zero": float(np.mean(np.asarray(zero_gap) > 0)),
            "win_rate_vs_corrupt": float(np.mean(np.asarray(corrupt_gap) > 0)),
            "win_rate_vs_current": float(np.mean(np.asarray(current_gap) > 0)),
        }
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= args.residual_scale < 1.0:
        raise ValueError("residual-scale must be in [0,1)")
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
    model.evidence_adapter.attention.residual_scale = float(args.residual_scale)
    set_probe_modes(model, training=False)

    rows: list[dict[str, Any]] = []
    eos_logit_max_abs_diff = {
        "keymask_null": 0.0,
        "hard_value": 0.0,
        "soft_value": 0.0,
    }
    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        reference_seed = args.seed + args.reference_seed_offset + row_index
        torch.manual_seed(reference_seed)
        torch.cuda.manual_seed_all(reference_seed)
        refs = model.sample_references(batch)
        generator = torch.Generator(device=device).manual_seed(reference_seed + 100000)
        with torch.autocast("cuda", dtype=amp_dtype):
            memory = model.build_memory(batch, refs=refs)
            teacher = model.teacher_forcing(batch, memory=memory)
            corrupted = memory.controlled(
                "corrupt_correspondence",
                generator=generator,
            )
            positions = torch.arange(
                teacher.token_hidden.shape[1],
                device=device,
            )
            corrupt_logits, _ = model.evidence_adapter.logits_from_hidden(
                model.transformer,
                teacher.token_hidden,
                corrupted,
                positions,
            )
            zero_logits = model.evidence_adapter.project_hidden(
                model.transformer,
                teacher.token_hidden,
            )
            query_skin = query_aligned_skin_weights(
                batch["target_skin_weights"],
                batch["faces"],
                refs,
            )[0].float()

        joint_count = int(batch["joint_count"][0])
        query_skin = query_skin[:, :joint_count]
        token_count = int(batch["attention_mask"][0].sum())
        parents = batch["target_parents"][0, :joint_count]
        decisions = parse_prefix_decisions(
            batch["input_ids"][0, :token_count],
            parents,
            branch_token=int(tokenizer.token_id_branch),
            eos_token=int(tokenizer.eos),
        )
        for decision in decisions:
            if decision.role not in {"branch", "eos"}:
                continue
            position = decision.prediction_position
            label = batch["input_ids"][:, position + 1]
            remaining_support = query_skin[
                :, decision.generated_joint_count :
            ].sum(dim=-1)
            active = remaining_support > 1.0e-8
            normal_coverage_logits = oracle_coverage_logits(
                model,
                teacher.token_hidden[:, position : position + 1],
                memory,
                active_anchors=active,
            )
            corrupt_coverage_logits = oracle_coverage_logits(
                model,
                teacher.token_hidden[:, position : position + 1],
                corrupted,
                active_anchors=active,
            )
            hard_mask = active.float()
            soft_mask = remaining_support.clamp(0.0, 1.0)
            hard_value_normal_logits = value_mask_logits(
                model,
                teacher.token_hidden[:, position : position + 1],
                memory,
                position=position,
                value_mask=hard_mask,
            )
            hard_value_corrupt_logits = value_mask_logits(
                model,
                teacher.token_hidden[:, position : position + 1],
                corrupted,
                position=position,
                value_mask=hard_mask,
            )
            soft_value_normal_logits = value_mask_logits(
                model,
                teacher.token_hidden[:, position : position + 1],
                memory,
                position=position,
                value_mask=soft_mask,
            )
            soft_value_corrupt_logits = value_mask_logits(
                model,
                teacher.token_hidden[:, position : position + 1],
                corrupted,
                position=position,
                value_mask=soft_mask,
            )
            if decision.role == "eos":
                candidates = {
                    "keymask_null": normal_coverage_logits,
                    "hard_value": hard_value_normal_logits,
                    "soft_value": soft_value_normal_logits,
                }
                for name, candidate in candidates.items():
                    eos_logit_max_abs_diff[name] = max(
                        eos_logit_max_abs_diff[name],
                        float((candidate - zero_logits[:, position]).abs().max()),
                    )
            rows.append(
                {
                    "asset_index": row_index,
                    "path": raw_batch["path"][0],
                    "role": decision.role,
                    "position": position,
                    "joint_count": joint_count,
                    "generated_joint_count": decision.generated_joint_count,
                    "active_anchor_fraction": float(active.float().mean()),
                    "motion_q90_rms": float(memory.raw_evidence.example_motion_q90_rms[0]),
                    "current_normal_nll": token_nll(
                        teacher.logits[:, position], label
                    ),
                    "zero_nll": token_nll(zero_logits[:, position], label),
                    "current_corrupt_nll": token_nll(
                        corrupt_logits[:, position], label
                    ),
                    "coverage_normal_nll": token_nll(
                        normal_coverage_logits, label
                    ),
                    "coverage_corrupt_nll": token_nll(
                        corrupt_coverage_logits, label
                    ),
                    "hard_value_normal_nll": token_nll(
                        hard_value_normal_logits, label
                    ),
                    "hard_value_corrupt_nll": token_nll(
                        hard_value_corrupt_logits, label
                    ),
                    "soft_value_normal_nll": token_nll(
                        soft_value_normal_logits, label
                    ),
                    "soft_value_corrupt_nll": token_nll(
                        soft_value_corrupt_logits, label
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "index": row_index,
                    "path": raw_batch["path"][0],
                    "decisions": len(decisions),
                }
            ),
            flush=True,
        )

    summary = {
        "assets": args.limit,
        "dataset_seed": args.seed,
        "reference_seed_offset": args.reference_seed_offset,
        "residual_scale": args.residual_scale,
        "eos_variant_logit_max_abs_diff_vs_zero": eos_logit_max_abs_diff,
        "roles": {
            role: summarize_role([row for row in rows if row["role"] == role])
            for role in ("branch", "eos")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "details": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
