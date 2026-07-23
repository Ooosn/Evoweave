#!/usr/bin/env python3
"""Trace normal-versus-zero motion evidence through a flat UniRig checkpoint."""

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
from torch.nn.functional import pad
from torch.utils.data import DataLoader


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
sys.path.insert(0, str(RIGWEAVE_SCRIPTS))
if sys.platform == "win32" and "resource" not in sys.modules:
    resource_stub = types.ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.getrusage = lambda _: types.SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = resource_stub


def _mean(values: list[float | int | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _target_role(tokenizer: Any, target: int, coordinate_ordinal: int) -> str:
    if target == int(tokenizer.eos):
        return "eos"
    if target == int(tokenizer.token_id_branch):
        return "branch"
    if 0 <= target < int(tokenizer.num_discrete):
        if coordinate_ordinal < 3:
            return "root_coordinate"
        if coordinate_ordinal < 6:
            return "first_child_coordinate"
        return "later_coordinate"
    return "part_or_class"


def _pair_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f = left.float()
    right_f = right.float()
    delta = left_f - right_f
    left_rms = left_f.square().mean().sqrt()
    right_rms = right_f.square().mean().sqrt()
    reference_rms = 0.5 * (left_rms + right_rms)
    return {
        "delta_rms": float(delta.square().mean().sqrt().item()),
        "reference_rms": float(reference_rms.item()),
        "relative_delta_rms": float(
            (delta.square().mean().sqrt() / reference_rms.clamp_min(1.0e-12)).item()
        ),
        "cosine": float(
            F.cosine_similarity(left_f.reshape(1, -1), right_f.reshape(1, -1)).item()
        ),
        "max_abs_diff": float(delta.abs().max().item()),
    }


def _motion_summary(frame_vertices: torch.Tensor, vertex_count: int) -> dict[str, float]:
    vertices = frame_vertices[0, :, : int(vertex_count)].float()
    delta = torch.linalg.vector_norm(vertices[1:] - vertices[:1], dim=-1).reshape(-1)
    return {
        "motion_mean": float(delta.mean().item()),
        "motion_rms": float(delta.square().mean().sqrt().item()),
        "motion_p90": float(torch.quantile(delta, 0.9).item()),
        "motion_max": float(delta.max().item()),
    }


def _grammar_stats(
    logits: torch.Tensor,
    possible_ids: torch.Tensor,
    target: int,
) -> dict[str, float | int]:
    selected = logits.index_select(0, possible_ids).float()
    target_offset = (possible_ids == int(target)).nonzero(as_tuple=False)
    if target_offset.numel() != 1:
        raise ValueError(f"target token {target} is absent from grammar candidates")
    offset = int(target_offset.item())
    target_logit = selected[offset]
    nll = torch.logsumexp(selected, dim=0) - target_logit
    pred_offset = int(selected.argmax().item())
    pred_id = int(possible_ids[pred_offset].item())
    return {
        "nll": float(nll.item()),
        "rank": int((selected > target_logit).sum().item() + 1),
        "probability": float(torch.exp(-nll).item()),
        "pred_id": pred_id,
        "correct": int(pred_id == int(target)),
    }


def _distribution_delta(
    left: torch.Tensor,
    right: torch.Tensor,
    possible_ids: torch.Tensor,
) -> dict[str, float | int]:
    left_logits = left.index_select(0, possible_ids).float()
    right_logits = right.index_select(0, possible_ids).float()
    left_logp = F.log_softmax(left_logits, dim=-1)
    right_logp = F.log_softmax(right_logits, dim=-1)
    left_prob = left_logp.exp()
    right_prob = right_logp.exp()
    mean_prob = 0.5 * (left_prob + right_prob)
    mean_logp = mean_prob.clamp_min(1.0e-30).log()
    js = 0.5 * (
        torch.sum(left_prob * (left_logp - mean_logp))
        + torch.sum(right_prob * (right_logp - mean_logp))
    )
    return {
        "js_divergence": float(js.item()),
        "total_variation": float((0.5 * (left_prob - right_prob).abs().sum()).item()),
        "top1_agreement": int(int(left_prob.argmax()) == int(right_prob.argmax())),
    }


@torch.no_grad()
def _teacher_trace(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    normal_condition: torch.Tensor,
    zero_condition: torch.Tensor,
) -> dict[str, Any]:
    condition = torch.cat([normal_condition, zero_condition], dim=0).to(
        dtype=model.transformer.dtype
    )
    input_ids = batch["input_ids"].repeat(2, 1)
    attention_mask = batch["attention_mask"].repeat(2, 1)
    token_embeds = model.token_inputs_embeds(input_ids, attention_mask)
    prompt = torch.cat([condition, token_embeds], dim=1)
    full_attention = pad(attention_mask, (condition.shape[1], 0, 0, 0), value=1.0)
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=full_attention,
        use_cache=False,
        output_hidden_states=True,
    )
    condition_length = int(condition.shape[1])
    token_hidden = output.hidden_states[-1][:, condition_length:]
    logits = model.apply_action_group_bias(
        output.logits[:, condition_length:],
        token_hidden,
        condition,
    )[:, :-1].float()
    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = -100

    positions: list[dict[str, Any]] = []
    coordinate_ordinal = 0
    valid_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
    for position in valid_positions:
        target = int(labels[0, position].item())
        prefix = input_ids[0, : position + 1].detach().cpu().numpy().astype(np.int64)
        possible = sorted(int(value) for value in tokenizer.next_posible_token(ids=prefix))
        possible_ids = torch.tensor(possible, device=logits.device, dtype=torch.long)
        role = _target_role(tokenizer, target, coordinate_ordinal)
        is_coordinate = 0 <= target < int(tokenizer.num_discrete)
        normal = _grammar_stats(logits[0, position], possible_ids, target)
        zero = _grammar_stats(logits[1, position], possible_ids, target)
        distribution = _distribution_delta(
            logits[0, position], logits[1, position], possible_ids
        )
        row: dict[str, Any] = {
            "position": int(position + 1),
            "target_id": target,
            "role": role,
            "normal": normal,
            "zero": zero,
            "zero_minus_normal_nll": float(zero["nll"] - normal["nll"]),
            "normal_minus_zero_rank": int(normal["rank"] - zero["rank"]),
            **distribution,
        }
        if is_coordinate:
            row["normal_abs_bin_error"] = abs(int(normal["pred_id"]) - target)
            row["zero_abs_bin_error"] = abs(int(zero["pred_id"]) - target)
            row["zero_minus_normal_abs_bin_error"] = (
                int(row["zero_abs_bin_error"]) - int(row["normal_abs_bin_error"])
            )
            coordinate_ordinal += 1
        positions.append(row)

    role_positions: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(positions):
        role_positions[str(row["role"])].append(row_index)
    all_position_ids = [int(row["position"]) - 1 for row in positions]

    hidden_by_layer: dict[str, dict[str, dict[str, float]]] = {}
    for layer_index, hidden in enumerate(output.hidden_states):
        token_states = hidden[:, condition_length:]
        hidden_by_layer[str(layer_index)] = {}
        for role, row_indices in role_positions.items():
            token_indices = torch.tensor(
                [all_position_ids[index] for index in row_indices],
                device=hidden.device,
                dtype=torch.long,
            )
            selected = token_states.index_select(1, token_indices)
            hidden_by_layer[str(layer_index)][role] = _pair_summary(
                selected[0], selected[1]
            )

    role_summary: dict[str, dict[str, float | int | None]] = {}
    for role, row_indices in role_positions.items():
        selected = [positions[index] for index in row_indices]
        coordinate_rows = [row for row in selected if "normal_abs_bin_error" in row]
        role_summary[role] = {
            "count": len(selected),
            "normal_nll": _mean([row["normal"]["nll"] for row in selected]),
            "zero_nll": _mean([row["zero"]["nll"] for row in selected]),
            "zero_minus_normal_nll": _mean(
                [row["zero_minus_normal_nll"] for row in selected]
            ),
            "normal_accuracy": _mean([row["normal"]["correct"] for row in selected]),
            "zero_accuracy": _mean([row["zero"]["correct"] for row in selected]),
            "normal_abs_bin_error": _mean(
                [row["normal_abs_bin_error"] for row in coordinate_rows]
            ),
            "zero_abs_bin_error": _mean(
                [row["zero_abs_bin_error"] for row in coordinate_rows]
            ),
            "zero_minus_normal_abs_bin_error": _mean(
                [row["zero_minus_normal_abs_bin_error"] for row in coordinate_rows]
            ),
            "js_divergence": _mean([row["js_divergence"] for row in selected]),
            "top1_agreement": _mean([row["top1_agreement"] for row in selected]),
        }

    return {
        "role_summary": role_summary,
        "hidden_by_layer": hidden_by_layer,
        "positions": positions,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    roles = sorted(
        {
            role
            for row in rows
            for role in row["teacher_trace"]["role_summary"]
        }
    )
    layer_names = sorted(
        {
            layer
            for row in rows
            for layer in row["teacher_trace"]["hidden_by_layer"]
        },
        key=int,
    )
    return {
        "rows": len(rows),
        "baseline_hitmax_rows": sum(bool(row.get("baseline_hitmax")) for row in rows),
        "motion": {
            key: _mean([row["motion"][key] for row in rows])
            for key in ("motion_mean", "motion_rms", "motion_p90", "motion_max")
        },
        "condition": {
            key: _mean([row["normal_vs_zero_condition"][key] for row in rows])
            for key in (
                "delta_rms",
                "reference_rms",
                "relative_delta_rms",
                "cosine",
                "max_abs_diff",
            )
        },
        "reverse_condition_max_abs_diff": _mean(
            [row["normal_vs_reverse_condition"]["max_abs_diff"] for row in rows]
        ),
        "roles": {
            role: {
                key: _mean(
                    [
                        row["teacher_trace"]["role_summary"].get(role, {}).get(key)
                        for row in rows
                    ]
                )
                for key in (
                    "normal_nll",
                    "zero_nll",
                    "zero_minus_normal_nll",
                    "normal_accuracy",
                    "zero_accuracy",
                    "normal_abs_bin_error",
                    "zero_abs_bin_error",
                    "zero_minus_normal_abs_bin_error",
                    "js_divergence",
                    "top1_agreement",
                )
            }
            for role in roles
        },
        "hidden_relative_delta_rms": {
            layer: {
                role: _mean(
                    [
                        row["teacher_trace"]["hidden_by_layer"]
                        .get(layer, {})
                        .get(role, {})
                        .get("relative_delta_rms")
                        for row in rows
                    ]
                )
                for role in roles
            }
            for layer in layer_names
        },
    }


def _load_baseline_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"baseline generation file has no rows list: {path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-generation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    from eval_dynamic_rig_ce import (
        CHECKPOINT_DEFAULTS,
        _build_dynamic_model,
        apply_checkpoint_eval_defaults,
    )
    from train_dynamic_rig import build_tokenizer, move_batch
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate

    model_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        unirig_checkpoint=args.unirig_checkpoint,
    )
    for name in CHECKPOINT_DEFAULTS:
        setattr(model_args, name, None)
    train_args = apply_checkpoint_eval_defaults(model_args)
    if model_args.condition_fusion != "dynamic":
        raise ValueError(
            "this audit requires the accepted flat dynamic condition route, "
            f"got condition_fusion={model_args.condition_fusion!r}"
        )

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
        motion_alignment_policy=model_args.motion_alignment_policy,
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
    baseline_rows = _load_baseline_rows(args.baseline_generation)
    if baseline_rows is not None and len(baseline_rows) != len(dataset):
        raise ValueError(
            f"baseline rows={len(baseline_rows)} do not match dataset rows={len(dataset)}"
        )

    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = _build_dynamic_model(model_args, tokenizer, device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        for index, batch in enumerate(loader):
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            batch = move_batch(batch, device)
            refs = model.sample_references(batch)
            normal_condition = model.build_condition(batch, control="normal", refs=refs)
            zero_condition = model.build_condition(batch, control="zero", refs=refs)
            reverse_condition = model.build_condition(batch, control="reverse", refs=refs)

            baseline_row = None if baseline_rows is None else baseline_rows[index]
            if baseline_row is not None:
                if str(batch["path"][0]) != str(baseline_row["path"]):
                    raise ValueError(f"row {index} path mismatch")
                selected_frames = (
                    batch["selected_frames"][0].detach().cpu().numpy().astype(int).tolist()
                )
                if selected_frames != baseline_row["selected_frames"]:
                    raise ValueError(f"row {index} selected-frame mismatch")

            row = {
                "index": index,
                "path": str(batch["path"][0]),
                "target_joint_count": int(batch["joint_count"][0].item()),
                "selected_frames": (
                    batch["selected_frames"][0].detach().cpu().numpy().astype(int).tolist()
                ),
                "baseline_hitmax": (
                    bool(baseline_row["dynamic"].get("hit_max_without_eos"))
                    if baseline_row is not None
                    else None
                ),
                "motion": _motion_summary(
                    batch["frame_vertices"], int(batch["vertex_count"][0].item())
                ),
                "normal_vs_zero_condition": _pair_summary(
                    normal_condition, zero_condition
                ),
                "normal_vs_reverse_condition": _pair_summary(
                    normal_condition, reverse_condition
                ),
                "teacher_trace": _teacher_trace(
                    model,
                    tokenizer,
                    batch,
                    normal_condition,
                    zero_condition,
                ),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "row": f"{index + 1}/{len(dataset)}",
                        "path": row["path"],
                        "joint_count": row["target_joint_count"],
                        "baseline_hitmax": row["baseline_hitmax"],
                        "motion_rms": row["motion"]["motion_rms"],
                        "condition_relative_delta": row["normal_vs_zero_condition"][
                            "relative_delta_rms"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    if baseline_rows is not None:
        groups["baseline_hitmax"] = [row for row in rows if row["baseline_hitmax"]]
        groups["baseline_legal"] = [row for row in rows if not row["baseline_hitmax"]]
    if len(rows) == 18:
        groups["hard6"] = rows[:6]
        groups["middle6"] = rows[6:12]
        groups["control6"] = rows[12:18]

    report = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sample_exposures": train_args.get("sample_exposures"),
            "baseline_generation": (
                str(args.baseline_generation) if args.baseline_generation else None
            ),
            "seed": args.seed,
            "rows": len(rows),
            "condition_fusion": model_args.condition_fusion,
            "use_motion_features": model_args.use_motion_features,
            "use_time_embedding": model_args.use_time_embedding,
        },
        "aggregate": {
            name: _aggregate_rows(group_rows) for name, group_rows in groups.items()
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
