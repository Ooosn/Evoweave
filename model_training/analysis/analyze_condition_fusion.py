#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import pad


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


CONDITION_LABELS = ("static", "fused", "zero_motion_fused", "dynamic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how a trained static-plus-motion condition fuser changes condition "
            "tokens and the first two teacher-forced joint predictions."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--generation-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().square().mean().sqrt().item())


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return _rms(numerator) / max(_rms(denominator), 1.0e-12)


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().flatten(), q).item())


def _condition_stats(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    zero_dynamic: torch.Tensor,
    fused: torch.Tensor,
    zero_fused: torch.Tensor,
) -> dict[str, float]:
    residual = fused - static
    zero_residual = zero_fused - static
    static_norm = static.float().norm(dim=-1).clamp_min(1.0e-12)
    residual_norm = residual.float().norm(dim=-1)
    token_ratio = residual_norm / static_norm
    token_cosine = F.cosine_similarity(static.float(), fused.float(), dim=-1)
    residual_energy = residual.float().square().sum(dim=-1).mean().clamp_min(1.0e-20)
    shared_energy = residual.float().mean(dim=1).square().sum(dim=-1).mean()
    motion_effect = fused - zero_fused
    motion_effect_energy = motion_effect.float().square().sum(dim=-1).mean().clamp_min(1.0e-20)
    motion_effect_shared_energy = motion_effect.float().mean(dim=1).square().sum(dim=-1).mean()
    return {
        "static_rms": _rms(static),
        "dynamic_rms": _rms(dynamic),
        "fused_rms": _rms(fused),
        "residual_rms": _rms(residual),
        "residual_rms_ratio": _ratio(residual, static),
        "zero_motion_residual_rms_ratio": _ratio(zero_residual, static),
        "motion_effect_dynamic_rms_ratio": _ratio(dynamic - zero_dynamic, static),
        "motion_effect_fused_rms_ratio": _ratio(fused - zero_fused, static),
        "residual_token_ratio_median": _quantile(token_ratio, 0.5),
        "residual_token_ratio_p90": _quantile(token_ratio, 0.9),
        "residual_token_ratio_max": float(token_ratio.max().item()),
        "static_fused_token_cosine_median": _quantile(token_cosine, 0.5),
        "static_fused_token_cosine_p10": _quantile(token_cosine, 0.1),
        "residual_shared_energy_fraction": float((shared_energy / residual_energy).item()),
        "motion_effect_fused_shared_energy_fraction": float(
            (motion_effect_shared_energy / motion_effect_energy).item()
        ),
    }


@torch.no_grad()
def _attention_stats(model: torch.nn.Module, static: torch.Tensor, dynamic: torch.Tensor) -> dict[str, float]:
    if len(model.condition_fuser.blocks) != 1:
        raise ValueError("this audit currently requires a one-block condition fuser")
    block = model.condition_fuser.blocks[0]
    query = block.static_norm(static.float())
    key_value = block.dynamic_norm(dynamic.float())
    _, weights = block.cross_attn(
        query,
        key_value,
        key_value,
        need_weights=True,
        average_attn_weights=False,
    )
    probability = weights.float().clamp_min(1.0e-12)
    key_count = int(probability.shape[-1])
    entropy = -(probability * probability.log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(max(key_count, 2))
    maximum = probability.max(dim=-1).values
    return {
        "attention_entropy_normalized_mean": float(normalized_entropy.mean().item()),
        "attention_entropy_normalized_p90": _quantile(normalized_entropy, 0.9),
        "attention_effective_key_count_mean": float(entropy.exp().mean().item()),
        "attention_max_weight_mean": float(maximum.mean().item()),
        "attention_max_weight_p90": _quantile(maximum, 0.9),
    }


def _teacher_logits(
    model: torch.nn.Module,
    conditions: list[torch.Tensor],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    # BOS, class, root xyz, and first-child xyz are sufficient for this audit.
    token_count = min(int(batch["input_ids"].shape[1]), 8)
    input_ids = batch["input_ids"][:, :token_count].repeat(len(conditions), 1)
    attention_mask = batch["attention_mask"][:, :token_count].repeat(len(conditions), 1)
    condition = torch.cat(conditions, dim=0).to(dtype=model.transformer.dtype)
    token_embeddings = model.token_inputs_embeds(input_ids, attention_mask)
    inputs = torch.cat([condition, token_embeddings], dim=1)
    full_attention = pad(attention_mask, (condition.shape[1], 0, 0, 0), value=1.0)
    output = model.transformer(
        inputs_embeds=inputs,
        attention_mask=full_attention,
        use_cache=False,
        output_hidden_states=False,
    )
    logits = output.logits[:, condition.shape[1] :]
    return logits[:, :-1].float(), input_ids[:, 1:]


def _coordinate_prediction_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_discrete: int,
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    coordinate_slices = {"root": slice(1, 4), "joint1": slice(4, 7)}
    for condition_index, condition_label in enumerate(CONDITION_LABELS):
        condition_result: dict[str, float | None] = {}
        for joint_label, position_slice in coordinate_slices.items():
            target = labels[condition_index, position_slice]
            joint_logits = logits[condition_index, position_slice, :num_discrete]
            if target.numel() != 3 or bool((target >= num_discrete).any()):
                condition_result[f"{joint_label}_bin_l2"] = None
                condition_result[f"{joint_label}_ce"] = None
                continue
            prediction = joint_logits.argmax(dim=-1)
            condition_result[f"{joint_label}_bin_l2"] = float(
                (prediction.float() - target.float()).square().sum().sqrt().item()
            )
            condition_result[f"{joint_label}_ce"] = float(
                F.cross_entropy(joint_logits, target, reduction="mean").item()
            )
        result[condition_label] = condition_result
    return result


def _load_generation_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"generation result has no rows list: {path}")
    return rows


def _generation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    block = row["dynamic"]
    metrics = block.get("metrics") or {}
    official = metrics.get("official") or {}
    topology = metrics.get("topology") or {}
    return {
        "free_has_eos": bool(block.get("has_eos", False)),
        "free_hitmax": bool(block.get("hit_max_without_eos", False)),
        "free_pred_joint_count": metrics.get("pred_joint_count"),
        "free_root_l2": metrics.get("root_l2"),
        "free_joint1_l2": metrics.get("joint1_l2"),
        "free_j2j": official.get("j2j"),
        "free_topology_f1": topology.get("edge_f1"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    return float(np.mean(values)) if values else None


def _correlation(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, float | int | None]:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if row.get(left) is not None
        and row.get(right) is not None
        and np.isfinite(row[left])
        and np.isfinite(row[right])
    ]
    if len(pairs) < 3:
        return {"count": len(pairs), "pearson": None}
    values = np.asarray(pairs, dtype=np.float64)
    if values[:, 0].std() == 0.0 or values[:, 1].std() == 0.0:
        return {"count": len(pairs), "pearson": None}
    return {"count": len(pairs), "pearson": float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (args.manifest, args.checkpoint, args.tokenizer_config, args.model_config, args.unirig_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        unirig_checkpoint=args.unirig_checkpoint,
    )
    for name in CHECKPOINT_DEFAULTS:
        setattr(model_args, name, None)
    train_args = apply_checkpoint_eval_defaults(model_args)
    if model_args.condition_fusion not in {
        "static_cross_attn",
        "static_cross_attn_zero",
        "anchor_motion_residual_zero",
    }:
        raise ValueError(f"checkpoint is not a supported condition-fusion run: {model_args.condition_fusion}")

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
    model = _build_dynamic_model(model_args, tokenizer, device)
    generation_rows = _load_generation_rows(args.generation_json)
    if generation_rows is not None and len(generation_rows) != len(dataset):
        raise ValueError(f"generation rows={len(generation_rows)} dataset rows={len(dataset)}")

    gate = float(torch.sigmoid(model.condition_fuser.gate_logit.detach()).item())
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        batch = dynamic_rig_collate([dataset[index]], pad_token=tokenizer.pad)
        batch = move_batch(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            references = model.sample_references(batch)
            if model_args.condition_fusion == "anchor_motion_residual_zero":
                frame_tokens, query_points = model.conditioner.tokenize_frames(
                    batch["frame_vertices"],
                    batch["faces"],
                    references,
                    vertex_normals=batch["vertex_normals"],
                    face_normals=batch["face_normals"],
                )
                zero_frame_tokens, zero_query_points = model.conditioner.tokenize_frames(
                    model._control_sequence(batch["frame_vertices"], "zero"),
                    batch["faces"],
                    references,
                    vertex_normals=model._control_sequence(batch["vertex_normals"], "zero"),
                    face_normals=model._control_sequence(batch["face_normals"], "zero"),
                )
                dynamic = model.conditioner.motion_encoder(frame_tokens, query_points=query_points)
                zero_dynamic = model.conditioner.motion_encoder(
                    zero_frame_tokens,
                    query_points=zero_query_points,
                )
                static = frame_tokens[:, 0]
            else:
                dynamic = model.conditioner(
                    batch["frame_vertices"],
                    batch["faces"],
                    references,
                    vertex_normals=batch["vertex_normals"],
                    face_normals=batch["face_normals"],
                )
                zero_dynamic = model.conditioner(
                    model._control_sequence(batch["frame_vertices"], "zero"),
                    batch["faces"],
                    references,
                    vertex_normals=model._control_sequence(batch["vertex_normals"], "zero"),
                    face_normals=model._control_sequence(batch["face_normals"], "zero"),
                )
                static = model.build_static_condition(batch).to(device=dynamic.device, dtype=dynamic.dtype)
            fused = model.condition_fuser(static, dynamic)
            zero_fused = model.condition_fuser(static, zero_dynamic)
            logits, labels = _teacher_logits(
                model,
                [static, fused, zero_fused, dynamic],
                batch,
            )

        row: dict[str, Any] = {
            "index": index,
            "path": str(batch["path"][0]),
            "sample": Path(batch["path"][0]).name,
            "query_frame": int(batch["selected_frames"][0, 0].item()),
            "target_joint_count": int(batch["joint_count"][0].item()),
            "gate": gate,
        }
        row.update(_condition_stats(static, dynamic, zero_dynamic, fused, zero_fused))
        if model_args.condition_fusion in {"static_cross_attn", "static_cross_attn_zero"}:
            row.update(_attention_stats(model, static, dynamic))
        coordinate_stats = _coordinate_prediction_stats(logits, labels, tokenizer.num_discrete)
        for condition_label, values in coordinate_stats.items():
            for key, value in values.items():
                row[f"{condition_label}_{key}"] = value
        row["fused_minus_static_root_bin_l2"] = (
            row["fused_root_bin_l2"] - row["static_root_bin_l2"]
            if row["fused_root_bin_l2"] is not None and row["static_root_bin_l2"] is not None
            else None
        )
        row["fused_minus_static_joint1_bin_l2"] = (
            row["fused_joint1_bin_l2"] - row["static_joint1_bin_l2"]
            if row["fused_joint1_bin_l2"] is not None and row["static_joint1_bin_l2"] is not None
            else None
        )
        if generation_rows is not None:
            generation_row = generation_rows[index]
            if Path(generation_row["path"]).name != row["sample"]:
                raise ValueError(f"generation sample mismatch at row {index}")
            if int(generation_row["query_frame"]) != row["query_frame"]:
                raise ValueError(f"generation query-frame mismatch at row {index}")
            if [int(value) for value in generation_row["selected_frames"]] != [
                int(value) for value in batch["selected_frames"][0].detach().cpu().tolist()
            ]:
                raise ValueError(f"generation selected-frame mismatch at row {index}")
            if int(generation_row["target_joint_count"]) != row["target_joint_count"]:
                raise ValueError(f"generation target-count mismatch at row {index}")
            row.update(_generation_metrics(generation_row))
        rows.append(row)
        print(f"condition audit {index + 1}/{len(dataset)} {row['sample']}", flush=True)

    mean_keys = [
        "residual_rms_ratio",
        "motion_effect_dynamic_rms_ratio",
        "motion_effect_fused_rms_ratio",
        "residual_shared_energy_fraction",
        "motion_effect_fused_shared_energy_fraction",
        "attention_entropy_normalized_mean",
        "attention_effective_key_count_mean",
        "static_root_bin_l2",
        "fused_root_bin_l2",
        "zero_motion_fused_root_bin_l2",
        "dynamic_root_bin_l2",
        "static_joint1_bin_l2",
        "fused_joint1_bin_l2",
        "zero_motion_fused_joint1_bin_l2",
        "dynamic_joint1_bin_l2",
        "fused_minus_static_root_bin_l2",
        "fused_minus_static_joint1_bin_l2",
    ]
    aggregate: dict[str, Any] = {"count": len(rows), "gate": gate}
    aggregate.update({f"mean_{key}": _mean(rows, key) for key in mean_keys})
    correlations = {
        f"residual_rms_ratio__{right}": _correlation(rows, "residual_rms_ratio", right)
        for right in (
            "fused_minus_static_root_bin_l2",
            "fused_minus_static_joint1_bin_l2",
            "free_hitmax",
            "free_root_l2",
            "free_joint1_l2",
            "free_j2j",
            "free_topology_f1",
        )
        if any(right in row for row in rows)
    }
    result = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "generation_json": str(args.generation_json) if args.generation_json else None,
            "seed": args.seed,
            "limit": args.limit,
            "checkpoint_condition_fusion": model_args.condition_fusion,
            "checkpoint_sample_exposures": train_args.get("sample_exposures"),
        },
        "aggregate": aggregate,
        "correlations": correlations,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_csv, rows)
    print(json.dumps({"aggregate": aggregate, "correlations": correlations}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
