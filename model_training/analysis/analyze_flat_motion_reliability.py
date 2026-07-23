#!/usr/bin/env python3
"""Test whether input-only motion reliability predicts flat UniRig failures."""

from __future__ import annotations

import argparse
import json
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
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


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _pair_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f = left.float()
    right_f = right.float()
    delta = left_f - right_f
    reference_rms = 0.5 * (_rms(left_f) + _rms(right_f))
    return {
        "delta_rms": float(_rms(delta).item()),
        "relative_delta_rms": float((_rms(delta) / reference_rms.clamp_min(1.0e-12)).item()),
        "cosine": float(F.cosine_similarity(left_f.reshape(1, -1), right_f.reshape(1, -1)).item()),
        "max_abs_diff": float(delta.abs().max().item()),
    }


def _best_rigid_residual(rest: torch.Tensor, posed: torch.Tensor) -> torch.Tensor:
    """Return posed points minus their best proper-rigid fit from rest."""

    if rest.shape != posed.shape or rest.ndim != 2 or rest.shape[-1] != 3:
        raise ValueError(f"expected matching (N,3) points, got {rest.shape} and {posed.shape}")
    rest_f = rest.float()
    posed_f = posed.float()
    rest_center = rest_f.mean(dim=0, keepdim=True)
    posed_center = posed_f.mean(dim=0, keepdim=True)
    x = rest_f - rest_center
    y = posed_f - posed_center
    u, _, vh = torch.linalg.svd(x.transpose(0, 1) @ y, full_matrices=False)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    fitted = x @ rotation + posed_center
    return posed_f - fitted


def _effective_rank(singular_values: torch.Tensor) -> float:
    energy = singular_values.float().square()
    total = energy.sum()
    if float(total.item()) <= 1.0e-20:
        return 0.0
    return float((total.square() / energy.square().sum().clamp_min(1.0e-20)).item())


def _safe_corr(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_f = left.float().reshape(-1)
    right_f = right.float().reshape(-1)
    if left_f.numel() < 2 or float(left_f.std().item()) <= 1.0e-12 or float(right_f.std().item()) <= 1.0e-12:
        return None
    return float(torch.corrcoef(torch.stack([left_f, right_f]))[0, 1].item())


def _motion_evidence_metrics(query_points: torch.Tensor) -> dict[str, float | None]:
    if query_points.ndim != 4 or query_points.shape[0] != 1 or query_points.shape[-1] != 3:
        raise ValueError(f"expected query_points=(1,T,Q,3), got {tuple(query_points.shape)}")
    points = query_points[0].float()
    rest = points[0]
    displacement = points[1:] - rest.unsqueeze(0)
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    anchor_rms = displacement.square().mean(dim=(0, 2)).sqrt()
    frame_rms = displacement.square().mean(dim=(1, 2)).sqrt()
    rigid_residuals = torch.stack(
        [_best_rigid_residual(rest, posed) for posed in points[1:]], dim=0
    )
    rigid_residual_norm = torch.linalg.vector_norm(rigid_residuals, dim=-1)
    flattened = displacement.reshape(displacement.shape[0], -1)
    singular_values = torch.linalg.svdvals(flattened)
    total_motion_rms = _rms(displacement)
    articulated_rms = _rms(rigid_residuals)
    anchor_energy = anchor_rms.square()
    anchor_energy_sum = anchor_energy.sum()
    anchor_participation = (
        anchor_energy_sum.square() / anchor_energy.square().sum().clamp_min(1.0e-20)
        if float(anchor_energy_sum.item()) > 1.0e-20
        else anchor_energy_sum
    )
    return {
        "motion_rms": float(total_motion_rms.item()),
        "motion_p50": float(torch.quantile(displacement_norm, 0.5).item()),
        "motion_p90": float(torch.quantile(displacement_norm, 0.9).item()),
        "motion_max": float(displacement_norm.max().item()),
        "moving_fraction_0p01": float((anchor_rms > 0.01).float().mean().item()),
        "moving_fraction_0p02": float((anchor_rms > 0.02).float().mean().item()),
        "moving_fraction_0p05": float((anchor_rms > 0.05).float().mean().item()),
        "anchor_motion_participation": float(anchor_participation.item() / anchor_rms.numel()),
        "frame_motion_cv": float((frame_rms.std() / frame_rms.mean().clamp_min(1.0e-12)).item()),
        "articulated_rms": float(articulated_rms.item()),
        "articulated_p90": float(torch.quantile(rigid_residual_norm, 0.9).item()),
        "articulated_to_motion_rms": float(
            (articulated_rms / total_motion_rms.clamp_min(1.0e-12)).item()
        ),
        "motion_mode_effective_rank": _effective_rank(singular_values),
    }


def _condition_locality_metrics(
    normal: torch.Tensor,
    zero: torch.Tensor,
    query_points: torch.Tensor,
) -> dict[str, float | None]:
    delta = (normal - zero)[0].float()
    token_delta = torch.linalg.vector_norm(delta, dim=-1)
    anchor_motion = (query_points[0, 1:].float() - query_points[0, :1].float()).square().mean(dim=(0, 2)).sqrt()
    shared = delta.mean(dim=0, keepdim=True).expand_as(delta)
    total_energy = delta.square().sum()
    shared_fraction = float(
        (shared.square().sum() / total_energy.clamp_min(1.0e-20)).item()
    )
    return {
        "token_delta_rms": float(_rms(delta).item()),
        "token_delta_shared_energy_fraction": shared_fraction,
        "anchor_motion_token_delta_correlation": _safe_corr(anchor_motion, token_delta),
    }


def _generation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    dynamic = row["dynamic"]
    metrics = dynamic.get("metrics") or {}
    topology = metrics.get("topology") or {}
    official = metrics.get("official") or {}
    ok = bool(dynamic.get("detokenize_ok")) and bool(dynamic.get("has_eos"))
    return {
        "ok": ok,
        "hitmax": bool(dynamic.get("hit_max_without_eos")),
        "pred_joint_count": metrics.get("pred_joint_count"),
        "joint_count_error": metrics.get("joint_count_error"),
        "root_l2": metrics.get("root_l2"),
        "joint1_l2": metrics.get("joint1_l2"),
        "topology_f1": topology.get("edge_f1") if ok else None,
        "topology_f1_failure_zero": float(topology.get("edge_f1") or 0.0) if ok else 0.0,
        "j2j": official.get("j2j") if ok else None,
    }


def _load_result(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"generation result has no rows: {path}")
    return rows


def _finite_mean(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _rank_auc(values: list[float], labels: list[int]) -> float | None:
    positives = [value for value, label in zip(values, labels, strict=True) if label]
    negatives = [value for value, label in zip(values, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def _summarize(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    baseline_hitmax = [int(not row["alpha_1p00"]["ok"]) for row in rows]
    gate_delta = [
        row["alpha_0p75"]["topology_f1_failure_zero"]
        - row["alpha_1p00"]["topology_f1_failure_zero"]
        for row in rows
    ]
    metrics = {}
    for name in metric_names:
        values = [row["evidence"].get(name) for row in rows]
        finite_pairs = [
            (float(value), baseline_hitmax[index], gate_delta[index])
            for index, value in enumerate(values)
            if value is not None and np.isfinite(float(value))
        ]
        finite_values = [item[0] for item in finite_pairs]
        labels = [item[1] for item in finite_pairs]
        deltas = [item[2] for item in finite_pairs]
        rho = spearmanr(finite_values, deltas).statistic if len(finite_values) >= 3 else None
        metrics[name] = {
            "mean": _finite_mean(values),
            "baseline_hitmax_mean": _finite_mean(
                [value for value, label in zip(values, baseline_hitmax, strict=True) if label]
            ),
            "baseline_legal_mean": _finite_mean(
                [value for value, label in zip(values, baseline_hitmax, strict=True) if not label]
            ),
            "hitmax_rank_auc_high_is_failure": _rank_auc(finite_values, labels),
            "spearman_with_gate_f1_delta": (
                float(rho) if rho is not None and np.isfinite(float(rho)) else None
            ),
        }
    return {
        "rows": len(rows),
        "alpha_1p00_legal": sum(row["alpha_1p00"]["ok"] for row in rows),
        "alpha_0p75_legal": sum(row["alpha_0p75"]["ok"] for row in rows),
        "alpha_1p00_f1_all": _finite_mean(
            [row["alpha_1p00"]["topology_f1_failure_zero"] for row in rows]
        ),
        "alpha_0p75_f1_all": _finite_mean(
            [row["alpha_0p75"]["topology_f1_failure_zero"] for row in rows]
        ),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--alpha075-result", type=Path, required=True)
    parser.add_argument("--alpha1-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=154)
    parser.add_argument("--failure-rows", type=int, default=77)
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
        args.alpha075_result,
        args.alpha1_result,
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
        raise ValueError("reliability audit requires the accepted dynamic condition route")

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
    alpha075_rows = _load_result(args.alpha075_result)
    alpha1_rows = _load_result(args.alpha1_result)
    if len(dataset) != len(alpha075_rows) or len(dataset) != len(alpha1_rows):
        raise ValueError("manifest and generation results have different row counts")
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
            frame_tokens, query_points = model.conditioner.tokenize_frames(
                batch["frame_vertices"],
                batch["faces"],
                refs,
                vertex_normals=batch.get("vertex_normals"),
                face_normals=batch.get("face_normals"),
            )
            normal = model.build_condition(batch, control="normal", refs=refs)
            zero = model.build_condition(batch, control="zero", refs=refs)
            path = str(batch["path"][0])
            selected_frames = batch["selected_frames"][0].detach().cpu().tolist()
            for result_row in (alpha075_rows[index], alpha1_rows[index]):
                if str(result_row["path"]) != path:
                    raise ValueError(f"row {index} path mismatch")
                if result_row["selected_frames"] != selected_frames:
                    raise ValueError(f"row {index} selected-frame mismatch")
            evidence = {
                **_motion_evidence_metrics(query_points),
                **_condition_locality_metrics(normal, zero, query_points),
                **{
                    f"normal_vs_zero_{key}": value
                    for key, value in _pair_summary(normal, zero).items()
                },
                **{
                    f"zero_vs_frame0_{key}": value
                    for key, value in _pair_summary(zero, frame_tokens[:, 0]).items()
                },
                **{
                    f"normal_vs_frame0_{key}": value
                    for key, value in _pair_summary(normal, frame_tokens[:, 0]).items()
                },
            }
            row = {
                "index": index,
                "path": path,
                "stratum": "westlake_baseline_hitmax" if index < args.failure_rows else "matched_legal_control",
                "target_joint_count": int(batch["joint_count"][0].item()),
                "selected_frames": selected_frames,
                "evidence": evidence,
                "alpha_0p75": _generation_metrics(alpha075_rows[index]),
                "alpha_1p00": _generation_metrics(alpha1_rows[index]),
            }
            rows.append(row)
            print(json.dumps({
                "row": f"{index + 1}/{len(dataset)}",
                "stratum": row["stratum"],
                "motion_rms": evidence["motion_rms"],
                "articulated_rms": evidence["articulated_rms"],
                "alpha075_ok": row["alpha_0p75"]["ok"],
                "alpha1_ok": row["alpha_1p00"]["ok"],
            }, sort_keys=True), flush=True)

    metric_names = sorted(rows[0]["evidence"])
    groups = {
        "all": rows,
        "selected_westlake_hitmax": rows[: args.failure_rows],
        "matched_legal_controls": rows[args.failure_rows :],
        "alpha1_hitmax": [row for row in rows if not row["alpha_1p00"]["ok"]],
        "alpha1_legal": [row for row in rows if row["alpha_1p00"]["ok"]],
    }
    report = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sample_exposures": train_args.get("sample_exposures"),
            "alpha075_result": str(args.alpha075_result),
            "alpha1_result": str(args.alpha1_result),
            "seed": args.seed,
            "rows": len(rows),
            "failure_rows": args.failure_rows,
        },
        "aggregate": {
            name: _summarize(group_rows, metric_names)
            for name, group_rows in groups.items()
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
