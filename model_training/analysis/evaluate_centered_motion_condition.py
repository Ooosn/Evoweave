#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
sys.path.insert(0, str(MODEL_TRAINING_ROOT))
sys.path.insert(0, str(RIGWEAVE_SCRIPTS))
if sys.platform == "win32" and "resource" not in sys.modules:
    resource_stub = types.ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.getrusage = lambda _: types.SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = resource_stub

from analysis.centered_motion_condition import build_centered_motion_condition  # noqa: E402
from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model, apply_checkpoint_eval_defaults  # noqa: E402
from eval_dynamic_rig_generation import _dynamic_generate, _run_model, _summarize  # noqa: E402
from train_dynamic_rig import build_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate query-pose static tokens plus the trained motion encoder's "
            "normal-minus-zero-motion response, without changing model weights."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def _parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("at least one centered-motion alpha is required")
    if any(not torch.isfinite(torch.tensor(alpha)) for alpha in alphas):
        raise ValueError(f"alphas must be finite, got {alphas}")
    return alphas


def _mode_name(alpha: float) -> str:
    text = f"{alpha:g}".replace("-", "m").replace(".", "p")
    return f"centered_{text}"


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _relative_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    return _rms(value) / max(_rms(reference), 1.0e-12)


def _shared_energy_fraction(value: torch.Tensor) -> float:
    x = value.detach().float()
    shared = x.mean(dim=1, keepdim=True).expand_as(x)
    return float(shared.square().sum().div(x.square().sum().clamp_min(1.0e-12)).cpu())


def _direct_condition(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    mode: str,
    alpha: float | None,
    refs: Any | None,
    return_branch_prior: bool,
) -> tuple[torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None], dict[str, float]]:
    if model.condition_fusion != "dynamic":
        raise ValueError(f"centered-motion intervention requires flat dynamic checkpoint, got {model.condition_fusion!r}")
    if refs is None:
        refs = model.sample_references(batch)
    frame_tokens, query_points = model.conditioner.tokenize_frames(
        batch["frame_vertices"],
        batch["faces"],
        refs,
        vertex_normals=batch.get("vertex_normals"),
        face_normals=batch.get("face_normals"),
    )

    if mode == "static":
        condition = frame_tokens[:, 0]
        stats = {"static_rms": _rms(condition)}
    elif mode == "centered":
        if alpha is None:
            raise ValueError("centered mode requires alpha")
        result = build_centered_motion_condition(
            frame_tokens,
            query_points,
            model.conditioner.motion_encoder,
            alpha=alpha,
        )
        condition = result.fused
        stats = {
            "static_rms": _rms(result.static),
            "normal_dynamic_rms": _rms(result.dynamic),
            "zero_motion_dynamic_rms": _rms(result.zero_motion_dynamic),
            "zero_motion_rewrite_relative_rms": _relative_rms(
                result.zero_motion_dynamic - result.static,
                result.static,
            ),
            "motion_delta_relative_rms": _relative_rms(result.motion_delta, result.static),
            "fused_residual_relative_rms": _relative_rms(result.fused - result.static, result.static),
            "motion_delta_shared_energy_fraction": _shared_energy_fraction(result.motion_delta),
        }
    else:
        raise ValueError(f"unsupported direct condition mode: {mode}")

    branch_prior = None
    if model.branch_prior is not None:
        branch_prior = model.branch_prior(condition)
        condition = torch.cat([condition, branch_prior["tokens"].to(dtype=condition.dtype)], dim=1)
    output: torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None]
    output = (condition, branch_prior) if return_branch_prior else condition
    return output, stats


def _install_condition_control(
    model: torch.nn.Module,
    *,
    mode: str,
    alpha: float | None,
    captures: list[dict[str, float]],
) -> Any:
    original = model.build_condition
    if mode == "dynamic":
        return original

    def controlled(
        batch: dict[str, Any],
        control: str = "normal",
        refs: Any | None = None,
        return_branch_prior: bool = False,
    ):
        if control != "normal":
            raise ValueError(f"direct centered-motion control does not support nested control={control!r}")
        output, stats = _direct_condition(
            model,
            batch,
            mode=mode,
            alpha=alpha,
            refs=refs,
            return_branch_prior=return_branch_prior,
        )
        captures.append(stats)
        return output

    model.build_condition = controlled
    return original


def _restore_condition(model: torch.nn.Module, original: Any) -> None:
    model.build_condition = types.MethodType(original.__func__, model)


def _aggregate_captures(captures: list[dict[str, float]]) -> dict[str, float]:
    if not captures:
        return {}
    keys = sorted(set.intersection(*(set(item) for item in captures)))
    return {key: float(sum(item[key] for item in captures) / len(captures)) for key in keys}


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main() -> None:
    args = parse_args()
    alphas = _parse_alphas(args.alphas)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (args.manifest, args.checkpoint, args.tokenizer_config, args.model_config, args.unirig_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    model_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        unirig_checkpoint=args.unirig_checkpoint,
    )
    for name in CHECKPOINT_DEFAULTS:
        setattr(model_args, name, None)
    train_args = apply_checkpoint_eval_defaults(model_args)
    if model_args.condition_fusion != "dynamic":
        raise ValueError(f"checkpoint is not the flat dynamic route: {model_args.condition_fusion!r}")

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

    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = _build_dynamic_model(model_args, tokenizer, device)
    generation_kwargs = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1}
    args.output_dir.mkdir(parents=True)

    modes: list[tuple[str, str, float | None]] = [("dynamic", "dynamic", None), ("static", "static", None)]
    modes.extend((_mode_name(alpha), "centered", alpha) for alpha in alphas)
    results: dict[str, dict[str, Any]] = {}
    condition_stats: dict[str, dict[str, float]] = {}

    for name, mode, alpha in modes:
        rows = [{"index": index, "path": str(path)} for index, path in enumerate(dataset.paths)]
        captures: list[dict[str, float]] = []
        original = _install_condition_control(model, mode=mode, alpha=alpha, captures=captures)
        try:
            _run_model(
                "dynamic",
                lambda batch, target_ids: _dynamic_generate(
                    model,
                    tokenizer,
                    batch,
                    args.max_new_tokens,
                    target_ids,
                    generation_kwargs,
                    count_guidance="none",
                    action_guidance="none",
                    branch_prior_guidance="none",
                    branch_parent_snap=False,
                ),
                rows,
                loader,
                tokenizer,
                device,
                amp_dtype,
                args.max_new_tokens,
                args.seed,
            )
        finally:
            _restore_condition(model, original)
        if mode != "dynamic" and len(captures) != len(rows):
            raise RuntimeError(f"condition capture count mismatch for {name}: {len(captures)} != {len(rows)}")

        payload = {
            "summary": {
                "manifest": str(args.manifest),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sample_exposures": train_args.get("sample_exposures"),
                "limit": len(rows),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "condition_mode": mode,
                "alpha": alpha,
                "dynamic": _summarize(rows, "dynamic"),
            },
            "rows": rows,
        }
        results[name] = payload
        condition_stats[name] = _aggregate_captures(captures)
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    token_comparison: dict[str, Any] = {}
    static_rows = results["static"]["rows"]
    for name, _, _ in modes:
        if name == "static":
            continue
        exact = 0
        divergences: list[int] = []
        for static_row, row in zip(static_rows, results[name]["rows"], strict=True):
            left = static_row["dynamic"].get("generated_ids") or []
            right = row["dynamic"].get("generated_ids") or []
            divergence = _first_divergence(left, right)
            if divergence is None:
                exact += 1
            else:
                divergences.append(divergence)
        token_comparison[name] = {
            "exact_generated_ids_vs_static": exact,
            "first_divergence_min": min(divergences) if divergences else None,
            "first_divergence_max": max(divergences) if divergences else None,
        }

    comparison = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "alphas": alphas,
            "formula": "query_static + alpha * (dynamic_normal - dynamic_query_repeated)",
        },
        "condition_stats": condition_stats,
        "token_comparison": token_comparison,
        "summaries": {name: payload["summary"]["dynamic"] for name, payload in results.items()},
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
