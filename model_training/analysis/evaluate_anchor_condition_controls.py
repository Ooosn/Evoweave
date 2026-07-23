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
from eval_dynamic_rig_generation import (  # noqa: E402
    _dynamic_generate,
    _run_model,
    _summarize,
)
from train_dynamic_rig import build_tokenizer  # noqa: E402


CONTROL_MODES = ("normal", "zero_motion", "static_bypass", "dynamic_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched free-generation controls for an anchor-motion-residual "
            "checkpoint without changing its parameters."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def _anchor_condition(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    mode: str,
    refs: Any | None,
    return_branch_prior: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    if mode not in {"static_bypass", "dynamic_only"}:
        raise ValueError(f"unsupported direct anchor condition mode: {mode}")
    if model.condition_fusion != "anchor_motion_residual_zero":
        raise ValueError(
            "direct anchor controls require condition_fusion='anchor_motion_residual_zero', "
            f"got {model.condition_fusion!r}"
        )
    if refs is None:
        refs = model.sample_references(batch)
    frame_tokens, query_points = model.conditioner.tokenize_frames(
        batch["frame_vertices"],
        batch["faces"],
        refs,
        vertex_normals=batch.get("vertex_normals"),
        face_normals=batch.get("face_normals"),
    )
    if mode == "static_bypass":
        condition = frame_tokens[:, 0]
    else:
        condition = model.conditioner.motion_encoder(frame_tokens, query_points=query_points)

    branch_prior = None
    if model.branch_prior is not None:
        branch_prior = model.branch_prior(condition)
        condition = torch.cat(
            [condition, branch_prior["tokens"].to(dtype=condition.dtype)],
            dim=1,
        )
    if return_branch_prior:
        return condition, branch_prior
    return condition


def _install_condition_control(model: torch.nn.Module, mode: str):
    original = model.build_condition

    if mode == "normal":
        return original

    if mode == "zero_motion":
        def controlled(
            batch: dict[str, Any],
            control: str = "normal",
            refs: Any | None = None,
            return_branch_prior: bool = False,
        ):
            del control
            return original(
                batch,
                control="zero",
                refs=refs,
                return_branch_prior=return_branch_prior,
            )
    else:
        def controlled(
            batch: dict[str, Any],
            control: str = "normal",
            refs: Any | None = None,
            return_branch_prior: bool = False,
        ):
            del control
            return _anchor_condition(
                model,
                batch,
                mode=mode,
                refs=refs,
                return_branch_prior=return_branch_prior,
            )

    model.build_condition = controlled
    return original


def _restore_condition(model: torch.nn.Module, original: Any) -> None:
    model.build_condition = types.MethodType(original.__func__, model)


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _token_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normal_rows = results["normal"]["rows"]
    comparison: dict[str, Any] = {}
    for mode in CONTROL_MODES[1:]:
        rows = results[mode]["rows"]
        divergences: list[int] = []
        exact = 0
        for normal_row, row in zip(normal_rows, rows, strict=True):
            left = normal_row["dynamic"].get("generated_ids") or []
            right = row["dynamic"].get("generated_ids") or []
            divergence = _first_divergence(left, right)
            if divergence is None:
                exact += 1
            else:
                divergences.append(divergence)
        comparison[mode] = {
            "rows": len(rows),
            "exact_generated_ids": exact,
            "first_divergence_indices": divergences,
            "first_divergence_min": min(divergences) if divergences else None,
            "first_divergence_max": max(divergences) if divergences else None,
        }
    return comparison


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
    if model_args.condition_fusion != "anchor_motion_residual_zero":
        raise ValueError(
            "checkpoint is not the anchor residual route: "
            f"condition_fusion={model_args.condition_fusion!r}"
        )

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
    generation_kwargs = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1}
    args.output_dir.mkdir(parents=True)

    results: dict[str, dict[str, Any]] = {}
    for mode in CONTROL_MODES:
        rows = [{"index": index, "path": str(path)} for index, path in enumerate(dataset.paths)]
        original = _install_condition_control(model, mode)
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

        payload = {
            "summary": {
                "manifest": str(args.manifest),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sample_exposures": train_args.get("sample_exposures"),
                "limit": len(rows),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "condition_control": mode,
                "dynamic": _summarize(rows, "dynamic"),
            },
            "rows": rows,
        }
        results[mode] = payload
        (args.output_dir / f"{mode}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    comparison = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "modes": list(CONTROL_MODES),
        },
        "token_comparison_to_normal": _token_comparison(results),
        "summaries": {mode: payload["summary"]["dynamic"] for mode, payload in results.items()},
    }
    (args.output_dir / "control_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
