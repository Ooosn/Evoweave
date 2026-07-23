#!/usr/bin/env python3
"""Evaluate confidence-style interpolation between zero and normal motion."""

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


def _parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not alphas:
        raise ValueError("at least one alpha is required")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError(f"motion gate alphas must be within [0, 1], got {alphas}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"motion gate alphas must be unique, got {alphas}")
    return alphas


def _alpha_name(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def _blend_conditions(
    normal: torch.Tensor,
    zero: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha == 1.0:
        return normal
    if alpha == 0.0:
        return zero
    return zero + float(alpha) * (normal - zero)


def _install_motion_gate(model: torch.nn.Module, alpha: float):
    if getattr(model, "condition_fusion", None) != "dynamic":
        raise ValueError(
            "motion gate sweep requires condition_fusion='dynamic', "
            f"got {getattr(model, 'condition_fusion', None)!r}"
        )
    if getattr(model, "branch_prior", None) is not None:
        raise ValueError("motion gate sweep does not support branch-prior condition tokens")
    original = model.build_condition

    def controlled(
        batch: dict[str, Any],
        control: str = "normal",
        refs: Any | None = None,
        return_branch_prior: bool = False,
    ):
        del control
        if refs is None:
            refs = model.sample_references(batch)
        normal = original(batch, control="normal", refs=refs)
        if float(alpha) == 1.0:
            condition = normal
        else:
            zero = original(batch, control="zero", refs=refs)
            condition = _blend_conditions(normal, zero, float(alpha))
        if return_branch_prior:
            return condition, None
        return condition

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


def _load_reference_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"reference result has no rows list: {path}")
    return rows


def _compare_to_reference(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(reference_rows):
        raise ValueError(
            f"generated rows={len(rows)} do not match reference rows={len(reference_rows)}"
        )
    exact = 0
    divergences: list[int] = []
    for index, (row, reference) in enumerate(zip(rows, reference_rows, strict=True)):
        if str(row["path"]) != str(reference["path"]):
            raise ValueError(f"row {index} path mismatch")
        if row["selected_frames"] != reference["selected_frames"]:
            raise ValueError(f"row {index} selected-frame mismatch")
        if row["target_ids"] != reference["target_ids"]:
            raise ValueError(f"row {index} target-token mismatch")
        generated = [int(value) for value in row["dynamic"].get("generated_ids") or []]
        expected = [int(value) for value in reference["dynamic"].get("generated_ids") or []]
        divergence = _first_divergence(generated, expected)
        if divergence is None:
            exact += 1
        else:
            divergences.append(divergence)
    return {
        "rows": len(rows),
        "exact_generated_ids": exact,
        "first_divergence_indices": divergences,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-normal", type=Path)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = _parse_alphas(args.alphas)
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
    if model_args.condition_fusion != "dynamic":
        raise ValueError(
            "this audit requires the accepted flat dynamic condition route, "
            f"got condition_fusion={model_args.condition_fusion!r}"
        )
    if model_args.branch_prior_proposals != 0:
        raise ValueError("this audit requires branch_prior_proposals=0")

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
    reference_rows = _load_reference_rows(args.reference_normal)
    args.output_dir.mkdir(parents=True)

    results: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        name = _alpha_name(alpha)
        rows = [{"index": index, "path": str(path)} for index, path in enumerate(dataset.paths)]
        original = _install_motion_gate(model, alpha)
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
                "motion_gate_alpha": alpha,
                "dynamic": _summarize(rows, "dynamic"),
            },
            "rows": rows,
        }
        results[name] = payload
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    reference_comparison = None
    if reference_rows is not None:
        alpha_one_name = _alpha_name(1.0)
        if alpha_one_name not in results:
            raise ValueError("reference validation requires alpha=1 in the sweep")
        reference_comparison = _compare_to_reference(
            results[alpha_one_name]["rows"], reference_rows
        )
        if reference_comparison["exact_generated_ids"] != len(reference_rows):
            raise RuntimeError(
                "alpha=1 generation does not exactly reproduce the fixed normal reference: "
                f"{reference_comparison}"
            )

    comparison = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "reference_normal": str(args.reference_normal) if args.reference_normal else None,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "alphas": list(alphas),
        },
        "reference_comparison": reference_comparison,
        "summaries": {name: payload["summary"]["dynamic"] for name, payload in results.items()},
    }
    (args.output_dir / "gate_sweep.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
