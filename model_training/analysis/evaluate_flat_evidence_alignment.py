#!/usr/bin/env python3
"""Intervene on global evidence-frame motion before flat UniRig generation."""

from __future__ import annotations

import argparse
import json
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any

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
from eval_dynamic_rig_generation import (  # noqa: E402
    _dynamic_generate,
    _run_model,
    _summarize,
)
from train_dynamic_rig import build_tokenizer  # noqa: E402


ALIGNMENT_MODES = ("normal", "center", "rigid")


def _proper_rotation(moving: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return the row-vector rotation that best maps moving onto reference."""

    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[-1] != 3:
        raise ValueError(
            f"expected matching (N,3) point sets, got {moving.shape} and {reference.shape}"
        )
    with torch.autocast(device_type=moving.device.type, enabled=False):
        x = moving.float() - moving.float().mean(dim=0, keepdim=True)
        y = reference.float() - reference.float().mean(dim=0, keepdim=True)
        u, _, vh = torch.linalg.svd(x.transpose(0, 1) @ y, full_matrices=False)
        rotation = u @ vh
        if torch.linalg.det(rotation) < 0:
            u = u.clone()
            u[:, -1] *= -1
            rotation = u @ vh
    return rotation


def _align_evidence_batch(batch: dict[str, Any], mode: str) -> dict[str, Any]:
    """Align evidence meshes to the query frame without changing frame zero."""

    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"unknown alignment mode {mode!r}")
    if mode == "normal":
        return batch

    vertices = batch["frame_vertices"]
    if vertices.ndim != 4 or vertices.shape[-1] != 3:
        raise ValueError(f"frame_vertices must be (B,T,N,3), got {tuple(vertices.shape)}")
    vertex_counts = batch.get("vertex_count")
    if vertex_counts is None:
        vertex_counts = torch.full(
            (vertices.shape[0],), vertices.shape[2], device=vertices.device, dtype=torch.long
        )
    face_counts = batch.get("face_count")
    if face_counts is None and batch.get("face_normals") is not None:
        face_counts = torch.full(
            (vertices.shape[0],),
            batch["face_normals"].shape[2],
            device=vertices.device,
            dtype=torch.long,
        )

    aligned = dict(batch)
    aligned_vertices = vertices.clone()
    aligned_vertex_normals = (
        None if batch.get("vertex_normals") is None else batch["vertex_normals"].clone()
    )
    aligned_face_normals = (
        None if batch.get("face_normals") is None else batch["face_normals"].clone()
    )

    for batch_index in range(vertices.shape[0]):
        vertex_count = int(vertex_counts[batch_index].item())
        if vertex_count < 3:
            raise ValueError(f"batch item {batch_index} has only {vertex_count} valid vertices")
        reference = vertices[batch_index, 0, :vertex_count].float()
        reference_center = reference.mean(dim=0, keepdim=True)
        for frame_index in range(1, vertices.shape[1]):
            moving = vertices[batch_index, frame_index, :vertex_count].float()
            moving_center = moving.mean(dim=0, keepdim=True)
            rotation = (
                torch.eye(3, device=vertices.device, dtype=torch.float32)
                if mode == "center"
                else _proper_rotation(moving, reference)
            )
            transformed = (moving - moving_center) @ rotation + reference_center
            aligned_vertices[batch_index, frame_index, :vertex_count] = transformed.to(
                dtype=vertices.dtype
            )
            if mode != "rigid":
                continue
            if aligned_vertex_normals is not None:
                normals = batch["vertex_normals"][batch_index, frame_index, :vertex_count].float()
                aligned_vertex_normals[batch_index, frame_index, :vertex_count] = F.normalize(
                    normals @ rotation, dim=-1, eps=1.0e-12
                ).to(dtype=aligned_vertex_normals.dtype)
            if aligned_face_normals is not None:
                assert face_counts is not None
                face_count = int(face_counts[batch_index].item())
                normals = batch["face_normals"][batch_index, frame_index, :face_count].float()
                aligned_face_normals[batch_index, frame_index, :face_count] = F.normalize(
                    normals @ rotation, dim=-1, eps=1.0e-12
                ).to(dtype=aligned_face_normals.dtype)

    aligned["frame_vertices"] = aligned_vertices
    if aligned_vertex_normals is not None:
        aligned["vertex_normals"] = aligned_vertex_normals
    if aligned_face_normals is not None:
        aligned["face_normals"] = aligned_face_normals
    return aligned


def _install_alignment(model: torch.nn.Module, mode: str):
    original = model.build_condition

    def controlled(
        batch: dict[str, Any],
        control: str = "normal",
        refs: Any | None = None,
        return_branch_prior: bool = False,
    ):
        return original(
            _align_evidence_batch(batch, mode),
            control=control,
            refs=refs,
            return_branch_prior=return_branch_prior,
        )

    model.build_condition = controlled
    return original


def _restore_condition(model: torch.nn.Module, original: Any) -> None:
    model.build_condition = types.MethodType(original.__func__, model)


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if int(left_token) != int(right_token):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare_modes(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normal_rows = results["normal"]["rows"]
    comparison: dict[str, Any] = {}
    for mode in ALIGNMENT_MODES[1:]:
        exact = 0
        divergences: list[int] = []
        for index, (normal, current) in enumerate(
            zip(normal_rows, results[mode]["rows"], strict=True)
        ):
            if normal["path"] != current["path"]:
                raise ValueError(f"row {index} path mismatch")
            normal_ids = [int(value) for value in normal["dynamic"].get("generated_ids") or []]
            current_ids = [int(value) for value in current["dynamic"].get("generated_ids") or []]
            divergence = _first_divergence(normal_ids, current_ids)
            if divergence is None:
                exact += 1
            else:
                divergences.append(divergence)
        comparison[mode] = {
            "rows": len(normal_rows),
            "exact_generated_ids": exact,
            "first_divergence_indices": divergences,
        }
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    if model_args.condition_fusion != "dynamic":
        raise ValueError("evidence alignment audit requires the accepted dynamic route")

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
    for mode in ALIGNMENT_MODES:
        rows = [{"index": index, "path": str(path)} for index, path in enumerate(dataset.paths)]
        original = _install_alignment(model, mode)
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
                "mode": mode,
                "limit": len(rows),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "dynamic": _summarize(rows, "dynamic"),
            },
            "rows": rows,
        }
        results[mode] = payload
        (args.output_dir / f"{mode}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    comparison = {
        "contract": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sample_exposures": train_args.get("sample_exposures"),
            "modes": list(ALIGNMENT_MODES),
            "limit": len(dataset),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "summaries": {
            mode: payload["summary"]["dynamic"] for mode, payload in results.items()
        },
        "token_comparison_to_normal": _compare_modes(results),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
