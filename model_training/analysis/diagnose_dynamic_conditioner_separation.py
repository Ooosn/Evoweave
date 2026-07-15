#!/usr/bin/env python3
"""Measure whether a trained dynamic conditioner preserves geometry differences."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def _pair_report(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"pair shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    left_f = left.float()
    right_f = right.float()
    delta = left_f - right_f
    denominator = 0.5 * (_rms(left_f) + _rms(right_f))
    left_centered = left_f - left_f.mean(dim=-2, keepdim=True)
    right_centered = right_f - right_f.mean(dim=-2, keepdim=True)
    centered_delta = left_centered - right_centered
    centered_denominator = 0.5 * (_rms(left_centered) + _rms(right_centered))
    return {
        "left_rms": _rms(left_f),
        "right_rms": _rms(right_f),
        "delta_rms": _rms(delta),
        "relative_delta_rms": _rms(delta) / max(denominator, 1.0e-12),
        "centered_delta_rms": _rms(centered_delta),
        "relative_centered_delta_rms": _rms(centered_delta) / max(centered_denominator, 1.0e-12),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                left_f.reshape(left_f.shape[0], -1),
                right_f.reshape(right_f.shape[0], -1),
                dim=-1,
            ).mean().item()
        ),
    }


def _slot_report(value: torch.Tensor) -> dict[str, float]:
    value_f = value.float()
    common = value_f.mean(dim=-2, keepdim=True)
    local = value_f - common
    common_energy = common.square().mean()
    local_energy = local.square().mean()
    return {
        "total_rms": _rms(value_f),
        "common_rms": _rms(common),
        "local_rms": _rms(local),
        "common_energy_fraction": float(
            (common_energy / (common_energy + local_energy).clamp_min(1.0e-20)).item()
        ),
    }


def _extract_prefixed_state(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _load_conditioner(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    scripts_dir = args.model_root / "rigweave" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from train_dynamic_rig import build_tokenizer, load_unirig

    from rigweave.dynamic_rig import AnchorWiseAlternatingMotionEncoder, FixedQuerySurfaceTokenizer
    from rigweave.dynamic_rig.model import DynamicRigConditioner

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(checkpoint.get("args", {}) or {})
    state = dict(checkpoint["model"])

    tokenizer = build_tokenizer(args.tokenizer_config)
    unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint)
    surface_tokenizer = FixedQuerySurfaceTokenizer(unirig.mesh_encoder, unirig.output_proj)
    motion_encoder = AnchorWiseAlternatingMotionEncoder(
        dim=int(unirig.hidden_size),
        depth=int(train_args.get("motion_depth", 12)),
        heads=int(train_args.get("motion_heads", 8)),
        register_tokens=int(train_args.get("register_tokens", 96)),
        max_frames=max(int(train_args.get("frames", 24)), 48),
        use_motion_features=bool(train_args.get("use_motion_features", False)),
        use_time_embedding=bool(train_args.get("use_time_embedding", False)),
        gradient_checkpointing=False,
    )
    conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)

    surface_state = _extract_prefixed_state(state, "conditioner.surface_tokenizer.")
    motion_state = _extract_prefixed_state(state, "conditioner.motion_encoder.")
    if not surface_state or not motion_state:
        raise RuntimeError(
            "checkpoint does not contain a complete conditioner: "
            f"surface_keys={len(surface_state)} motion_keys={len(motion_state)}"
        )
    surface_missing, surface_unexpected = surface_tokenizer.load_state_dict(surface_state, strict=False)
    motion_missing, motion_unexpected = motion_encoder.load_state_dict(motion_state, strict=False)
    if surface_missing or surface_unexpected or motion_missing or motion_unexpected:
        raise RuntimeError(
            "conditioner checkpoint mismatch: "
            f"surface_missing={surface_missing} surface_unexpected={surface_unexpected} "
            f"motion_missing={motion_missing} motion_unexpected={motion_unexpected}"
        )

    metadata = {
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sample_seen": checkpoint.get("sample_seen"),
        "train_args": train_args,
        "surface_state_keys": len(surface_state),
        "motion_state_keys": len(motion_state),
    }
    del checkpoint, state, unirig
    gc.collect()
    conditioner.to(device)
    conditioner.eval()
    return conditioner, metadata


@torch.no_grad()
def _trace_condition(
    conditioner: torch.nn.Module,
    batch: dict[str, Any],
    refs: Any,
) -> dict[str, torch.Tensor]:
    from rigweave.dynamic_rig.sampling import materialize_trackable_surface

    frame_tokens = []
    frame_query_points = []
    for frame_index in range(int(batch["frame_vertices"].shape[1])):
        samples = materialize_trackable_surface(
            batch["frame_vertices"][:, frame_index],
            batch["faces"],
            refs,
            vertex_normals=batch["vertex_normals"][:, frame_index],
            face_normals=batch["face_normals"][:, frame_index],
        )
        frame_tokens.append(
            conditioner.surface_tokenizer(
                samples.dense_points,
                samples.dense_normals,
                samples.query_points,
                samples.query_normals,
            )
        )
        frame_query_points.append(samples.query_points)

    surface_sequence = torch.stack(frame_tokens, dim=1)
    query_sequence = torch.stack(frame_query_points, dim=1)
    motion_encoder = conditioner.motion_encoder
    z = surface_sequence
    if motion_encoder.use_motion_features:
        features = motion_encoder._motion_features(query_sequence).to(dtype=z.dtype)
        z = z + motion_encoder.motion_feature_mlp(features)

    batch_size, frame_count, _, dim = z.shape
    if motion_encoder.register_tokens > 0:
        registers = motion_encoder.register.view(
            1, 1, motion_encoder.register_tokens, dim
        ).expand(batch_size, frame_count, -1, -1)
    else:
        registers = z.new_empty((batch_size, frame_count, 0, dim))
    canonical_role = motion_encoder.role_token[:, 0:1].expand(batch_size, 1, -1, -1)
    evidence_roles = motion_encoder.role_token[:, 1:2].expand(
        batch_size, max(0, frame_count - 1), -1, -1
    )
    roles = torch.cat([canonical_role, evidence_roles], dim=1)
    z = torch.cat([roles, registers, z], dim=2)
    if motion_encoder.use_time_embedding:
        z = z + motion_encoder.time_embed[:frame_count].view(1, frame_count, 1, dim)

    anchor_start = 1 + int(motion_encoder.register_tokens)
    stages = {
        "query_points_frame0": query_sequence[:, 0],
        "surface_tokens_frame0": surface_sequence[:, 0],
        "motion_input_frame0": z[:, 0, anchor_start:],
    }
    for block_index, block in enumerate(motion_encoder.blocks):
        z = block(z)
        stages[f"motion_block_{block_index:02d}_frame0"] = z[:, 0, anchor_start:]
    stages["motion_output_frame0"] = motion_encoder.norm(z)[:, 0, anchor_start:]
    return stages


def _make_batch(dataset: Any, index: int, device: torch.device) -> dict[str, Any]:
    from rigweave.dynamic_rig import puppeteer_dynamic_collate

    return _move_batch(puppeteer_dynamic_collate([dataset[int(index)]]), device)


def _sample_refs(
    batch: dict[str, Any],
    *,
    surface_samples: int,
    vertex_samples: int,
    query_tokens: int,
    seed: int,
) -> Any:
    from rigweave.dynamic_rig.sampling import sample_trackable_surface

    generator = torch.Generator(device=batch["frame_vertices"].device)
    generator.manual_seed(int(seed))
    return sample_trackable_surface(
        batch["frame_vertices"][:, 0],
        batch["faces"],
        num_samples=int(surface_samples),
        vertex_samples=int(vertex_samples),
        query_tokens=int(query_tokens),
        vertex_counts=batch.get("vertex_count"),
        face_counts=batch.get("face_count"),
        generator=generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--other-index", type=int, default=1)
    parser.add_argument("--pose-seed-a", type=int, default=2026071501)
    parser.add_argument("--pose-seed-b", type=int, default=2026071599)
    parser.add_argument("--reference-seed", type=int, default=20260715)
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_root / "rigweave" / "src"))
    sys.path.insert(0, str(args.model_root / "third_party_references" / "Puppeteer" / "skeleton"))
    from rigweave.dynamic_rig import PuppeteerDynamicRigDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditioner, metadata = _load_conditioner(args, device)
    train_args = metadata["train_args"]
    dataset_kwargs = {
        "frame_count": int(train_args.get("frames", 24)),
        "limit": max(int(args.index), int(args.other_index)) + 1,
        "random_query": True,
        "motion_fps_ratio": float(train_args.get("motion_fps_ratio", 0.7)),
        "motion_vertex_samples": int(train_args.get("motion_vertex_samples", 512)),
        "max_joints": int(train_args.get("n_max_joints", 101)),
    }
    dataset_a = PuppeteerDynamicRigDataset(args.manifest, seed=args.pose_seed_a, **dataset_kwargs)
    dataset_b = PuppeteerDynamicRigDataset(args.manifest, seed=args.pose_seed_b, **dataset_kwargs)
    batch_a = _make_batch(dataset_a, args.index, device)
    batch_b = _make_batch(dataset_b, args.index, device)
    batch_other = _make_batch(dataset_a, args.other_index, device)

    sample_kwargs = {
        "surface_samples": int(train_args.get("surface_samples", 65536)),
        "vertex_samples": int(train_args.get("vertex_samples", 8192)),
        "query_tokens": int(train_args.get("query_tokens", 1024)),
        "seed": args.reference_seed,
    }
    refs_a = _sample_refs(batch_a, **sample_kwargs)
    same_asset_a = _trace_condition(conditioner, batch_a, refs_a)
    same_asset_b = _trace_condition(conditioner, batch_b, refs_a)
    refs_other = _sample_refs(batch_other, **sample_kwargs)
    other_asset = _trace_condition(conditioner, batch_other, refs_other)

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "metadata": metadata,
        "same_asset": {
            "path": batch_a["path"][0],
            "selected_frames_a": batch_a["selected_frames"][0].detach().cpu().tolist(),
            "selected_frames_b": batch_b["selected_frames"][0].detach().cpu().tolist(),
            "stage_pairs": {
                name: _pair_report(same_asset_a[name], same_asset_b[name])
                for name in same_asset_a
            },
            "stage_structure_a": {
                name: _slot_report(value) for name, value in same_asset_a.items()
            },
            "stage_structure_b": {
                name: _slot_report(value) for name, value in same_asset_b.items()
            },
        },
        "cross_asset": {
            "path_a": batch_a["path"][0],
            "path_b": batch_other["path"][0],
            "stage_pairs": {
                name: _pair_report(same_asset_a[name], other_asset[name])
                for name in same_asset_a
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "same_asset": {
            name: values["relative_delta_rms"]
            for name, values in report["same_asset"]["stage_pairs"].items()
        },
        "cross_asset": {
            name: values["relative_delta_rms"]
            for name, values in report["cross_asset"]["stage_pairs"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
