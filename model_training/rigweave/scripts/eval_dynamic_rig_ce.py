#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from train_dynamic_rig import (
    build_tokenizer,
    load_unirig,
    move_batch,
)

CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "frames": 24,
    "surface_samples": 65536,
    "vertex_samples": 8192,
    "query_tokens": 1024,
    "register_tokens": 96,
    "motion_depth": 12,
    "motion_heads": 8,
    "use_motion_features": False,
    "use_time_embedding": False,
    "motion_fps_ratio": 0.7,
    "motion_vertex_samples": 512,
    "motion_alignment_policy": "none",
    "target_active_skin_only": False,
    "target_start_policy": "joint0",
    "target_root_policy": "legacy",
    "input_space_policy": "mesh_query_bbox",
    "active_skin_threshold": 1.0e-4,
    "condition_fusion": "dynamic",
    "condition_fusion_heads": 8,
    "condition_fusion_gate_init": 0.25,
    "condition_fusion_depth": 1,
    "condition_static_blend_weight": 0.0,
    "branch_prior_proposals": 0,
    "branch_prior_heads": 8,
    "branch_prior_loss_weight": 0.0,
    "branch_prior_coord_loss_weight": 0.0,
    "explicit_tree_loss_weight": 0.0,
    "explicit_tree_generated_prefix_weight": 0.0,
    "explicit_tree_generated_prefix_states": 4,
    "explicit_tree_generated_prefix_max_steps": 64,
    "explicit_tree_generated_prefix_max_rows": 4,
    "explicit_tree_oracle_prefix_weight": 0.0,
    "explicit_tree_oracle_prefix_states": 4,
    "explicit_tree_oracle_prefix_max_steps": 64,
    "explicit_tree_oracle_prefix_max_rows": 4,
    "explicit_tree_prefix_jitter_weight": 0.0,
    "explicit_tree_prefix_jitter_std": 0.0,
    "explicit_tree_depth": 4,
    "explicit_tree_heads": 8,
    "explicit_tree_topology_mode": "geometry",
    "explicit_tree_coordinate_mode": "absolute",
    "explicit_tree_action_eos_loss_weight": 1.0,
    "explicit_tree_action_child_loss_weight": 1.0,
    "explicit_tree_action_branch_loss_weight": 1.0,
    "explicit_tree_xyz_loss_weight": 1.0,
    "use_grammar_state_embedding": False,
    "use_action_group_bias": False,
    "use_condition_action_group_bias": False,
}


def apply_checkpoint_eval_defaults(args: argparse.Namespace) -> dict[str, Any]:
    """Fill eval architecture/sampling args from the training checkpoint.

    The dynamic condition shape is part of the checkpoint contract.  Keeping
    these defaults tied to the saved training args avoids accidentally
    evaluating a depth-12/register-96 model with an old depth-6/register-32
    command line default.
    """

    bool_names = {
        "use_motion_features",
        "use_time_embedding",
        "target_active_skin_only",
        "use_grammar_state_embedding",
        "use_action_group_bias",
        "use_condition_action_group_bias",
    }
    required_without_bool = [name for name in CHECKPOINT_DEFAULTS if name not in bool_names]
    if all(getattr(args, name, None) is not None for name in required_without_bool):
        for name in bool_names:
            if getattr(args, name, None) is None:
                setattr(args, name, False)
        return {}

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}) or {})
    for name, default_value in CHECKPOINT_DEFAULTS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, train_args.get(name, default_value))
    del ckpt
    gc.collect()
    return train_args


def _build_dynamic_model(args: argparse.Namespace, tokenizer: Any, device: torch.device) -> torch.nn.Module:
    from rigweave.dynamic_rig import (
        FixedQuerySurfaceTokenizer,
        FrameTypeAnchorWiseAlternatingMotionEncoder,
        LegacyTemporalMotionEncoder,
        TemporalMotionEncoder,
    )
    from rigweave.dynamic_rig.model import DynamicRigConditioner
    from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    legacy_motion_encoder = any(
        key.startswith("conditioner.motion_encoder.blocks.") and ".spatial." in key for key in state
    )
    frame_type_motion_encoder = (
        not legacy_motion_encoder
        and "conditioner.motion_encoder.frame_type_embed.weight" in state
        and "conditioner.motion_encoder.role_token" not in state
    )

    unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint)
    surface_tokenizer = FixedQuerySurfaceTokenizer(unirig.mesh_encoder, unirig.output_proj)
    if legacy_motion_encoder:
        motion_encoder_cls = LegacyTemporalMotionEncoder
    elif frame_type_motion_encoder:
        motion_encoder_cls = FrameTypeAnchorWiseAlternatingMotionEncoder
    else:
        motion_encoder_cls = TemporalMotionEncoder
    print(
        "[eval_dynamic_rig] motion_encoder="
        f"{motion_encoder_cls.__name__} checkpoint={args.checkpoint}",
        flush=True,
    )
    time_embed_key = "conditioner.motion_encoder.time_embed"
    checkpoint_max_frames = int(state[time_embed_key].shape[0]) if time_embed_key in state else 32
    motion_kwargs: dict[str, Any] = {
        "dim": unirig.hidden_size,
        "depth": args.motion_depth,
        "heads": args.motion_heads,
        "register_tokens": args.register_tokens,
        "max_frames": max(args.frames, checkpoint_max_frames),
    }
    if motion_encoder_cls is TemporalMotionEncoder:
        motion_kwargs["use_motion_features"] = getattr(args, "use_motion_features", False)
        motion_kwargs["use_time_embedding"] = getattr(args, "use_time_embedding", False)
    motion_encoder = motion_encoder_cls(**motion_kwargs)
    conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
    model = DynamicRigUniRigAR(
        unirig,
        conditioner,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        condition_fusion=args.condition_fusion,
        condition_fusion_heads=args.condition_fusion_heads,
        condition_fusion_gate_init=args.condition_fusion_gate_init,
        condition_fusion_depth=args.condition_fusion_depth,
        condition_static_blend_weight=args.condition_static_blend_weight,
        use_grammar_state_embedding=args.use_grammar_state_embedding,
        use_action_group_bias=args.use_action_group_bias,
        use_condition_action_group_bias=args.use_condition_action_group_bias,
        branch_prior_proposals=args.branch_prior_proposals,
        branch_prior_heads=args.branch_prior_heads,
        branch_prior_loss_weight=args.branch_prior_loss_weight,
        branch_prior_coord_loss_weight=args.branch_prior_coord_loss_weight,
        explicit_tree_loss_weight=args.explicit_tree_loss_weight,
        explicit_tree_generated_prefix_weight=args.explicit_tree_generated_prefix_weight,
        explicit_tree_generated_prefix_states=args.explicit_tree_generated_prefix_states,
        explicit_tree_generated_prefix_max_steps=args.explicit_tree_generated_prefix_max_steps,
        explicit_tree_generated_prefix_max_rows=args.explicit_tree_generated_prefix_max_rows,
        explicit_tree_oracle_prefix_weight=args.explicit_tree_oracle_prefix_weight,
        explicit_tree_oracle_prefix_states=args.explicit_tree_oracle_prefix_states,
        explicit_tree_oracle_prefix_max_steps=args.explicit_tree_oracle_prefix_max_steps,
        explicit_tree_oracle_prefix_max_rows=args.explicit_tree_oracle_prefix_max_rows,
        explicit_tree_prefix_jitter_weight=args.explicit_tree_prefix_jitter_weight,
        explicit_tree_prefix_jitter_std=args.explicit_tree_prefix_jitter_std,
        explicit_tree_depth=args.explicit_tree_depth,
        explicit_tree_heads=args.explicit_tree_heads,
        explicit_tree_topology_mode=args.explicit_tree_topology_mode,
        explicit_tree_coordinate_mode=args.explicit_tree_coordinate_mode,
        explicit_tree_action_eos_loss_weight=args.explicit_tree_action_eos_loss_weight,
        explicit_tree_action_child_loss_weight=args.explicit_tree_action_child_loss_weight,
        explicit_tree_action_branch_loss_weight=args.explicit_tree_action_branch_loss_weight,
        explicit_tree_xyz_loss_weight=args.explicit_tree_xyz_loss_weight,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    _move_dynamic_model_to_device(model, device)
    del ckpt, state
    gc.collect()
    if missing:
        print(f"[eval_dynamic_rig] WARNING missing dynamic keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[eval_dynamic_rig] WARNING unexpected dynamic keys: {len(unexpected)}", flush=True)
    model.eval()
    return model


def _move_dynamic_model_to_device(model: torch.nn.Module, device: torch.device) -> None:
    """Move the dynamic model without recursively revisiting shared modules.

    The dynamic wrapper intentionally shares UniRig's mesh encoder/output
    projection between `unirig_ar` and the dynamic surface tokenizer.  On some
    qlogin H100 sessions a single recursive `model.to(device)` has triggered a
    CUDA driver OOM even though the model is only about 2.5GB in fp32 and a
    module-wise move succeeds.  Moving the actual shared submodules once keeps
    the eval path equivalent while avoiding that driver/memory-manager corner.
    """

    model.conditioner.motion_encoder.to(device)
    model.condition_fuser.to(device)
    model.grammar_state_proj.to(device)
    model.action_group_bias_head.to(device)
    model.condition_action_group_bias_head.to(device)
    model.structure_count_head.to(device)
    model.structure_action_head.to(device)
    if getattr(model, "branch_prior", None) is not None:
        model.branch_prior.to(device)
    if getattr(model, "explicit_tree_decoder", None) is not None:
        model.explicit_tree_decoder.to(device)
    model.unirig_ar.mesh_encoder.to(device)
    model.unirig_ar.output_proj.to(device)
    model.unirig_ar.transformer.to(device)


def _control_batch(batch: dict[str, Any], control: str, seed: int) -> dict[str, Any]:
    if control == "normal":
        return batch
    out = dict(batch)
    order = None

    def edit_sequence(sequence: torch.Tensor | None) -> torch.Tensor | None:
        nonlocal order
        if sequence is None:
            return None
        frames = sequence.clone()
        if control == "zero":
            frames[:, 1:] = frames[:, :1]
        elif control == "shuffle":
            if frames.shape[1] > 2:
                if order is None:
                    gen = torch.Generator(device=frames.device)
                    gen.manual_seed(seed)
                    perm = torch.randperm(frames.shape[1] - 1, generator=gen, device=frames.device) + 1
                    order = torch.cat([torch.zeros(1, dtype=torch.long, device=frames.device), perm], dim=0)
                frames = frames[:, order]
        elif control == "reverse":
            if frames.shape[1] > 2:
                frames[:, 1:] = torch.flip(frames[:, 1:], dims=[1])
        else:
            raise ValueError(f"unknown control {control}")
        return frames

    out["frame_vertices"] = edit_sequence(batch["frame_vertices"])
    out["vertex_normals"] = edit_sequence(batch.get("vertex_normals"))
    out["face_normals"] = edit_sequence(batch.get("face_normals"))
    return out


@torch.no_grad()
def evaluate_dynamic(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    controls: list[str],
    seed: int,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for control in controls:
        losses: list[float] = []
        dis: list[float] = []
        for idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            batch = _control_batch(batch, control, seed=20260527 + idx)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + idx)
            torch.manual_seed(seed + idx)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(batch)
            losses.append(float(out["ce_loss"].detach().cpu()))
            dis.append(float(out["dis_loss"].detach().cpu()))
        metrics[control] = {
            "ce": sum(losses) / max(1, len(losses)),
            "dis": sum(dis) / max(1, len(dis)),
            "count": len(losses),
        }
    return metrics


@torch.no_grad()
def evaluate_static_base(
    unirig: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    losses: list[float] = []
    dis: list[float] = []
    unirig.to(device)
    unirig.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        static_batch = {
            "vertices": batch["frame_vertices"][:, 0],
            "normals": batch["vertex_normals"][:, 0],
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = unirig.training_step(static_batch)
        losses.append(float(out["ce_loss"].detach().cpu()))
        dis.append(float(out["dis_loss"].detach().cpu()))
    return {
        "ce": sum(losses) / max(1, len(losses)),
        "dis": sum(dis) / max(1, len(dis)),
        "count": len(losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DynamicRig CE and motion controls.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("EVOWEAVE_TEST_MANIFEST", "rigweave/configs/MISSING_TEST_MANIFEST.jsonl")),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, default=Path(os.environ.get("EVOWEAVE_TOKENIZER_CONFIG", "external/UniRig/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml")))
    parser.add_argument("--model-config", type=Path, default=Path(os.environ.get("EVOWEAVE_MODEL_CONFIG", "external/UniRig/configs/model/unirig_ar_350m_1024_81920_float32.yaml")))
    parser.add_argument("--unirig-checkpoint", type=Path, default=Path(os.environ.get("EVOWEAVE_UNIRIG_CKPT", "external/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt")))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--surface-samples", type=int, default=None)
    parser.add_argument("--vertex-samples", type=int, default=None)
    parser.add_argument("--query-tokens", type=int, default=None)
    parser.add_argument("--register-tokens", type=int, default=None)
    parser.add_argument("--motion-depth", type=int, default=None)
    parser.add_argument("--motion-heads", type=int, default=None)
    parser.add_argument("--use-motion-features", action="store_true", default=None)
    parser.add_argument("--use-time-embedding", action="store_true", default=None)
    parser.add_argument("--motion-fps-ratio", type=float, default=None)
    parser.add_argument("--motion-vertex-samples", type=int, default=None)
    parser.add_argument(
        "--motion-alignment-policy",
        choices=["none", "query_rigid"],
        default=None,
    )
    parser.add_argument("--target-active-skin-only", action="store_true", default=None)
    parser.add_argument("--active-skin-threshold", type=float, default=None)
    parser.add_argument("--target-start-policy", choices=["joint0"], default=None)
    parser.add_argument("--target-root-policy", choices=["legacy"], default=None)
    parser.add_argument(
        "--condition-fusion",
        choices=[
            "dynamic",
            "static_blend",
            "static_cross_attn",
            "static_cross_attn_zero",
            "anchor_motion_residual_zero",
        ],
        default=None,
    )
    parser.add_argument("--condition-fusion-heads", type=int, default=None)
    parser.add_argument("--condition-fusion-gate-init", type=float, default=None)
    parser.add_argument("--condition-fusion-depth", type=int, default=None)
    parser.add_argument("--condition-static-blend-weight", type=float, default=None)
    parser.add_argument("--branch-prior-proposals", type=int, default=None)
    parser.add_argument("--branch-prior-heads", type=int, default=None)
    parser.add_argument("--branch-prior-loss-weight", type=float, default=None)
    parser.add_argument("--branch-prior-coord-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-generated-prefix-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-generated-prefix-states", type=int, default=None)
    parser.add_argument("--explicit-tree-generated-prefix-max-steps", type=int, default=None)
    parser.add_argument("--explicit-tree-generated-prefix-max-rows", type=int, default=None)
    parser.add_argument("--explicit-tree-oracle-prefix-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-oracle-prefix-states", type=int, default=None)
    parser.add_argument("--explicit-tree-oracle-prefix-max-steps", type=int, default=None)
    parser.add_argument("--explicit-tree-oracle-prefix-max-rows", type=int, default=None)
    parser.add_argument("--explicit-tree-prefix-jitter-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-prefix-jitter-std", type=float, default=None)
    parser.add_argument("--explicit-tree-depth", type=int, default=None)
    parser.add_argument("--explicit-tree-heads", type=int, default=None)
    parser.add_argument(
        "--explicit-tree-topology-mode",
        choices=["geometry", "topology", "hybrid", "split", "planner", "topomlp"],
        default=None,
    )
    parser.add_argument("--explicit-tree-coordinate-mode", choices=["absolute", "parent_delta"], default=None)
    parser.add_argument("--explicit-tree-action-eos-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-action-child-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-action-branch-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-xyz-loss-weight", type=float, default=None)
    parser.add_argument("--use-grammar-state-embedding", action="store_true", default=None)
    parser.add_argument("--use-action-group-bias", action="store_true", default=None)
    parser.add_argument("--use-condition-action-group-bias", action="store_true", default=None)
    parser.add_argument("--random-query", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--skip-static-base", action="store_true")
    parser.add_argument("--controls", type=str, default="normal,zero,shuffle,reverse")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()
    train_args = apply_checkpoint_eval_defaults(args)

    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    tokenizer = build_tokenizer(args.tokenizer_config)
    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=args.frames,
        limit=args.limit,
        random_query=args.random_query,
        seed=args.seed,
        motion_fps_ratio=args.motion_fps_ratio,
        motion_vertex_samples=args.motion_vertex_samples,
        motion_alignment_policy=args.motion_alignment_policy,
        target_active_skin_only=args.target_active_skin_only,
        active_skin_threshold=args.active_skin_threshold,
        target_start_policy=args.target_start_policy,
        target_root_policy=args.target_root_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )

    dynamic_model = _build_dynamic_model(args, tokenizer, device)
    result: dict[str, Any] = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "eval_contract": {
            "frames": args.frames,
            "surface_samples": args.surface_samples,
            "vertex_samples": args.vertex_samples,
            "query_tokens": args.query_tokens,
            "register_tokens": args.register_tokens,
            "motion_depth": args.motion_depth,
            "motion_heads": args.motion_heads,
            "use_motion_features": args.use_motion_features,
            "use_time_embedding": args.use_time_embedding,
            "motion_fps_ratio": args.motion_fps_ratio,
            "motion_vertex_samples": args.motion_vertex_samples,
            "motion_alignment_policy": args.motion_alignment_policy,
            "target_active_skin_only": args.target_active_skin_only,
            "target_root_policy": args.target_root_policy,
            "active_skin_threshold": args.active_skin_threshold,
            "condition_fusion": args.condition_fusion,
            "condition_fusion_heads": args.condition_fusion_heads,
            "condition_fusion_gate_init": args.condition_fusion_gate_init,
            "condition_fusion_depth": args.condition_fusion_depth,
            "condition_static_blend_weight": args.condition_static_blend_weight,
            "branch_prior_proposals": args.branch_prior_proposals,
            "branch_prior_heads": args.branch_prior_heads,
            "branch_prior_loss_weight": args.branch_prior_loss_weight,
            "branch_prior_coord_loss_weight": args.branch_prior_coord_loss_weight,
            "explicit_tree_loss_weight": args.explicit_tree_loss_weight,
            "explicit_tree_generated_prefix_weight": args.explicit_tree_generated_prefix_weight,
            "explicit_tree_oracle_prefix_weight": args.explicit_tree_oracle_prefix_weight,
            "explicit_tree_prefix_jitter_weight": args.explicit_tree_prefix_jitter_weight,
            "explicit_tree_prefix_jitter_std": args.explicit_tree_prefix_jitter_std,
            "explicit_tree_depth": args.explicit_tree_depth,
            "explicit_tree_heads": args.explicit_tree_heads,
            "explicit_tree_topology_mode": args.explicit_tree_topology_mode,
            "explicit_tree_coordinate_mode": args.explicit_tree_coordinate_mode,
            "explicit_tree_action_weights": [
                args.explicit_tree_action_eos_loss_weight,
                args.explicit_tree_action_child_loss_weight,
                args.explicit_tree_action_branch_loss_weight,
            ],
            "explicit_tree_xyz_loss_weight": args.explicit_tree_xyz_loss_weight,
            "use_grammar_state_embedding": args.use_grammar_state_embedding,
            "use_action_group_bias": args.use_action_group_bias,
            "use_condition_action_group_bias": args.use_condition_action_group_bias,
            "random_query": args.random_query,
            "training_max_steps": train_args.get("max_steps"),
        },
        "dynamic": evaluate_dynamic(
            dynamic_model,
            loader,
            device,
            amp_dtype,
            [x.strip() for x in args.controls.split(",") if x.strip()],
            args.seed,
        ),
    }

    if not args.skip_static_base:
        base_unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint)
        result["static_base"] = evaluate_static_base(base_unirig, loader, device, amp_dtype)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
