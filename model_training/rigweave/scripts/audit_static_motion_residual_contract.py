#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.functional import pad

from train_dynamic_rig import (
    build_tokenizer,
    load_unirig,
    move_batch,
    move_dynamic_model_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the full static-condition plus motion-residual training contract."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--surface-samples", type=int, default=65536)
    parser.add_argument("--vertex-samples", type=int, default=8192)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--register-tokens", type=int, default=96)
    parser.add_argument("--motion-depth", type=int, default=12)
    parser.add_argument("--motion-heads", type=int, default=8)
    parser.add_argument("--motion-fps-ratio", type=float, default=0.7)
    parser.add_argument("--motion-vertex-samples", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=0.25)
    parser.add_argument(
        "--condition-fusion",
        choices=("static_cross_attn_zero", "anchor_motion_residual_zero"),
        default="static_cross_attn_zero",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def teacher_logits(model: Any, condition: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    condition = condition.to(dtype=model.transformer.dtype)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    inputs_embeds = model.token_inputs_embeds(input_ids, attention_mask)
    inputs_embeds = torch.cat([condition, inputs_embeds], dim=1)
    full_attention = pad(attention_mask, (condition.shape[1], 0, 0, 0), value=1.0)
    output = model.transformer(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention,
        use_cache=False,
        output_hidden_states=False,
    )
    logits = output.logits[:, condition.shape[1] :].reshape(
        input_ids.shape[0], -1, model.tokenizer.vocab_size
    )
    return logits[:, :-1]


def grad_l1(module: torch.nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().float().abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def audit_optimizer_coverage(model: Any) -> dict[str, int]:
    modules = {
        "motion": model.conditioner.motion_encoder,
        "ar": model.unirig_ar.transformer,
        "surface": model.conditioner.surface_tokenizer,
        "condition_fusion": model.condition_fuser,
    }
    groups = {
        name: [parameter for parameter in module.parameters() if parameter.requires_grad]
        for name, module in modules.items()
    }
    grouped = [parameter for parameters in groups.values() for parameter in parameters]
    grouped_ids = [id(parameter) for parameter in grouped]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer contract contains duplicate parameters")

    model_trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model_ids = {id(parameter) for parameter in model_trainable}
    grouped_id_set = set(grouped_ids)
    if model_ids != grouped_id_set:
        raise RuntimeError(
            "optimizer coverage mismatch: "
            f"missing={len(model_ids - grouped_id_set)} extra={len(grouped_id_set - model_ids)}"
        )
    return {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    } | {"total": sum(parameter.numel() for parameter in model_trainable)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full model contract audit")
    for path in (
        args.manifest,
        args.tokenizer_config,
        args.model_config,
        args.unirig_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    from rigweave.dynamic_rig import AnchorWiseAlternatingMotionEncoder, FixedQuerySurfaceTokenizer
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
    from rigweave.dynamic_rig.model import DynamicRigConditioner
    from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR

    tokenizer = build_tokenizer(args.tokenizer_config)
    unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint)
    surface_tokenizer = FixedQuerySurfaceTokenizer(unirig.mesh_encoder, unirig.output_proj)
    motion_encoder = AnchorWiseAlternatingMotionEncoder(
        dim=unirig.hidden_size,
        depth=args.motion_depth,
        heads=args.motion_heads,
        register_tokens=args.register_tokens,
        max_frames=max(args.frames, 48),
        use_motion_features=False,
        use_time_embedding=False,
        gradient_checkpointing=False,
    )
    conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
    model = DynamicRigUniRigAR(
        unirig,
        conditioner,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        condition_fusion=args.condition_fusion,
        condition_fusion_heads=8,
        condition_fusion_gate_init=args.gate_init,
        condition_fusion_depth=1,
        branch_prior_proposals=0,
    )
    model.condition_fuser.reset_parameters(
        gate_init=args.gate_init,
        zero_init_update=True,
    )
    optimizer_counts = audit_optimizer_coverage(model)
    move_dynamic_model_to_device(model, device)
    model.eval()

    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=args.frames,
        random_query=False,
        seed=args.seed,
        motion_fps_ratio=args.motion_fps_ratio,
        motion_vertex_samples=args.motion_vertex_samples,
        target_start_policy="joint0",
        target_root_policy="legacy",
    )
    index = args.manifest_index % len(dataset)
    batch = dynamic_rig_collate([dataset[index]], pad_token=tokenizer.pad)
    batch = move_batch(batch, device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        refs = model.sample_references(batch)
        if args.condition_fusion == "anchor_motion_residual_zero":
            frame_tokens, query_points = model.conditioner.tokenize_frames(
                batch["frame_vertices"],
                batch["faces"],
                refs,
                vertex_normals=batch["vertex_normals"],
                face_normals=batch["face_normals"],
            )
            dynamic_condition = model.conditioner.motion_encoder(
                frame_tokens,
                query_points=query_points,
            )
            static_condition = frame_tokens[:, 0]
        else:
            dynamic_condition = model.conditioner(
                batch["frame_vertices"],
                batch["faces"],
                refs,
                vertex_normals=batch["vertex_normals"],
                face_normals=batch["face_normals"],
            )
            static_condition = model.build_static_condition(batch).to(
                device=dynamic_condition.device,
                dtype=dynamic_condition.dtype,
            )
        fused_condition = model.condition_fuser(static_condition, dynamic_condition)
        if not torch.equal(fused_condition, static_condition):
            difference = float((fused_condition - static_condition).abs().max().item())
            raise RuntimeError(f"zero-residual condition is not exact static identity: max_diff={difference}")
        static_logits = teacher_logits(model, static_condition, batch)
        fused_logits = teacher_logits(model, fused_condition, batch)
        logit_max_diff = float((static_logits - fused_logits).abs().max().item())
        if logit_max_diff != 0.0:
            raise RuntimeError(f"initial teacher logits changed: max_diff={logit_max_diff}")
        wired_condition = model.build_condition(batch, refs=refs)
        if not torch.equal(wired_condition, fused_condition):
            difference = float((wired_condition - fused_condition).abs().max().item())
            raise RuntimeError(f"build_condition wiring mismatch: max_diff={difference}")

    del static_logits, fused_logits, static_condition, dynamic_condition, fused_condition
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        refs = model.sample_references(batch)
        condition = model.build_condition(batch, refs=refs)
        losses = model._ar_losses(
            condition,
            batch,
            include_loop_recovery=False,
            include_generated_prefix_recovery=False,
        )
        first_loss = losses["loss"]
    first_loss.backward()

    if args.condition_fusion == "anchor_motion_residual_zero":
        final_update = model.condition_fuser.update[-1]
    else:
        final_update = model.condition_fuser.blocks[0].update[-1]
    first_gradients = {
        "condition_fusion_final_update": grad_l1(final_update),
        "surface": grad_l1(model.conditioner.surface_tokenizer),
        "ar": grad_l1(model.unirig_ar.transformer),
        "motion": grad_l1(model.conditioner.motion_encoder),
    }
    for name in ("condition_fusion_final_update", "surface", "ar"):
        if not first_gradients[name] > 0.0:
            raise RuntimeError(f"required first-step gradient is absent: {name}")

    with torch.no_grad():
        for parameter in final_update.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1.0e-4)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        refs = model.sample_references(batch)
        condition = model.build_condition(batch, refs=refs)
        second_loss = model._ar_losses(
            condition,
            batch,
            include_loop_recovery=False,
            include_generated_prefix_recovery=False,
        )["loss"]
    second_loss.backward()
    second_motion_gradient = grad_l1(model.conditioner.motion_encoder)
    if not second_motion_gradient > 0.0:
        raise RuntimeError("motion encoder has no gradient after the residual output layer opens")

    result = {
        "status": "passed",
        "route": (
            "flat_anchor_motion_residual"
            if args.condition_fusion == "anchor_motion_residual_zero"
            else "flat_static_condition_motion_residual"
        ),
        "manifest": str(args.manifest),
        "manifest_index": index,
        "initial_condition_max_diff": 0.0,
        "initial_teacher_logit_max_diff": logit_max_diff,
        "first_loss": float(first_loss.detach().float().item()),
        "second_loss": float(second_loss.detach().float().item()),
        "first_step_gradient_l1": first_gradients,
        "second_step_motion_gradient_l1": second_motion_gradient,
        "optimizer_parameter_counts": optimizer_counts,
        "use_motion_features": False,
        "use_time_embedding": False,
        "condition_fusion": args.condition_fusion,
        "condition_fusion_depth": 1,
        "branch_prior_proposals": 0,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
