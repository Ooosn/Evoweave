#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dynamic_rig import (  # noqa: E402
    accounting_fields,
    build_tokenizer,
    cleanup_distributed,
    count_trainable,
    load_unirig,
    move_batch,
    move_dynamic_model_to_device,
    parse_sample_milestones,
    save_checkpoint,
    setup_distributed,
    unwrap_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the isolated Evoweave stack-close route.",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=15_903)
    parser.add_argument("--expected-val-rows", type=int, default=857)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--surface-samples", type=int, default=65_536)
    parser.add_argument("--vertex-samples", type=int, default=8_192)
    parser.add_argument("--query-tokens", type=int, default=1_024)
    parser.add_argument("--register-tokens", type=int, default=96)
    parser.add_argument("--motion-depth", type=int, default=12)
    parser.add_argument("--motion-heads", type=int, default=8)
    parser.add_argument("--motion-fps-ratio", type=float, default=0.7)
    parser.add_argument("--motion-vertex-samples", type=int, default=512)
    parser.add_argument(
        "--use-motion-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-time-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--random-sibling-order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1_667)
    parser.add_argument(
        "--sample-milestones",
        default="5000,10000,20000,30000,50000,80000",
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--lr-motion", type=float, default=1.0e-4)
    parser.add_argument("--lr-ar", type=float, default=1.0e-4)
    parser.add_argument("--lr-surface", type=float, default=1.0e-4)
    parser.add_argument("--lr-refresh", type=float, default=1.0e-4)
    parser.add_argument("--lr-stack-action", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--onecycle-div-factor", type=float, default=5.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10.0)
    parser.add_argument("--perturb-row-probability", type=float, default=0.5)
    parser.add_argument("--perturb-axial-fraction-max", type=float, default=0.05)
    parser.add_argument("--perturb-radial-fraction-max", type=float, default=0.05)
    parser.add_argument("--perturb-max-joints", type=int, default=4)
    parser.add_argument("--perturb-max-joint-fraction", type=float, default=0.08)
    parser.add_argument("--perturb-warmup-samples", type=int, default=5_000)
    parser.add_argument("--perturb-ramp-samples", type=int, default=15_000)
    parser.add_argument("--stack-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--stack-action-condition-dim", type=int, default=0)
    parser.add_argument("--stack-action-condition-heads", type=int, default=8)
    parser.add_argument(
        "--condition-refresh-layers",
        default="",
        help="Comma-separated zero-based OPT decoder layers. Empty disables refresh.",
    )
    parser.add_argument("--condition-refresh-dim", type=int, default=256)
    parser.add_argument("--condition-refresh-heads", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--initialize-stack-checkpoint", type=Path)
    parser.add_argument("--freeze-base-for-stack-action", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def _manifest_rows(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _parse_refresh_layers(value: str) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    layers = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not layers:
        raise ValueError("condition refresh layer list is empty")
    if tuple(sorted(set(layers))) != layers:
        raise ValueError(
            f"condition refresh layers must be sorted and unique, got {layers}"
        )
    return layers


def _validate_contract(args: argparse.Namespace, world_size: int) -> None:
    train_rows = _manifest_rows(args.train_manifest)
    val_rows = _manifest_rows(args.val_manifest)
    if args.limit_train <= 0 and train_rows != args.expected_train_rows:
        raise ValueError(
            f"train manifest rows={train_rows}, expected {args.expected_train_rows}"
        )
    if args.limit_val <= 0 and val_rows != args.expected_val_rows:
        raise ValueError(
            f"val manifest rows={val_rows}, expected {args.expected_val_rows}"
        )
    if args.query_tokens != 1_024:
        raise ValueError("stack-close formal route requires exactly 1024 condition tokens")
    if world_size == 2 and (
        args.batch_size != 3 or args.grad_accum_steps != 8
    ):
        raise ValueError(
            "2-GPU formal route requires micro batch 3 and grad accumulation 8 "
            "to preserve effective batch 48"
        )
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)
    if (
        args.initialize_stack_checkpoint is not None
        and not args.initialize_stack_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.initialize_stack_checkpoint)
    if args.resume_checkpoint is not None and args.initialize_stack_checkpoint is not None:
        raise ValueError("resume and initialization checkpoints are mutually exclusive")
    if args.stack_action_condition_dim < 0:
        raise ValueError("stack_action_condition_dim must be non-negative")
    if args.stack_action_condition_heads <= 0:
        raise ValueError("stack_action_condition_heads must be positive")
    if (
        args.stack_action_condition_dim > 0
        and args.stack_action_condition_dim % args.stack_action_condition_heads != 0
    ):
        raise ValueError(
            "stack_action_condition_dim must be divisible by "
            "stack_action_condition_heads"
        )
    if (
        args.stack_action_condition_dim > 0
        and args.stack_action_loss_weight <= 0.0
    ):
        raise ValueError(
            "condition-aware stack action requires a positive action loss weight"
        )
    if args.freeze_base_for_stack_action and (
        args.initialize_stack_checkpoint is None
        or args.stack_action_loss_weight <= 0.0
    ):
        raise ValueError(
            "freeze_base_for_stack_action requires an initialization checkpoint "
            "and a positive stack action loss weight"
        )


def _log(rank: int, message: str, log_path: Path | None = None) -> None:
    if rank != 0:
        return
    text = f"[stack_close] {message}"
    print(text, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def _audit_optimizer_coverage(
    model: torch.nn.Module,
    parameter_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    assigned: dict[int, str] = {}
    duplicates: list[str] = []
    for group in parameter_groups:
        group_name = str(group["name"])
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in assigned:
                duplicates.append(
                    f"{assigned[parameter_id]}+{group_name}"
                )
            assigned[parameter_id] = group_name

    missing = [
        (name, int(parameter.numel()))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    frozen_assigned = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and id(parameter) in assigned
    ]
    if duplicates or missing or frozen_assigned:
        raise RuntimeError(
            "optimizer coverage failure: "
            f"duplicate_groups={duplicates[:20]} "
            f"unassigned_trainable={missing[:20]} "
            f"frozen_assigned={frozen_assigned[:20]}"
        )

    group_counts = {
        str(group["name"]): int(
            sum(parameter.numel() for parameter in group["params"])
        )
        for group in parameter_groups
    }
    optimized = int(sum(group_counts.values()))
    trainable = count_trainable(model)
    if optimized != trainable:
        raise RuntimeError(
            f"optimizer owns {optimized:,} parameters, "
            f"but model has {trainable:,} trainable parameters"
        )
    return {
        "optimized_parameters": optimized,
        "group_parameters": group_counts,
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_steps: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for index, batch in enumerate(loader):
        if index >= max_steps:
            break
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            out = model(batch)
        for key in (
            "loss",
            "ce_loss",
            "token_acc",
            "coordinate_acc",
            "close_acc",
            "eos_acc",
            "stack_action_loss",
            "stack_action_acc",
        ):
            sums[key] = sums.get(key, 0.0) + float(out[key].detach().cpu())
        count += 1
    model.train(was_training)
    packed = torch.tensor(
        [
            *(sums.get(key, 0.0) for key in (
                "loss",
                "ce_loss",
                "token_acc",
                "coordinate_acc",
                "close_acc",
                "eos_acc",
                "stack_action_loss",
                "stack_action_acc",
            )),
            float(count),
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = int(packed[-1].item())
    if total_count <= 0:
        raise RuntimeError("validation loader produced no rows")
    names = (
        "val_loss",
        "val_ce",
        "val_token_acc",
        "val_coordinate_acc",
        "val_close_acc",
        "val_eos_acc",
        "val_stack_action_loss",
        "val_stack_action_acc",
    )
    return {
        name: float(packed[index].item() / total_count)
        for index, name in enumerate(names)
    } | {"val_count": total_count}


def main() -> None:
    args = _parser().parse_args()
    device, rank, local_rank, world_size = setup_distributed()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    run_log: Path | None = None
    metrics_log: Path | None = None
    try:
        _validate_contract(args, world_size)
        random.seed(args.seed + rank)
        np.random.seed((args.seed + rank) % (2**32))
        torch.manual_seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + rank)

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            logs = args.output_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            run_log = logs / "run.log"
            metrics_log = logs / "train.jsonl"
            if args.resume_checkpoint is None:
                run_log.write_text("", encoding="utf-8")
                metrics_log.write_text("", encoding="utf-8")

        from rigweave.dynamic_rig import (  # noqa: WPS433
            AnchorWiseAlternatingMotionEncoder,
            FixedQuerySurfaceTokenizer,
        )
        from rigweave.dynamic_rig.model import DynamicRigConditioner
        from rigweave.stack_close import (
            PrefixPerturbationConfig,
            StackCloseDynamicRigAR,
            StackCloseManifestDataset,
            StackCloseTokenizer,
            stack_close_collate,
        )
        refresh_layers = _parse_refresh_layers(
            args.condition_refresh_layers
        )
        route_parts = ["stack_close"]
        if args.stack_action_loss_weight > 0.0:
            route_parts.append("action")
        if refresh_layers:
            route_parts.append("condition_refresh")
        route_parts.append(
            "random_sibling" if args.random_sibling_order else "canonical_sibling"
        )
        if args.perturb_row_probability > 0.0:
            route_parts.append("perturb")
        route = "_".join(route_parts)

        _log(rank, f"device={device} world_size={world_size}", run_log)
        legacy_tokenizer = build_tokenizer(args.tokenizer_config)
        stack_tokenizer = StackCloseTokenizer(legacy_tokenizer)
        unirig = load_unirig(
            stack_tokenizer,
            args.model_config,
            args.unirig_checkpoint,
        )
        surface_tokenizer = FixedQuerySurfaceTokenizer(
            unirig.mesh_encoder,
            unirig.output_proj,
        )
        motion_encoder = AnchorWiseAlternatingMotionEncoder(
            dim=unirig.hidden_size,
            depth=args.motion_depth,
            heads=args.motion_heads,
            register_tokens=args.register_tokens,
            max_frames=max(args.frames, 48),
            use_motion_features=args.use_motion_features,
            use_time_embedding=args.use_time_embedding,
            gradient_checkpointing=True,
        )
        conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
        perturbation = PrefixPerturbationConfig(
            row_probability=args.perturb_row_probability,
            axial_fraction_max=args.perturb_axial_fraction_max,
            radial_fraction_max=args.perturb_radial_fraction_max,
            max_perturbed_joints=args.perturb_max_joints,
            max_joint_fraction=args.perturb_max_joint_fraction,
            warmup_samples=args.perturb_warmup_samples,
            ramp_samples=args.perturb_ramp_samples,
        )
        if refresh_layers:
            from rigweave.stack_close_refresh import (  # noqa: WPS433
                ConditionRefreshStackCloseDynamicRigAR,
            )

            model = ConditionRefreshStackCloseDynamicRigAR(
                unirig,
                conditioner,
                stack_tokenizer,
                perturbation=perturbation,
                stack_action_loss_weight=args.stack_action_loss_weight,
                stack_action_condition_dim=args.stack_action_condition_dim,
                stack_action_condition_heads=args.stack_action_condition_heads,
                num_surface_samples=args.surface_samples,
                vertex_samples=args.vertex_samples,
                query_tokens=args.query_tokens,
                refresh_layer_indices=refresh_layers,
                refresh_dim=args.condition_refresh_dim,
                refresh_heads=args.condition_refresh_heads,
            )
        else:
            model = StackCloseDynamicRigAR(
                unirig,
                conditioner,
                stack_tokenizer,
                perturbation=perturbation,
                stack_action_loss_weight=args.stack_action_loss_weight,
                stack_action_condition_dim=args.stack_action_condition_dim,
                stack_action_condition_heads=args.stack_action_condition_heads,
                num_surface_samples=args.surface_samples,
                vertex_samples=args.vertex_samples,
                query_tokens=args.query_tokens,
            )
        dynamic_checkpoint_loaded = False
        if args.initialize_stack_checkpoint is not None:
            initialization_payload = torch.load(
                args.initialize_stack_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            initialization_args = dict(initialization_payload.get("args", {}) or {})
            for key in (
                "frames",
                "register_tokens",
                "motion_depth",
                "motion_heads",
                "use_motion_features",
                "use_time_embedding",
            ):
                requested = getattr(args, key)
                loaded = initialization_args.get(key, requested)
                if loaded != requested:
                    raise ValueError(
                        f"initialization contract mismatch for {key}: "
                        f"checkpoint={loaded!r} requested={requested!r}"
                    )
            missing, unexpected = model.load_state_dict(
                initialization_payload["model"],
                strict=False,
            )
            allowed_missing = {
                key
                for key in model.state_dict()
                if key.startswith("stack_action_head.")
            }
            missing_set = set(missing)
            if (missing_set and missing_set != allowed_missing) or unexpected:
                raise RuntimeError(
                    "stack initialization mismatch "
                    f"missing={sorted(missing_set)} unexpected={unexpected}"
                )
            dynamic_checkpoint_loaded = True
            del initialization_payload
        if args.freeze_base_for_stack_action:
            model.requires_grad_(False)
            assert model.stack_action_head is not None
            model.stack_action_head.requires_grad_(True)
        move_dynamic_model_to_device(model, device)
        if model.stack_action_head is not None:
            model.stack_action_head.to(device)
        if refresh_layers:
            model.condition_refresh_adapters.to(device)

        train_dataset = StackCloseManifestDataset(
            args.train_manifest,
            legacy_tokenizer=legacy_tokenizer,
            stack_tokenizer=stack_tokenizer,
            frame_count=args.frames,
            limit=args.limit_train,
            random_query=True,
            random_sibling_order=args.random_sibling_order,
            seed=args.seed,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
        )
        val_dataset = StackCloseManifestDataset(
            args.val_manifest,
            legacy_tokenizer=legacy_tokenizer,
            stack_tokenizer=stack_tokenizer,
            frame_count=args.frames,
            limit=args.limit_val,
            random_query=False,
            random_sibling_order=False,
            seed=args.seed + 17,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if world_size > 1
            else None
        )
        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            if world_size > 1
            else None
        )
        collate = partial(
            stack_close_collate,
            pad_token=stack_tokenizer.pad,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=min(args.num_workers, 2),
            pin_memory=True,
            collate_fn=collate,
        )

        parameter_groups = []
        for name, module, learning_rate in (
            ("motion", motion_encoder, args.lr_motion),
            ("surface", surface_tokenizer, args.lr_surface),
            ("ar", unirig.transformer, args.lr_ar),
        ):
            parameters = [
                parameter for parameter in module.parameters() if parameter.requires_grad
            ]
            if parameters:
                parameter_groups.append(
                    {"params": parameters, "lr": learning_rate, "name": name}
                )
        if refresh_layers:
            refresh_parameters = [
                parameter
                for parameter in model.condition_refresh_adapters.parameters()
                if parameter.requires_grad
            ]
            if refresh_parameters:
                parameter_groups.append(
                    {
                        "params": refresh_parameters,
                        "lr": args.lr_refresh,
                        "name": "condition_refresh",
                    }
                )
        if model.stack_action_head is not None:
            parameter_groups.append(
                {
                    "params": list(model.stack_action_head.parameters()),
                    "lr": args.lr_stack_action,
                    "name": "stack_action",
                }
            )
        optimizer_audit = _audit_optimizer_coverage(
            model,
            parameter_groups,
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in parameter_groups],
            total_steps=args.max_steps,
            pct_start=args.onecycle_pct_start,
            anneal_strategy="cos",
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
        )

        effective_batch = world_size * args.batch_size * args.grad_accum_steps
        train_rows = len(train_dataset)
        args.effective_batch = effective_batch
        args.train_rows = train_rows
        sample_milestones = parse_sample_milestones(args.sample_milestones)
        step = 0
        if args.resume_checkpoint is not None:
            payload = torch.load(
                args.resume_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            missing, unexpected = model.load_state_dict(
                payload["model"],
                strict=True,
            )
            if missing or unexpected:
                raise RuntimeError(
                    f"strict resume mismatch missing={missing} unexpected={unexpected}"
                )
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            step = int(payload["step"])
            if scheduler.last_epoch != step:
                raise ValueError(
                    f"resume scheduler step={scheduler.last_epoch}, model step={step}"
                )

        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
        raw_model: StackCloseDynamicRigAR = unwrap_model(model)
        raw_model.set_training_sample_seen(step * effective_batch)

        if rank == 0:
            config = json.loads(json.dumps(vars(args), default=str))
            (args.output_dir / "args.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_jsonl(
                metrics_log,
                {
                    "event": "run_config",
                    "route": route,
                    "world_size": world_size,
                    "effective_batch": effective_batch,
                    "train_rows": train_rows,
                    "trainable_parameters": count_trainable(model),
                    "optimizer_audit": optimizer_audit,
                    "initialization": str(
                        args.initialize_stack_checkpoint or args.unirig_checkpoint
                    ),
                    "dynamic_checkpoint_loaded": dynamic_checkpoint_loaded,
                    "freeze_base_for_stack_action": bool(
                        args.freeze_base_for_stack_action
                    ),
                    "perturbation": vars(perturbation),
                    "condition_refresh": {
                        "enabled": bool(refresh_layers),
                        "layers": list(refresh_layers),
                        "dim": int(args.condition_refresh_dim),
                        "heads": int(args.condition_refresh_heads),
                    },
                    "stack_action": {
                        "enabled": model.stack_action_head is not None,
                        "loss_weight": float(args.stack_action_loss_weight),
                        "lr": float(args.lr_stack_action),
                        "condition_dim": int(args.stack_action_condition_dim),
                        "condition_heads": int(args.stack_action_condition_heads),
                    },
                },
            )
        _log(
            rank,
            f"route={route} train_rows={train_rows} "
            f"val_rows={len(val_dataset)} effective_batch={effective_batch} "
            f"trainable={count_trainable(model):,}",
            run_log,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch = 0
        accumulation = 0
        aggregate: dict[str, float] = {}
        aggregate_path = ""
        accumulation_started = time.perf_counter()
        saved_milestones = {
            milestone
            for milestone in sample_milestones
            if milestone <= step * effective_batch
        }
        while step < args.max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if accumulation == 0:
                    accumulation_started = time.perf_counter()
                    raw_model.set_training_sample_seen(step * effective_batch)
                batch = move_batch(batch, device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    output = model(batch)
                    loss = output["loss"] / args.grad_accum_steps
                loss.backward()
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        aggregate[key] = aggregate.get(key, 0.0) + float(
                            value.detach().cpu()
                        )
                aggregate_path = batch["path"][0]
                accumulation += 1
                if accumulation < args.grad_accum_steps:
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                microbatches = accumulation
                accumulation = 0
                raw_model.set_training_sample_seen(step * effective_batch)

                if rank == 0 and (step == 1 or step % args.log_every == 0):
                    row = {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "seconds": time.perf_counter() - accumulation_started,
                        "path": aggregate_path,
                        **accounting_fields(step, effective_batch, train_rows),
                        "lrs": {
                            group["name"]: group["lr"]
                            for group in optimizer.param_groups
                        },
                    }
                    for key, value in aggregate.items():
                        row[key] = value / microbatches
                    if device.type == "cuda":
                        row["gpu_peak_gb"] = (
                            torch.cuda.max_memory_allocated(device) / 1024**3
                        )
                        torch.cuda.reset_peak_memory_stats(device)
                    _write_jsonl(metrics_log, row)
                aggregate = {}
                aggregate_path = ""

                if rank == 0:
                    sample_seen = step * effective_batch
                    for milestone in sample_milestones:
                        if (
                            milestone in saved_milestones
                            or sample_seen < milestone
                        ):
                            continue
                        save_checkpoint(
                            args.output_dir
                            / f"checkpoint_sample_{milestone}.pt",
                            model,
                            optimizer,
                            step,
                            args,
                            scheduler,
                            include_optimizer=True,
                        )
                        saved_milestones.add(milestone)

                if args.save_every > 0 and step % args.save_every == 0 and rank == 0:
                    save_checkpoint(
                        args.output_dir / f"checkpoint_step_{step}.pt",
                        model,
                        optimizer,
                        step,
                        args,
                        scheduler,
                        include_optimizer=True,
                    )

                if args.val_every > 0 and step % args.val_every == 0:
                    stats = _evaluate(
                        model,
                        val_loader,
                        device=device,
                        amp_dtype=amp_dtype,
                        max_steps=args.val_steps,
                    )
                    if rank == 0:
                        _write_jsonl(
                            metrics_log,
                            {
                                "event": "validation",
                                "step": step,
                                **accounting_fields(
                                    step,
                                    effective_batch,
                                    train_rows,
                                ),
                                **stats,
                            },
                        )
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                if step >= args.max_steps:
                    break
            epoch += 1

        if rank == 0:
            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model,
                optimizer,
                step,
                args,
                scheduler,
                include_optimizer=True,
            )
        _log(rank, f"done step={step}", run_log)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
