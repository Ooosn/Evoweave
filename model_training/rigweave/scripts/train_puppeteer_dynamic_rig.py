#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
UNIRIG_ROOT = Path(os.environ.get("EVOWEAVE_UNIRIG_ROOT", REPO / "external" / "UniRig")).expanduser()
if str(REPO / "rigweave" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "rigweave" / "src"))
if str(UNIRIG_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIRIG_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dynamic_rig import (  # noqa: E402
    accounting_fields,
    build_tokenizer,
    cleanup_distributed,
    count_trainable,
    is_main,
    json_safe,
    load_unirig,
    move_batch,
    parse_sample_milestones,
    save_checkpoint,
    setup_distributed,
    trim_host_allocator,
    unwrap_model,
    write_json_log,
)
from rigweave.dynamic_rig import (  # noqa: E402
    AnchorWiseAlternatingMotionEncoder,
    DynamicRigConditioner,
    FixedQuerySurfaceTokenizer,
    PuppeteerDynamicRigDataset,
    PuppeteerDynamicRigModel,
    PuppeteerJointTokenizer,
    import_puppeteer_decoder,
    load_puppeteer_decoder_state,
    load_puppeteer_target_aware_pos_embed,
    puppeteer_dynamic_collate,
)


def log(rank: int, message: str) -> None:
    if is_main(rank):
        print(f"[puppeteer_dynamic] {message}", flush=True)


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported amp dtype: {name}")


def build_decoder_config(args: argparse.Namespace, SkeletonOPTConfig: type[Any]) -> Any:
    max_length = int(args.cond_length + args.n_max_joints * 4 + 2)
    vocab_size = int(args.n_discrete_size + 3)
    if args.tiny_random_decoder:
        ffn_dim = int(args.decoder_ffn_dim or args.decoder_hidden_size * 4)
        config = SkeletonOPTConfig(
            vocab_size=vocab_size,
            hidden_size=int(args.decoder_hidden_size),
            word_embed_proj_dim=int(args.decoder_hidden_size),
            ffn_dim=ffn_dim,
            num_hidden_layers=int(args.decoder_layers),
            num_attention_heads=int(args.decoder_heads),
            max_position_embeddings=max_length,
            n_positions=max_length,
            dropout=float(args.decoder_dropout),
            attention_dropout=float(args.decoder_attention_dropout),
            activation_dropout=float(args.decoder_activation_dropout),
            layerdrop=float(args.decoder_layerdrop),
        )
    else:
        config = SkeletonOPTConfig.from_pretrained(
            args.puppeteer_llm,
            local_files_only=args.local_files_only,
            n_positions=max_length,
            max_position_embeddings=max_length,
            vocab_size=vocab_size,
            _attn_implementation=args.attn_implementation,
        )
    config.joint_token = True
    config.bos_token_id = 0
    config.eos_token_id = 1
    config.pad_token_id = 2
    config._attn_implementation = args.attn_implementation
    config.n_discrete_size = int(args.n_discrete_size)
    config.bone_per_token = 4
    config.cond_length = int(args.cond_length)
    config.word_embed_proj_dim = int(config.hidden_size)
    return config


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def add_param_group(
    groups: list[dict[str, Any]],
    *,
    name: str,
    module: torch.nn.Module,
    lr: float,
    seen: set[int],
) -> None:
    params = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        ident = id(parameter)
        if ident in seen:
            continue
        seen.add(ident)
        params.append(parameter)
    if params:
        groups.append({"name": name, "params": params, "lr": float(lr)})


def add_parameter_group(
    groups: list[dict[str, Any]],
    *,
    name: str,
    parameter: torch.nn.Parameter | None,
    lr: float,
    seen: set[int],
) -> None:
    if parameter is None or not parameter.requires_grad:
        return
    ident = id(parameter)
    if ident in seen:
        return
    seen.add(ident)
    groups.append({"name": name, "params": [parameter], "lr": float(lr)})


def build_optimizer(model: PuppeteerDynamicRigModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    add_param_group(
        groups,
        name="surface",
        module=model.conditioner.surface_tokenizer,
        lr=args.lr_surface,
        seen=seen,
    )
    add_param_group(
        groups,
        name="motion",
        module=model.conditioner.motion_encoder,
        lr=args.lr_motion,
        seen=seen,
    )
    add_param_group(
        groups,
        name="prefix_projector",
        module=model.prefix_projector,
        lr=args.lr_prefix,
        seen=seen,
    )
    add_parameter_group(
        groups,
        name="target_aware_pos_embed",
        parameter=model.target_aware_pos_embed,
        lr=args.lr_decoder,
        seen=seen,
    )
    add_param_group(
        groups,
        name="decoder",
        module=model.decoder,
        lr=args.lr_decoder,
        seen=seen,
    )
    if not groups:
        raise RuntimeError("no trainable parameters; check freeze flags")
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay), betas=(args.adam_beta1, args.adam_beta2))


def optimizer_group_report(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "name": str(group.get("name", i)),
            "parameters": int(sum(p.numel() for p in group["params"])),
            "lr": float(group["lr"]),
        }
        for i, group in enumerate(optimizer.param_groups)
    ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.scheduler == "none":
        return None
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=max(1, int(args.max_steps)),
            pct_start=float(args.onecycle_pct_start),
            div_factor=float(args.onecycle_div_factor),
            final_div_factor=float(args.onecycle_final_div_factor),
        )
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.max_steps)))
    raise ValueError(f"unsupported scheduler: {args.scheduler}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_steps: int,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    local = torch.zeros(6, device=device, dtype=torch.float64)
    try:
        for i, batch in enumerate(loader):
            if i >= max_steps:
                break
            torch.manual_seed(2026070700 + i)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(2026070700 + i)
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
                out = model(batch)
            local[0] += float(out["loss"].detach().cpu())
            local[1] += float(out["token_acc"].detach().cpu())
            local[2] += float(out["coord_acc"].detach().cpu())
            local[3] += float(out["parent_acc"].detach().cpu())
            local[4] += float(out["eos_acc"].detach().cpu())
            local[5] += 1.0
    finally:
        model.train(was_training)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    count = int(local[5].item())
    if count <= 0:
        return {"val_loss": -1.0, "val_count": 0.0}
    return {
        "val_loss": float((local[0] / local[5]).item()),
        "val_token_acc": float((local[1] / local[5]).item()),
        "val_coord_acc": float((local[2] / local[5]).item()),
        "val_parent_acc": float((local[3] / local[5]).item()),
        "val_eos_acc": float((local[4] / local[5]).item()),
        "val_count": float(count),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Evoweave dynamic condition + Puppeteer SkeletonOPT decoder.")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(os.environ["EVOWEAVE_TRAIN_MANIFEST"]) if os.environ.get("EVOWEAVE_TRAIN_MANIFEST") else None,
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(os.environ["EVOWEAVE_VAL_MANIFEST"]) if os.environ.get("EVOWEAVE_VAL_MANIFEST") else None,
    )
    parser.add_argument("--tokenizer-config", type=Path, default=Path(os.environ.get("EVOWEAVE_TOKENIZER_CONFIG", "external/UniRig/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml")))
    parser.add_argument("--model-config", type=Path, default=Path(os.environ.get("EVOWEAVE_MODEL_CONFIG", "external/UniRig/configs/model/unirig_ar_350m_1024_81920_float32.yaml")))
    parser.add_argument("--unirig-checkpoint", type=Path, default=Path(os.environ.get("EVOWEAVE_UNIRIG_CKPT", "external/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt")))
    parser.add_argument("--puppeteer-root", type=Path, default=Path(os.environ.get("PUPPETEER_ROOT", REPO / "third_party_references" / "Puppeteer")))
    parser.add_argument("--puppeteer-checkpoint", type=Path, default=Path(os.environ["PUPPETEER_CHECKPOINT"]) if os.environ.get("PUPPETEER_CHECKPOINT") else None)
    parser.add_argument("--puppeteer-llm", type=str, default=os.environ.get("PUPPETEER_LLM", "facebook/opt-350m"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("EVOWEAVE_OUTPUT_DIR", "outputs/puppeteer_dynamic")))
    parser.add_argument("--limit-train", type=int, default=int(os.environ.get("RIGWEAVE_LIMIT_TRAIN", "0")))
    parser.add_argument("--limit-val", type=int, default=int(os.environ.get("RIGWEAVE_LIMIT_VAL", "64")))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--surface-samples", type=int, default=65536)
    parser.add_argument("--vertex-samples", type=int, default=8192)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--register-tokens", type=int, default=96)
    parser.add_argument("--motion-depth", type=int, default=12)
    parser.add_argument("--motion-heads", type=int, default=8)
    parser.add_argument("--use-motion-features", action="store_true")
    parser.add_argument("--use-time-embedding", action="store_true")
    parser.add_argument("--motion-checkpointing", action="store_true")
    parser.add_argument("--motion-fps-ratio", type=float, default=0.7)
    parser.add_argument("--motion-vertex-samples", type=int, default=512)
    parser.add_argument("--train-random-query", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--sample-milestones", type=str, default="5000,10000,20000,30000,50000,80000")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-save-optimizer", action="store_true")
    parser.add_argument("--lr-surface", type=float, default=1.0e-4)
    parser.add_argument("--lr-motion", type=float, default=1.0e-4)
    parser.add_argument("--lr-prefix", type=float, default=1.0e-4)
    parser.add_argument("--lr-decoder", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--scheduler", choices=["none", "onecycle", "cosine"], default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--onecycle-div-factor", type=float, default=5.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10.0)
    parser.add_argument("--n-discrete-size", type=int, default=128)
    parser.add_argument(
        "--n-max-joints",
        type=int,
        default=101,
        help="Default matches the released Puppeteer joint-token checkpoint position capacity.",
    )
    parser.add_argument("--target-coord-scale", type=float, default=0.25)
    parser.add_argument("--no-strict-target-range", action="store_true")
    parser.add_argument("--cond-length", type=int, default=257)
    parser.add_argument("--projector-heads", type=int, default=8)
    parser.add_argument("--attn-implementation", choices=["flash_attention_2"], default="flash_attention_2")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-resize-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-init-smoke", action="store_true", help="Allow decoder random initialization for code smoke only.")
    parser.add_argument("--tiny-random-decoder", action="store_true", help="Use a small random SkeletonOPT config for development smoke only.")
    parser.add_argument("--decoder-hidden-size", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-ffn-dim", type=int, default=0)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-attention-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-activation-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-layerdrop", type=float, default=0.0)
    parser.add_argument("--decoder-checkpointing", action="store_true")
    parser.add_argument("--freeze-surface-tokenizer", action="store_true")
    parser.add_argument("--freeze-conditioner", action="store_true")
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-forward", action="store_true")
    args = parser.parse_args()

    if args.train_manifest is None:
        raise ValueError("--train-manifest is required")
    if args.val_manifest is None:
        raise ValueError("--val-manifest is required")
    if not args.train_manifest.exists():
        raise FileNotFoundError(args.train_manifest)
    if not args.val_manifest.exists():
        raise FileNotFoundError(args.val_manifest)
    if args.puppeteer_checkpoint is None and not args.random_init_smoke:
        raise RuntimeError(
            "Puppeteer checkpoint is required for a formal Puppeteer run. "
            "Use --random-init-smoke only for development mechanics checks."
        )
    if args.tiny_random_decoder and not args.random_init_smoke:
        raise RuntimeError("--tiny-random-decoder is only allowed with --random-init-smoke")
    if args.n_max_joints > args.n_discrete_size:
        raise ValueError(
            f"n_max_joints={args.n_max_joints} must be <= n_discrete_size={args.n_discrete_size} "
            "because Puppeteer parent tokens reserve raw 0 for root and raw parent+1 for non-root parents"
        )

    device, rank, local_rank, world_size = setup_distributed()
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = args.output_dir / "train_metrics.jsonl"
    if is_main(rank):
        save_json(args.output_dir / "args.json", vars(args))

    try:
        log(rank, f"device={device} world_size={world_size} local_rank={local_rank}")
        log(rank, f"train_manifest={args.train_manifest}")
        log(rank, f"val_manifest={args.val_manifest}")
        log(rank, f"output_dir={args.output_dir}")

        stage_t0 = time.time()
        unirig_tokenizer = build_tokenizer(args.tokenizer_config)
        unirig = load_unirig(unirig_tokenizer, args.model_config, args.unirig_checkpoint)
        surface_tokenizer = FixedQuerySurfaceTokenizer(unirig.mesh_encoder, unirig.output_proj)
        motion_encoder = AnchorWiseAlternatingMotionEncoder(
            dim=unirig.hidden_size,
            depth=args.motion_depth,
            heads=args.motion_heads,
            register_tokens=args.register_tokens,
            max_frames=max(args.frames, 48),
            use_motion_features=args.use_motion_features,
            use_time_embedding=args.use_time_embedding,
            gradient_checkpointing=args.motion_checkpointing,
        )
        conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
        log(rank, f"Evoweave conditioner built in {time.time() - stage_t0:.2f}s")

        stage_t0 = time.time()
        SkeletonOPTConfig, SkeletonOPT = import_puppeteer_decoder(args.puppeteer_root)
        config = build_decoder_config(args, SkeletonOPTConfig)
        decoder = SkeletonOPT(config)
        if args.decoder_checkpointing:
            decoder.model.decoder.gradient_checkpointing = True
        decoder_report: dict[str, Any]
        target_aware_pos_embed = None
        if args.puppeteer_checkpoint is not None:
            decoder_report = load_puppeteer_decoder_state(
                decoder,
                args.puppeteer_checkpoint,
                allow_resize_positions=args.allow_resize_positions,
            )
            target_aware_pos_embed = load_puppeteer_target_aware_pos_embed(args.puppeteer_checkpoint)
            decoder_report["target_aware_pos_embed_loaded"] = target_aware_pos_embed is not None
            decoder_report["target_aware_pos_embed_shape"] = (
                list(target_aware_pos_embed.shape) if target_aware_pos_embed is not None else None
            )
        else:
            decoder_report = {
                "checkpoint": None,
                "random_init_smoke": True,
                "tiny_random_decoder": bool(args.tiny_random_decoder),
                "target_aware_pos_embed_loaded": False,
                "target_aware_pos_embed_shape": None,
            }
        if is_main(rank):
            save_json(args.output_dir / "puppeteer_decoder_load_report.json", decoder_report)
        log(rank, f"Puppeteer decoder built in {time.time() - stage_t0:.2f}s")

        joint_tokenizer = PuppeteerJointTokenizer(
            n_discrete_size=args.n_discrete_size,
            target_coord_scale=args.target_coord_scale,
            strict_range=not args.no_strict_target_range,
        )
        model = PuppeteerDynamicRigModel(
            conditioner=conditioner,
            decoder=decoder,
            tokenizer=joint_tokenizer,
            num_surface_samples=args.surface_samples,
            vertex_samples=args.vertex_samples,
            query_tokens=args.query_tokens,
            cond_length=args.cond_length,
            projector_heads=args.projector_heads,
            target_aware_pos_embed=target_aware_pos_embed,
        )
        if args.freeze_surface_tokenizer:
            set_requires_grad(model.conditioner.surface_tokenizer, False)
            model.conditioner.surface_tokenizer.eval()
        if args.freeze_conditioner:
            set_requires_grad(model.conditioner, False)
            model.conditioner.eval()
        if args.freeze_decoder:
            set_requires_grad(model.decoder, False)
            model.decoder.eval()
            if model.target_aware_pos_embed is not None:
                model.target_aware_pos_embed.requires_grad_(False)
        model.to(device)
        log(rank, f"trainable params={count_trainable(model):,}")

        train_dataset = PuppeteerDynamicRigDataset(
            args.train_manifest,
            frame_count=args.frames,
            limit=args.limit_train,
            random_query=args.train_random_query,
            seed=args.seed,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
            max_joints=args.n_max_joints,
        )
        val_dataset = PuppeteerDynamicRigDataset(
            args.val_manifest,
            frame_count=args.frames,
            limit=args.limit_val,
            random_query=False,
            seed=args.seed + 17,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
            max_joints=args.n_max_joints,
        )
        train_rows = len(train_dataset)
        effective_batch = int(world_size * args.batch_size * args.grad_accum_steps)
        args.effective_batch = effective_batch
        args.train_rows = train_rows
        sample_milestones = parse_sample_milestones(args.sample_milestones)
        if is_main(rank):
            save_json(
                args.output_dir / "training_contract.json",
                {
                    "decoder_family": "puppeteer_skeletonopt",
                    "target": "rootless target_joints_rootspace + target_parents",
                    "tail_tokens": False,
                    "tokenization": "joint-token: x,y,z,parent_index",
                    "max_joints": args.n_max_joints,
                    "train_raw_rows": train_dataset.raw_rows,
                    "train_rows_after_max_joint_filter": len(train_dataset),
                    "train_filtered_over_max_joints": train_dataset.filtered_over_max_joints,
                    "val_raw_rows": val_dataset.raw_rows,
                    "val_rows_after_max_joint_filter": len(val_dataset),
                    "val_filtered_over_max_joints": val_dataset.filtered_over_max_joints,
                    "condition": "Evoweave dynamic surface motion prefix",
                    "first_joint_contract": "joint0 is the single rootless skeleton root; no synthetic root",
                    "world_size": world_size,
                    "micro_batch_per_gpu": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "effective_batch": effective_batch,
                    "sample_milestones": sample_milestones,
                    "random_init_smoke": bool(args.random_init_smoke),
                },
            )
        log(
            rank,
            f"train rows={len(train_dataset)}/{train_dataset.raw_rows} "
            f"val rows={len(val_dataset)}/{val_dataset.raw_rows} "
            f"filtered_over_max_joints="
            f"{train_dataset.filtered_over_max_joints}/{val_dataset.filtered_over_max_joints} "
            f"max_joints={args.n_max_joints} effective_batch={effective_batch}",
        )

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=puppeteer_dynamic_collate,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=puppeteer_dynamic_collate,
            persistent_workers=args.num_workers > 0,
        )

        if args.preflight_only:
            batch = next(iter(train_loader))
            batch = move_batch(batch, device)
            token_batch = joint_tokenizer.make_batch(
                batch["target_joints"],
                batch["target_parents"],
                batch["joint_count"],
                batch["path"],
            )
            log(
                rank,
                "preflight batch "
                f"batch={len(batch['path'])} max_joints={int(batch['joint_count'].max().item())} "
                f"token_len={int(token_batch.input_ids.shape[1])} first={batch['path'][0]}",
            )
            if args.preflight_forward:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
                    out = model(batch)
                log(rank, f"preflight forward loss={float(out['loss'].detach().cpu()):.6f}")
            optimizer = build_optimizer(model, args)
            log(
                rank,
                "preflight optimizer groups="
                + ", ".join(f"{g['name']}:{g['parameters']:,}@{g['lr']}" for g in optimizer_group_report(optimizer)),
            )
            return

        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args)
        train_model: torch.nn.Module = model
        if world_size > 1:
            train_model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=args.ddp_find_unused_parameters,
            )

        log(
            rank,
            "optimizer groups="
            + ", ".join(f"{g['name']}:{g['parameters']:,}@{g['lr']}" for g in optimizer_group_report(optimizer)),
        )
        write_json_log(
            metrics_log if is_main(rank) else None,
            {
                "event": "start",
                "world_size": world_size,
                "effective_batch": effective_batch,
                "train_rows": train_rows,
                "random_init_smoke": bool(args.random_init_smoke),
                "decoder_checkpoint": str(args.puppeteer_checkpoint) if args.puppeteer_checkpoint else None,
                "optimizer_groups": optimizer_group_report(optimizer),
            },
        )

        saved_sample_milestones: set[int] = set()
        best_val_loss = float("inf")
        step = 0
        epoch = 0
        accum_count = 0
        accum_t0 = time.time()
        accum_sums = {"loss": 0.0, "token_acc": 0.0, "coord_acc": 0.0, "parent_acc": 0.0, "eos_acc": 0.0}
        accum_path = ""
        optimizer.zero_grad(set_to_none=True)
        train_model.train()
        while step < args.max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                batch = move_batch(batch, device)
                is_final_micro = accum_count + 1 >= max(1, args.grad_accum_steps)
                sync_context = (
                    train_model.no_sync()
                    if isinstance(train_model, DistributedDataParallel) and not is_final_micro
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
                        out = train_model(batch)
                        raw_loss = out["loss"]
                        loss = raw_loss / max(1, args.grad_accum_steps)
                    loss.backward()
                accum_sums["loss"] += float(raw_loss.detach().cpu())
                for key in ("token_acc", "coord_acc", "parent_acc", "eos_acc"):
                    accum_sums[key] += float(out[key].detach().cpu())
                accum_path = batch["path"][0]
                accum_count += 1
                if accum_count < max(1, args.grad_accum_steps):
                    continue

                step += 1
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                micro_count = max(1, accum_count)
                accum_count = 0

                if is_main(rank) and (step == 1 or step % args.log_every == 0):
                    accounting = accounting_fields(step, effective_batch, train_rows)
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "loss": accum_sums["loss"] / micro_count,
                        "token_acc": accum_sums["token_acc"] / micro_count,
                        "coord_acc": accum_sums["coord_acc"] / micro_count,
                        "parent_acc": accum_sums["parent_acc"] / micro_count,
                        "eos_acc": accum_sums["eos_acc"] / micro_count,
                        "seconds": round(time.time() - accum_t0, 3),
                        "path": accum_path,
                        "grad_accum": args.grad_accum_steps,
                        "micro_batch_per_gpu": args.batch_size,
                        "lrs": {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)},
                        **accounting,
                    }
                    if device.type == "cuda":
                        row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                        torch.cuda.reset_peak_memory_stats(device)
                    write_json_log(metrics_log, row)
                accum_sums = {"loss": 0.0, "token_acc": 0.0, "coord_acc": 0.0, "parent_acc": 0.0, "eos_acc": 0.0}
                accum_path = ""
                accum_t0 = time.time()

                if is_main(rank):
                    sample_seen = int(step * effective_batch)
                    for milestone in sample_milestones:
                        if milestone in saved_sample_milestones or sample_seen < milestone:
                            continue
                        path = args.output_dir / f"checkpoint_sample_{milestone}.pt"
                        save_checkpoint(
                            path,
                            train_model,
                            optimizer,
                            step,
                            args,
                            scheduler,
                            include_optimizer=not args.no_save_optimizer,
                        )
                        saved_sample_milestones.add(milestone)
                        write_json_log(
                            metrics_log,
                            {
                                "event": "sample_milestone_checkpoint",
                                "step": step,
                                "sample_milestone": milestone,
                                "path": str(path),
                                **accounting_fields(step, effective_batch, train_rows),
                            },
                        )

                if is_main(rank) and args.save_every > 0 and step % args.save_every == 0:
                    path = args.output_dir / f"checkpoint_step_{step}.pt"
                    save_checkpoint(
                        path,
                        train_model,
                        optimizer,
                        step,
                        args,
                        scheduler,
                        include_optimizer=not args.no_save_optimizer,
                    )
                    write_json_log(metrics_log, {"event": "step_checkpoint", "step": step, "path": str(path)})

                if args.val_every > 0 and step % args.val_every == 0:
                    stats = evaluate(train_model, val_loader, device, max_steps=args.val_steps, amp_dtype=amp_dtype)
                    if is_main(rank):
                        row = {"event": "eval", "step": step, **accounting_fields(step, effective_batch, train_rows), **stats}
                        write_json_log(metrics_log, row)
                        if stats["val_loss"] >= 0.0 and stats["val_loss"] < best_val_loss:
                            best_val_loss = stats["val_loss"]
                            path = args.output_dir / "checkpoint_best_val.pt"
                            save_checkpoint(
                                path,
                                train_model,
                                optimizer,
                                step,
                                args,
                                scheduler,
                                include_optimizer=not args.no_save_optimizer,
                            )
                            write_json_log(metrics_log, {"event": "best_val_checkpoint", "step": step, "path": str(path), **stats})
                    train_model.train()

                if step >= args.max_steps:
                    break
            epoch += 1

        if is_main(rank):
            final_path = args.output_dir / "checkpoint_final.pt"
            save_checkpoint(
                final_path,
                train_model,
                optimizer,
                step,
                args,
                scheduler,
                include_optimizer=not args.no_save_optimizer,
            )
            write_json_log(metrics_log, {"event": "done", "step": step, "path": str(final_path)})
        trim_host_allocator()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
