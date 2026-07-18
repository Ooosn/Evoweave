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
    JointCountMixtureSampler,
    TopologyFamilyMixtureSampler,
    PuppeteerDynamicRigDataset,
    PuppeteerDynamicRigModel,
    PuppeteerJointTokenizer,
    import_puppeteer_decoder,
    load_puppeteer_decoder_state,
    load_puppeteer_target_aware_pos_embed,
    load_parent_topology_signatures,
    puppeteer_dynamic_collate,
    parse_joint_count_bin_uppers,
)
from rigweave.dynamic_rig.puppeteer_diagnostics import (  # noqa: E402
    condition_path_audit,
    gradient_path_audit,
    pose_target_contract_audit,
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
    if args.decoder_norm_style != "config":
        config.do_layer_norm_before = args.decoder_norm_style == "pre"
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


def decoder_block_modules(model: PuppeteerDynamicRigModel) -> list[torch.nn.Module]:
    decoder = model.decoder.model.decoder
    modules: list[torch.nn.Module] = [decoder.layers]
    final_layer_norm = getattr(decoder, "final_layer_norm", None)
    if final_layer_norm is not None:
        modules.append(final_layer_norm)
    return modules


def add_parameters_group(
    groups: list[dict[str, Any]],
    *,
    name: str,
    parameters: list[torch.nn.Parameter],
    lr: float,
    seen: set[int],
) -> None:
    params = []
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        params.append(parameter)
    if params:
        groups.append({"name": name, "params": params, "lr": float(lr)})


def suppress_decoder_block_gradients(model: PuppeteerDynamicRigModel) -> int:
    suppressed = 0
    for module in decoder_block_modules(model):
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            suppressed += int(parameter.numel())
            parameter.grad = None
    return suppressed


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
        name="joint_slot_embedding",
        parameter=model.target_aware_pos_embed,
        lr=args.lr_decoder,
        seen=seen,
    )
    if args.lr_decoder_blocks >= 0:
        add_parameters_group(
            groups,
            name="decoder_blocks",
            parameters=[parameter for module in decoder_block_modules(model) for parameter in module.parameters()],
            lr=args.lr_decoder_blocks,
            seen=seen,
        )
    add_param_group(
        groups,
        name="decoder_token_path" if args.lr_decoder_blocks >= 0 else "decoder",
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


def load_full_model_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    """Initialize the complete model with an exact Evoweave checkpoint state."""

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError(f"{checkpoint} is not an Evoweave training checkpoint with a model state")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "checkpoint": str(checkpoint),
        "source_step": int(payload.get("step", -1)),
        "source_sample_seen": payload.get("sample_seen"),
        "strict": True,
        "optimizer_loaded": False,
        "scheduler_loaded": False,
    }


def validate_query_preserving_baseline_args(args: argparse.Namespace) -> None:
    if not args.require_query_preserving_baseline_contract:
        return

    errors: list[str] = []
    if int(args.query_tokens) != 1024:
        errors.append(f"query_tokens must be 1024, got {args.query_tokens}")
    if int(args.cond_length) != int(args.query_tokens):
        errors.append(
            "cond_length must equal query_tokens for a one-to-one condition path, "
            f"got cond_length={args.cond_length} query_tokens={args.query_tokens}"
        )
    if args.condition_projection != "identity":
        errors.append(
            "condition_projection must be identity; the learned-query cross-attention "
            "projector is not approved for the baseline"
        )
    if args.decoder_norm_style != "pre":
        errors.append(
            "decoder_norm_style must be pre; the inherited Puppeteer post-LN config "
            "failed the controlled training check"
        )
    if args.scheduler != "onecycle":
        errors.append(
            "scheduler must be onecycle; sustained constant full learning rate collapsed "
            "the direct-condition control"
        )
    if args.no_joint_slot_embedding:
        errors.append("joint-slot embedding must be enabled for the joint-token baseline")
    if not args.train_random_query:
        errors.append("train_random_query must be enabled so input and target use sampled query poses")
    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(f"query-preserving Puppeteer baseline contract failed:\n  - {details}")


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
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(os.environ["EVOWEAVE_INIT_CHECKPOINT"]) if os.environ.get("EVOWEAVE_INIT_CHECKPOINT") else None,
        help="Strictly initialize the complete model state; optimizer and scheduler start fresh.",
    )
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
    parser.add_argument(
        "--joint-count-balance-alpha",
        type=float,
        default=0.0,
        help=(
            "Mix natural row-uniform sampling with uniform-over-joint-count-bin sampling. "
            "0 preserves the baseline distribution; 1 makes configured bins equiprobable."
        ),
    )
    parser.add_argument(
        "--joint-count-bin-uppers",
        type=str,
        default="10,20,40,60,80,101",
        help="Inclusive upper bounds for joint-count-balanced sampling bins.",
    )
    parser.add_argument(
        "--topology-balance-alpha",
        type=float,
        default=0.0,
        help=(
            "Mix natural row-uniform sampling with uniform-over-exact-target-parent-topology "
            "sampling. Mutually exclusive with joint-count balancing."
        ),
    )
    parser.add_argument(
        "--topology-scan-workers",
        type=int,
        default=8,
        help="Workers used to read target_parents while constructing topology families.",
    )
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
    parser.add_argument("--lr-decoder-blocks", type=float, default=-1.0)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--scheduler", choices=["none", "onecycle", "cosine"], default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--onecycle-div-factor", type=float, default=5.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10.0)
    parser.add_argument("--n-discrete-size", type=int, default=128)
    parser.add_argument(
        "--token-loss-reduction",
        choices=["token_mean", "sequence_mean"],
        default="token_mean",
        help="token_mean reproduces Puppeteer/HF CE; sequence_mean gives every skeleton equal CE mass.",
    )
    parser.add_argument(
        "--termination-decision-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary EOS-versus-next-joint loss at legal joint boundaries; uses existing decoder logits.",
    )
    parser.add_argument(
        "--n-max-joints",
        type=int,
        default=101,
        help="Default matches the released Puppeteer joint-token checkpoint position capacity.",
    )
    parser.add_argument("--target-coord-scale", type=float, default=0.25)
    parser.add_argument("--no-strict-target-range", action="store_true")
    parser.add_argument("--cond-length", type=int, default=1024)
    parser.add_argument("--projector-heads", type=int, default=8)
    parser.add_argument(
        "--condition-projection",
        choices=["cross_attention", "identity"],
        default="identity",
    )
    parser.add_argument("--attn-implementation", choices=["flash_attention_2"], default="flash_attention_2")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-resize-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-init", action="store_true", help="Train this decoder route from random initialization.")
    parser.add_argument("--random-init-smoke", action="store_true", help="Deprecated alias for --random-init.")
    parser.add_argument("--tiny-random-decoder", action="store_true", help="Use a small random SkeletonOPT config for development smoke only.")
    parser.add_argument("--decoder-hidden-size", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-ffn-dim", type=int, default=0)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-attention-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-activation-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-layerdrop", type=float, default=0.0)
    parser.add_argument(
        "--decoder-norm-style",
        choices=["config", "pre", "post"],
        default="pre",
        help="Keep the backbone norm placement or explicitly select pre/post layer norm.",
    )
    parser.add_argument(
        "--require-query-preserving-baseline-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the validated 1024-token identity condition path, pre-LN decoder, "
            "joint-slot embedding, random query poses, and OneCycle schedule. Disable only "
            "for an explicitly named diagnostic experiment."
        ),
    )
    parser.add_argument("--decoder-checkpointing", action="store_true")
    parser.add_argument("--decoder-block-warmup-steps", type=int, default=0)
    parser.add_argument("--no-joint-slot-embedding", action="store_true")
    parser.add_argument("--freeze-surface-tokenizer", action="store_true")
    parser.add_argument("--freeze-conditioner", action="store_true")
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-forward", action="store_true")
    parser.add_argument("--preflight-contract-sanity", action="store_true")
    parser.add_argument("--preflight-contract-max-positions", type=int, default=32)
    parser.add_argument("--preflight-contract-max-diff", type=float, default=3.0e-2)
    parser.add_argument("--preflight-pose-audit", action="store_true")
    parser.add_argument("--preflight-condition-audit", action="store_true")
    parser.add_argument("--preflight-gradient-audit", action="store_true")
    args = parser.parse_args()
    args.random_init = bool(args.random_init or args.random_init_smoke)
    validate_query_preserving_baseline_args(args)

    if args.train_manifest is None:
        raise ValueError("--train-manifest is required")
    if args.val_manifest is None:
        raise ValueError("--val-manifest is required")
    if not args.train_manifest.exists():
        raise FileNotFoundError(args.train_manifest)
    if not args.val_manifest.exists():
        raise FileNotFoundError(args.val_manifest)
    if args.puppeteer_checkpoint is not None and args.init_checkpoint is not None:
        raise ValueError("--puppeteer-checkpoint and --init-checkpoint are mutually exclusive")
    if args.puppeteer_checkpoint is None and args.init_checkpoint is None and not args.random_init:
        raise RuntimeError(
            "Set --puppeteer-checkpoint for decoder-only pretrained initialization, "
            "set --init-checkpoint for complete-model initialization, "
            "or set --random-init to train this route from scratch."
        )
    if args.tiny_random_decoder and not args.random_init:
        raise RuntimeError("--tiny-random-decoder requires --random-init")
    if args.decoder_block_warmup_steps < 0:
        raise ValueError("--decoder-block-warmup-steps must be non-negative")
    if args.decoder_block_warmup_steps > 0 and args.lr_decoder_blocks < 0:
        raise ValueError("--decoder-block-warmup-steps requires an explicit --lr-decoder-blocks")
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
        if args.require_query_preserving_baseline_contract and not bool(config.do_layer_norm_before):
            raise RuntimeError(
                "resolved decoder config is post-LN despite the required pre-LN baseline contract"
            )
        decoder = SkeletonOPT(config)
        if args.decoder_checkpointing:
            decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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
                "random_init": True,
                "tiny_random_decoder": bool(args.tiny_random_decoder),
                "target_aware_pos_embed_loaded": False,
                "target_aware_pos_embed_shape": None,
            }
        decoder_report["resolved_config"] = {
            "hidden_size": int(config.hidden_size),
            "ffn_dim": int(config.ffn_dim),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "dropout": float(config.dropout),
            "attention_dropout": float(config.attention_dropout),
            "activation_dropout": float(config.activation_dropout),
            "layerdrop": float(config.layerdrop),
            "do_layer_norm_before": bool(config.do_layer_norm_before),
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
            condition_projection=args.condition_projection,
            max_joints=args.n_max_joints,
            use_joint_slot_embedding=not args.no_joint_slot_embedding,
            target_aware_pos_embed=target_aware_pos_embed,
            token_loss_reduction=args.token_loss_reduction,
            termination_decision_loss_weight=args.termination_decision_loss_weight,
        )
        full_init_report = None
        if args.init_checkpoint is not None:
            full_init_report = load_full_model_checkpoint(model, args.init_checkpoint)
            if is_main(rank):
                save_json(args.output_dir / "full_model_init_report.json", full_init_report)
            log(
                rank,
                "complete model initialized strictly from "
                f"{full_init_report['checkpoint']} source_step={full_init_report['source_step']}",
            )
        if args.require_query_preserving_baseline_contract:
            if model.condition_projection != "identity":
                raise RuntimeError(
                    f"resolved condition projection is {model.condition_projection!r}, expected 'identity'"
                )
            if model.cond_length != model.query_tokens:
                raise RuntimeError(
                    "resolved condition length does not preserve the query-token sequence: "
                    f"cond_length={model.cond_length} query_tokens={model.query_tokens}"
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
        joint_count_bin_uppers = parse_joint_count_bin_uppers(args.joint_count_bin_uppers)
        if not 0.0 <= float(args.joint_count_balance_alpha) <= 1.0:
            raise ValueError(
                f"--joint-count-balance-alpha must be in [0, 1], got {args.joint_count_balance_alpha}"
            )
        if not 0.0 <= float(args.topology_balance_alpha) <= 1.0:
            raise ValueError(
                f"--topology-balance-alpha must be in [0, 1], got {args.topology_balance_alpha}"
            )
        if int(args.topology_scan_workers) < 0:
            raise ValueError(
                f"--topology-scan-workers must be non-negative, got {args.topology_scan_workers}"
            )
        if (
            float(args.joint_count_balance_alpha) > 0.0
            and float(args.topology_balance_alpha) > 0.0
        ):
            raise ValueError(
                "joint-count balancing and topology-family balancing are mutually exclusive"
            )
        if float(args.joint_count_balance_alpha) > 0.0 and joint_count_bin_uppers[-1] < args.n_max_joints:
            raise ValueError(
                "final joint-count bin must cover n_max_joints when balancing is enabled: "
                f"last_bin={joint_count_bin_uppers[-1]} n_max_joints={args.n_max_joints}"
            )
        effective_batch = int(world_size * args.batch_size * args.grad_accum_steps)
        args.effective_batch = effective_batch
        args.train_rows = train_rows
        sample_milestones = parse_sample_milestones(args.sample_milestones)
        joint_count_sampling: dict[str, object] = {
            "mode": "disabled",
            "mixture_alpha": 0.0,
            "bin_upper_bounds": list(joint_count_bin_uppers),
        }
        topology_sampling: dict[str, object] = {
            "mode": "disabled",
            "mixture_alpha": 0.0,
        }
        if float(args.topology_balance_alpha) > 0.0:
            topology_signatures = load_parent_topology_signatures(
                train_dataset.paths,
                num_workers=args.topology_scan_workers,
            )
            train_sampler = TopologyFamilyMixtureSampler(
                topology_signatures,
                mixture_alpha=args.topology_balance_alpha,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
            topology_sampling = train_sampler.report()
        elif float(args.joint_count_balance_alpha) > 0.0:
            train_sampler = JointCountMixtureSampler(
                train_dataset.manifest_joint_counts,
                bin_upper_bounds=joint_count_bin_uppers,
                mixture_alpha=args.joint_count_balance_alpha,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
            joint_count_sampling = train_sampler.report()
        else:
            train_sampler = (
                DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                if world_size > 1
                else None
            )
            topology_sampling = {
                "mode": "natural_row_uniform",
                "mixture_alpha": 0.0,
            }

        if is_main(rank):
            save_json(
                args.output_dir / "training_contract.json",
                {
                    "decoder_family": "puppeteer_skeletonopt",
                    "target": "rootless target_joints_rootspace + target_parents",
                    "tail_tokens": False,
                    "tokenization": "joint-token: x,y,z,parent_index",
                    "max_joints": args.n_max_joints,
                    "joint_slot_embedding": not args.no_joint_slot_embedding,
                    "train_raw_rows": train_dataset.raw_rows,
                    "train_rows_after_max_joint_filter": len(train_dataset),
                    "train_filtered_over_max_joints": train_dataset.filtered_over_max_joints,
                    "val_raw_rows": val_dataset.raw_rows,
                    "val_rows_after_max_joint_filter": len(val_dataset),
                    "val_filtered_over_max_joints": val_dataset.filtered_over_max_joints,
                    "condition": "Evoweave dynamic surface motion prefix",
                    "query_tokens": args.query_tokens,
                    "condition_length": args.cond_length,
                    "condition_projection": args.condition_projection,
                    "decoder_norm_style": args.decoder_norm_style,
                    "resolved_do_layer_norm_before": bool(config.do_layer_norm_before),
                    "scheduler": args.scheduler,
                    "query_preserving_baseline_contract_required": bool(
                        args.require_query_preserving_baseline_contract
                    ),
                    "first_joint_contract": "joint0 is the single rootless skeleton root; no synthetic root",
                    "token_loss_reduction": args.token_loss_reduction,
                    "termination_decision_loss_weight": args.termination_decision_loss_weight,
                    "joint_count_sampling": joint_count_sampling,
                    "topology_sampling": topology_sampling,
                    "world_size": world_size,
                    "micro_batch_per_gpu": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "effective_batch": effective_batch,
                    "sample_milestones": sample_milestones,
                    "random_init": bool(args.random_init),
                    "full_model_init": full_init_report,
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
        log(
            rank,
            "condition contract: "
            f"projection={args.condition_projection} "
            f"query_tokens={args.query_tokens} cond_length={args.cond_length} "
            f"decoder_norm={'pre' if config.do_layer_norm_before else 'post'} "
            f"scheduler={args.scheduler}",
        )
        log(rank, f"joint-count sampling: {json.dumps(joint_count_sampling, sort_keys=True)}")
        log(rank, f"topology sampling: {json.dumps(topology_sampling, sort_keys=True)}")

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
            def release_preflight_memory() -> None:
                trim_host_allocator()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            batch = next(iter(train_loader))
            batch = move_batch(batch, device)
            preflight_audit: dict[str, Any] = {}
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
            if args.preflight_pose_audit:
                preflight_audit["pose_target_contract"] = pose_target_contract_audit(batch)
                log(
                    rank,
                    "preflight pose target contract="
                    + json.dumps(preflight_audit["pose_target_contract"], sort_keys=True),
                )
            if args.preflight_condition_audit:
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=device.type == "cuda" and amp_dtype != torch.float32,
                ):
                    preflight_audit["condition_path"] = condition_path_audit(model, batch)
                log(rank, "preflight condition path=" + json.dumps(preflight_audit["condition_path"], sort_keys=True))
                release_preflight_memory()
            if args.preflight_forward:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=device.type == "cuda" and amp_dtype != torch.float32,
                ):
                    out = model(batch)
                log(rank, f"preflight forward loss={float(out['loss'].detach().cpu()):.6f}")
                del out
                release_preflight_memory()
            if args.preflight_contract_sanity:
                if int(batch["frame_vertices"].shape[0]) != 1:
                    raise ValueError("--preflight-contract-sanity requires --batch-size 1")
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
                    sanity = model.teacher_forcing_generation_alignment(
                        batch,
                        max_positions=args.preflight_contract_max_positions,
                    )
                log(rank, "preflight contract sanity=" + json.dumps(sanity, sort_keys=True))
                if int(sanity["checked_positions"]) <= 0 or float(sanity["max_abs_logit_diff"]) > float(args.preflight_contract_max_diff):
                    raise RuntimeError(f"teacher-forcing/generation logits are not aligned: {sanity}")
                release_preflight_memory()
            if args.preflight_gradient_audit:
                release_preflight_memory()
                was_training = model.training
                model.train()
                torch.manual_seed(args.seed + 991)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + 991)
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=device.type == "cuda" and amp_dtype != torch.float32,
                ):
                    gradient_out = model(batch)
                preflight_audit["gradient_path"] = gradient_path_audit(model, gradient_out["loss"])
                model.train(was_training)
                log(rank, "preflight gradient path=" + json.dumps(preflight_audit["gradient_path"], sort_keys=True))
            if preflight_audit and is_main(rank):
                save_json(args.output_dir / "preflight_audit.json", preflight_audit)
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
                "random_init": bool(args.random_init),
                "decoder_checkpoint": str(args.puppeteer_checkpoint) if args.puppeteer_checkpoint else None,
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                "optimizer_groups": optimizer_group_report(optimizer),
            },
        )

        saved_sample_milestones: set[int] = set()
        best_val_loss = float("inf")
        step = 0
        epoch = 0
        accum_count = 0
        accum_t0 = time.time()
        tracked_metrics = (
            "token_acc",
            "coord_acc",
            "parent_acc",
            "eos_acc",
            "token_ce_loss",
            "termination_decision_loss",
            "termination_stop_acc",
            "termination_continue_acc",
        )
        accum_sums = {"loss": 0.0, **{key: 0.0 for key in tracked_metrics}}
        accum_path = ""
        decoder_block_grads_suppressed = 0
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
                    if step < args.decoder_block_warmup_steps:
                        decoder_block_grads_suppressed = suppress_decoder_block_gradients(unwrap_model(train_model))
                    else:
                        decoder_block_grads_suppressed = 0
                accum_sums["loss"] += float(raw_loss.detach().cpu())
                for key in tracked_metrics:
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
                        **{key: accum_sums[key] / micro_count for key in tracked_metrics},
                        "seconds": round(time.time() - accum_t0, 3),
                        "path": accum_path,
                        "grad_accum": args.grad_accum_steps,
                        "micro_batch_per_gpu": args.batch_size,
                        "decoder_block_warmup_active": step <= args.decoder_block_warmup_steps,
                        "decoder_block_grads_suppressed": decoder_block_grads_suppressed,
                        "lrs": {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)},
                        **accounting,
                    }
                    if device.type == "cuda":
                        row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                        torch.cuda.reset_peak_memory_stats(device)
                    write_json_log(metrics_log, row)
                accum_sums = {"loss": 0.0, **{key: 0.0 for key in tracked_metrics}}
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
