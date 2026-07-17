#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager
import gc
import importlib.util
import json
import os
import resource
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


REPO = Path(__file__).resolve().parents[2]
UNIRIG_ROOT = Path(os.environ.get("EVOWEAVE_UNIRIG_ROOT", REPO / "external" / "UniRig")).expanduser()
if str(UNIRIG_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIRIG_ROOT))
if str(REPO / "rigweave" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "rigweave" / "src"))


RUN_LOG_PATH: Path | None = None


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _attrify(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _attrify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_attrify(v) for v in value]
    return value


def load_yaml(path: str | Path) -> AttrDict:
    with Path(path).open(encoding="utf-8") as f:
        return _attrify(yaml.safe_load(f))


def trim_host_allocator() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def move_dynamic_model_to_device(model: torch.nn.Module, device: torch.device) -> None:
    """Move shared dynamic/UniRig modules once.

    The dynamic surface tokenizer shares UniRig's Michelangelo mesh encoder and
    output projection with the AR wrapper.  A recursive ``model.to(device)`` can
    revisit those shared modules through multiple paths and has caused CUDA
    driver OOMs on qlogin H100 sessions before the first batch.  Moving the real
    submodules explicitly keeps the train path aligned with eval while avoiding
    duplicate traversal.
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
    for name, parameter in model.named_parameters(recurse=False):
        parameter.data = parameter.data.to(device)
        if parameter.grad is not None:
            parameter.grad = parameter.grad.to(device)
    for name, buffer in model.named_buffers(recurse=False):
        setattr(model, name, buffer.to(device))


def assert_module_on_device(model: torch.nn.Module, device: torch.device) -> None:
    wrong: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.device != device:
            wrong.append(f"param:{name}:{parameter.device}")
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            wrong.append(f"buffer:{name}:{buffer.device}")
    if wrong:
        preview = ", ".join(wrong[:40])
        suffix = "" if len(wrong) <= 40 else f", ... total={len(wrong)}"
        raise RuntimeError(f"module has tensors off {device}: {preview}{suffix}")


def validate_init_checkpoint_keys(missing: list[str], unexpected: list[str]) -> None:
    """Fail fast if an init checkpoint drops any core route weights.

    Older checkpoints do not contain auxiliary heads that were added for the
    current EOS/free-generation diagnosis.  Those are allowed to initialize from
    their explicit module defaults.  Missing UniRig AR, surface tokenizer, or
    alternating motion encoder weights are not allowed, because that would turn
    a continuation run into a silent partial reinitialization.
    """

    allowed_missing_prefixes = (
        "condition_fuser.",
        "structure_count_head.",
        "structure_action_head.",
        "grammar_state_proj.",
        "action_group_bias_head.",
        "condition_action_group_bias_head.",
        "branch_prior.",
        "explicit_tree_decoder.",
    )
    bad_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        preview_missing = ", ".join(bad_missing[:40])
        preview_unexpected = ", ".join(unexpected[:40])
        raise RuntimeError(
            "init checkpoint key mismatch: "
            f"bad_missing={len(bad_missing)} [{preview_missing}] "
            f"unexpected={len(unexpected)} [{preview_unexpected}]"
        )


def initialize_explicit_tree_topology_from_geometry(model: torch.nn.Module) -> int:
    """Seed coordinate-free topology modules from the trained geometry decoder path."""

    decoder = getattr(model, "explicit_tree_decoder", None)
    if decoder is None:
        return 0
    copied = 0
    pairs = (
        ("topology_state_mlp", "state_mlp"),
        ("topology_step_mlp", "step_mlp"),
        ("topology_decoder", "decoder"),
        ("topology_out_norm", "out_norm"),
        ("topology_action_head", "action_head"),
        ("topology_parent_query", "parent_query"),
    )
    for dst_name, src_name in pairs:
        dst = getattr(decoder, dst_name, None)
        src = getattr(decoder, src_name, None)
        if dst is None or src is None:
            continue
        dst.load_state_dict(src.state_dict())
        copied += 1
    parent_key = getattr(decoder, "topology_parent_key", None)
    if isinstance(parent_key, torch.nn.Sequential):
        for module in parent_key:
            if isinstance(module, torch.nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
                with torch.no_grad():
                    module.weight.zero_()
                    module.weight.diagonal().fill_(1.0)
                    if module.bias is not None:
                        module.bias.zero_()
                copied += 1
    return copied


def log(rank: int, message: str) -> None:
    if is_main(rank):
        text = f"[dynamic_rig] {message}"
        print(text, flush=True)
        if RUN_LOG_PATH is not None:
            with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(text + "\n")


def _read_proc_status(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with Path(f"/proc/{pid}/status").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith(("VmSize:", "VmRSS:", "VmPeak:", "VmHWM:")):
                    key, rest = line.split(":", 1)
                    parts = rest.strip().split()
                    if parts:
                        values[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return values


def _child_pids(parent_pid: int) -> list[int]:
    children: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            with (proc / "stat").open(encoding="utf-8") as f:
                fields = f.read().split()
            if len(fields) > 3 and int(fields[3]) == parent_pid:
                child = int(proc.name)
                children.append(child)
                children.extend(_child_pids(child))
        except OSError:
            continue
    return children


def _fmt_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f}G"


def log_process_memory(rank: int, label: str, enabled: bool) -> None:
    if not enabled:
        return
    pid = os.getpid()
    pids = [pid] + _child_pids(pid)
    rows: list[str] = []
    total_vms = 0
    total_rss = 0
    for child_pid in pids:
        status = _read_proc_status(child_pid)
        vms = status.get("VmSize", 0)
        rss = status.get("VmRSS", 0)
        total_vms += vms
        total_rss += rss
        try:
            comm = Path(f"/proc/{child_pid}/comm").read_text(encoding="utf-8").strip()
        except OSError:
            comm = "?"
        rows.append(f"{child_pid}:{comm}:vms={_fmt_gib(vms)} rss={_fmt_gib(rss)}")
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    log(
        rank,
        "proc_mem "
        f"label={label} pid={pid} nproc={len(pids)} "
        f"tree_vms={_fmt_gib(total_vms)} tree_rss={_fmt_gib(total_rss)} "
        f"self_maxrss={rusage.ru_maxrss}KB rows={' | '.join(rows[:16])}",
    )


def write_json_log(path: Path | None, row: dict[str, Any]) -> None:
    text = json.dumps(row, ensure_ascii=False)
    print(text, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def parse_sample_milestones(raw: str) -> list[int]:
    if not raw.strip():
        return []
    milestones: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"sample milestone must be positive, got {value}")
        milestones.append(value)
    return sorted(set(milestones))


def accounting_fields(step: int, effective_batch: int, train_rows: int) -> dict[str, float | int]:
    sample_seen = int(step * effective_batch)
    epoch_equivalent = float(sample_seen / max(1, train_rows))
    return {
        "sample_seen": sample_seen,
        "optimizer_samples_seen": sample_seen,
        "effective_batch": int(effective_batch),
        "train_rows": int(train_rows),
        "epoch_equivalent": epoch_equivalent,
    }


def load_unirig(tokenizer: Any, model_config: Path, checkpoint: Path) -> Any:
    if os.environ.get("RIGWEAVE_DISABLE_LIGHTNING_IMPORT", "0") == "1" and "lightning.pytorch" not in sys.modules:
        # UniRig's model base class inherits LightningModule, but this training
        # script never uses Lightning Trainer APIs.  Stubbing avoids importing
        # lightning/torchmetrics just to construct an nn.Module.
        lightning_module = types.ModuleType("lightning")
        lightning_pytorch = types.ModuleType("lightning.pytorch")
        lightning_pytorch.LightningModule = torch.nn.Module
        lightning_module.pytorch = lightning_pytorch
        sys.modules["lightning"] = lightning_module
        sys.modules["lightning.pytorch"] = lightning_pytorch

    from transformers import AutoConfig, AutoModelForCausalLM, OPTConfig, OPTForCausalLM

    original_from_pretrained = AutoConfig.from_pretrained
    original_from_config = AutoModelForCausalLM.from_config

    def fast_opt_config(cls: type, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path).startswith("facebook/opt"):
            kwargs.setdefault("local_files_only", True)
            return OPTConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def fast_opt_model(cls: type, config: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(config, OPTConfig):
            return OPTForCausalLM(config)
        return original_from_config(config, *args, **kwargs)

    AutoConfig.from_pretrained = classmethod(fast_opt_config)
    AutoModelForCausalLM.from_config = classmethod(fast_opt_model)
    from src.model.parse import get_model

    cfg = load_yaml(model_config)
    llm_cfg = cfg.get("llm")
    if isinstance(llm_cfg, dict) and llm_cfg.get("_attn_implementation") == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            llm_cfg["_attn_implementation"] = "sdpa"
            print("[dynamic_rig] flash_attn not installed; using OPT sdpa attention", flush=True)
    cfg["tokenizer"] = tokenizer
    with pushd(UNIRIG_ROOT):
        model = get_model(**cfg)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    del ckpt, state
    trim_host_allocator()
    if missing:
        print(f"[dynamic_rig] WARNING missing UniRig keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[dynamic_rig] WARNING unexpected UniRig keys: {len(unexpected)}", flush=True)
    return model


def build_tokenizer(tokenizer_config: Path) -> Any:
    if os.environ.get("RIGWEAVE_DISABLE_OPEN3D", "0") == "1" and "open3d" not in sys.modules:
        # UniRig's tokenizer imports an exporter module that tries to import
        # open3d. Training only needs tokenization, so avoid a slow optional
        # visualization dependency during every launch.
        sys.modules["open3d"] = None
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    cfg = load_yaml(tokenizer_config)
    with pushd(UNIRIG_ROOT):
        return get_tokenizer(TokenizerConfig.parse(cfg))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    *,
    include_optimizer: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    effective_batch = int(getattr(args, "effective_batch", 0) or 0)
    train_rows = int(getattr(args, "train_rows", 0) or 0)
    sample_seen = int(step * effective_batch) if effective_batch > 0 else None
    payload = {
        "step": step,
        "model": raw.state_dict(),
        "args": json_safe(vars(args)),
    }
    if sample_seen is not None:
        payload["sample_seen"] = sample_seen
        payload["effective_batch"] = effective_batch
        payload["train_rows"] = train_rows
        payload["epoch_equivalent"] = float(sample_seen / max(1, train_rows))
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if include_optimizer and scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    trim_host_allocator()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    raw_model = unwrap_model(model)
    eval_disabled_weights = [
        "decision_loss_weight",
        "loop_recovery_loss_weight",
        "prefix_decision_recovery_weight",
        "prefix_token_recovery_weight",
        "prefix_action_recovery_weight",
        "generated_prefix_recovery_weight",
        "explicit_tree_generated_prefix_weight",
        "structure_count_loss_weight",
        "structure_action_loss_weight",
        "latent_align_weight",
        "motion_contrast_weight",
        "condition_control_ce_weight",
    ]
    previous_weights = {
        name: getattr(raw_model, name)
        for name in eval_disabled_weights
        if hasattr(raw_model, name)
    }
    for name in previous_weights:
        setattr(raw_model, name, 0.0)
    model.eval()
    local_loss_sum = 0.0
    local_ce_sum = 0.0
    local_eos_loss_sum = 0.0
    local_eos_acc_sum = 0.0
    local_count = 0
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            for i, batch in enumerate(loader):
                if i >= max_steps:
                    break
                torch.manual_seed(2026052800 + i)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(2026052800 + i)
                batch = move_batch(batch, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    out = model(batch)
                local_loss_sum += float(out["loss"].detach().cpu())
                local_ce_sum += float(out["ce_loss"].detach().cpu())
                local_eos_loss_sum += float(out.get("eos_loss", torch.zeros(())).detach().cpu())
                local_eos_acc_sum += float(out.get("eos_acc", torch.zeros(())).detach().cpu())
                local_count += 1
    finally:
        for name, value in previous_weights.items():
            setattr(raw_model, name, value)
        model.train(was_training)
    stats = torch.tensor(
        [local_loss_sum, local_ce_sum, local_eos_loss_sum, local_eos_acc_sum, float(local_count)],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    count = int(stats[4].item())
    if count <= 0:
        return {"val_loss": -1.0, "val_ce": -1.0}
    return {
        "val_loss": float((stats[0] / stats[4]).item()),
        "val_ce": float((stats[1] / stats[4]).item()),
        "val_eos_loss": float((stats[2] / stats[4]).item()),
        "val_eos_acc": float((stats[3] / stats[4]).item()),
        "val_count": count,
    }


def main() -> None:
    global RUN_LOG_PATH

    parser = argparse.ArgumentParser(description="Train the locked Dynamic Rig Phase1 route.")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(os.environ.get("EVOWEAVE_TRAIN_MANIFEST", "rigweave/configs/MISSING_TRAIN_MANIFEST.jsonl")),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(os.environ.get("EVOWEAVE_VAL_MANIFEST", "rigweave/configs/MISSING_VAL_MANIFEST.jsonl")),
    )
    parser.add_argument("--tokenizer-config", type=Path, default=Path("external/UniRig/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("external/UniRig/configs/model/unirig_ar_350m_1024_81920_float32.yaml"))
    parser.add_argument("--unirig-checkpoint", type=Path, default=Path("external/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--init-dynamic-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--init-surface-tokenizer-from-dynamic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("rigweave/outputs/dynamic_rig_runs/trackable_surface_ar"))
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=64)
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
    parser.add_argument("--lr-motion", type=float, default=1.0e-4)
    parser.add_argument("--lr-ar", type=float, default=1.0e-4)
    parser.add_argument("--lr-surface", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--scheduler", choices=["none", "onecycle"], default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--onecycle-div-factor", type=float, default=5.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument(
        "--sample-milestones",
        type=str,
        default="5000,10000,20000,30000,50000,80000",
        help=(
            "Comma-separated optimizer sample-seen milestones.  The trainer "
            "saves checkpoint_sample_<N>.pt when step * effective_batch crosses N. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--no-save-optimizer",
        action="store_true",
        help="Save model-only checkpoints for qlogin diagnostics; full resume state is intentionally omitted.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-process-memory",
        action="store_true",
        default=os.environ.get("RIGWEAVE_LOG_PROCESS_MEMORY", "0") == "1",
        help="Log /proc VmSize/RSS for the rank process tree at key launch/training stages.",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--freeze-ar", action="store_true")
    parser.add_argument(
        "--freeze-conditioner",
        action="store_true",
        help=(
            "Freeze the dynamic mesh conditioner. Use this for causal probes "
            "that train only newly added decode heads on top of an already "
            "trained condition representation."
        ),
    )
    parser.add_argument("--train-surface-tokenizer", dest="train_surface_tokenizer", action="store_true", default=True)
    parser.add_argument("--freeze-surface-tokenizer", dest="train_surface_tokenizer", action="store_false")
    parser.add_argument("--motion-fps-ratio", type=float, default=0.7)
    parser.add_argument("--motion-vertex-samples", type=int, default=512)
    parser.add_argument("--train-random-query", dest="train_random_query", action="store_true", default=True)
    parser.add_argument("--no-train-random-query", dest="train_random_query", action="store_false")
    parser.add_argument("--target-active-skin-only", action="store_true")
    parser.add_argument("--active-skin-threshold", type=float, default=1.0e-4)
    parser.add_argument("--target-start-policy", choices=["joint0"], default="joint0")
    parser.add_argument("--target-root-policy", choices=["legacy"], default="legacy")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--latent-align-weight", type=float, default=0.0)
    parser.add_argument("--motion-contrast-weight", type=float, default=0.0)
    parser.add_argument("--motion-contrast-margin", type=float, default=0.05)
    parser.add_argument("--contrast-controls", type=str, default="zero,reverse")
    parser.add_argument("--condition-control-ce-weight", type=float, default=0.0)
    parser.add_argument("--condition-control-ce-controls", type=str, default="zero,shuffle")
    parser.add_argument(
        "--condition-control-ce-every",
        type=int,
        default=1,
        help=(
            "Run condition-control CE every N optimizer steps. "
            "Use 0 to disable it during training even when the weight is nonzero."
        ),
    )
    parser.add_argument("--eos-loss-weight", type=float, default=1.0)
    parser.add_argument("--decision-loss-weight", type=float, default=0.0)
    parser.add_argument("--loop-recovery-loss-weight", type=float, default=0.0)
    parser.add_argument("--loop-recovery-repeats", type=int, default=4)
    parser.add_argument("--prefix-decision-recovery-weight", type=float, default=0.0)
    parser.add_argument("--prefix-decision-recovery-states", type=int, default=4)
    parser.add_argument("--prefix-decision-recovery-variants", type=int, default=1)
    parser.add_argument("--prefix-decision-recovery-jitter", type=int, default=4)
    parser.add_argument("--prefix-token-recovery-weight", type=float, default=0.0)
    parser.add_argument("--prefix-token-recovery-states", type=int, default=4)
    parser.add_argument("--prefix-token-recovery-variants", type=int, default=1)
    parser.add_argument("--prefix-token-recovery-jitter", type=int, default=4)
    parser.add_argument("--prefix-token-recovery-max-rows", type=int, default=4)
    parser.add_argument("--prefix-action-recovery-weight", type=float, default=0.0)
    parser.add_argument("--prefix-action-recovery-states", type=int, default=4)
    parser.add_argument("--prefix-action-recovery-variants", type=int, default=1)
    parser.add_argument("--prefix-action-recovery-jitter", type=int, default=4)
    parser.add_argument("--prefix-action-recovery-max-rows", type=int, default=4)
    parser.add_argument("--generated-prefix-recovery-weight", type=float, default=0.0)
    parser.add_argument("--generated-prefix-recovery-states", type=int, default=4)
    parser.add_argument("--generated-prefix-recovery-max-new-tokens", type=int, default=128)
    parser.add_argument("--generated-prefix-recovery-max-rows", type=int, default=4)
    parser.add_argument(
        "--generated-prefix-recovery-every",
        type=int,
        default=1,
        help=(
            "Run generated-prefix recovery every N optimizer steps. "
            "Use 0 to disable it during training even when the weight is nonzero."
        ),
    )
    parser.add_argument("--structure-count-loss-weight", type=float, default=0.0)
    parser.add_argument("--structure-action-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--condition-fusion",
        choices=["dynamic", "static_blend", "static_cross_attn", "static_cross_attn_zero"],
        default="dynamic",
    )
    parser.add_argument("--condition-fusion-heads", type=int, default=8)
    parser.add_argument("--condition-fusion-gate-init", type=float, default=0.25)
    parser.add_argument("--condition-fusion-depth", type=int, default=1)
    parser.add_argument("--condition-static-blend-weight", type=float, default=0.0)
    parser.add_argument("--branch-prior-proposals", type=int, default=32)
    parser.add_argument("--branch-prior-heads", type=int, default=8)
    parser.add_argument("--branch-prior-loss-weight", type=float, default=1.0)
    parser.add_argument("--branch-prior-coord-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--freeze-branch-prior",
        action="store_true",
        help="Freeze coarse branch-prior proposal tokens/heads for isolated decoder-head probes.",
    )
    parser.add_argument(
        "--freeze-explicit-tree-non-xyz",
        action="store_true",
        help="Freeze all explicit-tree decoder parameters except xyz_head for coordinate calibration probes.",
    )
    parser.add_argument("--explicit-tree-loss-weight", type=float, default=0.0)
    parser.add_argument("--explicit-tree-generated-prefix-weight", type=float, default=0.0)
    parser.add_argument("--explicit-tree-generated-prefix-states", type=int, default=4)
    parser.add_argument("--explicit-tree-generated-prefix-max-steps", type=int, default=64)
    parser.add_argument("--explicit-tree-generated-prefix-max-rows", type=int, default=4)
    parser.add_argument("--explicit-tree-oracle-prefix-weight", type=float, default=0.0)
    parser.add_argument("--explicit-tree-oracle-prefix-states", type=int, default=4)
    parser.add_argument("--explicit-tree-oracle-prefix-max-steps", type=int, default=64)
    parser.add_argument("--explicit-tree-oracle-prefix-max-rows", type=int, default=4)
    parser.add_argument(
        "--explicit-tree-generated-prefix-every",
        type=int,
        default=1,
        help="Run explicit-tree generated-prefix recovery every N optimizer steps; <=0 disables it.",
    )
    parser.add_argument("--explicit-tree-prefix-jitter-weight", type=float, default=0.0)
    parser.add_argument("--explicit-tree-prefix-jitter-std", type=float, default=0.0)
    parser.add_argument("--explicit-tree-depth", type=int, default=4)
    parser.add_argument("--explicit-tree-heads", type=int, default=8)
    parser.add_argument(
        "--explicit-tree-topology-mode",
        choices=["geometry", "topology", "hybrid", "split", "planner", "topomlp"],
        default="geometry",
        help="Use geometry prefix/key features or coordinate-free topology features for the explicit-tree decoder.",
    )
    parser.add_argument(
        "--explicit-tree-coordinate-mode",
        choices=["absolute", "parent_delta"],
        default="absolute",
        help="Predict explicit-tree coordinates as absolute xyz or as an offset from the selected parent.",
    )
    parser.add_argument(
        "--init-explicit-tree-topology-from-geometry",
        action="store_true",
        help="Seed split/planner topology modules from the trained geometry explicit-tree path after loading init checkpoint.",
    )
    parser.add_argument("--explicit-tree-action-eos-loss-weight", type=float, default=1.0)
    parser.add_argument("--explicit-tree-action-child-loss-weight", type=float, default=1.0)
    parser.add_argument("--explicit-tree-action-branch-loss-weight", type=float, default=1.0)
    parser.add_argument("--explicit-tree-xyz-loss-weight", type=float, default=1.0)
    parser.add_argument("--reset-condition-fuser", action="store_true")
    parser.add_argument("--use-grammar-state-embedding", action="store_true")
    parser.add_argument("--use-action-group-bias", action="store_true")
    parser.add_argument("--use-condition-action-group-bias", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    device, rank, local_rank, world_size = setup_distributed()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    metrics_log: Path | None = None
    if is_main(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        RUN_LOG_PATH = log_dir / "run.log"
        RUN_LOG_PATH.write_text("", encoding="utf-8")

    try:
        from functools import partial

        from rigweave.dynamic_rig import AnchorWiseAlternatingMotionEncoder, FixedQuerySurfaceTokenizer
        from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
        from rigweave.dynamic_rig.model import DynamicRigConditioner
        from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR

        log(rank, f"device={device} world_size={world_size}")
        log(rank, "build tokenizer and official UniRig")
        setup_t0 = time.time()
        stage_t0 = time.time()
        tokenizer = build_tokenizer(args.tokenizer_config)
        log(rank, f"tokenizer built in {time.time() - stage_t0:.2f}s")
        log_process_memory(rank, "after_tokenizer", args.log_process_memory)
        stage_t0 = time.time()
        unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint)
        log(rank, f"official UniRig loaded in {time.time() - stage_t0:.2f}s")
        log_process_memory(rank, "after_unirig_load_cpu", args.log_process_memory)
        stage_t0 = time.time()

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
        model = DynamicRigUniRigAR(
            unirig,
            conditioner,
            tokenizer,
            num_surface_samples=args.surface_samples,
            vertex_samples=args.vertex_samples,
            query_tokens=args.query_tokens,
            latent_align_weight=args.latent_align_weight,
            motion_contrast_weight=args.motion_contrast_weight,
            motion_contrast_margin=args.motion_contrast_margin,
            contrast_controls=tuple(x.strip() for x in args.contrast_controls.split(",") if x.strip()),
            condition_control_ce_weight=args.condition_control_ce_weight,
            condition_control_ce_controls=tuple(
                x.strip() for x in args.condition_control_ce_controls.split(",") if x.strip()
            ),
            eos_loss_weight=args.eos_loss_weight,
            decision_loss_weight=args.decision_loss_weight,
            loop_recovery_loss_weight=args.loop_recovery_loss_weight,
            loop_recovery_repeats=args.loop_recovery_repeats,
            prefix_decision_recovery_weight=args.prefix_decision_recovery_weight,
            prefix_decision_recovery_states=args.prefix_decision_recovery_states,
            prefix_decision_recovery_variants=args.prefix_decision_recovery_variants,
            prefix_decision_recovery_jitter=args.prefix_decision_recovery_jitter,
            prefix_token_recovery_weight=args.prefix_token_recovery_weight,
            prefix_token_recovery_states=args.prefix_token_recovery_states,
            prefix_token_recovery_variants=args.prefix_token_recovery_variants,
            prefix_token_recovery_jitter=args.prefix_token_recovery_jitter,
            prefix_token_recovery_max_rows=args.prefix_token_recovery_max_rows,
            prefix_action_recovery_weight=args.prefix_action_recovery_weight,
            prefix_action_recovery_states=args.prefix_action_recovery_states,
            prefix_action_recovery_variants=args.prefix_action_recovery_variants,
            prefix_action_recovery_jitter=args.prefix_action_recovery_jitter,
            prefix_action_recovery_max_rows=args.prefix_action_recovery_max_rows,
            generated_prefix_recovery_weight=args.generated_prefix_recovery_weight,
            generated_prefix_recovery_states=args.generated_prefix_recovery_states,
            generated_prefix_recovery_max_new_tokens=args.generated_prefix_recovery_max_new_tokens,
            generated_prefix_recovery_max_rows=args.generated_prefix_recovery_max_rows,
            structure_count_loss_weight=args.structure_count_loss_weight,
            structure_action_loss_weight=args.structure_action_loss_weight,
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
        log(rank, f"dynamic route modules built in {time.time() - stage_t0:.2f}s")
        log_process_memory(rank, "after_dynamic_modules_cpu", args.log_process_memory)
        stage_t0 = time.time()

        if not args.train_surface_tokenizer:
            for p in surface_tokenizer.parameters():
                p.requires_grad_(False)
            surface_tokenizer.eval()
        if args.freeze_conditioner:
            for p in conditioner.parameters():
                p.requires_grad_(False)
            conditioner.eval()
        if args.freeze_branch_prior and getattr(model, "branch_prior", None) is not None:
            for p in model.branch_prior.parameters():
                p.requires_grad_(False)
            model.branch_prior.eval()
        if args.freeze_explicit_tree_non_xyz and getattr(model, "explicit_tree_decoder", None) is not None:
            for name, p in model.explicit_tree_decoder.named_parameters():
                p.requires_grad_("xyz_head." in name)
        if args.freeze_ar:
            for p in unirig.transformer.parameters():
                p.requires_grad_(False)

        move_dynamic_model_to_device(model, device)
        log(rank, f"model moved to device in {time.time() - stage_t0:.2f}s")
        log_process_memory(rank, "after_model_to_device", args.log_process_memory)
        stage_t0 = time.time()
        if args.init_checkpoint is not None:
            init_t0 = time.time()
            ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            state = dict(ckpt["model"])
            role_key = "conditioner.motion_encoder.role_token"
            frame_type_key = "conditioner.motion_encoder.frame_type_embed.weight"
            if role_key not in state and frame_type_key in state and role_key in model.state_dict():
                state[role_key] = state[frame_type_key].view(1, 2, 1, -1).clone()
                log(rank, "initialized role_token from frame_type_embed.weight in init checkpoint")
            missing, unexpected = model.load_state_dict(state, strict=False)
            validate_init_checkpoint_keys(list(missing), list(unexpected))
            del ckpt, state
            trim_host_allocator()
            log(
                rank,
                f"init checkpoint={args.init_checkpoint} missing={len(missing)} "
                f"unexpected={len(unexpected)} loaded_in={time.time() - init_t0:.2f}s",
            )
            log_process_memory(rank, "after_init_checkpoint", args.log_process_memory)
        if args.init_explicit_tree_topology_from_geometry:
            copied = initialize_explicit_tree_topology_from_geometry(model)
            log(rank, f"initialized explicit-tree topology modules from geometry path copied={copied}")
        stage_t0 = time.time()
        if args.init_dynamic_encoder_checkpoint is not None:
            init_t0 = time.time()
            ckpt = torch.load(args.init_dynamic_encoder_checkpoint, map_location="cpu", weights_only=False)
            state = dict(ckpt["model"])
            prefixes = ["conditioner.motion_encoder."]
            if args.init_surface_tokenizer_from_dynamic:
                prefixes.append("conditioner.surface_tokenizer.")
            subset = {k: v for k, v in state.items() if any(k.startswith(prefix) for prefix in prefixes)}
            subset_len = len(subset)
            role_key = "conditioner.motion_encoder.role_token"
            frame_type_key = "conditioner.motion_encoder.frame_type_embed.weight"
            if role_key not in subset and frame_type_key in state and role_key in model.state_dict():
                subset[role_key] = state[frame_type_key].view(1, 2, 1, -1).clone()
                log(rank, "initialized dynamic role_token from frame_type_embed.weight")
            missing, unexpected = model.load_state_dict(subset, strict=False)
            bad_dynamic_missing = [key for key in missing if any(key.startswith(prefix) for prefix in prefixes)]
            if bad_dynamic_missing or unexpected:
                raise RuntimeError(
                    "init dynamic encoder checkpoint key mismatch: "
                    f"bad_missing={len(bad_dynamic_missing)} [{', '.join(bad_dynamic_missing[:40])}] "
                    f"unexpected={len(unexpected)} [{', '.join(unexpected[:40])}]"
                )
            loaded = len(subset) - len(unexpected)
            del ckpt, state, subset
            trim_host_allocator()
            log(
                rank,
                "init dynamic encoder checkpoint="
                f"{args.init_dynamic_encoder_checkpoint} loaded={loaded} "
                f"subset={subset_len} missing={len(missing)} unexpected={len(unexpected)} "
                f"surface={args.init_surface_tokenizer_from_dynamic} loaded_in={time.time() - init_t0:.2f}s",
            )
            log_process_memory(rank, "after_init_dynamic_encoder", args.log_process_memory)
        stage_t0 = time.time()
        if args.reset_condition_fuser:
            model.condition_fuser.reset_parameters(
                gate_init=args.condition_fusion_gate_init,
                zero_init_update=args.condition_fusion == "static_cross_attn_zero",
            )
            log(
                rank,
                "reset condition_fuser "
                f"gate_init={args.condition_fusion_gate_init} "
                f"zero_update={args.condition_fusion == 'static_cross_attn_zero'}",
            )
        assert_module_on_device(model, device)

        train_dataset = DynamicRigManifestDataset(
            args.train_manifest,
            tokenizer,
            frame_count=args.frames,
            limit=args.limit_train,
            random_query=args.train_random_query,
            seed=args.seed,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
            target_active_skin_only=args.target_active_skin_only,
            active_skin_threshold=args.active_skin_threshold,
            target_start_policy=args.target_start_policy,
            target_root_policy=args.target_root_policy,
        )
        val_dataset = DynamicRigManifestDataset(
            args.val_manifest,
            tokenizer,
            frame_count=args.frames,
            limit=args.limit_val,
            random_query=False,
            seed=args.seed + 17,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
            target_active_skin_only=args.target_active_skin_only,
            active_skin_threshold=args.active_skin_threshold,
            target_start_policy=args.target_start_policy,
            target_root_policy=args.target_root_policy,
        )
        log(rank, f"datasets built in {time.time() - stage_t0:.2f}s")
        log_process_memory(rank, "after_datasets", args.log_process_memory)
        stage_t0 = time.time()
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        collate = partial(dynamic_rig_collate, pad_token=tokenizer.pad)
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
            num_workers=max(0, min(args.num_workers, 2)),
            pin_memory=True,
            collate_fn=collate,
        )
        log(rank, f"dataloaders built in {time.time() - stage_t0:.2f}s")
        log_process_memory(rank, "after_dataloaders", args.log_process_memory)
        stage_t0 = time.time()

        motion_params = [p for p in motion_encoder.parameters() if p.requires_grad]
        ar_params = [p for p in unirig.transformer.parameters() if p.requires_grad]
        surface_params = [p for p in surface_tokenizer.parameters() if p.requires_grad]
        aux_params = [
            p for module in (model.structure_count_head, model.structure_action_head)
            for p in module.parameters()
            if p.requires_grad
        ]
        fusion_params = [p for p in model.condition_fuser.parameters() if p.requires_grad]
        grammar_state_params = [p for p in model.grammar_state_proj.parameters() if p.requires_grad]
        action_group_bias_params = [p for p in model.action_group_bias_head.parameters() if p.requires_grad]
        condition_action_group_bias_params = [
            p for p in model.condition_action_group_bias_head.parameters()
            if p.requires_grad
        ]
        branch_prior_params = []
        if getattr(model, "branch_prior", None) is not None:
            branch_prior_params = [p for p in model.branch_prior.parameters() if p.requires_grad]
        explicit_tree_params = []
        if getattr(model, "explicit_tree_decoder", None) is not None:
            explicit_tree_params = [p for p in model.explicit_tree_decoder.parameters() if p.requires_grad]
        groups = []
        if motion_params:
            groups.append({"params": motion_params, "lr": args.lr_motion, "name": "motion"})
        if ar_params:
            groups.append({"params": ar_params, "lr": args.lr_ar, "name": "ar"})
        if surface_params:
            groups.append({"params": surface_params, "lr": args.lr_surface, "name": "surface"})
        if aux_params:
            groups.append({"params": aux_params, "lr": args.lr_ar, "name": "aux_heads"})
        if fusion_params:
            groups.append({"params": fusion_params, "lr": args.lr_motion, "name": "condition_fusion"})
        if grammar_state_params:
            groups.append({"params": grammar_state_params, "lr": args.lr_ar, "name": "grammar_state"})
        if action_group_bias_params:
            groups.append({"params": action_group_bias_params, "lr": args.lr_ar, "name": "action_group_bias"})
        if condition_action_group_bias_params:
            groups.append({
                "params": condition_action_group_bias_params,
                "lr": args.lr_ar,
                "name": "condition_action_group_bias",
            })
        if branch_prior_params:
            groups.append({"params": branch_prior_params, "lr": args.lr_motion, "name": "branch_prior"})
        if explicit_tree_params:
            groups.append({"params": explicit_tree_params, "lr": args.lr_ar, "name": "explicit_tree"})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        scheduler = None
        if args.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[group["lr"] for group in groups],
                total_steps=args.max_steps,
                pct_start=args.onecycle_pct_start,
                anneal_strategy="cos",
                div_factor=args.onecycle_div_factor,
                final_div_factor=args.onecycle_final_div_factor,
            )

        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        train_model = unwrap_model(model)
        log(rank, f"optimizer/ddp setup done in {time.time() - stage_t0:.2f}s total_setup={time.time() - setup_t0:.2f}s")
        log_process_memory(rank, "after_optimizer_ddp", args.log_process_memory)

        effective_batch = int(world_size * args.batch_size * args.grad_accum_steps)
        train_rows = int(len(train_dataset))
        sample_milestones = parse_sample_milestones(args.sample_milestones)
        setattr(args, "effective_batch", effective_batch)
        setattr(args, "train_rows", train_rows)
        setattr(args, "sample_milestones_parsed", sample_milestones)
        if is_main(rank):
            args_json = json.loads(json.dumps(vars(args), default=str))
            (args.output_dir / "args.json").write_text(json.dumps(args_json, indent=2) + "\n", encoding="utf-8")
            metrics_log = log_dir / "train.log"
            metrics_log.write_text("", encoding="utf-8")
            write_json_log(
                metrics_log,
                {
                    "event": "run_config",
                    "args": args_json,
                    "world_size": world_size,
                    "effective_batch": effective_batch,
                    "train_rows": train_rows,
                    "sample_milestones": sample_milestones,
                },
            )
        saved_sample_milestones: set[int] = set()
        log(rank, f"train rows={len(train_dataset)} val rows={len(val_dataset)}")
        log(rank, f"trainable params={count_trainable(model):,}")
        log(
            rank,
            "trainable groups="
            f"motion:{sum(p.numel() for p in motion_params):,} "
            f"ar:{sum(p.numel() for p in ar_params):,} "
            f"surface:{sum(p.numel() for p in surface_params):,} "
            f"aux_heads:{sum(p.numel() for p in aux_params):,} "
            f"condition_fusion:{sum(p.numel() for p in fusion_params):,} "
            f"grammar_state:{sum(p.numel() for p in grammar_state_params):,} "
            f"action_group_bias:{sum(p.numel() for p in action_group_bias_params):,} "
            f"condition_action_group_bias:{sum(p.numel() for p in condition_action_group_bias_params):,} "
            f"branch_prior:{sum(p.numel() for p in branch_prior_params):,} "
            f"explicit_tree:{sum(p.numel() for p in explicit_tree_params):,}",
        )
        log(
            rank,
            "clean_route_flags="
            f"use_motion_features={args.use_motion_features} "
            f"use_time_embedding={args.use_time_embedding}",
        )
        log(
            rank,
            "sampling_contract="
            f"frames={args.frames} motion_fps_ratio={args.motion_fps_ratio} "
            f"motion_vertex_samples={args.motion_vertex_samples} "
            f"train_random_query={args.train_random_query} "
            f"target_active_skin_only={args.target_active_skin_only} "
            f"target_start_policy={args.target_start_policy} "
            f"target_root_policy={args.target_root_policy} "
            f"active_skin_threshold={args.active_skin_threshold} "
            "query_frame=random rootless_npz=target_joints_rootspace "
            "input_space=frame_vertices_rootspace query_normalization=query_mesh",
        )
        log(
            rank,
            "architecture_contract="
            f"query_tokens={args.query_tokens} register_tokens={args.register_tokens} "
            f"motion_depth={args.motion_depth} motion_heads={args.motion_heads} "
            f"motion_checkpointing={args.motion_checkpointing} "
            f"surface_tokenizer={'trainable' if args.train_surface_tokenizer else 'frozen'} "
            f"conditioner={'frozen' if args.freeze_conditioner else 'trainable'} "
            f"ar_decoder={'frozen' if args.freeze_ar else 'trainable'} "
            f"condition_fusion={args.condition_fusion} "
            f"condition_fusion_gate_init={args.condition_fusion_gate_init} "
            f"condition_fusion_depth={args.condition_fusion_depth} "
            f"condition_static_blend_weight={args.condition_static_blend_weight} "
            f"use_grammar_state_embedding={args.use_grammar_state_embedding} "
            f"use_action_group_bias={args.use_action_group_bias} "
            f"use_condition_action_group_bias={args.use_condition_action_group_bias} "
            f"branch_prior_proposals={args.branch_prior_proposals} "
            f"branch_prior_heads={args.branch_prior_heads} "
            f"branch_prior={'frozen' if args.freeze_branch_prior else 'trainable'} "
            f"branch_prior_loss_weight={args.branch_prior_loss_weight} "
            f"branch_prior_coord_loss_weight={args.branch_prior_coord_loss_weight} "
            f"explicit_tree_loss_weight={args.explicit_tree_loss_weight} "
            f"explicit_tree_generated_prefix_weight={args.explicit_tree_generated_prefix_weight} "
            f"explicit_tree_generated_prefix_states={args.explicit_tree_generated_prefix_states} "
            f"explicit_tree_generated_prefix_max_steps={args.explicit_tree_generated_prefix_max_steps} "
            f"explicit_tree_generated_prefix_max_rows={args.explicit_tree_generated_prefix_max_rows} "
            f"explicit_tree_generated_prefix_every={args.explicit_tree_generated_prefix_every} "
            f"explicit_tree_oracle_prefix_weight={args.explicit_tree_oracle_prefix_weight} "
            f"explicit_tree_oracle_prefix_states={args.explicit_tree_oracle_prefix_states} "
            f"explicit_tree_oracle_prefix_max_steps={args.explicit_tree_oracle_prefix_max_steps} "
            f"explicit_tree_oracle_prefix_max_rows={args.explicit_tree_oracle_prefix_max_rows} "
            f"explicit_tree_prefix_jitter_weight={args.explicit_tree_prefix_jitter_weight} "
            f"explicit_tree_prefix_jitter_std={args.explicit_tree_prefix_jitter_std} "
            f"explicit_tree_depth={args.explicit_tree_depth} "
            f"explicit_tree_heads={args.explicit_tree_heads} "
            f"explicit_tree_topology_mode={args.explicit_tree_topology_mode} "
            f"explicit_tree_coordinate_mode={args.explicit_tree_coordinate_mode} "
            f"explicit_tree_action_weights=("
            f"{args.explicit_tree_action_eos_loss_weight},"
            f"{args.explicit_tree_action_child_loss_weight},"
            f"{args.explicit_tree_action_branch_loss_weight}) "
            f"explicit_tree_xyz_loss_weight={args.explicit_tree_xyz_loss_weight}",
        )
        log(
            rank,
            f"micro_batch_per_gpu={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
            f"effective_batch={effective_batch} train_rows={train_rows} "
            f"sample_milestones={sample_milestones}",
        )
        log(
            rank,
            "auxiliary_losses="
            f"eos_loss_weight={args.eos_loss_weight} "
            f"decision_loss_weight={args.decision_loss_weight} "
            f"loop_recovery_loss_weight={args.loop_recovery_loss_weight} "
            f"loop_recovery_repeats={args.loop_recovery_repeats} "
            f"prefix_decision_recovery_weight={args.prefix_decision_recovery_weight} "
            f"prefix_decision_recovery_states={args.prefix_decision_recovery_states} "
            f"prefix_decision_recovery_variants={args.prefix_decision_recovery_variants} "
            f"prefix_decision_recovery_jitter={args.prefix_decision_recovery_jitter} "
            f"prefix_token_recovery_weight={args.prefix_token_recovery_weight} "
            f"prefix_token_recovery_states={args.prefix_token_recovery_states} "
            f"prefix_token_recovery_variants={args.prefix_token_recovery_variants} "
            f"prefix_token_recovery_jitter={args.prefix_token_recovery_jitter} "
            f"prefix_token_recovery_max_rows={args.prefix_token_recovery_max_rows} "
            f"prefix_action_recovery_weight={args.prefix_action_recovery_weight} "
            f"prefix_action_recovery_states={args.prefix_action_recovery_states} "
            f"prefix_action_recovery_variants={args.prefix_action_recovery_variants} "
            f"prefix_action_recovery_jitter={args.prefix_action_recovery_jitter} "
            f"prefix_action_recovery_max_rows={args.prefix_action_recovery_max_rows} "
            f"generated_prefix_recovery_weight={args.generated_prefix_recovery_weight} "
            f"generated_prefix_recovery_states={args.generated_prefix_recovery_states} "
            f"generated_prefix_recovery_max_new_tokens={args.generated_prefix_recovery_max_new_tokens} "
            f"generated_prefix_recovery_max_rows={args.generated_prefix_recovery_max_rows} "
            f"generated_prefix_recovery_every={args.generated_prefix_recovery_every} "
            f"explicit_tree_generated_prefix_weight={args.explicit_tree_generated_prefix_weight} "
            f"explicit_tree_generated_prefix_every={args.explicit_tree_generated_prefix_every} "
            f"explicit_tree_oracle_prefix_weight={args.explicit_tree_oracle_prefix_weight} "
            f"structure_count_loss_weight={args.structure_count_loss_weight} "
            f"structure_action_loss_weight={args.structure_action_loss_weight} "
            f"condition_control_ce_weight={args.condition_control_ce_weight} "
            f"condition_control_ce_controls={args.condition_control_ce_controls} "
            f"condition_control_ce_every={args.condition_control_ce_every}",
        )

        step = 0
        epoch = 0
        best_val_ce = float("inf")
        best_val_step = -1
        best_val_eos_acc = -float("inf")
        best_val_eos_step = -1
        accum_count = 0
        accum_t0 = time.time()
        accum_loss_sum = 0.0
        accum_ce_sum = 0.0
        accum_dis_sum = 0.0
        accum_aux_sums: dict[str, float] = {}
        accum_path = ""
        optimizer.zero_grad(set_to_none=True)
        model.train()
        base_generated_prefix_weight = float(getattr(train_model, "generated_prefix_recovery_weight", 0.0))
        base_explicit_tree_generated_prefix_weight = float(
            getattr(train_model, "explicit_tree_generated_prefix_weight", 0.0)
        )
        base_condition_control_ce_weight = float(getattr(train_model, "condition_control_ce_weight", 0.0))
        first_batch_mem_logged = False
        if not args.train_surface_tokenizer:
            surface_tokenizer.eval()
        while step < args.max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if accum_count == 0:
                    accum_t0 = time.time()
                    if args.generated_prefix_recovery_every <= 0:
                        train_model.generated_prefix_recovery_weight = 0.0
                    elif base_generated_prefix_weight > 0.0 and (step + 1) % args.generated_prefix_recovery_every == 0:
                        train_model.generated_prefix_recovery_weight = base_generated_prefix_weight
                    else:
                        train_model.generated_prefix_recovery_weight = 0.0
                    if args.explicit_tree_generated_prefix_every <= 0:
                        train_model.explicit_tree_generated_prefix_weight = 0.0
                    elif (
                        base_explicit_tree_generated_prefix_weight > 0.0
                        and (step + 1) % args.explicit_tree_generated_prefix_every == 0
                    ):
                        train_model.explicit_tree_generated_prefix_weight = base_explicit_tree_generated_prefix_weight
                    else:
                        train_model.explicit_tree_generated_prefix_weight = 0.0
                    if args.condition_control_ce_every <= 0:
                        train_model.condition_control_ce_weight = 0.0
                    elif base_condition_control_ce_weight > 0.0 and (step + 1) % args.condition_control_ce_every == 0:
                        train_model.condition_control_ce_weight = base_condition_control_ce_weight
                    else:
                        train_model.condition_control_ce_weight = 0.0
                batch = move_batch(batch, device)
                if not first_batch_mem_logged:
                    log_process_memory(rank, "after_first_batch_to_device", args.log_process_memory)
                    first_batch_mem_logged = True
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    out = model(batch)
                    raw_loss = out["loss"]
                    loss = raw_loss / max(1, args.grad_accum_steps)
                loss.backward()
                accum_loss_sum += float(raw_loss.detach().cpu())
                accum_ce_sum += float(out["ce_loss"].detach().cpu())
                accum_dis_sum += float(out["dis_loss"].detach().cpu())
                for key, value in out.items():
                    if key in {"loss", "ce_loss", "dis_loss"}:
                        continue
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        accum_aux_sums[key] = accum_aux_sums.get(key, 0.0) + float(value.detach().cpu())
                accum_path = batch["path"][0]

                accum_count += 1
                if accum_count < max(1, args.grad_accum_steps):
                    continue

                step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                micro_count = max(1, accum_count)
                accum_count = 0

                if is_main(rank) and (step == 1 or step % args.log_every == 0):
                    accounting = accounting_fields(step, effective_batch, train_rows)
                    if device.type == "cuda":
                        mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)
                        mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                        mem_peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                    else:
                        mem_alloc = 0.0
                        mem_reserved = 0.0
                        mem_peak = 0.0
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "loss": accum_loss_sum / micro_count,
                        "ce": accum_ce_sum / micro_count,
                        "dis": accum_dis_sum / micro_count,
                        "seconds": round(time.time() - accum_t0, 3),
                        "gpu_gb": round(mem_peak, 3),
                        "gpu_alloc_gb": round(mem_alloc, 3),
                        "gpu_reserved_gb": round(mem_reserved, 3),
                        "gpu_peak_gb": round(mem_peak, 3),
                        "path": accum_path,
                        "grad_accum": args.grad_accum_steps,
                        "micro_batch_per_gpu": args.batch_size,
                        **accounting,
                        "generated_prefix_recovery_active": bool(
                            getattr(train_model, "generated_prefix_recovery_weight", 0.0) > 0.0
                        ),
                        "generated_prefix_recovery_module_weight": float(
                            getattr(train_model, "generated_prefix_recovery_weight", 0.0)
                        ),
                        "generated_prefix_recovery_every": args.generated_prefix_recovery_every,
                        "explicit_tree_generated_prefix_active": bool(
                            getattr(train_model, "explicit_tree_generated_prefix_weight", 0.0) > 0.0
                        ),
                        "explicit_tree_generated_prefix_module_weight": float(
                            getattr(train_model, "explicit_tree_generated_prefix_weight", 0.0)
                        ),
                        "explicit_tree_generated_prefix_every": args.explicit_tree_generated_prefix_every,
                        "explicit_tree_oracle_prefix_active": bool(
                            getattr(train_model, "explicit_tree_oracle_prefix_weight", 0.0) > 0.0
                        ),
                        "explicit_tree_oracle_prefix_module_weight": float(
                            getattr(train_model, "explicit_tree_oracle_prefix_weight", 0.0)
                        ),
                        "condition_control_ce_active": bool(
                            getattr(train_model, "condition_control_ce_weight", 0.0) > 0.0
                        ),
                        "condition_control_ce_module_weight": float(
                            getattr(train_model, "condition_control_ce_weight", 0.0)
                        ),
                        "condition_control_ce_every": args.condition_control_ce_every,
                        "lrs": {
                            group.get("name", str(i)): group["lr"]
                            for i, group in enumerate(optimizer.param_groups)
                        },
                    }
                    for key, value_sum in accum_aux_sums.items():
                        if key.endswith("_loss"):
                            row[key.removesuffix("_loss")] = value_sum / micro_count
                        elif key.endswith("_acc"):
                            row[key] = value_sum / micro_count
                        elif key.endswith("_count"):
                            row[key] = value_sum
                        else:
                            row[key] = value_sum / micro_count
                    write_json_log(metrics_log, row)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                accum_loss_sum = 0.0
                accum_ce_sum = 0.0
                accum_dis_sum = 0.0
                accum_aux_sums = {}
                accum_path = ""

                if is_main(rank):
                    sample_seen = int(step * effective_batch)
                    for milestone in sample_milestones:
                        if milestone in saved_sample_milestones or sample_seen < milestone:
                            continue
                        save_path = args.output_dir / f"checkpoint_sample_{milestone}.pt"
                        save_checkpoint(
                            save_path,
                            model,
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
                                "step": step,
                                **accounting_fields(step, effective_batch, train_rows),
                                "event": "sample_milestone_checkpoint",
                                "sample_milestone": milestone,
                                "path": str(save_path),
                                "include_optimizer": not args.no_save_optimizer,
                            },
                        )

                if is_main(rank) and args.save_every > 0 and step % args.save_every == 0:
                    save_path = args.output_dir / f"checkpoint_step_{step}.pt"
                    save_checkpoint(
                        save_path,
                        model,
                        optimizer,
                        step,
                        args,
                        scheduler,
                        include_optimizer=not args.no_save_optimizer,
                    )
                    write_json_log(
                        metrics_log,
                        {
                            "step": step,
                            **accounting_fields(step, effective_batch, train_rows),
                            "event": "checkpoint_saved",
                            "path": str(save_path),
                            "include_optimizer": not args.no_save_optimizer,
                        },
                    )

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                trim_host_allocator()

                if args.val_every > 0 and step % args.val_every == 0:
                    metrics = evaluate(model, val_loader, device, args.val_steps, amp_dtype)
                    if not args.train_surface_tokenizer:
                        surface_tokenizer.eval()
                    if is_main(rank):
                        write_json_log(
                            metrics_log,
                            {"step": step, **accounting_fields(step, effective_batch, train_rows), **metrics},
                        )
                        val_ce = float(metrics.get("val_ce", float("inf")))
                        if val_ce < best_val_ce:
                            best_val_ce = val_ce
                            best_val_step = step
                            save_checkpoint(
                                args.output_dir / "checkpoint_best_val.pt",
                                model,
                                optimizer,
                                step,
                                args,
                                scheduler,
                                include_optimizer=not args.no_save_optimizer,
                            )
                            save_checkpoint(
                                args.output_dir / f"checkpoint_best_val_step_{step}.pt",
                                model,
                                optimizer,
                                step,
                                args,
                                scheduler,
                                include_optimizer=not args.no_save_optimizer,
                            )
                            write_json_log(
                                metrics_log,
                                {
                                    "step": step,
                                    **accounting_fields(step, effective_batch, train_rows),
                                    "best_val_ce": best_val_ce,
                                    "best_val_step": best_val_step,
                                    "event": "best_val_checkpoint",
                                },
                            )
                        val_eos_acc = float(metrics.get("val_eos_acc", -float("inf")))
                        if val_eos_acc > best_val_eos_acc:
                            best_val_eos_acc = val_eos_acc
                            best_val_eos_step = step
                            save_checkpoint(
                                args.output_dir / "checkpoint_best_eos.pt",
                                model,
                                optimizer,
                                step,
                                args,
                                scheduler,
                                include_optimizer=not args.no_save_optimizer,
                            )
                            save_checkpoint(
                                args.output_dir / f"checkpoint_best_eos_step_{step}.pt",
                                model,
                                optimizer,
                                step,
                                args,
                                scheduler,
                                include_optimizer=not args.no_save_optimizer,
                            )
                            write_json_log(
                                metrics_log,
                                {
                                    "step": step,
                                    **accounting_fields(step, effective_batch, train_rows),
                                    "best_val_eos_acc": best_val_eos_acc,
                                    "best_val_eos_step": best_val_eos_step,
                                    "event": "best_eos_checkpoint",
                                },
                            )

                if step >= args.max_steps:
                    break
            epoch += 1

        if is_main(rank):
            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model,
                optimizer,
                step,
                args,
                scheduler,
                include_optimizer=not args.no_save_optimizer,
            )
        log(rank, "done")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
