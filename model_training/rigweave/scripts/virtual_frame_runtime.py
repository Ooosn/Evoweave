"""Initialization, exposure accounting and runtime shared by virtual-frame runs."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import itertools
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest, VariableFrameBatchSampler
from rigweave.dynamic_rig.surface_tokenizer import FixedQuerySurfaceTokenizer
from rigweave.dynamic_rig.virtual_frame_rig import VirtualFrameRig, VirtualFrameRigConfig


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_virtual_model(config: dict[str, Any], candidate: str, device: torch.device) -> tuple[VirtualFrameRig, dict[str, Any]]:
    initial = config["initialization"]
    upstream_root = config["runtime"]["unirig_root"]
    revision = subprocess.check_output(["git", "-C", upstream_root, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", upstream_root, "status", "--porcelain"], text=True).strip()
    if revision != initial["upstream_source_commit"] or status:
        raise ValueError("upstream implementation must be the pinned clean revision")
    checkpoint = Path(initial["checkpoint"]).resolve()
    if checkpoint != Path(initial["official_checkpoint"]).resolve():
        raise ValueError("new virtual-frame runs may initialize only from the declared official upstream checkpoint")
    if file_sha256(checkpoint) != initial["sha256"]:
        raise ValueError("upstream initialization checkpoint hash changed")
    random.seed(initial["seed"])
    np.random.seed(initial["seed"] % 2**32)
    torch.manual_seed(initial["seed"])
    import yaml
    from transformers import OPTConfig, OPTModel
    from src.model.michelangelo.get_model import get_encoder_simplified

    upstream_config = yaml.safe_load(Path(initial["model_config"]).read_text())
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()}
    llm_args = dict(upstream_config["llm"])
    llm_args.pop("pretrained_model_name_or_path")
    llm_args["vocab_size"] = state["transformer.model.decoder.embed_tokens.weight"].shape[0]
    llm_config = OPTConfig.from_pretrained(config["runtime"]["opt_config_root"], local_files_only=True, **llm_args)
    llm_config.torch_dtype = torch.float32
    llm_config.pre_norm = True
    backbone = OPTModel(llm_config)
    encoder_args = dict(upstream_config["mesh_encoder"])
    if encoder_args.pop("__target__") != "michelangelo_encoder":
        raise ValueError("the matched upstream recipe requires Michelangelo surface encoding")
    mesh_encoder = get_encoder_simplified(**encoder_args)
    output_projection = nn.Linear(mesh_encoder.width, llm_config.hidden_size)
    loaded_keys = set()
    load_report = {}
    for prefix, module in (("transformer.model.", backbone), ("mesh_encoder.", mesh_encoder), ("output_proj.", output_projection)):
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(selected, strict=True)
        loaded_keys.update(key for key in state if key.startswith(prefix))
        load_report[prefix] = {"strict": True, "tensors": len(selected), "parameters": sum(value.numel() for value in selected.values())}
    load_report["unused_upstream_keys"] = sorted(set(state) - loaded_keys)
    del payload, state, selected
    backbone.requires_grad_(True)
    backbone.get_input_embeddings().requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
        for name in ("dropout", "attention_dropout", "activation_dropout", "layerdrop"):
            if isinstance(getattr(module, name, None), (int, float)):
                setattr(module, name, 0.)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    architecture = VirtualFrameRigConfig(**{**config["architecture"], **config["candidates"][candidate]})
    if architecture.dim != backbone.get_input_embeddings().embedding_dim:
        raise ValueError("official backbone width differs from the virtual-frame model")
    # All comparisons first construct the same largest head. Removing optional
    # layers preserves identical random initial weights for every shared key.
    init_depth = initial["position_depth"]
    if architecture.position_depth > init_depth:
        raise ValueError("candidate exceeds the common initialization head depth")
    init_config = replace(architecture, position_depth=init_depth, position_self_attention=True)
    torch.manual_seed(initial["seed"])
    model = VirtualFrameRig(FixedQuerySurfaceTokenizer(mesh_encoder, output_projection), backbone, init_config)
    model.config = architecture
    model.positions.config = architecture
    model.positions.layers = nn.ModuleList(list(model.positions.layers)[:architecture.position_depth])
    if not architecture.position_self_attention:
        for layer in model.positions.layers:
            layer.self_attention = None
            layer.self_norm = None
    model.to(device)
    report = {
        "official_checkpoint": str(checkpoint), "sha256": initial["sha256"],
        "seed": initial["seed"], "common_position_initialization_depth": init_depth,
        "upstream_source_commit": revision, "upstream_source_clean": True,
        "model_config_sha256": file_sha256(initial["model_config"]),
        "load_report": load_report, "architecture": asdict(architecture),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "groups": {name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
                   for name in ("surface_tokenizer", "motion_encoder", "structure", "positions")},
        "backbone_attention": getattr(backbone.config, "_attn_implementation", None),
        "old_method_trained_weights_loaded": False,
    }
    return model, report


class FrameBudgetStream:
    """Replay the shared T/B schedule, including an exactly truncated final batch."""

    def __init__(self, dataset, *, rank: int, world_size: int, seed: int, frame_budget: int,
                 batch_cap: int, min_frames: int, max_frames: int, state: dict[str, Any] | None = None):
        self.base = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0)
        self.sampler = VariableFrameBatchSampler(
            self.base, batch_size=batch_cap, min_frames=min_frames, max_frames=max_frames, seed=seed,
            frame_budget=frame_budget, max_batch_size=batch_cap, drop_last=False,
        )
        self.contract = {"rank": rank, "world_size": world_size, "seed": seed, "frame_budget": frame_budget,
                         "batch_cap": batch_cap, "min_frames": min_frames, "max_frames": max_frames,
                         "dataset_rows": len(dataset)}
        if state is not None and state["contract"] != self.contract:
            raise ValueError("resume sampler configuration or rank differs")
        self.epoch = int(state["epoch"]) if state else 0
        self.next_batch = int(state["next_batch"]) if state else 0
        self.pending = [FrameSampleRequest(**x) for x in state.get("pending", [])] if state else []
        self._iterator = self._replay()

    def _replay(self):
        self.base.set_epoch(self.epoch)
        self.sampler.set_epoch(self.epoch)
        return itertools.islice(iter(self.sampler), self.next_batch, None)

    def take(self, max_assets: int | None = None) -> list[FrameSampleRequest]:
        if max_assets is not None and max_assets < 1:
            raise ValueError("a sampler request must permit a positive number of assets")
        if self.pending:
            batch, self.pending = self.pending, []
        else:
            try:
                batch = next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self.next_batch = 0
                self._iterator = self._replay()
                batch = next(self._iterator)
            self.next_batch += 1
        if max_assets is not None and len(batch) > max_assets:
            batch, self.pending = batch[:max_assets], batch[max_assets:]
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {"contract": self.contract, "epoch": self.epoch, "next_batch": self.next_batch,
                "pending": [x._asdict() for x in self.pending]}


def exposure_learning_rate(samples: int, total: int, *, peak: float = 1e-4, warmup_fraction: float = .1,
                           div_factor: float = 5., final_div_factor: float = 10.) -> tuple[float, float]:
    if total < 1 or not 0 <= samples <= total or not 0 < warmup_fraction < 1:
        raise ValueError("invalid exposure schedule")
    progress = samples / total
    initial, final = peak / div_factor, peak / div_factor / final_div_factor
    if progress <= warmup_fraction:
        fraction = .5 * (1 - math.cos(math.pi * progress / warmup_fraction))
        return initial + fraction * (peak - initial), .95 - .10 * fraction
    fraction = .5 * (1 - math.cos(math.pi * (progress - warmup_fraction) / (1 - warmup_fraction)))
    return peak + fraction * (final - peak), .85 + .10 * fraction


def set_exposure_lr(optimizer, samples: int, config: dict[str, Any]) -> tuple[float, float]:
    plan = config["optimization"]
    lr, beta1 = exposure_learning_rate(
        samples, plan["schedule_samples"], peak=plan["peak_lr"], warmup_fraction=plan["warmup_fraction"],
        div_factor=plan["div_factor"], final_div_factor=plan["final_div_factor"],
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["betas"] = (beta1, group["betas"][1])
    return lr, beta1


def normalize_asset_gradients(parameters, *, global_assets: int, world_size: int) -> None:
    if global_assets < 1 or world_size < 1:
        raise ValueError("gradient normalization needs positive actual counts")
    multiplier = world_size / global_assets
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(multiplier)


def finite_gradient_groups(model: VirtualFrameRig) -> dict[str, Any]:
    report = {}
    for name in ("surface_tokenizer", "motion_encoder", "structure", "positions"):
        parameters = [(key, p) for key, p in getattr(model, name).named_parameters() if p.requires_grad]
        present = [(key, p.grad) for key, p in parameters if p.grad is not None]
        if not present or any(not bool(torch.isfinite(g).all()) for _, g in present):
            raise RuntimeError(f"missing or nonfinite gradients in {name}")
        norm = sum(g.detach().float().square().sum() for _, g in present).sqrt().item()
        if not norm > 0:
            raise RuntimeError(f"zero gradient group: {name}")
        report[name] = {"norm": norm, "with_gradient": len(present), "trainable_tensors": len(parameters),
                        "unused": [key for key, p in parameters if p.grad is None]}
    return report


def virtual_runtime_environment(config: dict[str, Any], devices: list[int]) -> dict[str, str]:
    runtime = config["runtime"]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("EVOWEAVE_", "RIGWEAVE_"))}
    environment.update({
        "PATH": runtime["conda_bin"] + os.pathsep + environment.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join((str(Path(runtime["repo"]) / "model_training/rigweave/src"),
                                      runtime["unirig_root"], runtime["extra_pythonpath"])),
        "EVOWEAVE_UNIRIG_ROOT": runtime["unirig_root"], "EVOWEAVE_OPT_CONFIG_ROOT": runtime["opt_config_root"],
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)), "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "RIGWEAVE_DISABLE_OPEN3D": "1", "RIGWEAVE_DISABLE_LIGHTNING_IMPORT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_MODULE_LOADING": "LAZY",
    })
    return environment
