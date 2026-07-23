#!/usr/bin/env python3
"""Bounded adapter fit for aligned-versus-corrupted motion-evidence testing."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_ROOT = SCRIPT_DIR.parent
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
if str(RIGWEAVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from eval_dynamic_rig_ce import (  # noqa: E402
    CHECKPOINT_DEFAULTS,
    _build_dynamic_model,
    apply_checkpoint_eval_defaults,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402

from rigweave.dynamic_rig.data import (  # noqa: E402
    DynamicRigManifestDataset,
    dynamic_rig_collate,
)
from rigweave.motion_evidence import TopologyMotionEvidenceUniRigAR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--valid-limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--static-prefix-steps", type=int, default=4)
    args = parser.parse_args()
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def build_dataset(
    args: argparse.Namespace,
    manifest: Path,
    tokenizer: Any,
    *,
    limit: int,
    seed: int,
) -> DynamicRigManifestDataset:
    return DynamicRigManifestDataset(
        manifest,
        tokenizer,
        frame_count=args.frames,
        limit=limit,
        random_query=False,
        seed=seed,
        motion_fps_ratio=args.motion_fps_ratio,
        motion_vertex_samples=args.motion_vertex_samples,
        motion_alignment_policy=args.motion_alignment_policy,
        target_active_skin_only=args.target_active_skin_only,
        active_skin_threshold=args.active_skin_threshold,
        target_start_policy=args.target_start_policy,
        target_root_policy=args.target_root_policy,
        input_space_policy=args.input_space_policy,
    )


def make_loader(
    dataset: DynamicRigManifestDataset,
    tokenizer: Any,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )


def shifted_region_losses(
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    *,
    static_prefix_steps: int,
) -> dict[str, tuple[torch.Tensor, int]]:
    shifted_logits = logits[:, :-1]
    labels = input_ids[:, 1:].clone()
    valid = attention_mask[:, 1:] != 0
    labels[~valid] = -100
    losses = nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    positions = torch.arange(losses.shape[1], device=losses.device)[None]
    root_mask = valid & (positions < static_prefix_steps)
    later_mask = valid & (positions >= static_prefix_steps)

    def reduce(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        count = int(mask.sum().item())
        if count == 0:
            return losses.new_zeros(()), 0
        return losses[mask].mean(), count

    return {
        "all": reduce(valid),
        "root": reduce(root_mask),
        "later": reduce(later_mask),
    }


def set_probe_modes(model: TopologyMotionEvidenceUniRigAR, *, training: bool) -> None:
    model.transformer.eval()
    model.conditioner.surface_tokenizer.eval()
    model.conditioner.value_encoder.train(training)
    model.evidence_adapter.train(training)


@torch.no_grad()
def evaluate_controls(
    model: TopologyMotionEvidenceUniRigAR,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    seed: int,
) -> dict[str, Any]:
    set_probe_modes(model, training=False)
    controls = ("normal", "zero", "corrupt")
    regions = ("all", "root", "later")
    weighted_loss = {
        control: {region: 0.0 for region in regions} for control in controls
    }
    token_count = {region: 0 for region in regions}
    rows: list[dict[str, Any]] = []
    query_tokens_all_1024 = True
    root_logit_max_abs_diff = 0.0

    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        torch.manual_seed(seed + row_index)
        torch.cuda.manual_seed_all(seed + row_index)
        refs = model.sample_references(batch)
        generator = torch.Generator(device=device).manual_seed(seed + 100000 + row_index)
        with torch.autocast("cuda", dtype=amp_dtype):
            memory = model.build_memory(batch, refs=refs)
            normal = model.teacher_forcing(batch, memory=memory)
            corrupt_memory = memory.controlled(
                "corrupt_correspondence",
                generator=generator,
            )
            positions = torch.arange(
                normal.token_hidden.shape[1],
                device=normal.token_hidden.device,
            )
            corrupt_logits, _ = model.evidence_adapter.logits_from_hidden(
                model.transformer,
                normal.token_hidden,
                corrupt_memory,
                positions,
            )
            static_steps = min(model.evidence_adapter.static_prefix_steps, corrupt_logits.shape[1])
            if static_steps:
                corrupt_logits = torch.cat(
                    (
                        normal.baseline_logits[:, :static_steps],
                        corrupt_logits[:, static_steps:],
                    ),
                    dim=1,
                )
            logits_by_control = {
                "normal": normal.logits,
                "zero": normal.baseline_logits,
                "corrupt": corrupt_logits,
            }
            row_losses = {
                control: shifted_region_losses(
                    logits,
                    batch["input_ids"],
                    batch["attention_mask"],
                    static_prefix_steps=model.evidence_adapter.static_prefix_steps,
                )
                for control, logits in logits_by_control.items()
            }

        row_values: dict[str, dict[str, float]] = {}
        for control in controls:
            row_values[control] = {}
            for region in regions:
                loss, count = row_losses[control][region]
                row_values[control][region] = float(loss)
                weighted_loss[control][region] += float(loss) * count
                if control == "normal":
                    token_count[region] += count
        root_steps = min(model.evidence_adapter.static_prefix_steps, normal.logits.shape[1])
        if root_steps:
            root_logit_max_abs_diff = max(
                root_logit_max_abs_diff,
                float(
                    (
                        normal.logits[:, :root_steps].float()
                        - normal.baseline_logits[:, :root_steps].float()
                    )
                    .abs()
                    .max()
                ),
            )
        query_tokens_all_1024 = query_tokens_all_1024 and memory.static_tokens.shape[1] == 1024
        rows.append(
            {
                "index": row_index,
                "path": raw_batch["path"][0],
                "joint_count": int(raw_batch["joint_count"][0]),
                "confidence": float(memory.confidence[0]),
                "motion_q90_rms": float(memory.raw_evidence.example_motion_q90_rms[0]),
                "loss": row_values,
                "later_corrupt_minus_normal": (
                    row_values["corrupt"]["later"] - row_values["normal"]["later"]
                ),
                "later_zero_minus_normal": (
                    row_values["zero"]["later"] - row_values["normal"]["later"]
                ),
            }
        )

    mean_loss = {
        control: {
            region: weighted_loss[control][region] / max(token_count[region], 1)
            for region in regions
        }
        for control in controls
    }
    later_corrupt_gaps = [row["later_corrupt_minus_normal"] for row in rows]
    later_zero_gaps = [row["later_zero_minus_normal"] for row in rows]
    high_conf_rows = [row for row in rows if row["confidence"] >= 0.5]
    return {
        "rows": len(rows),
        "mean_token_ce": mean_loss,
        "later_corrupt_minus_normal": mean_loss["corrupt"]["later"] - mean_loss["normal"]["later"],
        "later_zero_minus_normal": mean_loss["zero"]["later"] - mean_loss["normal"]["later"],
        "later_normal_win_rate_vs_corrupt": (
            sum(gap > 0.0 for gap in later_corrupt_gaps) / max(len(later_corrupt_gaps), 1)
        ),
        "later_normal_win_rate_vs_zero": (
            sum(gap > 0.0 for gap in later_zero_gaps) / max(len(later_zero_gaps), 1)
        ),
        "high_confidence_rows": len(high_conf_rows),
        "high_confidence_later_corrupt_minus_normal": (
            sum(row["later_corrupt_minus_normal"] for row in high_conf_rows)
            / max(len(high_conf_rows), 1)
        ),
        "root_logit_max_abs_diff": root_logit_max_abs_diff,
        "query_tokens_all_1024": query_tokens_all_1024,
        "details": rows,
    }


def main() -> None:
    args = parse_args()
    if args.train_limit <= 0 or args.valid_limit <= 0 or args.steps <= 0:
        raise ValueError("train-limit, valid-limit, and steps must be positive")
    if args.static_prefix_steps != 4:
        raise ValueError("flat UniRig contract requires four static class/root prediction steps")
    apply_checkpoint_eval_defaults(args)
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = build_tokenizer(args.tokenizer_config)
    train_dataset = build_dataset(
        args,
        args.train_manifest,
        tokenizer,
        limit=args.train_limit,
        seed=args.seed,
    )
    valid_dataset = build_dataset(
        args,
        args.valid_manifest,
        tokenizer,
        limit=args.valid_limit,
        seed=args.seed,
    )
    train_loader = make_loader(train_dataset, tokenizer, shuffle=True, seed=args.seed)
    valid_loader = make_loader(valid_dataset, tokenizer, shuffle=False, seed=args.seed)

    baseline = _build_dynamic_model(args, tokenizer, device)
    model = TopologyMotionEvidenceUniRigAR(
        baseline.unirig_ar,
        baseline.conditioner.surface_tokenizer,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        evidence_heads=8,
        evidence_static_prefix_steps=args.static_prefix_steps,
    )
    model.conditioner.value_encoder.to(device)
    model.evidence_adapter.to(device)
    del baseline
    gc.collect()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.conditioner.value_encoder, model.evidence_adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    before = evaluate_controls(
        model,
        valid_loader,
        device,
        amp_dtype,
        seed=args.seed + 200000,
    )
    set_probe_modes(model, training=True)
    iterator = iter(train_loader)
    train_log: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            teacher = model.teacher_forcing(batch)
            regions = shifted_region_losses(
                teacher.logits,
                batch["input_ids"],
                batch["attention_mask"],
                static_prefix_steps=model.evidence_adapter.static_prefix_steps,
            )
            loss = regions["later"][0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "later_ce": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "gate": float(torch.tanh(model.evidence_adapter.attention.gate.detach())),
            }
            train_log.append(record)
            print(json.dumps(record), flush=True)

    after = evaluate_controls(
        model,
        valid_loader,
        device,
        amp_dtype,
        seed=args.seed + 200000,
    )
    elapsed = time.perf_counter() - started
    backbone_gradients = sum(
        parameter.grad is not None for parameter in model.transformer.parameters()
    )
    result = {
        "scope": "development learnability probe; backbone and surface tokenizer frozen",
        "checkpoint": str(args.checkpoint),
        "train_manifest": str(args.train_manifest),
        "valid_manifest": str(args.valid_manifest),
        "train_assets": len(train_dataset),
        "valid_assets": len(valid_dataset),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "static_prefix_steps": args.static_prefix_steps,
        "before": before,
        "after": after,
        "train_log": train_log,
        "backbone_parameters_with_gradients": backbone_gradients,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "elapsed_seconds": elapsed,
        "acceptance": {
            "root_exact": after["root_logit_max_abs_diff"] == 0.0,
            "correct_beats_corrupt_later": after["later_corrupt_minus_normal"] > 0.0,
            "correct_beats_zero_later": after["later_zero_minus_normal"] > 0.0,
            "majority_correct_beats_corrupt": after["later_normal_win_rate_vs_corrupt"] > 0.5,
            "backbone_frozen": backbone_gradients == 0,
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "value_encoder": model.conditioner.value_encoder.state_dict(),
            "evidence_adapter": model.evidence_adapter.state_dict(),
            "probe": {
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "static_prefix_steps": args.static_prefix_steps,
            },
        },
        args.output_dir / "adapter_probe.pt",
    )
    print(json.dumps({key: value for key, value in result.items() if key not in {"before", "after"}}, indent=2))
    print(json.dumps({"before": before, "after": after}, indent=2))


if __name__ == "__main__":
    main()
