#!/usr/bin/env python3
"""Fit and test prefix-aware motion evidence on the real flat-UniRig baseline.

This development preflight keeps the skeleton transformer and the learned
motion-value path frozen.  It trains only the prefix-to-surface support/null
head, then compares aligned, zero, and correspondence-corrupted motion at
continue, branch, and EOS decisions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_ROOT = SCRIPT_DIR.parent
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
if str(RIGWEAVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from analyze_prefix_motion_coverage import parse_prefix_decisions  # noqa: E402
from eval_dynamic_rig_ce import (  # noqa: E402
    CHECKPOINT_DEFAULTS,
    _build_dynamic_model,
    apply_checkpoint_eval_defaults,
)
from preflight_motion_evidence_decoder import _generation_consistency  # noqa: E402
from preflight_motion_evidence_learnability import (  # noqa: E402
    build_dataset,
    make_loader,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402

from rigweave.motion_evidence import (  # noqa: E402
    CoverageAwareTopologyMotionEvidenceUniRigAR,
    PrefixSupportTargets,
    prefix_support_distribution_loss,
)


CONTROLS = ("normal", "zero", "corrupt")
REGIONS = ("all", "later", "continue", "branch", "eos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-probe", type=Path, required=True)
    parser.add_argument("--support-head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--valid-limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--coverage-bias-strength", type=float, default=0.5)
    parser.add_argument("--evidence-residual-scale", type=float, default=0.1)
    parser.add_argument("--static-prefix-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--consistency-rows", type=int, default=2)
    parser.add_argument("--prefix-lengths", default="4,8,16")
    args = parser.parse_args()
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def _load_probe_initialization(
    model: CoverageAwareTopologyMotionEvidenceUniRigAR,
    evidence_path: Path,
    support_path: Path,
) -> dict[str, Any]:
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    if not {"value_encoder", "evidence_adapter", "probe"} <= set(evidence):
        raise KeyError("evidence probe lacks value_encoder, evidence_adapter, or probe")
    probe = evidence["probe"]
    expected_scale = float(model.evidence_adapter.attention.residual_scale)
    if not math.isclose(
        float(probe["evidence_residual_scale"]),
        expected_scale,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("evidence probe residual scale does not match this model")
    if int(probe["static_prefix_steps"]) != model.evidence_adapter.static_prefix_steps:
        raise ValueError("evidence probe static-prefix contract does not match")
    model.conditioner.value_encoder.load_state_dict(evidence["value_encoder"], strict=True)

    legacy = evidence["evidence_adapter"]
    required_legacy = {
        "attention.query_norm.weight",
        "attention.query_norm.bias",
        "attention.key_norm.weight",
        "attention.key_norm.bias",
        "attention.cross_attention.in_proj_weight",
        "attention.cross_attention.out_proj.weight",
    }
    if set(legacy) != required_legacy:
        missing = sorted(required_legacy - set(legacy))
        extra = sorted(set(legacy) - required_legacy)
        raise KeyError(f"unexpected evidence-adapter state: missing={missing}, extra={extra}")
    attention = model.evidence_adapter.attention
    qkv = legacy["attention.cross_attention.in_proj_weight"]
    if qkv.shape != (3 * attention.hidden_size, attention.hidden_size):
        raise ValueError("legacy evidence attention has an incompatible hidden width")
    query, key, value = qkv.chunk(3, dim=0)
    with torch.no_grad():
        attention.query_norm.weight.copy_(legacy["attention.query_norm.weight"])
        attention.query_norm.bias.copy_(legacy["attention.query_norm.bias"])
        attention.key_norm.weight.copy_(legacy["attention.key_norm.weight"])
        attention.key_norm.bias.copy_(legacy["attention.key_norm.bias"])
        attention.query_projection.weight.copy_(query)
        attention.key_projection.weight.copy_(key)
        attention.value_projection.weight.copy_(value)
        attention.output_projection.weight.copy_(
            legacy["attention.cross_attention.out_proj.weight"]
        )

    support = torch.load(support_path, map_location="cpu", weights_only=False)
    if not {"head", "probe"} <= set(support):
        raise KeyError("support checkpoint lacks head or probe metadata")
    support_state = support["head"]
    required_support = {
        "logit_bias",
        "query_norm.weight",
        "query_norm.bias",
        "key_norm.weight",
        "key_norm.bias",
        "query_projection.weight",
        "key_projection.weight",
    }
    if set(support_state) != required_support:
        missing = sorted(required_support - set(support_state))
        extra = sorted(set(support_state) - required_support)
        raise KeyError(f"unexpected support-head state: missing={missing}, extra={extra}")
    head = attention.support_head
    if support_state["query_projection.weight"].shape != head.query_projection.weight.shape:
        raise ValueError("support checkpoint projection size does not match")
    with torch.no_grad():
        head.query_norm.weight.copy_(support_state["query_norm.weight"])
        head.query_norm.bias.copy_(support_state["query_norm.bias"])
        head.key_norm.weight.copy_(support_state["key_norm.weight"])
        head.key_norm.bias.copy_(support_state["key_norm.bias"])
        head.query_projection.weight.copy_(support_state["query_projection.weight"])
        head.key_projection.weight.copy_(support_state["key_projection.weight"])
        head.null_projection.weight.zero_()
        # The old binary head added one bias to every anchor.  Subtracting the
        # same value from the new null logit preserves that relative intercept.
        head.null_projection.bias.fill_(-float(support_state["logit_bias"].item()))
    return {
        "evidence_probe": str(evidence_path),
        "support_head_checkpoint": str(support_path),
        "legacy_evidence_probe": probe,
        "legacy_support_probe": support["probe"],
    }


def _logits_from_memory(
    model: CoverageAwareTopologyMotionEvidenceUniRigAR,
    teacher: Any,
    memory: Any,
) -> torch.Tensor:
    positions = torch.arange(
        teacher.token_hidden.shape[1],
        device=teacher.token_hidden.device,
    )
    logits, _ = model.evidence_adapter.logits_from_hidden(
        model.transformer,
        teacher.token_hidden,
        memory,
        positions,
    )
    static_steps = min(model.evidence_adapter.static_prefix_steps, logits.shape[1])
    if static_steps:
        logits = torch.cat(
            (teacher.baseline_logits[:, :static_steps], logits[:, static_steps:]),
            dim=1,
        )
    zero_rows = memory.anchor_confidence.amax(dim=1) == 0
    if bool(zero_rows.any()):
        logits = torch.where(
            zero_rows[:, None, None],
            teacher.baseline_logits,
            logits,
        )
    return logits


def _region_losses(
    logits: torch.Tensor,
    batch: dict[str, Any],
    tokenizer: Any,
    *,
    static_prefix_steps: int,
) -> dict[str, tuple[torch.Tensor, int]]:
    labels = batch["input_ids"][:, 1:].clone()
    valid = batch["attention_mask"][:, 1:] != 0
    labels[~valid] = -100
    losses = nn.functional.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    positions = torch.arange(losses.shape[1], device=losses.device)[None]
    masks = {
        "all": valid,
        "later": valid & (positions >= int(static_prefix_steps)),
        "continue": torch.zeros_like(valid),
        "branch": torch.zeros_like(valid),
        "eos": torch.zeros_like(valid),
    }
    batch_size = int(logits.shape[0])
    for batch_index in range(batch_size):
        joint_count = int(batch["joint_count"][batch_index])
        token_count = int(batch["attention_mask"][batch_index].sum())
        decisions = parse_prefix_decisions(
            batch["input_ids"][batch_index, :token_count],
            batch["target_parents"][batch_index, :joint_count],
            branch_token=int(tokenizer.token_id_branch),
            eos_token=int(tokenizer.eos),
        )
        for decision in decisions:
            masks[decision.role][batch_index, decision.prediction_position] = True

    def reduce(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        count = int(mask.sum().item())
        if count == 0:
            return losses.new_zeros(()), 0
        return losses[mask].mean(), count

    return {name: reduce(mask) for name, mask in masks.items()}


def _support_statistics(
    logits: torch.Tensor,
    targets: PrefixSupportTargets,
    attention_mask: torch.Tensor,
    *,
    static_prefix_steps: int,
) -> dict[str, float]:
    output = prefix_support_distribution_loss(
        logits,
        targets,
        attention_mask,
        static_prefix_steps=static_prefix_steps,
    )
    probabilities = nn.functional.softmax(logits.float(), dim=-1)
    predicts_token = torch.zeros_like(targets.valid_mask)
    predicts_token[:, :-1] = attention_mask[:, 1:] != 0
    positions = torch.arange(logits.shape[1], device=logits.device)[None]
    valid = (
        targets.valid_mask
        & predicts_token
        & (positions >= int(static_prefix_steps))
    )
    null_state = valid & (targets.null_probability > 0.5)
    branch = valid & targets.branch_decision_mask & ~null_state
    regular = valid & ~targets.branch_decision_mask & ~null_state
    anchor_probability = probabilities[..., :-1]
    assigned = (anchor_probability * targets.anchor_distribution.float()).sum(dim=-1)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].mean()) if bool(mask.any()) else math.nan

    return {
        "loss": float(output["prefix_support_loss"]),
        "null_probability_regular": masked_mean(probabilities[..., -1], regular),
        "null_probability_branch": masked_mean(probabilities[..., -1], branch),
        "null_probability_null_state": masked_mean(probabilities[..., -1], null_state),
        "assigned_probability_regular": masked_mean(assigned, regular),
        "assigned_probability_branch": masked_mean(assigned, branch),
        "regular_positions": float(regular.sum()),
        "branch_positions": float(branch.sum()),
        "null_positions": float(null_state.sum()),
    }


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _gradient_summary(module: nn.Module) -> dict[str, float | bool]:
    total = 0.0
    finite = True
    tensors = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        total += float(gradient.abs().sum())
        finite = finite and bool(torch.isfinite(gradient).all())
        tensors += 1
    return {"abs_sum": total, "finite": finite, "tensors": tensors}


@torch.no_grad()
def evaluate(
    model: CoverageAwareTopologyMotionEvidenceUniRigAR,
    loader: Any,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    seed: int,
    consistency_rows: int,
    prefix_lengths: list[int],
) -> dict[str, Any]:
    model.eval()
    weighted_loss = {
        control: {region: 0.0 for region in REGIONS} for control in CONTROLS
    }
    counts = {region: 0 for region in REGIONS}
    support_rows: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []
    root_logit_max_abs_diff = 0.0
    zero_logit_max_abs_diff = 0.0
    consistency: list[dict[str, dict[str, float]]] = []
    alignment_loss = 0.0
    alignment_branch_fraction = 0.0
    boundary_loss = 0.0
    query_tokens_all_1024 = True

    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        torch.manual_seed(seed + row_index)
        torch.cuda.manual_seed_all(seed + row_index)
        refs = model.sample_references(batch)
        generator = torch.Generator(device=device).manual_seed(seed + 100000 + row_index)
        with torch.autocast("cuda", dtype=amp_dtype):
            memory = model.build_memory(batch, refs=refs)
            teacher = model.teacher_forcing(batch, memory=memory)
            targets = model._prefix_support_targets(batch, refs)
            support_logits = model.evidence_adapter.attention.support_logits(
                teacher.token_hidden,
                memory.static_tokens,
            )
            alignment = model.coverage_attention_alignment_loss(
                batch,
                memory,
                teacher.token_hidden,
                targets,
            )
            boundary = model.boundary_auxiliary_loss(batch, refs, memory)
            zero_logits = _logits_from_memory(
                model,
                teacher,
                memory.controlled("zero"),
            )
            corrupt_logits = _logits_from_memory(
                model,
                teacher,
                memory.controlled("corrupt_correspondence", generator=generator),
            )
            logits_by_control = {
                "normal": teacher.logits,
                "zero": zero_logits,
                "corrupt": corrupt_logits,
            }
            losses = {
                control: _region_losses(
                    logits,
                    batch,
                    tokenizer,
                    static_prefix_steps=model.evidence_adapter.static_prefix_steps,
                )
                for control, logits in logits_by_control.items()
            }

        row_loss: dict[str, dict[str, float]] = {}
        for control in CONTROLS:
            row_loss[control] = {}
            for region in REGIONS:
                loss, count = losses[control][region]
                row_loss[control][region] = float(loss)
                weighted_loss[control][region] += float(loss) * count
                if control == "normal":
                    counts[region] += count
        support_rows.append(
            _support_statistics(
                support_logits,
                targets,
                batch["attention_mask"],
                static_prefix_steps=model.evidence_adapter.static_prefix_steps,
            )
        )
        root_steps = min(model.evidence_adapter.static_prefix_steps, teacher.logits.shape[1])
        if root_steps:
            root_logit_max_abs_diff = max(
                root_logit_max_abs_diff,
                float(
                    (
                        teacher.logits[:, :root_steps].float()
                        - teacher.baseline_logits[:, :root_steps].float()
                    ).abs().max()
                ),
            )
        zero_logit_max_abs_diff = max(
            zero_logit_max_abs_diff,
            float((zero_logits.float() - teacher.baseline_logits.float()).abs().max()),
        )
        if row_index < consistency_rows:
            with torch.autocast("cuda", dtype=amp_dtype):
                consistency.append(
                    _generation_consistency(
                        model,
                        batch,
                        memory,
                        teacher.logits,
                        teacher.baseline_logits,
                        prefix_lengths,
                    )
                )
        alignment_loss += float(alignment["attention_alignment_loss"])
        alignment_branch_fraction += float(
            alignment["attention_alignment_branch_valid_fraction"]
        )
        boundary_loss += float(boundary["boundary_loss"])
        query_tokens_all_1024 = query_tokens_all_1024 and memory.static_tokens.shape[1] == 1024
        details.append(
            {
                "index": row_index,
                "path": raw_batch["path"][0],
                "joint_count": int(raw_batch["joint_count"][0]),
                "confidence": float(memory.confidence[0]),
                "motion_q90_rms": float(memory.raw_evidence.example_motion_q90_rms[0]),
                "loss": row_loss,
                "branch_corrupt_minus_normal": (
                    row_loss["corrupt"]["branch"] - row_loss["normal"]["branch"]
                    if losses["normal"]["branch"][1]
                    else math.nan
                ),
                "eos_normal_minus_zero": (
                    row_loss["normal"]["eos"] - row_loss["zero"]["eos"]
                ),
            }
        )

    mean_loss = {
        control: {
            region: weighted_loss[control][region] / max(counts[region], 1)
            for region in REGIONS
        }
        for control in CONTROLS
    }
    branch_rows = [
        row for row in details if math.isfinite(row["branch_corrupt_minus_normal"])
    ]
    max_consistency = 0.0
    for row in consistency:
        for metrics in row.values():
            max_consistency = max(
                max_consistency,
                metrics["qe_max_abs_diff"],
                metrics["adapter_increment_max_abs_diff"],
            )
    return {
        "rows": len(details),
        "decision_counts": counts,
        "mean_token_ce": mean_loss,
        "normal_advantage": {
            "later_vs_corrupt": mean_loss["corrupt"]["later"] - mean_loss["normal"]["later"],
            "later_vs_zero": mean_loss["zero"]["later"] - mean_loss["normal"]["later"],
            "branch_vs_corrupt": mean_loss["corrupt"]["branch"] - mean_loss["normal"]["branch"],
            "branch_vs_zero": mean_loss["zero"]["branch"] - mean_loss["normal"]["branch"],
            "continue_vs_corrupt": mean_loss["corrupt"]["continue"] - mean_loss["normal"]["continue"],
            "eos_harm_vs_zero": mean_loss["normal"]["eos"] - mean_loss["zero"]["eos"],
        },
        "branch_normal_win_rate_vs_corrupt": (
            sum(row["branch_corrupt_minus_normal"] > 0.0 for row in branch_rows)
            / max(len(branch_rows), 1)
        ),
        "support": {
            key: _mean_finite([row[key] for row in support_rows])
            for key in support_rows[0]
        },
        "root_logit_max_abs_diff": root_logit_max_abs_diff,
        "zero_logit_max_abs_diff": zero_logit_max_abs_diff,
        "query_tokens_all_1024": query_tokens_all_1024,
        "attention_alignment_loss": alignment_loss / max(len(details), 1),
        "attention_alignment_branch_valid_fraction": (
            alignment_branch_fraction / max(len(details), 1)
        ),
        "boundary_loss": boundary_loss / max(len(details), 1),
        "teacher_generation_max_abs_diff": max_consistency,
        "teacher_generation_details": consistency,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    if args.train_limit <= 0 or args.valid_limit <= 0 or args.steps <= 0:
        raise ValueError("train-limit, valid-limit, and steps must be positive")
    if args.static_prefix_steps != 4:
        raise ValueError("flat UniRig requires four static prefix positions")
    apply_checkpoint_eval_defaults(args)
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    prefix_lengths = [
        int(value) for value in args.prefix_lengths.split(",") if value.strip()
    ]
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
    model = CoverageAwareTopologyMotionEvidenceUniRigAR(
        baseline.unirig_ar,
        baseline.conditioner.surface_tokenizer,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        evidence_heads=8,
        evidence_residual_scale=args.evidence_residual_scale,
        evidence_static_prefix_steps=args.static_prefix_steps,
        coverage_bias_strength=args.coverage_bias_strength,
        support_projection_size=args.projection_size,
    )
    initialization = _load_probe_initialization(
        model,
        args.evidence_probe,
        args.support_head_checkpoint,
    )
    model.conditioner.value_encoder.to(device)
    model.evidence_adapter.to(device)
    del baseline
    gc.collect()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    support_head = model.evidence_adapter.attention.support_head
    for parameter in support_head.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in support_head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    before = evaluate(
        model,
        valid_loader,
        tokenizer,
        device,
        amp_dtype,
        seed=args.seed + 200000,
        consistency_rows=args.consistency_rows,
        prefix_lengths=prefix_lengths,
    )

    iterator = iter(train_loader)
    train_log: list[dict[str, float | int]] = []
    last_gradient: dict[str, dict[str, float | bool]] = {}
    model.eval()
    support_head.train()
    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            refs = model.sample_references(batch)
            memory = model.build_memory(batch, refs=refs)
            teacher = model.teacher_forcing(
                batch,
                memory=memory.controlled("zero"),
            )
            targets = model._prefix_support_targets(batch, refs)
        optimizer.zero_grad(set_to_none=True)
        support_logits = support_head(
            teacher.token_hidden.detach(),
            memory.static_tokens.detach(),
        )
        output = prefix_support_distribution_loss(
            support_logits,
            targets,
            batch["attention_mask"],
            static_prefix_steps=model.evidence_adapter.static_prefix_steps,
        )
        loss = output["prefix_support_loss"]
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        last_gradient = {
            "query_projection": _gradient_summary(support_head.query_projection),
            "key_projection": _gradient_summary(support_head.key_projection),
            "null_projection": _gradient_summary(support_head.null_projection),
        }
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "prefix_support_loss": float(loss.detach()),
                "null_probability_supported": float(
                    output["prefix_support_null_probability_supported"].detach()
                ),
                "null_probability_null_state": float(
                    output["prefix_support_null_probability_terminal"].detach()
                ),
                "branch_valid_fraction": float(
                    output["prefix_support_branch_valid_fraction"].detach()
                ),
                "grad_norm": float(grad_norm),
                "query_gradient_abs_sum": float(
                    last_gradient["query_projection"]["abs_sum"]
                ),
                "key_gradient_abs_sum": float(
                    last_gradient["key_projection"]["abs_sum"]
                ),
                "null_gradient_abs_sum": float(
                    last_gradient["null_projection"]["abs_sum"]
                ),
            }
            train_log.append(record)
            print(json.dumps(record), flush=True)

    after = evaluate(
        model,
        valid_loader,
        tokenizer,
        device,
        amp_dtype,
        seed=args.seed + 200000,
        consistency_rows=args.consistency_rows,
        prefix_lengths=prefix_lengths,
    )
    elapsed = time.perf_counter() - started
    result = {
        "scope": "development preflight; only prefix support/null head updated",
        "checkpoint": str(args.checkpoint),
        "train_manifest": str(args.train_manifest),
        "valid_manifest": str(args.valid_manifest),
        "train_assets": len(train_dataset),
        "valid_assets": len(valid_dataset),
        "initialization": initialization,
        "configuration": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "projection_size": args.projection_size,
            "coverage_bias_strength": args.coverage_bias_strength,
            "evidence_residual_scale": args.evidence_residual_scale,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        },
        "before": before,
        "after": after,
        "train_log": train_log,
        "last_gradient": last_gradient,
        "acceptance": {
            "support_loss_improved": after["support"]["loss"] < before["support"]["loss"],
            "terminal_null_learned": after["support"]["null_probability_null_state"] > 0.8,
            "supported_motion_not_abstained": after["support"]["null_probability_regular"] < 0.2,
            "branch_correct_beats_corrupt": after["normal_advantage"]["branch_vs_corrupt"] > 0.0,
            "branch_correct_beats_zero": after["normal_advantage"]["branch_vs_zero"] > 0.0,
            "eos_not_harmed": after["normal_advantage"]["eos_harm_vs_zero"] <= 0.02,
            "zero_motion_exact_noop": after["zero_logit_max_abs_diff"] == 0.0,
            "root_exact": after["root_logit_max_abs_diff"] == 0.0,
            "query_tokens_all_1024": after["query_tokens_all_1024"],
            "teacher_generation_consistent": after["teacher_generation_max_abs_diff"] <= 0.02,
            "support_gradients_finite_nonzero": all(
                bool(value["finite"]) and float(value["abs_sum"]) > 0.0
                for value in last_gradient.values()
            ),
        },
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }
    torch.save(
        {
            "support_head": support_head.state_dict(),
            "configuration": result["configuration"],
            "initialization": initialization,
        },
        args.output_dir / "coverage_support_probe.pt",
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "acceptance": result["acceptance"],
                "before_support": before["support"],
                "after_support": after["support"],
                "before_advantage": before["normal_advantage"],
                "after_advantage": after["normal_advantage"],
                "elapsed_seconds": elapsed,
                "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
