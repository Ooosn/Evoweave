#!/usr/bin/env python3
"""Real-checkpoint preflight for the isolated Q/E decoder route."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--prefix-lengths", default="4,8,16")
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
    parser.add_argument("--motion-alignment-policy", default=None)
    parser.add_argument("--input-space-policy", default=None)
    parser.add_argument("--target-start-policy", default=None)
    parser.add_argument("--target-root-policy", default=None)
    parser.add_argument("--target-active-skin-only", action="store_true", default=None)
    parser.add_argument("--active-skin-threshold", type=float, default=None)
    parser.add_argument("--condition-fusion", default=None)
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
    parser.add_argument("--explicit-tree-topology-mode", default=None)
    parser.add_argument("--explicit-tree-coordinate-mode", default=None)
    parser.add_argument("--explicit-tree-action-eos-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-action-child-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-action-branch-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-xyz-loss-weight", type=float, default=None)
    parser.add_argument("--use-grammar-state-embedding", action="store_true", default=None)
    parser.add_argument("--use-action-group-bias", action="store_true", default=None)
    parser.add_argument("--use-condition-action-group-bias", action="store_true", default=None)
    return parser.parse_args()


def shifted_ce(
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = -100
    return nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels)


def finite_grad_sum(module: nn.Module) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(grad.abs().sum())
        finite = finite and bool(torch.isfinite(grad).all())
    return total, finite


def _generation_consistency(
    model: TopologyMotionEvidenceUniRigAR,
    batch: dict[str, Any],
    memory: Any,
    teacher_logits: torch.Tensor,
    teacher_baseline_logits: torch.Tensor,
    prefix_lengths: list[int],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    valid_length = int(batch["attention_mask"][0].sum().item())
    for requested in prefix_lengths:
        prefix_length = min(int(requested), valid_length)
        if prefix_length <= 0:
            continue
        prefix_ids = batch["input_ids"][:, :prefix_length]
        token_embeds = model.transformer.get_input_embeddings()(prefix_ids).to(
            dtype=model.transformer.dtype
        )
        prompt = torch.cat(
            (memory.static_tokens.to(dtype=model.transformer.dtype), token_embeds),
            dim=1,
        )
        prompt_mask = torch.ones(prompt.shape[:2], device=prompt.device)
        transformer_output = model.transformer(
            inputs_embeds=prompt,
            attention_mask=prompt_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        generation_logits = model.evidence_adapter.generation_step(
            model.transformer,
            transformer_output,
            memory,
            prefix_position=prefix_length - 1,
        )
        teacher_qe = teacher_logits[:, prefix_length - 1].float()
        teacher_static = teacher_baseline_logits[:, prefix_length - 1].float()
        generation_static = transformer_output.logits[:, -1].float()
        qe_delta = generation_logits.float() - teacher_qe
        static_delta = generation_static - teacher_static
        adapter_delta = (generation_logits.float() - generation_static) - (
            teacher_qe - teacher_static
        )
        results[str(prefix_length)] = {
            "qe_max_abs_diff": float(qe_delta.abs().max()),
            "static_max_abs_diff": float(static_delta.abs().max()),
            "adapter_increment_max_abs_diff": float(adapter_delta.abs().max()),
        }
    return results


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    apply_checkpoint_eval_defaults(args)
    prefix_lengths = [int(value) for value in args.prefix_lengths.split(",") if value.strip()]
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = build_tokenizer(args.tokenizer_config)
    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=args.frames,
        limit=args.limit,
        random_query=False,
        seed=args.seed,
        motion_fps_ratio=args.motion_fps_ratio,
        motion_vertex_samples=args.motion_vertex_samples,
        motion_alignment_policy=args.motion_alignment_policy,
        target_active_skin_only=args.target_active_skin_only,
        active_skin_threshold=args.active_skin_threshold,
        target_start_policy=args.target_start_policy,
        target_root_policy=args.target_root_policy,
        input_space_policy=args.input_space_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )
    baseline = _build_dynamic_model(args, tokenizer, device)
    model = TopologyMotionEvidenceUniRigAR(
        baseline.unirig_ar,
        baseline.conditioner.surface_tokenizer,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        evidence_heads=8,
    )
    model.conditioner.value_encoder.to(device)
    model.evidence_adapter.to(device)
    model.eval()
    baseline.eval()

    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        torch.manual_seed(args.seed + row_index)
        torch.cuda.manual_seed_all(args.seed + row_index)
        row_started = time.perf_counter()
        refs = model.sample_references(batch)
        corrupt_generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + row_index)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            dynamic_condition = baseline.build_condition(batch, refs=refs)
            dynamic_losses = baseline._ar_losses(
                dynamic_condition,
                batch,
                include_loop_recovery=False,
                include_generated_prefix_recovery=False,
            )
            memory = model.build_memory(batch, refs=refs)
            normal = model.teacher_forcing(batch, memory=memory)
            zero_memory = memory.controlled("zero")
            zero = model.teacher_forcing(batch, memory=zero_memory)
            corrupt_memory = memory.controlled(
                "corrupt_correspondence",
                generator=corrupt_generator,
            )
            corrupted = model.teacher_forcing(batch, memory=corrupt_memory)
            static_ce = shifted_ce(normal.baseline_logits, batch["input_ids"], batch["attention_mask"])
            normal_ce = shifted_ce(normal.logits, batch["input_ids"], batch["attention_mask"])
            zero_ce = shifted_ce(zero.logits, batch["input_ids"], batch["attention_mask"])
            corrupt_ce = shifted_ce(corrupted.logits, batch["input_ids"], batch["attention_mask"])
            zero_logit_diff = float((zero.logits.float() - zero.baseline_logits.float()).abs().max())
            normal_hidden_delta = normal.refined_hidden.float() - normal.token_hidden.float()
            first_positions = min(
                model.evidence_adapter.static_prefix_steps,
                normal_hidden_delta.shape[1],
            )
            root_delta = torch.sqrt(normal_hidden_delta[:, :first_positions].square().mean())
            later_delta = torch.sqrt(normal_hidden_delta[:, first_positions:].square().mean())
            consistency = _generation_consistency(
                model,
                batch,
                memory,
                normal.logits,
                normal.baseline_logits,
                prefix_lengths,
            )

        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            gradient_memory = model.build_memory(batch, refs=refs)
            gradient_teacher = model.teacher_forcing(batch, memory=gradient_memory)
            gradient_loss = shifted_ce(
                gradient_teacher.logits,
                batch["input_ids"],
                batch["attention_mask"],
            )
        gradient_loss.backward()
        adapter_grad, adapter_finite = finite_grad_sum(model.evidence_adapter)
        value_grad, value_finite = finite_grad_sum(model.conditioner.value_encoder)
        transformer_grad, transformer_finite = finite_grad_sum(model.transformer)
        model.zero_grad(set_to_none=True)

        rows.append(
            {
                "index": row_index,
                "path": raw_batch["path"][0],
                "target_token_count": int(raw_batch["attention_mask"][0].sum().item()),
                "target_joint_count": int(raw_batch["joint_count"][0].item()),
                "query_tokens": int(memory.static_tokens.shape[1]),
                "source_edge_count": int(memory.raw_evidence.source_edge_counts[0].item()),
                "motion_q90_rms": float(memory.raw_evidence.example_motion_q90_rms[0]),
                "confidence": float(memory.confidence[0]),
                "dynamic_baseline_ce": float(dynamic_losses["ce_loss"]),
                "static_q_ce": float(static_ce),
                "normal_qe_ce_untrained_adapter": float(normal_ce),
                "zero_e_ce": float(zero_ce),
                "corrupted_e_ce_untrained_adapter": float(corrupt_ce),
                "zero_logit_max_abs_diff": zero_logit_diff,
                "root_hidden_delta_rms": float(root_delta),
                "later_hidden_delta_rms": float(later_delta) if later_delta.numel() else math.nan,
                "teacher_generation_max_abs_diff": consistency,
                "gradient": {
                    "adapter_abs_sum": adapter_grad,
                    "value_encoder_abs_sum": value_grad,
                    "transformer_abs_sum": transformer_grad,
                    "all_finite": adapter_finite and value_finite and transformer_finite,
                },
                "elapsed_seconds": time.perf_counter() - row_started,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    elapsed = time.perf_counter() - started
    summary = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "rows": len(rows),
        "weights_changed": False,
        "query_tokens_all_1024": all(row["query_tokens"] == 1024 for row in rows),
        "zero_evidence_exact_noop": all(row["zero_logit_max_abs_diff"] == 0.0 for row in rows),
        "all_gradients_finite_nonzero": all(
            row["gradient"]["all_finite"]
            and row["gradient"]["adapter_abs_sum"] > 0.0
            and row["gradient"]["value_encoder_abs_sum"] > 0.0
            and row["gradient"]["transformer_abs_sum"] > 0.0
            for row in rows
        ),
        "max_teacher_generation_qe_logit_diff": max(
            metrics["qe_max_abs_diff"]
            for row in rows
            for metrics in row["teacher_generation_max_abs_diff"].values()
        ),
        "max_teacher_generation_static_logit_diff": max(
            metrics["static_max_abs_diff"]
            for row in rows
            for metrics in row["teacher_generation_max_abs_diff"].values()
        ),
        "max_teacher_generation_adapter_increment_diff": max(
            metrics["adapter_increment_max_abs_diff"]
            for row in rows
            for metrics in row["teacher_generation_max_abs_diff"].values()
        ),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }
    result = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    del model, baseline
    gc.collect()


if __name__ == "__main__":
    main()
