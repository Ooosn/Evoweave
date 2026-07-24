#!/usr/bin/env python3
"""Learn whether a flat skeleton prefix can localize unexplained mesh support.

The baseline, surface tokenizer, and autoregressive transformer remain frozen.
Only a small prefix-to-anchor support head is optimized.  Skin weights provide
training-only targets; shuffled anchors and shuffled prefix states test whether
the head learned aligned coverage rather than a sequence-position template.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
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
from preflight_motion_evidence_learnability import (  # noqa: E402
    build_dataset,
    make_loader,
    set_probe_modes,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402

from rigweave.motion_evidence import (  # noqa: E402
    TopologyMotionEvidenceUniRigAR,
    query_aligned_skin_weights,
)


class PrefixRemainingSupportHead(nn.Module):
    """Predict independent remaining-support probabilities at query anchors."""

    def __init__(self, hidden_size: int, projection_size: int = 128) -> None:
        super().__init__()
        if hidden_size <= 0 or projection_size <= 0:
            raise ValueError("hidden and projection sizes must be positive")
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.scale = float(projection_size) ** -0.5

    def forward(
        self,
        prefix_states: torch.Tensor,
        static_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_states.ndim != 3 or static_tokens.ndim != 3:
            raise ValueError("prefix_states and static_tokens must be rank-3")
        if prefix_states.shape[0] != static_tokens.shape[0]:
            raise ValueError("prefix and static-token batch sizes differ")
        query = self.query_projection(self.query_norm(prefix_states.float()))
        key = self.key_projection(self.key_norm(static_tokens.float()))
        return torch.matmul(query, key.transpose(1, 2)) * self.scale + self.logit_bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--valid-limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--support-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def decision_targets(
    batch: dict[str, Any],
    tokenizer: Any,
    query_skin: torch.Tensor,
) -> tuple[torch.LongTensor, torch.Tensor, list[str]]:
    if query_skin.shape[0] != 1:
        raise ValueError("coverage preflight uses batch_size=1")
    joint_count = int(batch["joint_count"][0])
    token_count = int(batch["attention_mask"][0].sum())
    parents = batch["target_parents"][0, :joint_count]
    decisions = parse_prefix_decisions(
        batch["input_ids"][0, :token_count],
        parents,
        branch_token=int(tokenizer.token_id_branch),
        eos_token=int(tokenizer.eos),
    )
    positions = torch.tensor(
        [decision.prediction_position for decision in decisions],
        device=query_skin.device,
        dtype=torch.long,
    )
    targets = torch.stack(
        [
            query_skin[0, :, decision.generated_joint_count : joint_count]
            .sum(dim=-1)
            .clamp(0.0, 1.0)
            for decision in decisions
        ],
        dim=0,
    )
    return positions, targets, [decision.role for decision in decisions]


def role_balanced_support_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    roles: list[str],
    *,
    max_positive_weight: float = 32.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("coverage logits and targets must have identical shape")
    positive_fraction = targets.mean(dim=-1)
    positive_weight = (
        (1.0 - positive_fraction) / positive_fraction.clamp_min(eps)
    ).clamp(1.0, max_positive_weight)
    per_anchor = -(
        positive_weight[:, None] * targets * nn.functional.logsigmoid(logits)
        + (1.0 - targets) * nn.functional.logsigmoid(-logits)
    )
    per_position = per_anchor.mean(dim=-1)
    role_losses = []
    for role in ("continue", "branch", "eos"):
        mask = torch.tensor(
            [value == role for value in roles],
            device=logits.device,
            dtype=torch.bool,
        )
        if bool(mask.any()):
            role_losses.append(per_position[mask].mean())
    if not role_losses:
        raise ValueError("no topology decisions were available for coverage loss")
    return torch.stack(role_losses).mean()


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"count": int(labels.size), "auroc": float("nan"), "auprc": float("nan")}
    return {
        "count": int(labels.size),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


@torch.no_grad()
def frozen_features(
    model: TopologyMotionEvidenceUniRigAR,
    batch: dict[str, Any],
    tokenizer: Any,
    *,
    seed: int,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    refs = model.sample_references(batch)
    with torch.autocast("cuda", dtype=amp_dtype):
        memory = model.build_memory(batch, refs=refs)
        teacher = model.teacher_forcing(batch, memory=memory.controlled("zero"))
        query_skin = query_aligned_skin_weights(
            batch["target_skin_weights"],
            batch["faces"],
            refs,
        ).float()
    positions, targets, roles = decision_targets(batch, tokenizer, query_skin)
    return (
        teacher.token_hidden[:, positions].detach(),
        memory.static_tokens.detach(),
        targets.detach(),
        roles,
    )


@torch.no_grad()
def evaluate(
    head: PrefixRemainingSupportHead,
    model: TopologyMotionEvidenceUniRigAR,
    loader: Any,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    seed: int,
    support_threshold: float,
) -> dict[str, Any]:
    head.eval()
    set_probe_modes(model, training=False)
    controls = ("correct", "anchor_shuffle", "prefix_reverse", "constant_keys")
    labels_by_control: dict[str, list[np.ndarray]] = {name: [] for name in controls}
    scores_by_control: dict[str, list[np.ndarray]] = {name: [] for name in controls}
    per_decision_auc: dict[str, list[float]] = {name: [] for name in controls}
    eos_probabilities: dict[str, list[float]] = {name: [] for name in controls}
    losses: list[float] = []
    rows: list[dict[str, Any]] = []

    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        prefix, static, targets, roles = frozen_features(
            model,
            batch,
            tokenizer,
            seed=seed + row_index,
            amp_dtype=amp_dtype,
        )
        generator = torch.Generator(device=device).manual_seed(seed + 100000 + row_index)
        permutation = torch.randperm(static.shape[1], device=device, generator=generator)
        inputs = {
            "correct": (prefix, static),
            "anchor_shuffle": (prefix, static[:, permutation]),
            "prefix_reverse": (prefix.flip(dims=(1,)), static),
            "constant_keys": (prefix, torch.zeros_like(static)),
        }
        logits = {name: head(*values)[0] for name, values in inputs.items()}
        losses.append(float(role_balanced_support_loss(logits["correct"], targets, roles)))
        hard = (targets >= support_threshold).cpu().numpy().astype(np.int8)
        role_array = np.asarray(roles)
        non_eos = role_array != "eos"
        eos = role_array == "eos"
        row_payload: dict[str, Any] = {
            "index": row_index,
            "path": raw_batch["path"][0],
            "decisions": len(roles),
        }
        for name in controls:
            probabilities = logits[name].sigmoid().cpu().numpy()
            labels_by_control[name].append(hard[non_eos].reshape(-1))
            scores_by_control[name].append(probabilities[non_eos].reshape(-1))
            for decision_index in np.flatnonzero(non_eos):
                decision_labels = hard[decision_index]
                if np.unique(decision_labels).size >= 2:
                    per_decision_auc[name].append(
                        float(
                            roc_auc_score(
                                decision_labels,
                                probabilities[decision_index],
                            )
                        )
                    )
            if bool(eos.any()):
                eos_probabilities[name].extend(
                    probabilities[eos].reshape(-1).tolist()
                )
            row_payload[f"{name}_mean_probability"] = float(probabilities.mean())
        rows.append(row_payload)

    result: dict[str, Any] = {
        "assets": len(rows),
        "role_balanced_loss": float(np.mean(losses)),
        "controls": {},
        "details": rows,
    }
    for name in controls:
        labels = np.concatenate(labels_by_control[name])
        scores = np.concatenate(scores_by_control[name])
        metrics = binary_metrics(labels, scores)
        decision_auc = np.asarray(per_decision_auc[name], dtype=np.float64)
        eos_values = np.asarray(eos_probabilities[name], dtype=np.float64)
        metrics.update(
            {
                "per_decision_auroc_count": int(decision_auc.size),
                "per_decision_auroc_median": float(np.median(decision_auc)),
                "per_decision_auroc_q10": float(np.quantile(decision_auc, 0.10)),
                "eos_probability_mean": float(eos_values.mean()),
                "eos_probability_q90": float(np.quantile(eos_values, 0.90)),
                "eos_probability_max": float(eos_values.max()),
            }
        )
        result["controls"][name] = metrics
    return result


def main() -> None:
    args = parse_args()
    if args.train_limit <= 0 or args.valid_limit <= 0 or args.steps <= 0:
        raise ValueError("limits and steps must be positive")
    if not 0.0 < args.support_threshold < 1.0:
        raise ValueError("support-threshold must be in (0,1)")
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
        evidence_static_prefix_steps=4,
    )
    model.conditioner.value_encoder.to(device)
    model.evidence_adapter.to(device)
    del baseline
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    set_probe_modes(model, training=False)

    head = PrefixRemainingSupportHead(
        int(model.unirig_ar.hidden_size),
        projection_size=args.projection_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    before = evaluate(
        head,
        model,
        valid_loader,
        tokenizer,
        device,
        amp_dtype,
        seed=args.seed + 200000,
        support_threshold=args.support_threshold,
    )

    head.train()
    iterator = iter(train_loader)
    train_log: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        prefix, static, targets, roles = frozen_features(
            model,
            batch,
            tokenizer,
            seed=args.seed + step,
            amp_dtype=amp_dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = head(prefix, static)[0]
        loss = role_balanced_support_loss(logits, targets, roles)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
            }
            train_log.append(row)
            print(json.dumps(row), flush=True)

    after = evaluate(
        head,
        model,
        valid_loader,
        tokenizer,
        device,
        amp_dtype,
        seed=args.seed + 200000,
        support_threshold=args.support_threshold,
    )
    elapsed = time.perf_counter() - started
    result = {
        "probe": {
            "train_limit": args.train_limit,
            "valid_limit": args.valid_limit,
            "steps": args.steps,
            "projection_size": args.projection_size,
            "learning_rate": args.learning_rate,
            "support_threshold": args.support_threshold,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "backbone_trainable_parameters": 0,
            "head_trainable_parameters": sum(
                parameter.numel() for parameter in head.parameters()
            ),
        },
        "before": before,
        "after": after,
        "train_log": train_log,
    }
    torch.save(
        {"head": head.state_dict(), "probe": result["probe"]},
        args.output_dir / "remaining_support_head.pt",
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
