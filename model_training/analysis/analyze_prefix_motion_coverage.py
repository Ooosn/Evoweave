#!/usr/bin/env python3
"""Audit whether ungenerated skeleton regions retain observable motion evidence.

This is a training-only oracle diagnostic.  Skin weights define which query
anchors belong to joints that have not yet appeared in a ground-truth flat
UniRig prefix.  They are never proposed as an inference input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_ROOT = SCRIPT_DIR.parent
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
if str(RIGWEAVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from eval_dynamic_rig_ce import (  # noqa: E402
    CHECKPOINT_DEFAULTS,
    apply_checkpoint_eval_defaults,
)
from preflight_motion_evidence_learnability import (  # noqa: E402
    build_dataset,
    make_loader,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402

from rigweave.dynamic_rig.sampling import sample_trackable_surface  # noqa: E402
from rigweave.motion_evidence import (  # noqa: E402
    TopologyLocalMotionEvidence,
    query_aligned_skin_weights,
)


@dataclass(frozen=True)
class PrefixDecision:
    role: str
    prediction_position: int
    generated_joint_count: int
    next_joint_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--reference-seed-offset", type=int, default=200000)
    parser.add_argument("--observable-threshold", type=float, default=0.1)
    for name in CHECKPOINT_DEFAULTS:
        parser.set_defaults(**{name: None})
    return parser.parse_args()


def parse_prefix_decisions(
    input_ids: torch.LongTensor,
    target_parents: torch.LongTensor,
    *,
    branch_token: int,
    eos_token: int,
) -> list[PrefixDecision]:
    """Locate topology decisions after completed joints in a flat token row."""

    ids = [int(value) for value in input_ids.tolist()]
    parents = [int(value) for value in target_parents.tolist()]
    cursor = 2  # BOS and class token.
    decisions: list[PrefixDecision] = []
    for joint_index, parent in enumerate(parents):
        is_branch = joint_index > 0 and parent != joint_index - 1
        if is_branch:
            if cursor >= len(ids) or ids[cursor] != int(branch_token):
                raise ValueError(
                    f"missing branch token for joint {joint_index} at token {cursor}"
                )
            decisions.append(
                PrefixDecision(
                    role="branch",
                    prediction_position=cursor - 1,
                    generated_joint_count=joint_index,
                    next_joint_index=joint_index,
                )
            )
            cursor += 1
            if cursor + 6 > len(ids):
                raise ValueError("branch row ends before parent/child coordinates")
            cursor += 6
        else:
            if joint_index > 0:
                decisions.append(
                    PrefixDecision(
                        role="continue",
                        prediction_position=cursor - 1,
                        generated_joint_count=joint_index,
                        next_joint_index=joint_index,
                    )
                )
            if cursor + 3 > len(ids):
                raise ValueError("row ends before joint coordinates")
            cursor += 3

    if cursor >= len(ids) or ids[cursor] != int(eos_token):
        raise ValueError(f"expected EOS at token {cursor}")
    if cursor != len(ids) - 1:
        raise ValueError("tokens follow the terminal EOS")
    decisions.append(
        PrefixDecision(
            role="eos",
            prediction_position=cursor - 1,
            generated_joint_count=len(parents),
            next_joint_index=None,
        )
    )
    return decisions


def descendant_indices(parents: list[int], root: int) -> list[int]:
    descendants: list[int] = []
    for candidate in range(root, len(parents)):
        cursor = candidate
        while cursor >= 0 and cursor != root:
            cursor = parents[cursor]
        if cursor == root:
            descendants.append(candidate)
    return descendants


def support_metrics(
    support: torch.Tensor,
    total_support: torch.Tensor,
    anchor_confidence: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> dict[str, float | None]:
    support_mass = float(support.sum())
    total_mass = float(total_support.sum())
    observable_mass = float((support * anchor_confidence).sum())
    total_observable_mass = float((total_support * anchor_confidence).sum())
    return {
        "support_mass": support_mass,
        "support_fraction_of_total": (
            support_mass / total_mass if total_mass > eps else None
        ),
        "support_observability": (
            observable_mass / support_mass if support_mass > eps else None
        ),
        "observable_fraction_of_total": (
            observable_mass / total_observable_mass
            if total_observable_mass > eps
            else None
        ),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_role(
    rows: list[dict[str, Any]],
    *,
    observable_threshold: float,
) -> dict[str, Any]:
    metric_names = (
        "next_support_observability",
        "subtree_support_observability",
        "remaining_support_fraction_of_total",
        "remaining_support_observability",
        "remaining_observable_fraction_of_total",
    )
    result: dict[str, Any] = {"decisions": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row[name] is not None]
        result[name] = summarize(values)
        result[f"{name}_present_rate"] = len(values) / max(len(rows), 1)

    if rows and rows[0]["role"] != "eos":
        next_low = [
            row
            for row in rows
            if row["next_support_observability"] is None
            or float(row["next_support_observability"]) < observable_threshold
        ]
        subtree_rescued = [
            row
            for row in next_low
            if row["subtree_support_observability"] is not None
            and float(row["subtree_support_observability"]) >= observable_threshold
        ]
        remaining_rescued = [
            row
            for row in next_low
            if row["remaining_support_observability"] is not None
            and float(row["remaining_support_observability"]) >= observable_threshold
        ]
        result["next_joint_low_or_missing_rate"] = len(next_low) / len(rows)
        result["subtree_rescue_rate_within_next_low"] = (
            len(subtree_rescued) / len(next_low) if next_low else 0.0
        )
        result["remaining_region_rescue_rate_within_next_low"] = (
            len(remaining_rescued) / len(next_low) if next_low else 0.0
        )
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= args.observable_threshold <= 1.0:
        raise ValueError("observable-threshold must be in [0,1]")
    apply_checkpoint_eval_defaults(args)

    device = torch.device("cuda:0")
    tokenizer = build_tokenizer(args.tokenizer_config)
    dataset = build_dataset(
        args,
        args.valid_manifest,
        tokenizer,
        limit=args.limit,
        seed=args.seed,
    )
    loader = make_loader(dataset, tokenizer, shuffle=False, seed=args.seed)
    extractor = TopologyLocalMotionEvidence().to(device)

    decision_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    for row_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        reference_seed = args.seed + args.reference_seed_offset + row_index
        torch.manual_seed(reference_seed)
        torch.cuda.manual_seed_all(reference_seed)
        refs = sample_trackable_surface(
            batch["frame_vertices"][:, 0],
            batch["faces"],
            num_samples=args.surface_samples,
            vertex_samples=args.vertex_samples,
            query_tokens=args.query_tokens,
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
        )
        evidence = extractor(
            batch["frame_vertices"],
            batch["faces"],
            refs,
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
        )
        query_skin = query_aligned_skin_weights(
            batch["target_skin_weights"],
            batch["faces"],
            refs,
        )[0].float()
        joint_count = int(batch["joint_count"][0])
        query_skin = query_skin[:, :joint_count]
        parents_tensor = batch["target_parents"][0, :joint_count]
        parents = [int(value) for value in parents_tensor.tolist()]
        token_count = int(batch["attention_mask"][0].sum())
        decisions = parse_prefix_decisions(
            batch["input_ids"][0, :token_count],
            parents_tensor,
            branch_token=int(tokenizer.token_id_branch),
            eos_token=int(tokenizer.eos),
        )
        total_support = query_skin.sum(dim=-1)
        anchor_confidence = evidence.anchor_confidence[0].float()
        path = raw_batch["path"][0]

        for decision in decisions:
            generated = decision.generated_joint_count
            remaining_support = query_skin[:, generated:].sum(dim=-1)
            remaining = support_metrics(
                remaining_support,
                total_support,
                anchor_confidence,
            )
            next_metrics: dict[str, float | None]
            subtree_metrics: dict[str, float | None]
            if decision.next_joint_index is None:
                next_metrics = support_metrics(
                    torch.zeros_like(total_support),
                    total_support,
                    anchor_confidence,
                )
                subtree_metrics = next_metrics
            else:
                next_index = decision.next_joint_index
                next_metrics = support_metrics(
                    query_skin[:, next_index],
                    total_support,
                    anchor_confidence,
                )
                subtree = descendant_indices(parents, next_index)
                subtree_metrics = support_metrics(
                    query_skin[:, subtree].sum(dim=-1),
                    total_support,
                    anchor_confidence,
                )

            decision_rows.append(
                {
                    "asset_index": row_index,
                    "path": path,
                    "role": decision.role,
                    "prediction_position": decision.prediction_position,
                    "generated_joint_count": generated,
                    "joint_count": joint_count,
                    "motion_q90_rms": float(evidence.example_motion_q90_rms[0]),
                    "next_support_observability": next_metrics[
                        "support_observability"
                    ],
                    "subtree_support_observability": subtree_metrics[
                        "support_observability"
                    ],
                    "remaining_support_fraction_of_total": remaining[
                        "support_fraction_of_total"
                    ],
                    "remaining_support_observability": remaining[
                        "support_observability"
                    ],
                    "remaining_observable_fraction_of_total": remaining[
                        "observable_fraction_of_total"
                    ],
                }
            )

        asset_rows.append(
            {
                "index": row_index,
                "path": path,
                "joint_count": joint_count,
                "branch_decisions": sum(d.role == "branch" for d in decisions),
                "continue_decisions": sum(d.role == "continue" for d in decisions),
                "motion_q90_rms": float(evidence.example_motion_q90_rms[0]),
                "observable_anchor_fraction": float(
                    (anchor_confidence > 0.0).float().mean()
                ),
            }
        )
        print(json.dumps(asset_rows[-1]), flush=True)

    by_role = {
        role: summarize_role(
            [row for row in decision_rows if row["role"] == role],
            observable_threshold=args.observable_threshold,
        )
        for role in ("continue", "branch", "eos")
    }
    eos_rows = [row for row in decision_rows if row["role"] == "eos"]
    eos_complete = all(
        row["remaining_support_fraction_of_total"] == 0.0 for row in eos_rows
    )
    summary = {
        "assets": len(asset_rows),
        "dataset_seed": args.seed,
        "reference_seed_offset": args.reference_seed_offset,
        "observable_threshold": args.observable_threshold,
        "query_tokens": args.query_tokens,
        "eos_oracle_coverage_complete": eos_complete,
        "roles": by_role,
    }
    payload = {
        "summary": summary,
        "assets": asset_rows,
        "decisions": decision_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
