#!/usr/bin/env python3
"""Trace condition influence through the flat decoder under a fixed self prefix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnose_flat_autoregressive_failures import (
    _first_mismatch,
    _load_generation_rows,
    _move_batch,
    _possible_groups,
    _seed_all,
)


def _mean(values: list[float | int | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _distance_bin(position: int, mismatch_position: int) -> str:
    distance = int(position) - int(mismatch_position)
    if distance < 0:
        return "before_first_mismatch"
    if distance <= 3:
        return "mismatch_to_plus_3"
    if distance <= 15:
        return "plus_4_to_15"
    if distance <= 63:
        return "plus_16_to_63"
    if distance <= 255:
        return "plus_64_to_255"
    if distance <= 511:
        return "plus_256_to_511"
    return "plus_512_and_later"


def _pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
    left_f = left.float()
    right_f = right.float()
    left_norm = torch.linalg.vector_norm(left_f, dim=-1)
    right_norm = torch.linalg.vector_norm(right_f, dim=-1)
    delta_norm = torch.linalg.vector_norm(left_f - right_f, dim=-1)
    denominator = 0.5 * (left_norm + right_norm)
    return {
        "relative_l2": delta_norm / denominator.clamp_min(1.0e-12),
        "cosine": F.cosine_similarity(left_f, right_f, dim=-1),
        "delta_rms": (left_f - right_f).square().mean(dim=-1).sqrt(),
        "hidden_rms": 0.5
        * (
            left_f.square().mean(dim=-1).sqrt()
            + right_f.square().mean(dim=-1).sqrt()
        ),
    }


def _summarize_pair_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.mean().item())
        for name, value in metrics.items()
    }


def _distribution_metrics(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
) -> dict[str, float | int]:
    left_log_probability = F.log_softmax(left_logits.float(), dim=-1)
    right_log_probability = F.log_softmax(right_logits.float(), dim=-1)
    left_probability = left_log_probability.exp()
    right_probability = right_log_probability.exp()
    mean_probability = 0.5 * (left_probability + right_probability)
    mean_log_probability = mean_probability.clamp_min(1.0e-30).log()
    js = 0.5 * (
        torch.sum(left_probability * (left_log_probability - mean_log_probability))
        + torch.sum(right_probability * (right_log_probability - mean_log_probability))
    )
    return {
        "js_divergence_nats": float(js.item()),
        "total_variation": float(
            (0.5 * torch.sum(torch.abs(left_probability - right_probability))).item()
        ),
        "top1_agreement": int(
            int(left_probability.argmax().item())
            == int(right_probability.argmax().item())
        ),
    }


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"positions": 0}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"position", "distance_bin"} and isinstance(value, (int, float))
        }
    )
    return {
        "positions": len(rows),
        **{key: _mean([row.get(key) for row in rows]) for key in keys},
    }


def _select_success_controls(
    hitmax_indices: list[int],
    success_indices: list[int],
    target_joint_counts: list[int],
    limit: int,
) -> list[int]:
    available = set(int(index) for index in success_indices)
    selected: list[int] = []
    for hitmax_index in hitmax_indices:
        if not available or len(selected) >= int(limit):
            break
        closest = min(
            available,
            key=lambda index: (
                abs(target_joint_counts[index] - target_joint_counts[hitmax_index]),
                index,
            ),
        )
        selected.append(int(closest))
        available.remove(closest)
    if len(selected) < int(limit):
        selected.extend(sorted(available)[: int(limit) - len(selected)])
    return selected


@torch.no_grad()
def _trace_row(
    model: torch.nn.Module,
    tokenizer: Any,
    correct_condition: torch.Tensor,
    swapped_condition: torch.Tensor,
    generated_ids: list[int],
    target_ids: list[int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    mismatch = _first_mismatch(tokenizer, target_ids, generated_ids)
    mismatch_position = int(mismatch["position"])
    if mismatch_position < 2:
        raise ValueError(f"invalid first mismatch: {mismatch}")

    condition = torch.cat(
        [
            correct_condition.to(device),
            swapped_condition.to(device),
        ],
        dim=0,
    ).to(dtype=model.transformer.dtype)
    input_ids = torch.tensor(
        [generated_ids, generated_ids],
        device=device,
        dtype=torch.long,
    )
    token_attention = torch.ones_like(input_ids)
    token_embeds = model.token_inputs_embeds(input_ids, token_attention)
    prompt = torch.cat([condition, token_embeds], dim=1)
    attention_mask = torch.ones(
        prompt.shape[:2],
        device=device,
        dtype=torch.long,
    )
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    condition_length = int(condition.shape[1])
    token_logits = output.logits[:, condition_length:]
    final_token_hidden = output.hidden_states[-1][:, condition_length:]
    token_logits = model.apply_action_group_bias(
        token_logits,
        final_token_hidden,
        condition,
    )

    has_eos = bool(generated_ids and generated_ids[-1] == int(tokenizer.eos))
    decision_positions = list(range(2, len(generated_ids)))
    if not has_eos:
        decision_positions.append(len(generated_ids))
    decision_indices = torch.tensor(
        [position - 1 for position in decision_positions],
        device=device,
        dtype=torch.long,
    )
    distance_bins = [
        _distance_bin(position, mismatch_position)
        for position in decision_positions
    ]

    hidden_by_layer: dict[str, dict[str, Any]] = {}
    for layer_index, hidden in enumerate(output.hidden_states):
        selected = hidden[:, condition_length:].index_select(1, decision_indices)
        metrics = _pair_metrics(selected[0], selected[1])
        layer_bins: dict[str, list[int]] = defaultdict(list)
        for row_index, name in enumerate(distance_bins):
            layer_bins[name].append(row_index)
        hidden_by_layer[str(layer_index)] = {
            name: _summarize_pair_metrics(
                {
                    key: value[
                        torch.tensor(indices, device=value.device, dtype=torch.long)
                    ]
                    for key, value in metrics.items()
                }
            )
            | {"positions": len(indices)}
            for name, indices in sorted(layer_bins.items())
        }

    logit_rows: list[dict[str, Any]] = []
    for position, distance_name in zip(
        decision_positions,
        distance_bins,
        strict=True,
    ):
        prefix = np.asarray(generated_ids[:position], dtype=np.int64)
        possible = sorted(
            int(value)
            for value in tokenizer.next_posible_token(ids=prefix)
        )
        possible_ids = torch.tensor(possible, device=device, dtype=torch.long)
        left_logits = token_logits[0, position - 1].index_select(0, possible_ids)
        right_logits = token_logits[1, position - 1].index_select(0, possible_ids)
        row: dict[str, Any] = {
            "position": int(position),
            "distance_bin": distance_name,
            **_distribution_metrics(left_logits, right_logits),
        }

        grouped = _possible_groups(tokenizer, set(possible))
        if len(grouped) > 1:
            group_ids = sorted(grouped)
            left_group_logits = torch.stack(
                [
                    torch.logsumexp(
                        token_logits[0, position - 1].index_select(
                            0,
                            torch.tensor(
                                grouped[group],
                                device=device,
                                dtype=torch.long,
                            ),
                        ).float(),
                        dim=0,
                    )
                    for group in group_ids
                ]
            )
            right_group_logits = torch.stack(
                [
                    torch.logsumexp(
                        token_logits[1, position - 1].index_select(
                            0,
                            torch.tensor(
                                grouped[group],
                                device=device,
                                dtype=torch.long,
                            ),
                        ).float(),
                        dim=0,
                    )
                    for group in group_ids
                ]
            )
            group_metrics = _distribution_metrics(
                left_group_logits,
                right_group_logits,
            )
            row.update(
                {
                    f"group_{key}": value
                    for key, value in group_metrics.items()
                }
            )
        if int(tokenizer.eos) in possible:
            left_probability = F.softmax(left_logits.float(), dim=-1)
            right_probability = F.softmax(right_logits.float(), dim=-1)
            eos_offset = possible.index(int(tokenizer.eos))
            row["correct_eos_probability"] = float(
                left_probability[eos_offset].item()
            )
            row["swapped_eos_probability"] = float(
                right_probability[eos_offset].item()
            )
            row["absolute_eos_probability_delta"] = abs(
                row["swapped_eos_probability"]
                - row["correct_eos_probability"]
            )
        logit_rows.append(row)

    logit_bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in logit_rows:
        logit_bins[row["distance_bin"]].append(row)

    condition_metrics = _summarize_pair_metrics(
        _pair_metrics(condition[0], condition[1])
    )
    del output, prompt, token_embeds, token_logits, final_token_hidden
    return {
        "first_mismatch": mismatch,
        "generated_token_count": len(generated_ids) - 2,
        "has_eos": has_eos,
        "condition_pair": condition_metrics,
        "hidden_by_layer": hidden_by_layer,
        "logits_by_distance": {
            name: _aggregate_metric_rows(rows)
            for name, rows in sorted(logit_bins.items())
        },
    }


def _aggregate_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "hitmax": [row for row in rows if row["hitmax"]],
        "success": [row for row in rows if not row["hitmax"]],
    }
    output: dict[str, Any] = {}
    for group_name, group_rows in groups.items():
        layer_names = sorted(
            {
                layer
                for row in group_rows
                for layer in row["trace"]["hidden_by_layer"]
            },
            key=int,
        )
        distance_names = sorted(
            {
                distance
                for row in group_rows
                for distance in row["trace"]["logits_by_distance"]
            }
        )
        output[group_name] = {
            "row_count": len(group_rows),
            "condition_pair": {
                key: _mean(
                    [row["trace"]["condition_pair"].get(key) for row in group_rows]
                )
                for key in (
                    "relative_l2",
                    "cosine",
                    "delta_rms",
                    "hidden_rms",
                )
            },
            "hidden_by_layer": {
                layer: {
                    distance: {
                        metric: _mean(
                            [
                                row["trace"]["hidden_by_layer"]
                                .get(layer, {})
                                .get(distance, {})
                                .get(metric)
                                for row in group_rows
                            ]
                        )
                        for metric in (
                            "relative_l2",
                            "cosine",
                            "delta_rms",
                            "hidden_rms",
                        )
                    }
                    for distance in distance_names
                    if any(
                        distance
                        in row["trace"]["hidden_by_layer"].get(layer, {})
                        for row in group_rows
                    )
                }
                for layer in layer_names
            },
            "logits_by_distance": {
                distance: {
                    metric: _mean(
                        [
                            row["trace"]["logits_by_distance"]
                            .get(distance, {})
                            .get(metric)
                            for row in group_rows
                        ]
                    )
                    for metric in (
                        "js_divergence_nats",
                        "total_variation",
                        "top1_agreement",
                        "group_js_divergence_nats",
                        "group_total_variation",
                        "group_top1_agreement",
                        "absolute_eos_probability_delta",
                    )
                }
                for distance in distance_names
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-result", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--limit", type=int, default=52)
    parser.add_argument("--hitmax-limit", type=int, default=10)
    parser.add_argument("--success-limit", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scripts_dir = args.model_root / "rigweave" / "scripts"
    source_dir = args.model_root / "rigweave" / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))

    from eval_dynamic_rig_ce import (
        CHECKPOINT_DEFAULTS,
        _build_dynamic_model,
        apply_checkpoint_eval_defaults,
    )
    from eval_dynamic_rig_generation import _count_prefix_structure
    from rigweave.dynamic_rig.data import (
        DynamicRigManifestDataset,
        dynamic_rig_collate,
    )
    from train_dynamic_rig import build_tokenizer

    generation_summary, generation_rows = _load_generation_rows(
        args.generation_result
    )
    if len(generation_rows) != int(args.limit):
        raise ValueError(
            f"generation result has {len(generation_rows)} rows, "
            f"expected {args.limit}"
        )

    namespace = argparse.Namespace(
        checkpoint=args.checkpoint,
        tokenizer_config=args.tokenizer_config,
        model_config=args.model_config,
        unirig_checkpoint=args.unirig_checkpoint,
    )
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(namespace, name):
            setattr(namespace, name, None)
    apply_checkpoint_eval_defaults(namespace)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args.tokenizer_config)
    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=namespace.frames,
        limit=args.limit,
        random_query=False,
        seed=args.seed,
        motion_fps_ratio=namespace.motion_fps_ratio,
        motion_vertex_samples=namespace.motion_vertex_samples,
        target_active_skin_only=namespace.target_active_skin_only,
        active_skin_threshold=namespace.active_skin_threshold,
        target_start_policy=namespace.target_start_policy,
        target_root_policy=namespace.target_root_policy,
        input_space_policy=namespace.input_space_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )
    model = _build_dynamic_model(namespace, tokenizer, device)
    model.eval()

    condition_cache: list[torch.Tensor] = []
    target_ids_cache: list[list[int]] = []
    target_joint_counts: list[int] = []
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for index, (batch, generation_row) in enumerate(
            zip(loader, generation_rows, strict=True)
        ):
            batch = _move_batch(batch, device)
            if str(batch["path"][0]) != str(generation_row["path"]):
                raise ValueError(f"row {index} path mismatch")
            selected_frames = (
                batch["selected_frames"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )
            if selected_frames != generation_row["selected_frames"]:
                raise ValueError(f"row {index} selected-frame mismatch")
            target_ids = (
                batch["input_ids"][0][batch["attention_mask"][0].bool()]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )
            if target_ids != generation_row["target_ids"]:
                raise ValueError(f"row {index} target-token mismatch")
            _seed_all(args.seed + index, device)
            references = model.sample_references(batch)
            condition = model.build_condition(batch, refs=references)
            condition_cache.append(
                condition.detach().to(device="cpu", dtype=torch.bfloat16)
            )
            target_ids_cache.append(target_ids)
            target_joint_counts.append(
                int(_count_prefix_structure(tokenizer, target_ids)[0])
            )
            print(
                f"[condition] {index + 1}/{len(generation_rows)}",
                flush=True,
            )

        hitmax_indices = [
            index
            for index, row in enumerate(generation_rows)
            if bool(row["dynamic"].get("hit_max_without_eos"))
        ][: int(args.hitmax_limit)]
        success_indices = [
            index
            for index, row in enumerate(generation_rows)
            if not bool(row["dynamic"].get("hit_max_without_eos"))
        ]
        selected_success = _select_success_controls(
            hitmax_indices,
            success_indices,
            target_joint_counts,
            int(args.success_limit),
        )
        selected_indices = hitmax_indices + selected_success

        rows: list[dict[str, Any]] = []
        for selected_offset, index in enumerate(selected_indices):
            generated = [
                int(value)
                for value in generation_rows[index]["dynamic"]["generated_ids"]
            ]
            trace = _trace_row(
                model,
                tokenizer,
                condition_cache[index],
                condition_cache[(index + 1) % len(condition_cache)],
                generated,
                target_ids_cache[index],
                device=device,
            )
            row = {
                "index": int(index),
                "path": generation_rows[index]["path"],
                "hitmax": bool(
                    generation_rows[index]["dynamic"].get(
                        "hit_max_without_eos"
                    )
                ),
                "target_joint_count": int(target_joint_counts[index]),
                "swapped_condition_index": int(
                    (index + 1) % len(condition_cache)
                ),
                "trace": trace,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "trace": f"{selected_offset + 1}/{len(selected_indices)}",
                        "index": index,
                        "hitmax": row["hitmax"],
                        "target_joint_count": row["target_joint_count"],
                        "first_mismatch": trace["first_mismatch"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "generation_result": str(args.generation_result),
        "generation_summary": generation_summary,
        "seed": int(args.seed),
        "selection": {
            "hitmax_indices": hitmax_indices,
            "success_indices": selected_success,
            "success_matching": "greedy nearest target joint count",
        },
        "aggregate": _aggregate_traces(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    "row_count": values["row_count"],
                    "condition_pair": values["condition_pair"],
                    "final_layer": values["hidden_by_layer"][
                        str(max(map(int, values["hidden_by_layer"])))
                    ],
                    "logits_by_distance": values["logits_by_distance"],
                }
                for name, values in report["aggregate"].items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
