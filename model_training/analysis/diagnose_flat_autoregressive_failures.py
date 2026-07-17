#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader


def _stratum(index: int) -> str:
    if index < 16:
        return "train_low"
    if index < 32:
        return "train_common"
    return "valid_low"


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_generation_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows or "dynamic" not in rows[0]:
        raise ValueError(f"{path} is not a flat dynamic generation result")
    return payload["summary"], rows


def _teacher_logits(
    model: torch.nn.Module,
    batch: dict[str, Any],
    cond: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    cond = cond.to(dtype=model.transformer.dtype)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    input_embeds = model.token_inputs_embeds(input_ids, attention_mask)
    prompt = torch.cat([cond, input_embeds], dim=1)
    full_attention = pad(attention_mask, (cond.shape[1], 0, 0, 0), value=1)
    need_hidden = bool(getattr(model, "uses_action_group_bias", False))
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=full_attention,
        use_cache=False,
        output_hidden_states=need_hidden,
    )
    logits = output.logits[:, cond.shape[1] :]
    token_hidden = output.hidden_states[-1][:, cond.shape[1] :] if need_hidden else None
    logits = model.apply_action_group_bias(logits, token_hidden, cond)
    logits = logits[:, :-1].float()
    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = -100
    return logits, labels, input_ids


def _token_group(tokenizer: Any, token: int) -> int:
    if token == int(tokenizer.eos):
        return 0
    if token == int(tokenizer.token_id_branch):
        return 1
    if 0 <= token < int(tokenizer.num_discrete):
        return 2
    return 3


def _group_name(group: int) -> str:
    return ("eos", "branch", "coordinate", "part_or_class")[group]


def _possible_groups(tokenizer: Any, possible: set[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for token in sorted(possible):
        groups[_token_group(tokenizer, int(token))].append(int(token))
    return dict(groups)


def _target_role(tokenizer: Any, target: int, coordinate_ordinal: int) -> str:
    group = _token_group(tokenizer, target)
    if group == 0:
        return "eos"
    if group == 1:
        return "branch"
    if group == 2:
        if coordinate_ordinal < 3:
            return ("joint0_x", "joint0_y", "joint0_z")[coordinate_ordinal]
        return "coordinate"
    return "part_or_class"


def _rank(logits: torch.Tensor, target: int) -> int:
    return int((logits > logits[target]).sum().item() + 1)


def _masked_target_stats(
    logits: torch.Tensor,
    possible: set[int],
    target: int,
) -> dict[str, float | int | bool]:
    if target not in possible:
        return {
            "target_possible": False,
            "target_nll": float("inf"),
            "target_rank": -1,
            "target_probability": 0.0,
            "pred_id": -1,
        }
    possible_ids = torch.tensor(sorted(possible), device=logits.device, dtype=torch.long)
    possible_logits = logits.index_select(0, possible_ids).float()
    target_logit = logits[target].float()
    nll = torch.logsumexp(possible_logits, dim=0) - target_logit
    pred_offset = int(possible_logits.argmax().item())
    return {
        "target_possible": True,
        "target_nll": float(nll.item()),
        "target_rank": int((possible_logits > target_logit).sum().item() + 1),
        "target_probability": float(torch.exp(-nll).item()),
        "pred_id": int(possible_ids[pred_offset].item()),
    }


def _group_stats(
    tokenizer: Any,
    logits: torch.Tensor,
    possible: set[int],
    target_token: int | None,
) -> dict[str, Any]:
    grouped = _possible_groups(tokenizer, possible)
    group_ids = sorted(grouped)
    scores = torch.stack(
        [
            torch.logsumexp(
                logits[torch.tensor(grouped[group], device=logits.device, dtype=torch.long)].float(),
                dim=0,
            )
            for group in group_ids
        ]
    )
    probabilities = torch.softmax(scores, dim=0)
    predicted_offset = int(scores.argmax().item())
    predicted_group = int(group_ids[predicted_offset])
    output: dict[str, Any] = {
        "possible_groups": [_group_name(group) for group in group_ids],
        "predicted_group": _group_name(predicted_group),
        "group_probabilities": {
            _group_name(group): float(probabilities[offset].item())
            for offset, group in enumerate(group_ids)
        },
    }
    if target_token is not None:
        target_group = _token_group(tokenizer, int(target_token))
        output["target_group"] = _group_name(target_group)
        if target_group in group_ids:
            target_offset = group_ids.index(target_group)
            target_score = scores[target_offset]
            output["target_group_rank"] = int((scores > target_score).sum().item() + 1)
            output["target_group_nll"] = float(
                (torch.logsumexp(scores, dim=0) - target_score).item()
            )
            output["target_group_correct"] = int(predicted_group == target_group)
        else:
            output["target_group_rank"] = -1
            output["target_group_nll"] = float("inf")
            output["target_group_correct"] = 0
    return output


def _teacher_report(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    cond: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.LongTensor]:
    logits, labels, input_ids = _teacher_logits(model, batch, cond)
    valid_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
    positions: list[dict[str, Any]] = []
    coordinate_ordinal = 0
    for position in valid_positions:
        target = int(labels[0, position].item())
        prefix = input_ids[0, : position + 1].detach().cpu().numpy().astype(np.int64)
        possible = set(int(value) for value in tokenizer.next_posible_token(ids=prefix))
        row_logits = logits[0, position]
        full_nll = torch.logsumexp(row_logits, dim=0) - row_logits[target]
        masked = _masked_target_stats(row_logits, possible, target)
        role = _target_role(tokenizer, target, coordinate_ordinal)
        if _token_group(tokenizer, target) == 2:
            coordinate_ordinal += 1
        group_report = _group_stats(tokenizer, row_logits, possible, target)
        positions.append(
            {
                "position": int(position + 1),
                "target_id": target,
                "role": role,
                "full_pred_id": int(row_logits.argmax().item()),
                "full_correct": int(int(row_logits.argmax().item()) == target),
                "full_target_nll": float(full_nll.item()),
                "full_target_rank": _rank(row_logits, target),
                **{f"grammar_{key}": value for key, value in masked.items()},
                **group_report,
            }
        )

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float | int | None]:
        if not selected:
            return {
                "count": 0,
                "full_accuracy": None,
                "grammar_accuracy": None,
                "full_target_nll": None,
                "grammar_target_nll": None,
                "target_group_accuracy": None,
            }
        group_rows = [row for row in selected if len(row["possible_groups"]) > 1]
        return {
            "count": len(selected),
            "full_accuracy": float(np.mean([row["full_correct"] for row in selected])),
            "grammar_accuracy": float(
                np.mean([int(row["grammar_pred_id"] == row["target_id"]) for row in selected])
            ),
            "full_target_nll": float(np.mean([row["full_target_nll"] for row in selected])),
            "grammar_target_nll": float(
                np.mean([row["grammar_target_nll"] for row in selected])
            ),
            "target_group_accuracy": (
                float(np.mean([row["target_group_correct"] for row in group_rows]))
                if group_rows
                else None
            ),
        }

    role_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positions:
        role_groups[row["role"]].append(row)
        if row["role"].startswith("joint0_"):
            role_groups["joint0_coord"].append(row)
        if row["role"] in {"joint0_x", "joint0_y", "joint0_z", "coordinate"}:
            role_groups["all_coord"].append(row)
        if len(row["possible_groups"]) > 1:
            role_groups["all_decisions"].append(row)

    first_full_wrong = next((row for row in positions if not row["full_correct"]), None)
    first_grammar_wrong = next(
        (row for row in positions if row["grammar_pred_id"] != row["target_id"]),
        None,
    )
    return (
        {
            "all": summarize(positions),
            "by_role": {
                role: summarize(role_rows)
                for role, role_rows in sorted(role_groups.items())
            },
            "first_full_wrong": first_full_wrong,
            "first_grammar_wrong": first_grammar_wrong,
            "positions": positions,
        },
        logits,
        labels,
    )


def _condition_likelihood(
    tokenizer: Any,
    labels: torch.LongTensor,
    logits_by_condition: dict[str, torch.Tensor],
) -> dict[str, Any]:
    valid = labels[0] != -100
    target_values = labels[0]
    coordinate_mask = valid & (target_values >= 0) & (target_values < int(tokenizer.num_discrete))
    coordinate_positions = coordinate_mask.nonzero(as_tuple=False).flatten()
    joint0_mask = torch.zeros_like(valid)
    joint0_mask[coordinate_positions[:3]] = True
    eos_mask = valid & (target_values == int(tokenizer.eos))
    masks = {
        "all": valid,
        "all_coord": coordinate_mask,
        "joint0_coord": joint0_mask,
        "eos": eos_mask,
    }
    report: dict[str, Any] = {}
    for role, mask in masks.items():
        if not bool(mask.any()):
            continue
        target = target_values[mask]
        condition_rows: dict[str, Any] = {}
        for name, logits in logits_by_condition.items():
            selected = logits[0, mask]
            nll = torch.nn.functional.cross_entropy(selected, target, reduction="none")
            target_logits = selected.gather(1, target[:, None]).squeeze(1)
            condition_rows[name] = {
                "nll": float(nll.mean().item()),
                "target_rank": float(
                    ((selected > target_logits[:, None]).sum(dim=1) + 1).float().mean().item()
                ),
                "top1_accuracy": float((selected.argmax(dim=1) == target).float().mean().item()),
            }
        report[role] = {
            "positions": int(mask.sum().item()),
            **condition_rows,
            "swapped_minus_correct_nll": float(
                condition_rows["swapped"]["nll"] - condition_rows["correct"]["nll"]
            ),
            "zero_minus_correct_nll": float(
                condition_rows["zero"]["nll"] - condition_rows["correct"]["nll"]
            ),
        }
    return report


def _first_mismatch(
    tokenizer: Any,
    target_ids: list[int],
    generated_ids: list[int],
) -> dict[str, Any]:
    compare_count = min(len(target_ids), len(generated_ids))
    mismatch = next(
        (index for index in range(compare_count) if int(target_ids[index]) != int(generated_ids[index])),
        compare_count if len(target_ids) != len(generated_ids) else -1,
    )
    if mismatch < 0:
        return {
            "position": -1,
            "target_id": None,
            "generated_id": None,
            "target_role": None,
        }
    coordinate_ordinal = sum(
        0 <= int(token) < int(tokenizer.num_discrete)
        for token in target_ids[:mismatch]
    )
    target = int(target_ids[mismatch]) if mismatch < len(target_ids) else None
    generated = int(generated_ids[mismatch]) if mismatch < len(generated_ids) else None
    role = (
        _target_role(tokenizer, target, coordinate_ordinal)
        if target is not None
        else "target_exhausted"
    )
    return {
        "position": mismatch,
        "target_id": target,
        "generated_id": generated,
        "target_role": role,
    }


@torch.no_grad()
def _manual_free_trace(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    cond: torch.Tensor,
    target_ids: list[int],
    saved_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    from eval_dynamic_rig_generation import _count_prefix_structure

    cond = cond.to(dtype=model.transformer.dtype)
    start_tokens = torch.tensor(
        [tokenizer.bos, tokenizer.cls_name_to_token(batch["cls"][0])],
        device=cond.device,
        dtype=torch.long,
    )
    start_mask = torch.ones((1, start_tokens.numel()), device=cond.device, dtype=torch.long)
    start_embeds = model.token_inputs_embeds(start_tokens.unsqueeze(0), start_mask)
    prompt = torch.cat([cond, start_embeds], dim=1)
    attention_mask = torch.ones((1, prompt.shape[1]), device=cond.device, dtype=torch.long)
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=bool(getattr(model, "uses_action_group_bias", False)),
    )
    past = output.past_key_values
    next_logits = output.logits[:, -1, :].float()
    if bool(getattr(model, "uses_action_group_bias", False)):
        next_logits = model.apply_action_group_bias_row(
            next_logits,
            output.hidden_states[-1][:, -1],
            cond,
        )

    generated: list[int] = []
    events: list[dict[str, Any]] = []
    first_saved_divergence: dict[str, Any] | None = None
    for step in range(max_new_tokens):
        prefix_ids = start_tokens.detach().cpu().tolist() + generated
        possible = set(
            int(value)
            for value in tokenizer.next_posible_token(ids=np.asarray(prefix_ids, dtype=np.int64))
        )
        possible_ids = torch.tensor(sorted(possible), device=cond.device, dtype=torch.long)
        possible_logits = next_logits[0].index_select(0, possible_ids)
        selected_offset = int(possible_logits.argmax().item())
        selected = int(possible_ids[selected_offset].item())
        groups = _possible_groups(tokenizer, possible)
        target_position = len(prefix_ids)
        target = int(target_ids[target_position]) if target_position < len(target_ids) else None
        saved = int(saved_ids[target_position]) if target_position < len(saved_ids) else None
        if first_saved_divergence is None and selected != saved:
            top_count = min(5, int(possible_ids.numel()))
            top_logits, top_offsets = possible_logits.topk(top_count)
            first_saved_divergence = {
                "step": step,
                "position": target_position,
                "manual_id": selected,
                "saved_id": saved,
                "manual_logit": float(next_logits[0, selected].item()),
                "saved_logit": (
                    float(next_logits[0, saved].item())
                    if saved is not None
                    else None
                ),
                "manual_minus_saved_logit": (
                    float((next_logits[0, selected] - next_logits[0, saved]).item())
                    if saved is not None
                    else None
                ),
                "top_possible": [
                    {
                        "id": int(possible_ids[offset].item()),
                        "logit": float(logit.item()),
                    }
                    for logit, offset in zip(top_logits, top_offsets, strict=True)
                ],
            }
        if len(groups) > 1 or int(tokenizer.eos) in possible:
            joint_count, branch_count = _count_prefix_structure(tokenizer, prefix_ids)
            group_report = _group_stats(tokenizer, next_logits[0], possible, target)
            events.append(
                {
                    "step": step,
                    "position": target_position,
                    "joint_count_before": int(joint_count),
                    "branch_count_before": int(branch_count),
                    "selected_id": selected,
                    "selected_group": _group_name(_token_group(tokenizer, selected)),
                    "target_id": target,
                    **group_report,
                }
            )
        generated.append(selected)
        if selected == int(tokenizer.eos):
            break
        next_token = torch.tensor([[selected]], device=cond.device, dtype=torch.long)
        attention_mask = torch.ones(
            (1, prompt.shape[1] + len(generated)),
            device=cond.device,
            dtype=torch.long,
        )
        output = model.transformer(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=bool(getattr(model, "uses_action_group_bias", False)),
        )
        past = output.past_key_values
        next_logits = output.logits[:, -1, :].float()
        if bool(getattr(model, "uses_action_group_bias", False)):
            next_logits = model.apply_action_group_bias_row(
                next_logits,
                output.hidden_states[-1][:, -1],
                cond,
            )

    ids = start_tokens.detach().cpu().tolist() + generated
    eos_events = [
        event
        for event in events
        if "eos" in event["group_probabilities"]
    ]
    target_joint_count, _target_branch_count = _count_prefix_structure(tokenizer, target_ids)
    at_target_count = [
        event
        for event in eos_events
        if int(event["joint_count_before"]) == int(target_joint_count)
    ]
    return {
        "generated_ids": ids,
        "has_eos": bool(ids and ids[-1] == int(tokenizer.eos)),
        "steps": len(generated),
        "first_saved_divergence": first_saved_divergence,
        "decision_events": events,
        "eos_decision_count": len(eos_events),
        "max_eos_group_probability": (
            max(event["group_probabilities"]["eos"] for event in eos_events)
            if eos_events
            else None
        ),
        "max_eos_probability_at_target_joint_count": (
            max(event["group_probabilities"]["eos"] for event in at_target_count)
            if at_target_count
            else None
        ),
    }


def _mean(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["stratum"]].append(row)
        groups["hitmax" if row["hitmax"] else "success"].append(row)

    output: dict[str, Any] = {}
    for name, group_rows in groups.items():
        role_counts = Counter(
            row["first_free_mismatch"]["target_role"]
            for row in group_rows
        )
        output[name] = {
            "row_count": len(group_rows),
            "hitmax_count": sum(bool(row["hitmax"]) for row in group_rows),
            "teacher_token_accuracy": _mean(
                [row["teacher"]["all"]["grammar_accuracy"] for row in group_rows]
            ),
            "teacher_joint0_coord_accuracy": _mean(
                [
                    row["teacher"]["by_role"]["joint0_coord"]["grammar_accuracy"]
                    for row in group_rows
                ]
            ),
            "teacher_joint0_coord_nll": _mean(
                [
                    row["teacher"]["by_role"]["joint0_coord"]["grammar_target_nll"]
                    for row in group_rows
                ]
            ),
            "teacher_eos_accuracy": _mean(
                [row["teacher"]["by_role"]["eos"]["grammar_accuracy"] for row in group_rows]
            ),
            "teacher_eos_nll": _mean(
                [row["teacher"]["by_role"]["eos"]["grammar_target_nll"] for row in group_rows]
            ),
            "condition_joint0_swapped_minus_correct_nll": _mean(
                [
                    row["condition_likelihood"]["joint0_coord"]["swapped_minus_correct_nll"]
                    for row in group_rows
                ]
            ),
            "condition_joint0_zero_minus_correct_nll": _mean(
                [
                    row["condition_likelihood"]["joint0_coord"]["zero_minus_correct_nll"]
                    for row in group_rows
                ]
            ),
            "first_free_mismatch_roles": dict(role_counts),
            "free_max_eos_group_probability": _mean(
                [row["manual_trace"]["max_eos_group_probability"] for row in group_rows]
            ),
            "free_max_eos_probability_at_target_joint_count": _mean(
                [
                    row["manual_trace"]["max_eos_probability_at_target_joint_count"]
                    for row in group_rows
                ]
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace flat UniRig teacher-forced and free-generation failures.")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-result", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--limit", type=int, default=52)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scripts_dir = args.model_root / "rigweave" / "scripts"
    source_dir = args.model_root / "rigweave" / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))

    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model, apply_checkpoint_eval_defaults
    from eval_dynamic_rig_generation import _count_prefix_structure
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
    from train_dynamic_rig import build_tokenizer

    generation_summary, generation_rows = _load_generation_rows(args.generation_result)
    if len(generation_rows) != args.limit:
        raise ValueError(
            f"generation result has {len(generation_rows)} rows, expected limit={args.limit}"
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

    rows: list[dict[str, Any]] = []
    condition_cache: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for index, (batch, generation_row) in enumerate(zip(loader, generation_rows, strict=True)):
            batch = _move_batch(batch, device)
            if str(batch["path"][0]) != str(generation_row["path"]):
                raise ValueError(
                    f"row {index} path mismatch: {batch['path'][0]} vs {generation_row['path']}"
                )
            selected = batch["selected_frames"][0].detach().cpu().numpy().astype(int).tolist()
            if selected != generation_row["selected_frames"]:
                raise ValueError(f"row {index} selected-frame mismatch")
            if int(selected[0]) != int(generation_row["query_frame"]):
                raise ValueError(f"row {index} query-frame mismatch")
            query_center = (
                batch["query_center"][0].detach().cpu().numpy().astype(np.float64)
            )
            saved_query_center = np.asarray(
                generation_row["query_center"],
                dtype=np.float64,
            )
            if not np.allclose(
                query_center,
                saved_query_center,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise ValueError(f"row {index} query-center mismatch")
            query_scale = float(batch["query_scale"][0].detach().cpu())
            saved_query_scale = float(generation_row["query_scale"])
            if not np.isclose(
                query_scale,
                saved_query_scale,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise ValueError(f"row {index} query-scale mismatch")
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
            refs = model.sample_references(batch)
            cond = model.build_condition(batch, refs=refs)
            condition_cache.append(cond.detach().cpu())
            teacher, _logits, _labels = _teacher_report(model, tokenizer, batch, cond)
            expected_generated = generation_row["dynamic"]["generated_ids"]
            manual = _manual_free_trace(
                model,
                tokenizer,
                batch,
                cond,
                target_ids,
                expected_generated,
                int(generation_row["max_new_tokens"]),
            )
            manual_ids = manual.pop("generated_ids")
            manual["matches_saved_generation"] = bool(manual_ids == expected_generated)
            manual["first_saved_mismatch"] = _first_mismatch(
                tokenizer,
                expected_generated,
                manual_ids,
            )
            generated_block = generation_row["dynamic"]
            hitmax = bool(generated_block.get("hit_max_without_eos"))
            target_joint_count, target_branch_count = _count_prefix_structure(tokenizer, target_ids)
            row = {
                "index": index,
                "path": batch["path"][0],
                "stratum": _stratum(index),
                "hitmax": hitmax,
                "target_joint_count": int(target_joint_count),
                "target_branch_count": int(target_branch_count),
                "query_frame": int(selected[0]),
                "query_center": query_center.tolist(),
                "query_scale": query_scale,
                "teacher": teacher,
                "first_free_mismatch": _first_mismatch(
                    tokenizer,
                    target_ids,
                    expected_generated,
                ),
                "manual_trace": manual,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "index": index,
                        "hitmax": hitmax,
                        "target_joint_count": int(target_joint_count),
                        "first_free_mismatch": row["first_free_mismatch"],
                        "teacher_joint0": teacher["by_role"]["joint0_coord"],
                        "teacher_eos": teacher["by_role"]["eos"],
                        "trace_matches": manual["matches_saved_generation"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not manual["matches_saved_generation"]:
                raise RuntimeError(
                    "manual greedy trace differs from the saved generation at "
                    f"row {index}: {manual['first_saved_divergence']}"
                )

        for index, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            correct = condition_cache[index].to(device)
            swapped = condition_cache[(index + 1) % len(condition_cache)].to(device)
            zero = torch.zeros_like(correct)
            logits_by_condition: dict[str, torch.Tensor] = {}
            labels = None
            for name, cond in (("correct", correct), ("swapped", swapped), ("zero", zero)):
                logits, current_labels, _input_ids = _teacher_logits(model, batch, cond)
                logits_by_condition[name] = logits
                if labels is None:
                    labels = current_labels
                elif not torch.equal(labels, current_labels):
                    raise ValueError("condition intervention changed labels")
            assert labels is not None
            rows[index]["condition_likelihood"] = _condition_likelihood(
                tokenizer,
                labels,
                logits_by_condition,
            )
            print(
                json.dumps(
                    {
                        "event": "condition_likelihood",
                        "index": index,
                        "joint0": rows[index]["condition_likelihood"]["joint0_coord"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "generation_result": str(args.generation_result),
        "manifest": str(args.manifest),
        "seed": int(args.seed),
        "generation_summary": generation_summary,
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
