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
    branch_mask = valid & (target_values == int(tokenizer.token_id_branch))
    eos_mask = valid & (target_values == int(tokenizer.eos))
    masks = {
        "all": valid,
        "all_coord": coordinate_mask,
        "joint0_coord": joint0_mask,
        "branch": branch_mask,
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


def _condition_group_likelihood(
    tokenizer: Any,
    labels: torch.LongTensor,
    input_ids: torch.LongTensor,
    logits_by_condition: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare structural action likelihoods under fixed GT prefixes."""

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
    for position in valid_positions:
        target = int(labels[0, position].item())
        prefix = input_ids[0, : position + 1].detach().cpu().numpy().astype(np.int64)
        possible = set(int(value) for value in tokenizer.next_posible_token(ids=prefix))
        grouped = _possible_groups(tokenizer, possible)
        if len(grouped) <= 1:
            continue
        target_group = _token_group(tokenizer, target)
        if target_group not in grouped:
            raise ValueError(
                f"target group {_group_name(target_group)} is not grammar-valid "
                f"at GT position {position + 1}"
            )
        group_ids = sorted(grouped)
        target_offset = group_ids.index(target_group)
        condition_rows: dict[str, Any] = {}
        for name, logits in logits_by_condition.items():
            row_logits = logits[0, position]
            scores = torch.stack(
                [
                    torch.logsumexp(
                        row_logits[
                            torch.tensor(
                                grouped[group],
                                device=row_logits.device,
                                dtype=torch.long,
                            )
                        ].float(),
                        dim=0,
                    )
                    for group in group_ids
                ]
            )
            target_score = scores[target_offset]
            condition_rows[name] = {
                "nll": float((torch.logsumexp(scores, dim=0) - target_score).item()),
                "top1_accuracy": float(int(int(scores.argmax().item()) == target_offset)),
            }
        row = {
            "position": int(position + 1),
            "target_group": _group_name(target_group),
            **condition_rows,
            "swapped_minus_correct_nll": float(
                condition_rows["swapped"]["nll"] - condition_rows["correct"]["nll"]
            ),
            "zero_minus_correct_nll": float(
                condition_rows["zero"]["nll"] - condition_rows["correct"]["nll"]
            ),
        }
        rows["all_decisions"].append(row)
        rows[_group_name(target_group)].append(row)

    report: dict[str, Any] = {}
    for role, role_rows in sorted(rows.items()):
        report[role] = {
            "positions": len(role_rows),
            "correct": {
                "nll": _mean([row["correct"]["nll"] for row in role_rows]),
                "top1_accuracy": _mean(
                    [row["correct"]["top1_accuracy"] for row in role_rows]
                ),
            },
            "swapped": {
                "nll": _mean([row["swapped"]["nll"] for row in role_rows]),
                "top1_accuracy": _mean(
                    [row["swapped"]["top1_accuracy"] for row in role_rows]
                ),
            },
            "zero": {
                "nll": _mean([row["zero"]["nll"] for row in role_rows]),
                "top1_accuracy": _mean(
                    [row["zero"]["top1_accuracy"] for row in role_rows]
                ),
            },
            "swapped_minus_correct_nll": _mean(
                [row["swapped_minus_correct_nll"] for row in role_rows]
            ),
            "zero_minus_correct_nll": _mean(
                [row["zero_minus_correct_nll"] for row in role_rows]
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


def _repetition_report(generated_ids: list[int]) -> dict[str, Any]:
    tokens = [int(value) for value in generated_ids[2:]]
    ngram_unique_ratio: dict[str, float | None] = {}
    for width in (3, 6, 12, 24):
        count = len(tokens) - width + 1
        ngram_unique_ratio[str(width)] = (
            float(
                len(
                    {
                        tuple(tokens[start : start + width])
                        for start in range(count)
                    }
                )
                / count
            )
            if count > 0
            else None
        )

    terminal_same_token_run = 0
    if tokens:
        terminal_token = tokens[-1]
        for token in reversed(tokens):
            if token != terminal_token:
                break
            terminal_same_token_run += 1

    best_period = None
    best_coverage = 0
    for period in range(1, min(257, len(tokens))):
        matching = 0
        for position in range(len(tokens) - 1, period - 1, -1):
            if tokens[position] != tokens[position - period]:
                break
            matching += 1
        if matching < period:
            continue
        coverage = matching + period if matching else 0
        if coverage > best_coverage:
            best_period = period
            best_coverage = coverage
    return {
        "generated_token_count": len(tokens),
        "ngram_unique_ratio": ngram_unique_ratio,
        "terminal_same_token_run": int(terminal_same_token_run),
        "best_suffix_period": best_period,
        "best_suffix_periodic_token_count": int(best_coverage),
        "best_suffix_periodic_coverage": (
            float(best_coverage / len(tokens)) if tokens else None
        ),
    }


@torch.no_grad()
def _sequence_next_logits(
    model: torch.nn.Module,
    cond: torch.Tensor,
    ids: list[int],
) -> torch.Tensor:
    cond = cond.to(dtype=model.transformer.dtype)
    input_ids = torch.tensor(
        [ids],
        device=cond.device,
        dtype=torch.long,
    )
    token_attention = torch.ones_like(input_ids)
    token_embeds = model.token_inputs_embeds(input_ids, token_attention)
    prompt = torch.cat([cond, token_embeds], dim=1)
    attention_mask = torch.ones((1, prompt.shape[1]), device=cond.device, dtype=torch.long)
    need_hidden = bool(getattr(model, "uses_action_group_bias", False))
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=need_hidden,
    )
    logits = output.logits[:, cond.shape[1] :]
    hidden = output.hidden_states[-1][:, cond.shape[1] :] if need_hidden else None
    logits = model.apply_action_group_bias(logits, hidden, cond)
    return logits[0].float()


@torch.no_grad()
def _greedy_continue_from_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    cond: torch.Tensor,
    prefix_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    """Greedily continue a grammar-valid flat UniRig prefix."""

    if len(prefix_ids) < 2:
        raise ValueError("forced prefix must contain BOS and class tokens")
    for position in range(2, len(prefix_ids)):
        possible = set(
            int(value)
            for value in tokenizer.next_posible_token(
                ids=np.asarray(prefix_ids[:position], dtype=np.int64)
            )
        )
        if int(prefix_ids[position]) not in possible:
            raise ValueError(
                f"forced token {prefix_ids[position]} is not grammar-valid "
                f"at position {position}"
            )

    cond = cond.to(dtype=model.transformer.dtype)
    ids = [int(value) for value in prefix_ids]
    input_ids = torch.tensor([ids], device=cond.device, dtype=torch.long)
    input_attention = torch.ones_like(input_ids)
    token_embeds = model.token_inputs_embeds(input_ids, input_attention)
    prompt = torch.cat([cond, token_embeds], dim=1)
    attention_mask = torch.ones(
        (1, prompt.shape[1]),
        device=cond.device,
        dtype=torch.long,
    )
    need_hidden = bool(getattr(model, "uses_action_group_bias", False))
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=need_hidden,
    )
    past = output.past_key_values
    next_logits = output.logits[:, -1, :]
    next_hidden = output.hidden_states[-1][:, -1] if need_hidden else None
    next_logits = model.apply_action_group_bias_row(next_logits, next_hidden, cond)

    while len(ids) - 2 < int(max_new_tokens):
        possible = set(
            int(value)
            for value in tokenizer.next_posible_token(
                ids=np.asarray(ids, dtype=np.int64)
            )
        )
        possible_ids = torch.tensor(
            sorted(possible),
            device=next_logits.device,
            dtype=torch.long,
        )
        selected_offset = int(
            next_logits[0].index_select(0, possible_ids).argmax().item()
        )
        token = int(possible_ids[selected_offset].item())
        ids.append(token)
        if token == int(tokenizer.eos):
            break

        next_token_ids = torch.tensor(
            [[token]],
            device=cond.device,
            dtype=torch.long,
        )
        if bool(getattr(model, "use_grammar_state_embedding", False)):
            next_embed = model.next_token_embed_with_state(ids, cond.device)
        else:
            next_embed = model.token_inputs_embeds(
                next_token_ids,
                torch.ones_like(next_token_ids),
            )
        attention_mask = torch.ones(
            (1, cond.shape[1] + len(ids)),
            device=cond.device,
            dtype=torch.long,
        )
        output = model.transformer(
            inputs_embeds=next_embed,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=need_hidden,
        )
        past = output.past_key_values
        next_logits = output.logits[:, -1, :]
        next_hidden = output.hidden_states[-1][:, -1] if need_hidden else None
        next_logits = model.apply_action_group_bias_row(
            next_logits,
            next_hidden,
            cond,
        )
    return ids


def _target_prefix_through_joint_count(
    tokenizer: Any,
    target_ids: list[int],
    joint_count: int,
) -> list[int] | None:
    from eval_dynamic_rig_generation import _count_prefix_structure

    for end in range(2, len(target_ids) + 1):
        prefix = target_ids[:end]
        count, _ = _count_prefix_structure(tokenizer, prefix)
        if int(count) >= int(joint_count):
            if prefix[-1] == int(tokenizer.eos):
                return prefix[:-1]
            return prefix
    return None


def _compact_generation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    official = metrics.get("official", {})
    topology = metrics.get("topology", {})
    return {
        "detokenize_ok": bool(metrics.get("detokenize_ok", False)),
        "pred_joint_count": metrics.get("pred_joint_count"),
        "joint_count_error": metrics.get("joint_count_error"),
        "joint_count_abs_error": metrics.get("joint_count_abs_error"),
        "j2j": official.get("j2j"),
        "j2b": official.get("j2b"),
        "b2b": official.get("b2b"),
        "topology_edge_f1": topology.get("edge_f1"),
    }


def _prefix_repair_rollout(
    model: torch.nn.Module,
    tokenizer: Any,
    cond: torch.Tensor,
    target_ids: list[int],
    saved_ids: list[int],
    first_mismatch_position: int,
    max_new_tokens: int,
    target: Any,
    continuous_range: tuple[float, float],
) -> dict[str, Any]:
    from eval_dynamic_rig_generation import (
        _count_prefix_structure,
        _output_metrics,
    )

    reproduction_budget = min(32, int(max_new_tokens))
    reproduced = _greedy_continue_from_prefix(
        model,
        tokenizer,
        cond,
        saved_ids[:2],
        reproduction_budget,
    )
    expected_reproduction = saved_ids[: len(reproduced)]
    saved_reproduction_mismatch = next(
        (
            position
            for position in range(
                min(len(reproduced), len(expected_reproduction))
            )
            if int(reproduced[position])
            != int(expected_reproduction[position])
        ),
        (
            min(len(reproduced), len(expected_reproduction))
            if len(reproduced) != len(expected_reproduction)
            else -1
        ),
    )
    repeated = _greedy_continue_from_prefix(
        model,
        tokenizer,
        cond,
        saved_ids[:2],
        reproduction_budget,
    )

    if first_mismatch_position < 2:
        raise ValueError(
            f"invalid free-generation mismatch position {first_mismatch_position}"
        )
    target_joints_before_mismatch, _ = _count_prefix_structure(
        tokenizer,
        target_ids[:first_mismatch_position],
    )
    target_joint_through_mismatch = min(
        int(target_joints_before_mismatch) + 1,
        int(_count_prefix_structure(tokenizer, target_ids)[0]),
    )
    interventions: dict[str, list[int]] = {
        "repair_first_mismatch_token": (
            saved_ids[:first_mismatch_position]
            + [int(target_ids[first_mismatch_position])]
        ),
    }
    through_mismatch = _target_prefix_through_joint_count(
        tokenizer,
        target_ids,
        target_joint_through_mismatch,
    )
    if through_mismatch is not None:
        interventions["gt_through_first_mismatched_joint"] = through_mismatch
    for joint_count in (1, 2, 4):
        prefix = _target_prefix_through_joint_count(
            tokenizer,
            target_ids,
            joint_count,
        )
        if prefix is not None:
            interventions[f"gt_first_{joint_count}_joints"] = prefix

    reports: dict[str, Any] = {}
    generated_by_prefix: dict[tuple[int, ...], dict[str, Any]] = {}
    for name, prefix in interventions.items():
        prefix_key = tuple(int(value) for value in prefix)
        if prefix_key in generated_by_prefix:
            reports[name] = generated_by_prefix[prefix_key]
            continue
        generated = _greedy_continue_from_prefix(
            model,
            tokenizer,
            cond,
            prefix,
            max_new_tokens,
        )
        has_eos = bool(generated and generated[-1] == int(tokenizer.eos))
        generated_joint_count, generated_branch_count = _count_prefix_structure(
            tokenizer,
            generated,
        )
        metrics: dict[str, Any]
        if has_eos:
            try:
                prediction = tokenizer.detokenize(
                    np.asarray(generated, dtype=np.int64)
                )
                metrics = _compact_generation_metrics(
                    _output_metrics(
                        prediction,
                        target,
                        continuous_range,
                    )
                )
            except Exception as exc:
                metrics = {
                    "detokenize_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            metrics = _compact_generation_metrics(
                _output_metrics(None, target, continuous_range)
            )
        report = {
            "forced_prefix_length": len(prefix),
            "forced_generated_tokens": len(prefix) - 2,
            "max_new_tokens": int(max_new_tokens),
            "has_eos": has_eos,
            "reached_probe_budget_without_eos": bool(
                not has_eos and len(generated) - 2 >= int(max_new_tokens)
            ),
            "generated_new_tokens": len(generated) - 2,
            "generated_joint_count": int(generated_joint_count),
            "generated_branch_count": int(generated_branch_count),
            "generated_ids": generated,
            "metrics": metrics,
        }
        generated_by_prefix[prefix_key] = report
        reports[name] = report
    return {
        "baseline_reproduction_tokens": int(reproduction_budget),
        "saved_baseline_reproduction_exact": bool(
            saved_reproduction_mismatch < 0
        ),
        "saved_baseline_reproduction_first_mismatch": int(
            saved_reproduction_mismatch
        ),
        "same_process_reproduction_exact": bool(repeated == reproduced),
        "probe_max_new_tokens": int(max_new_tokens),
        "interventions": reports,
    }


@torch.no_grad()
def _saved_prefix_trace(
    model: torch.nn.Module,
    tokenizer: Any,
    cond: torch.Tensor,
    target_ids: list[int],
    saved_ids: list[int],
    target_joint_count: int,
) -> dict[str, Any]:
    from eval_dynamic_rig_generation import _count_prefix_structure

    if len(saved_ids) < 3:
        raise ValueError("saved generation does not contain a generated token")
    if saved_ids[:2] != target_ids[:2]:
        raise ValueError(
            f"saved generation prefix {saved_ids[:2]} differs from target prefix {target_ids[:2]}"
        )

    next_logits = _sequence_next_logits(model, cond, saved_ids)
    token_events: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for position in range(2, len(saved_ids)):
        prefix_ids = saved_ids[:position]
        possible = set(
            int(value)
            for value in tokenizer.next_posible_token(ids=np.asarray(prefix_ids, dtype=np.int64))
        )
        selected = int(saved_ids[position])
        if selected not in possible:
            raise ValueError(
                f"saved token {selected} is not grammar-valid at position {position}"
            )
        row_logits = next_logits[position - 1]
        masked = _masked_target_stats(row_logits, possible, selected)
        token_events.append(
            {
                "position": position,
                "saved_id": selected,
                "saved_group": _group_name(_token_group(tokenizer, selected)),
                "saved_token_rank": int(masked["target_rank"]),
                "saved_token_probability": float(masked["target_probability"]),
                "masked_pred_id": int(masked["pred_id"]),
                "masked_pred_matches_saved": bool(
                    int(masked["pred_id"]) == selected
                ),
            }
        )
        groups = _possible_groups(tokenizer, possible)
        target = int(target_ids[position]) if position < len(target_ids) else None
        if len(groups) > 1 or int(tokenizer.eos) in possible:
            joint_count, branch_count = _count_prefix_structure(tokenizer, prefix_ids)
            group_report = _group_stats(tokenizer, row_logits, possible, selected)
            events.append(
                {
                    "position": position,
                    "joint_count_before": int(joint_count),
                    "branch_count_before": int(branch_count),
                    "saved_id": selected,
                    "saved_group": _group_name(_token_group(tokenizer, selected)),
                    "saved_token_rank": int(masked["target_rank"]),
                    "saved_token_probability": float(masked["target_probability"]),
                    "masked_pred_id": int(masked["pred_id"]),
                    "masked_pred_matches_saved": bool(int(masked["pred_id"]) == selected),
                    "target_id": target,
                    "target_matches_saved": bool(target == selected),
                    **group_report,
                }
            )

    if saved_ids[-1] != int(tokenizer.eos):
        prefix_ids = saved_ids
        possible = set(
            int(value)
            for value in tokenizer.next_posible_token(ids=np.asarray(prefix_ids, dtype=np.int64))
        )
        row_logits = next_logits[-1]
        groups = _possible_groups(tokenizer, possible)
        joint_count, branch_count = _count_prefix_structure(tokenizer, prefix_ids)
        possible_ids = torch.tensor(sorted(possible), device=row_logits.device, dtype=torch.long)
        predicted = int(
            possible_ids[row_logits.index_select(0, possible_ids).argmax()].item()
        )
        events.append(
            {
                "position": len(saved_ids),
                "joint_count_before": int(joint_count),
                "branch_count_before": int(branch_count),
                "saved_id": None,
                "saved_group": None,
                "saved_token_rank": None,
                "saved_token_probability": None,
                "masked_pred_id": predicted,
                "masked_pred_matches_saved": None,
                "target_id": None,
                "target_matches_saved": None,
                "terminal_probe": True,
                **_group_stats(tokenizer, row_logits, possible, None),
            }
        )

    eos_events = [
        event
        for event in events
        if "eos" in event["group_probabilities"]
    ]
    at_target_count = [
        event
        for event in eos_events
        if int(event["joint_count_before"]) == int(target_joint_count)
    ]
    generated_joint_count, generated_branch_count = _count_prefix_structure(
        tokenizer,
        saved_ids,
    )
    return {
        "has_eos": bool(saved_ids and saved_ids[-1] == int(tokenizer.eos)),
        "generated_new_tokens": len(saved_ids) - 2,
        "generated_joint_count": int(generated_joint_count),
        "generated_branch_count": int(generated_branch_count),
        "joint_count_error": int(generated_joint_count) - int(target_joint_count),
        "token_events": token_events,
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
        "selected_eos_at_target_joint_count": any(
            event.get("saved_id") == int(tokenizer.eos)
            for event in at_target_count
        ),
        "first_event_after_target_joint_count": next(
            (
                event
                for event in eos_events
                if int(event["joint_count_before"]) > int(target_joint_count)
            ),
            None,
        ),
    }


def _compact_prefix_trace(trace: dict[str, Any]) -> dict[str, Any]:
    selected_events = [
        event
        for event in trace["decision_events"]
        if event.get("saved_token_probability") is not None
    ]
    eos_events = [
        event
        for event in trace["decision_events"]
        if "eos" in event["group_probabilities"]
    ]
    at_target = [
        event
        for event in eos_events
        if int(event["joint_count_before"])
        == int(trace["generated_joint_count"] - trace["joint_count_error"])
    ]
    first_after_target = trace.get("first_event_after_target_joint_count")
    return {
        "max_eos_group_probability": trace["max_eos_group_probability"],
        "max_eos_probability_at_target_joint_count": trace[
            "max_eos_probability_at_target_joint_count"
        ],
        "mean_eos_group_probability": _mean(
            [event["group_probabilities"]["eos"] for event in eos_events]
        ),
        "first_after_target_eos_probability": (
            first_after_target["group_probabilities"]["eos"]
            if first_after_target is not None
            else None
        ),
        "mean_saved_token_nll": _mean(
            [
                -np.log(max(float(event["saved_token_probability"]), 1.0e-30))
                for event in selected_events
            ]
        ),
        "masked_top1_matches_saved_rate": _mean(
            [int(event["masked_pred_matches_saved"]) for event in selected_events]
        ),
        "group_top1_matches_saved_rate": _mean(
            [
                int(event["predicted_group"] == event["saved_group"])
                for event in selected_events
            ]
        ),
        "target_count_events": [
            {
                "position": int(event["position"]),
                "joint_count_before": int(event["joint_count_before"]),
                "saved_group": event["saved_group"],
                "predicted_group": event["predicted_group"],
                "eos_probability": float(event["group_probabilities"]["eos"]),
            }
            for event in at_target
        ],
    }


def _compare_prefix_traces(
    correct: dict[str, Any],
    intervention: dict[str, Any],
    first_mismatch_position: int,
) -> dict[str, Any]:
    correct_events = {
        int(event["position"]): event for event in correct["decision_events"]
    }
    intervention_events = {
        int(event["position"]): event for event in intervention["decision_events"]
    }
    if correct_events.keys() != intervention_events.keys():
        raise ValueError("condition intervention changed self-prefix decision positions")
    pairs = [
        (correct_events[position], intervention_events[position])
        for position in sorted(correct_events)
    ]
    correct_tokens = {
        int(event["position"]): event for event in correct["token_events"]
    }
    intervention_tokens = {
        int(event["position"]): event for event in intervention["token_events"]
    }
    if correct_tokens.keys() != intervention_tokens.keys():
        raise ValueError("condition intervention changed self-prefix token positions")
    token_pairs = [
        (correct_tokens[position], intervention_tokens[position])
        for position in sorted(correct_tokens)
    ]
    correct_target = correct["max_eos_probability_at_target_joint_count"]
    intervention_target = intervention["max_eos_probability_at_target_joint_count"]

    def distance_bin(position: int) -> str:
        distance = int(position) - int(first_mismatch_position)
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

    def summarize_pairs(selected: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        selected_with_token = [
            pair
            for pair in selected
            if pair[0].get("saved_token_probability") is not None
            and pair[1].get("saved_token_probability") is not None
        ]
        selected_with_eos = [
            pair
            for pair in selected
            if "eos" in pair[0]["group_probabilities"]
            and "eos" in pair[1]["group_probabilities"]
        ]
        return {
            "decision_positions": len(selected),
            "masked_top1_agreement": _mean(
                [
                    int(left["masked_pred_id"] == right["masked_pred_id"])
                    for left, right in selected
                ]
            ),
            "group_top1_agreement": _mean(
                [
                    int(left["predicted_group"] == right["predicted_group"])
                    for left, right in selected
                ]
            ),
            "saved_token_nll_delta": _mean(
                [
                    -np.log(max(float(right["saved_token_probability"]), 1.0e-30))
                    + np.log(max(float(left["saved_token_probability"]), 1.0e-30))
                    for left, right in selected_with_token
                ]
            ),
            "mean_abs_eos_probability_delta": _mean(
                [
                    abs(
                        float(right["group_probabilities"]["eos"])
                        - float(left["group_probabilities"]["eos"])
                    )
                    for left, right in selected_with_eos
                ]
            ),
            "mean_signed_eos_probability_delta": _mean(
                [
                    float(right["group_probabilities"]["eos"])
                    - float(left["group_probabilities"]["eos"])
                    for left, right in selected_with_eos
                ]
            ),
        }

    def summarize_token_pairs(
        selected: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "token_positions": len(selected),
            "masked_top1_agreement": _mean(
                [
                    int(left["masked_pred_id"] == right["masked_pred_id"])
                    for left, right in selected
                ]
            ),
            "saved_token_nll_delta": _mean(
                [
                    -np.log(max(float(right["saved_token_probability"]), 1.0e-30))
                    + np.log(max(float(left["saved_token_probability"]), 1.0e-30))
                    for left, right in selected
                ]
            ),
        }

    pairs_by_distance: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for pair in pairs:
        pairs_by_distance[distance_bin(int(pair[0]["position"]))].append(pair)
    token_pairs_by_distance: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    coordinate_pairs_by_distance: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for pair in token_pairs:
        name = distance_bin(int(pair[0]["position"]))
        token_pairs_by_distance[name].append(pair)
        if pair[0]["saved_group"] == "coordinate":
            coordinate_pairs_by_distance[name].append(pair)
    overall = summarize_pairs(pairs)
    coordinate_pairs = [
        pair for pair in token_pairs if pair[0]["saved_group"] == "coordinate"
    ]
    return {
        **overall,
        "all_tokens": summarize_token_pairs(token_pairs),
        "coordinate_tokens": summarize_token_pairs(coordinate_pairs),
        "by_distance_from_first_mismatch": {
            name: summarize_pairs(bin_pairs)
            for name, bin_pairs in sorted(pairs_by_distance.items())
        },
        "all_tokens_by_distance_from_first_mismatch": {
            name: summarize_token_pairs(bin_pairs)
            for name, bin_pairs in sorted(token_pairs_by_distance.items())
        },
        "coordinate_tokens_by_distance_from_first_mismatch": {
            name: summarize_token_pairs(bin_pairs)
            for name, bin_pairs in sorted(coordinate_pairs_by_distance.items())
        },
        "target_count_eos_probability_delta": (
            float(intervention_target) - float(correct_target)
            if intervention_target is not None and correct_target is not None
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
            "gt_prefix_decision_swapped_minus_correct_nll": _mean(
                [
                    row["condition_group_likelihood"]["all_decisions"][
                        "swapped_minus_correct_nll"
                    ]
                    for row in group_rows
                ]
            ),
            "gt_prefix_decision_zero_minus_correct_nll": _mean(
                [
                    row["condition_group_likelihood"]["all_decisions"][
                        "zero_minus_correct_nll"
                    ]
                    for row in group_rows
                ]
            ),
            "gt_prefix_eos_swapped_minus_correct_nll": _mean(
                [
                    row["condition_group_likelihood"]["eos"][
                        "swapped_minus_correct_nll"
                    ]
                    for row in group_rows
                ]
            ),
            "gt_prefix_eos_zero_minus_correct_nll": _mean(
                [
                    row["condition_group_likelihood"]["eos"][
                        "zero_minus_correct_nll"
                    ]
                    for row in group_rows
                ]
            ),
            "first_free_mismatch_roles": dict(role_counts),
            "free_max_eos_group_probability": _mean(
                [
                    row["saved_prefix_trace"]["max_eos_group_probability"]
                    for row in group_rows
                ]
            ),
            "free_max_eos_probability_at_target_joint_count": _mean(
                [
                    row["saved_prefix_trace"]["max_eos_probability_at_target_joint_count"]
                    for row in group_rows
                ]
            ),
            "free_generated_joint_count": _mean(
                [
                    row["saved_prefix_trace"]["generated_joint_count"]
                    for row in group_rows
                ]
            ),
            "free_joint_count_error": _mean(
                [
                    row["saved_prefix_trace"]["joint_count_error"]
                    for row in group_rows
                ]
            ),
            "generation_trigram_unique_ratio": _mean(
                [
                    row["generation_repetition"]["ngram_unique_ratio"]["3"]
                    for row in group_rows
                ]
            ),
            "generation_terminal_same_token_run": _mean(
                [
                    row["generation_repetition"]["terminal_same_token_run"]
                    for row in group_rows
                ]
            ),
            "generation_suffix_periodic_coverage": _mean(
                [
                    row["generation_repetition"][
                        "best_suffix_periodic_coverage"
                    ]
                    for row in group_rows
                ]
            ),
            "free_selected_eos_at_target_joint_count_rate": _mean(
                [
                    int(
                        row["saved_prefix_trace"][
                            "selected_eos_at_target_joint_count"
                        ]
                    )
                    for row in group_rows
                ]
            ),
            "self_prefix_correct_eos_probability_at_target_count": _mean(
                [
                    row["self_prefix_condition"]["correct"][
                        "max_eos_probability_at_target_joint_count"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_swapped_eos_probability_at_target_count": _mean(
                [
                    row["self_prefix_condition"]["swapped"][
                        "max_eos_probability_at_target_joint_count"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_zero_eos_probability_at_target_count": _mean(
                [
                    row["self_prefix_condition"]["zero"][
                        "max_eos_probability_at_target_joint_count"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_swapped_target_eos_probability_delta": _mean(
                [
                    row["self_prefix_condition"]["swapped_vs_correct"][
                        "target_count_eos_probability_delta"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_zero_target_eos_probability_delta": _mean(
                [
                    row["self_prefix_condition"]["zero_vs_correct"][
                        "target_count_eos_probability_delta"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_swapped_group_top1_agreement": _mean(
                [
                    row["self_prefix_condition"]["swapped_vs_correct"][
                        "group_top1_agreement"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_zero_group_top1_agreement": _mean(
                [
                    row["self_prefix_condition"]["zero_vs_correct"][
                        "group_top1_agreement"
                    ]
                    for row in group_rows
                ]
            ),
            "self_prefix_swapped_coordinate_top1_agreement": _mean(
                [
                    row["self_prefix_condition"]["swapped_vs_correct"][
                        "coordinate_tokens"
                    ]["masked_top1_agreement"]
                    for row in group_rows
                ]
            ),
            "self_prefix_zero_coordinate_top1_agreement": _mean(
                [
                    row["self_prefix_condition"]["zero_vs_correct"][
                        "coordinate_tokens"
                    ]["masked_top1_agreement"]
                    for row in group_rows
                ]
            ),
        }
        repair_names = sorted(
            {
                repair_name
                for row in group_rows
                for repair_name in row.get("prefix_repair_probe", {})
                .get("interventions", {})
            }
        )
        if repair_names:
            output[name]["prefix_repair_probe"] = {}
            for repair_name in repair_names:
                repair_rows = [
                    row["prefix_repair_probe"]["interventions"][repair_name]
                    for row in group_rows
                    if repair_name
                    in row.get("prefix_repair_probe", {}).get("interventions", {})
                ]
                output[name]["prefix_repair_probe"][repair_name] = {
                    "row_count": len(repair_rows),
                    "eos_rate": _mean(
                        [int(row["has_eos"]) for row in repair_rows]
                    ),
                    "detokenize_rate": _mean(
                        [
                            int(row["metrics"].get("detokenize_ok", False))
                            for row in repair_rows
                        ]
                    ),
                    "generated_joint_count": _mean(
                        [row["generated_joint_count"] for row in repair_rows]
                    ),
                    "joint_count_abs_error": _mean(
                        [
                            row["metrics"].get("joint_count_abs_error")
                            for row in repair_rows
                        ]
                    ),
                    "j2j": _mean(
                        [row["metrics"].get("j2j") for row in repair_rows]
                    ),
                    "topology_edge_f1": _mean(
                        [
                            row["metrics"].get("topology_edge_f1")
                            for row in repair_rows
                        ]
                    ),
                }
            probe_rows = [
                row["prefix_repair_probe"]
                for row in group_rows
                if "prefix_repair_probe" in row
            ]
            output[name]["prefix_repair_contract"] = {
                "row_count": len(probe_rows),
                "saved_baseline_reproduction_exact_rate": _mean(
                    [
                        int(row["saved_baseline_reproduction_exact"])
                        for row in probe_rows
                    ]
                ),
                "same_process_reproduction_exact_rate": _mean(
                    [
                        int(row["same_process_reproduction_exact"])
                        for row in probe_rows
                    ]
                ),
                "condition_repeat_exact_rate": _mean(
                    [
                        int(
                            group_row.get(
                                "condition_repeat_exact",
                                False,
                            )
                        )
                        for group_row in group_rows
                        if "prefix_repair_probe" in group_row
                    ]
                ),
                "condition_repeat_max_abs_diff": _mean(
                    [
                        group_row.get("condition_repeat_max_abs_diff")
                        for group_row in group_rows
                        if "prefix_repair_probe" in group_row
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
    parser.add_argument(
        "--prefix-repair-probe",
        action="store_true",
        help="Run bounded greedy continuations from repaired hitmax prefixes.",
    )
    parser.add_argument(
        "--prefix-repair-margin",
        type=int,
        default=128,
        help="Additional generated-token budget beyond the target sequence length.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scripts_dir = args.model_root / "rigweave" / "scripts"
    source_dir = args.model_root / "rigweave" / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))

    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model, apply_checkpoint_eval_defaults
    from eval_dynamic_rig_generation import (
        _continuous_range,
        _count_prefix_structure,
    )
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
    continuous_range = _continuous_range(tokenizer)

    rows: list[dict[str, Any]] = []
    condition_cache: list[torch.Tensor] = []
    target_ids_cache: list[list[int]] = []
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
            target_ids_cache.append(target_ids)
            if target_ids != generation_row["target_ids"]:
                raise ValueError(f"row {index} target-token mismatch")

            _seed_all(args.seed + index, device)
            refs = model.sample_references(batch)
            cond = model.build_condition(batch, refs=refs)
            condition_cache.append(cond.detach().cpu())
            teacher, _logits, _labels = _teacher_report(model, tokenizer, batch, cond)
            expected_generated = generation_row["dynamic"]["generated_ids"]
            generated_block = generation_row["dynamic"]
            hitmax = bool(generated_block.get("hit_max_without_eos"))
            target_joint_count, target_branch_count = _count_prefix_structure(tokenizer, target_ids)
            max_new_tokens = int(generation_row["max_new_tokens"])
            has_eos = int(tokenizer.eos) in expected_generated
            if has_eos and expected_generated[-1] != int(tokenizer.eos):
                raise ValueError(f"row {index} contains a non-terminal EOS token")
            expected_hitmax = (
                len(expected_generated) - 2 >= max_new_tokens
                and not has_eos
            )
            if hitmax != expected_hitmax:
                raise ValueError(
                    f"row {index} inconsistent saved hitmax state: "
                    f"flag={hitmax} derived={expected_hitmax}"
                )
            saved_prefix_trace = _saved_prefix_trace(
                model,
                tokenizer,
                cond,
                target_ids,
                expected_generated,
                int(target_joint_count),
            )
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
                "generation_repetition": _repetition_report(
                    expected_generated
                ),
                "saved_prefix_trace": saved_prefix_trace,
            }
            if args.prefix_repair_probe and hitmax:
                _seed_all(args.seed + index, device)
                repeated_refs = model.sample_references(batch)
                repeated_cond = model.build_condition(
                    batch,
                    refs=repeated_refs,
                )
                condition_abs_diff = (
                    repeated_cond.detach().float()
                    - cond.detach().float()
                ).abs()
                row["condition_repeat_exact"] = bool(
                    torch.equal(repeated_cond, cond)
                )
                row["condition_repeat_max_abs_diff"] = float(
                    condition_abs_diff.max().item()
                )
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
                        "free_generated_joint_count": saved_prefix_trace[
                            "generated_joint_count"
                        ],
                        "free_eos_probability_at_target_count": saved_prefix_trace[
                            "max_eos_probability_at_target_joint_count"
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        for index, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            correct = condition_cache[index].to(device)
            swapped = condition_cache[(index + 1) % len(condition_cache)].to(device)
            zero = torch.zeros_like(correct)
            logits_by_condition: dict[str, torch.Tensor] = {}
            labels = None
            input_ids = None
            for name, cond in (("correct", correct), ("swapped", swapped), ("zero", zero)):
                logits, current_labels, current_input_ids = _teacher_logits(model, batch, cond)
                logits_by_condition[name] = logits
                if labels is None:
                    labels = current_labels
                    input_ids = current_input_ids
                elif not torch.equal(labels, current_labels):
                    raise ValueError("condition intervention changed labels")
                elif not torch.equal(input_ids, current_input_ids):
                    raise ValueError("condition intervention changed input IDs")
            assert labels is not None and input_ids is not None
            rows[index]["condition_likelihood"] = _condition_likelihood(
                tokenizer,
                labels,
                logits_by_condition,
            )
            rows[index]["condition_group_likelihood"] = _condition_group_likelihood(
                tokenizer,
                labels,
                input_ids,
                logits_by_condition,
            )

            generated_ids = generation_rows[index]["dynamic"]["generated_ids"]
            correct_trace = rows[index]["saved_prefix_trace"]
            swapped_trace = _saved_prefix_trace(
                model,
                tokenizer,
                swapped,
                target_ids_cache[index],
                generated_ids,
                rows[index]["target_joint_count"],
            )
            zero_trace = _saved_prefix_trace(
                model,
                tokenizer,
                zero,
                target_ids_cache[index],
                generated_ids,
                rows[index]["target_joint_count"],
            )
            rows[index]["self_prefix_condition"] = {
                "correct": _compact_prefix_trace(correct_trace),
                "swapped": _compact_prefix_trace(swapped_trace),
                "zero": _compact_prefix_trace(zero_trace),
                "swapped_vs_correct": _compare_prefix_traces(
                    correct_trace,
                    swapped_trace,
                    rows[index]["first_free_mismatch"]["position"],
                ),
                "zero_vs_correct": _compare_prefix_traces(
                    correct_trace,
                    zero_trace,
                    rows[index]["first_free_mismatch"]["position"],
                ),
            }
            if args.prefix_repair_probe and rows[index]["hitmax"]:
                target = tokenizer.detokenize(
                    np.asarray(target_ids_cache[index], dtype=np.int64)
                )
                target_new_tokens = max(len(target_ids_cache[index]) - 2, 1)
                probe_max_new_tokens = min(
                    int(generation_rows[index]["max_new_tokens"]),
                    int(target_new_tokens + args.prefix_repair_margin),
                )
                rows[index]["prefix_repair_probe"] = _prefix_repair_rollout(
                    model,
                    tokenizer,
                    correct,
                    target_ids_cache[index],
                    generated_ids,
                    rows[index]["first_free_mismatch"]["position"],
                    probe_max_new_tokens,
                    target,
                    continuous_range,
                )
            print(
                json.dumps(
                    {
                        "event": "condition_likelihood",
                        "index": index,
                        "joint0": rows[index]["condition_likelihood"]["joint0_coord"],
                        "gt_prefix_decisions": rows[index][
                            "condition_group_likelihood"
                        ].get("all_decisions"),
                        "self_prefix_condition": rows[index][
                            "self_prefix_condition"
                        ],
                        "prefix_repair_probe": (
                            {
                                name: {
                                    "has_eos": result["has_eos"],
                                    "generated_joint_count": result[
                                        "generated_joint_count"
                                    ],
                                    "metrics": result["metrics"],
                                }
                                for name, result in rows[index]
                                .get("prefix_repair_probe", {})
                                .get("interventions", {})
                                .items()
                            }
                            or None
                        ),
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
