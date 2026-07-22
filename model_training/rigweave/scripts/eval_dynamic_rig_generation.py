#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import LogitsProcessor, LogitsProcessorList

from eval_dynamic_rig_ce import _build_dynamic_model, apply_checkpoint_eval_defaults
from train_dynamic_rig import build_tokenizer, load_unirig, move_batch


def _parents_from_output(output: Any) -> list[int | None]:
    if getattr(output, "parents", None) is not None:
        return [None if p is None else int(p) for p in output.parents]
    return output._get_parents()


def _detokenize_target(tokenizer: Any, batch: dict[str, Any]) -> Any:
    ids = batch["input_ids"][0][batch["attention_mask"][0].bool()].detach().cpu().numpy().astype(np.int64)
    return tokenizer.detokenize(ids)


def _count_prefix_structure(tokenizer: Any, ids: list[int]) -> tuple[int, int]:
    num_discrete = int(tokenizer.num_discrete)
    branch_token = int(getattr(tokenizer, "token_id_branch"))
    eos = int(tokenizer.eos)
    state = "expect_bos"
    coords_needed = 0
    joints = 0
    branches = 0
    for token in ids:
        token = int(token)
        if token == eos:
            break
        if state == "expect_bos":
            state = "expect_cls_or_part_or_joint"
        elif state == "expect_cls_or_part_or_joint":
            if 0 <= token < num_discrete:
                coords_needed = 2
                state = "expect_coords"
            elif token == getattr(tokenizer, "token_id_cls_none", -1) or token in getattr(tokenizer, "cls_token_id", {}).values():
                state = "expect_part_or_joint"
            else:
                state = "expect_joint"
        elif state == "expect_part_or_joint":
            if 0 <= token < num_discrete:
                coords_needed = 2
                state = "expect_coords"
            else:
                state = "expect_part_or_joint"
        elif state == "expect_branch_or_part_or_joint":
            if token == branch_token:
                branches += 1
                state = "expect_joint"
            elif 0 <= token < num_discrete:
                coords_needed = 2
                state = "expect_coords"
            else:
                state = "expect_joint"
        elif state == "expect_joint":
            if 0 <= token < num_discrete:
                coords_needed = 5
                state = "expect_coords"
        elif state == "expect_coords":
            if 0 <= token < num_discrete:
                coords_needed -= 1
                if coords_needed <= 0:
                    joints += 1
                    state = "expect_branch_or_part_or_joint"
    return joints, branches


def _apply_branch_prior_count_bias_to_scores(
    *,
    tokenizer: Any,
    start_tokens: torch.Tensor,
    generated: list[int],
    scores: torch.Tensor,
    expected_branch_count: float,
    branch_margin: float,
    branch_under_bias: float,
    branch_over_penalty: float,
    min_completed_joints: int,
) -> torch.Tensor:
    branch = int(getattr(tokenizer, "token_id_branch"))
    full = np.asarray(start_tokens.detach().cpu().tolist() + generated, dtype=np.int64)
    try:
        possible = set(int(x) for x in tokenizer.next_posible_token(ids=full))
    except Exception:
        return scores
    if branch not in possible:
        return scores
    joints, branches = _count_prefix_structure(tokenizer, [int(x) for x in full.tolist()])
    if joints < int(min_completed_joints):
        return scores
    if branches + float(branch_margin) < float(expected_branch_count):
        scores[:, branch] = scores[:, branch] + float(branch_under_bias)
    elif branches >= float(expected_branch_count) + float(branch_margin):
        scores[:, branch] = scores[:, branch] - float(branch_over_penalty)
    return scores


def _branch_coordinate_state(
    *,
    tokenizer: Any,
    start_tokens: torch.Tensor,
    generated: list[int],
) -> tuple[int, int, bool] | None:
    """Return `(branch_index, coord_dim, is_child_coord)` inside BRANCH coords.

    UniRig serializes one branch event as:
        BRANCH, parent_x, parent_y, parent_z, child_x, child_y, child_z

    The branch prior proposals are trained in the same serialized branch order,
    so branch_index selects the proposal, coord_dim selects x/y/z, and
    is_child_coord tells whether to use the proposal parent/root or child xyz.
    """

    num_discrete = int(tokenizer.num_discrete)
    branch = int(getattr(tokenizer, "token_id_branch"))
    full = start_tokens.detach().cpu().tolist() + [int(x) for x in generated]
    branch_positions = [i for i, token in enumerate(full) if int(token) == branch]
    if not branch_positions:
        return None
    last_branch = int(branch_positions[-1])
    suffix = [int(x) for x in full[last_branch + 1 :]]
    if len(suffix) >= 6:
        return None
    if not all(0 <= token < num_discrete for token in suffix):
        return None
    branch_index = int(sum(1 for token in full[:last_branch] if int(token) == branch))
    coord_offset = len(suffix)
    return branch_index, coord_offset % 3, coord_offset >= 3


def _discretize_branch_prior_xyz(tokenizer: Any, xyz: np.ndarray | torch.Tensor) -> np.ndarray:
    lo, hi = _continuous_range(tokenizer)
    arr = np.asarray(xyz, dtype=np.float32)
    tokens = (arr - float(lo)) / max(float(hi - lo), 1.0e-12)
    tokens = tokens * float(tokenizer.num_discrete)
    return np.clip(np.rint(tokens), 0, int(tokenizer.num_discrete) - 1).astype(np.int64)


def _apply_branch_prior_coordinate_bias_to_scores(
    *,
    tokenizer: Any,
    start_tokens: torch.Tensor,
    generated: list[int],
    scores: torch.Tensor,
    root_xyz: np.ndarray,
    child_xyz: np.ndarray,
    exist_probs: np.ndarray | None,
    coord_bias: float,
    coord_sigma: float,
    min_exist_prob: float,
) -> torch.Tensor:
    state = _branch_coordinate_state(tokenizer=tokenizer, start_tokens=start_tokens, generated=generated)
    if state is None:
        return scores
    branch_index, coord_dim, is_child_coord = state
    if branch_index < 0 or branch_index >= int(root_xyz.shape[0]):
        return scores
    if exist_probs is not None and float(exist_probs[branch_index]) < float(min_exist_prob):
        return scores

    target_xyz = child_xyz[branch_index] if is_child_coord else root_xyz[branch_index]
    target_token = int(_discretize_branch_prior_xyz(tokenizer, target_xyz)[coord_dim])
    num_discrete = int(tokenizer.num_discrete)
    arange = torch.arange(num_discrete, device=scores.device, dtype=torch.float32)
    sigma = max(float(coord_sigma), 1.0e-6)
    bias = -0.5 * ((arange - float(target_token)) / sigma).pow(2) * float(coord_bias)
    scores[:, :num_discrete] = scores[:, :num_discrete] + bias.to(dtype=scores.dtype).unsqueeze(0)
    return scores


def _target_structure_counts_from_ids(tokenizer: Any, ids: np.ndarray) -> tuple[int, int]:
    branch_token = int(getattr(tokenizer, "token_id_branch"))
    branch_count = int(np.sum(ids == branch_token))
    try:
        out = tokenizer.detokenize(ids.astype(np.int64))
        return int(len(out.joints)), branch_count
    except Exception:
        joints, branches = _count_prefix_structure(tokenizer, [int(x) for x in ids.tolist()])
        return int(joints), int(branches)


def _possible_action_groups(tokenizer: Any, possible: set[int]) -> set[int]:
    num_discrete = int(tokenizer.num_discrete)
    eos = int(tokenizer.eos)
    branch = int(getattr(tokenizer, "token_id_branch"))
    groups: set[int] = set()
    if eos in possible:
        groups.add(0)
    if branch in possible:
        groups.add(1)
    if any(0 <= x < num_discrete for x in possible):
        groups.add(2)
    if any(x >= num_discrete and x not in {eos, branch} for x in possible):
        groups.add(3)
    return groups


def _tokens_for_action_group(tokenizer: Any, possible: set[int], group: int) -> list[int]:
    num_discrete = int(tokenizer.num_discrete)
    eos = int(tokenizer.eos)
    branch = int(getattr(tokenizer, "token_id_branch"))
    if group == 0:
        return [eos] if eos in possible else []
    if group == 1:
        return [branch] if branch in possible else []
    if group == 2:
        return sorted(x for x in possible if 0 <= x < num_discrete)
    if group == 3:
        return sorted(x for x in possible if x >= num_discrete and x not in {eos, branch})
    return []


def _filter_logits_to_ids(logits: torch.Tensor, ids: set[int] | list[int]) -> torch.Tensor:
    out = torch.full_like(logits, -float("inf"))
    if ids:
        out[..., list(ids)] = logits[..., list(ids)]
    return out


def _fuse_dynamic_static_logits(
    *,
    tokenizer: Any,
    possible: set[int],
    dynamic_logits: torch.Tensor,
    static_logits: torch.Tensor,
    prior_weight: float,
    prior_scope: str,
) -> torch.Tensor:
    if prior_scope == "all":
        fused = dynamic_logits + float(prior_weight) * static_logits
        return _filter_logits_to_ids(fused, possible)

    num_discrete = int(tokenizer.num_discrete)
    eos = int(tokenizer.eos)
    if prior_scope == "eos":
        prior_ids = [eos] if eos in possible else []
    elif prior_scope == "structure":
        # Do not let the static prior choose coordinate bins.  It may only bias
        # grammar/structure tokens such as EOS, branch, class, or part tokens.
        prior_ids = [int(x) for x in possible if not (0 <= int(x) < num_discrete)]
    else:
        raise ValueError(f"unknown logit prior scope {prior_scope!r}")

    fused = dynamic_logits.clone()
    if prior_ids:
        fused[..., prior_ids] = fused[..., prior_ids] + float(prior_weight) * static_logits[..., prior_ids]
    return _filter_logits_to_ids(fused, possible)


def _sample_or_argmax(logits: torch.Tensor, generation_kwargs: dict[str, Any] | None) -> torch.Tensor:
    kwargs = generation_kwargs or {}
    logits = logits.float()
    temperature = kwargs.get("temperature")
    if temperature is not None and float(temperature) > 0:
        logits = logits / float(temperature)
    if kwargs.get("do_sample", False):
        top_k = kwargs.get("top_k")
        if top_k is not None and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
            values, indices = torch.topk(logits, int(top_k), dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(dim=-1, index=indices, src=values)
            logits = filtered
        probs = torch.softmax(logits, dim=-1)
        if torch.isfinite(probs).all() and float(probs.sum()) > 0:
            return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


class StructureCountGuidanceLogitsProcessor(LogitsProcessor):
    """Use global joint/branch count estimates to stabilize skeleton termination."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        start_tokens: torch.Tensor,
        joint_count: float,
        branch_count: float,
        joint_margin: float = 1.0,
        early_eos_penalty: float = 6.0,
        eos_bias: float = 6.0,
        branch_over_penalty: float = 4.0,
        branch_under_bias: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.start_tokens = start_tokens.detach().cpu().to(dtype=torch.long)
        self.joint_count = float(joint_count)
        self.branch_count = float(branch_count)
        self.joint_margin = float(joint_margin)
        self.early_eos_penalty = float(early_eos_penalty)
        self.eos_bias = float(eos_bias)
        self.branch_over_penalty = float(branch_over_penalty)
        self.branch_under_bias = float(branch_under_bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        start = self.start_tokens.to(device=input_ids.device)
        for batch_idx, sequence in enumerate(input_ids):
            full = torch.cat([start, sequence.detach().to(dtype=torch.long)]).detach().cpu().numpy().astype(np.int64)
            try:
                possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=full))
            except Exception:
                continue
            joints, branches = _count_prefix_structure(self.tokenizer, [int(x) for x in full.tolist()])
            if eos in possible:
                if joints < self.joint_count - self.joint_margin:
                    scores[batch_idx, eos] -= self.early_eos_penalty
                elif joints >= self.joint_count:
                    over = min(max(joints - self.joint_count, 0.0), 8.0)
                    scores[batch_idx, eos] += self.eos_bias * (1.0 + 0.25 * over)
            if branch in possible:
                if branches >= self.branch_count:
                    scores[batch_idx, branch] -= self.branch_over_penalty
                elif branches + 1.0 < self.branch_count:
                    scores[batch_idx, branch] += self.branch_under_bias
        return scores


class BranchPriorCountGuidanceLogitsProcessor(LogitsProcessor):
    """Use the learned coarse branch prior to bias branch decisions.

    This is a model-side structural prior, not an alternate decoder.  The branch
    prior is trained to predict how many branch events should exist from the
    dynamic condition.  During free generation this processor only nudges the
    UniRig grammar decision token `BRANCH`; it never chooses coordinates and it
    never edits an already generated sequence.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        start_tokens: torch.Tensor,
        expected_branch_count: float,
        branch_margin: float = 0.5,
        branch_under_bias: float = 1.5,
        branch_over_penalty: float = 6.0,
        min_completed_joints: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.start_tokens = start_tokens.detach().cpu().to(dtype=torch.long)
        self.expected_branch_count = max(float(expected_branch_count), 0.0)
        self.branch_margin = max(float(branch_margin), 0.0)
        self.branch_under_bias = float(branch_under_bias)
        self.branch_over_penalty = float(branch_over_penalty)
        self.min_completed_joints = max(int(min_completed_joints), 0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        start = self.start_tokens.to(device=input_ids.device)
        for batch_idx, sequence in enumerate(input_ids):
            full = torch.cat([start, sequence.detach().to(dtype=torch.long)]).detach().cpu().numpy().astype(np.int64)
            try:
                possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=full))
            except Exception:
                continue
            if branch not in possible:
                continue
            joints, branches = _count_prefix_structure(self.tokenizer, [int(x) for x in full.tolist()])
            if joints < self.min_completed_joints:
                continue
            if branches + self.branch_margin < self.expected_branch_count:
                scores[batch_idx, branch] += self.branch_under_bias
            elif branches >= self.expected_branch_count + self.branch_margin:
                scores[batch_idx, branch] -= self.branch_over_penalty
        return scores


class BranchPriorCoordinateGuidanceLogitsProcessor(LogitsProcessor):
    """Use coarse branch parent/child xyz proposals as a soft coordinate prior."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        start_tokens: torch.Tensor,
        root_xyz: torch.Tensor,
        child_xyz: torch.Tensor,
        exist_logits: torch.Tensor | None = None,
        coord_bias: float = 2.0,
        coord_sigma: float = 6.0,
        min_exist_prob: float = 0.2,
    ) -> None:
        self.tokenizer = tokenizer
        self.start_tokens = start_tokens.detach().cpu().to(dtype=torch.long)
        self.root_xyz = root_xyz.detach().float().cpu().numpy()
        self.child_xyz = child_xyz.detach().float().cpu().numpy()
        self.exist_probs = None if exist_logits is None else exist_logits.detach().sigmoid().float().cpu().numpy()
        self.coord_bias = float(coord_bias)
        self.coord_sigma = float(coord_sigma)
        self.min_exist_prob = float(min_exist_prob)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx, sequence in enumerate(input_ids):
            row_scores = scores[batch_idx : batch_idx + 1]
            row_scores = _apply_branch_prior_coordinate_bias_to_scores(
                tokenizer=self.tokenizer,
                start_tokens=self.start_tokens,
                generated=[int(x) for x in sequence.detach().cpu().tolist()],
                scores=row_scores,
                root_xyz=self.root_xyz,
                child_xyz=self.child_xyz,
                exist_probs=self.exist_probs,
                coord_bias=self.coord_bias,
                coord_sigma=self.coord_sigma,
                min_exist_prob=self.min_exist_prob,
            )
            scores[batch_idx : batch_idx + 1] = row_scores
        return scores


def _parse_completed_discrete_joints(tokenizer: Any, ids: list[int]) -> list[tuple[int, int, int]]:
    """Parse completed child-joint coordinate triples from a token prefix.

    This intentionally mirrors UniRig's branch serialization at the discrete
    token level: a non-branch continuation emits one child triplet, while a
    branch emits parent triplet + child triplet. It is used only for decoding
    constraints, not as a replacement for the official detokenizer.
    """

    num_discrete = int(tokenizer.num_discrete)
    branch = int(getattr(tokenizer, "token_id_branch"))
    eos = int(tokenizer.eos)
    pad_id = int(tokenizer.pad)
    joints: list[tuple[int, int, int]] = []
    i = 0
    while i < len(ids):
        token = int(ids[i])
        if token in {eos, pad_id}:
            break
        if token == branch:
            coords = ids[i + 1 : i + 7]
            if len(coords) < 6 or not all(0 <= int(x) < num_discrete for x in coords):
                break
            joints.append((int(coords[3]), int(coords[4]), int(coords[5])))
            i += 7
            continue
        if 0 <= token < num_discrete:
            coords = ids[i : i + 3]
            if len(coords) < 3 or not all(0 <= int(x) < num_discrete for x in coords):
                break
            joints.append((int(coords[0]), int(coords[1]), int(coords[2])))
            i += 3
            continue
        i += 1
    return joints


class BranchParentSnapLogitsProcessor(LogitsProcessor):
    """Constrain branch parent coordinates to point at an existing joint.

    UniRig serializes a branch as:
        BRANCH, parent_x, parent_y, parent_z, child_x, child_y, child_z

    The parent triplet is not a new joint; it is a pointer represented by
    coordinates. If free generation emits a parent coordinate that is slightly
    closer to the wrong previous joint, detokenization builds the wrong
    topology even when child joint positions look plausible. This processor
    makes that hidden pointer contract explicit during decoding.
    """

    def __init__(self, *, tokenizer: Any, start_tokens: torch.Tensor) -> None:
        self.tokenizer = tokenizer
        self.start_tokens = start_tokens.detach().cpu().to(dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_discrete = int(self.tokenizer.num_discrete)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        start = self.start_tokens.to(device=input_ids.device)
        for batch_idx, sequence in enumerate(input_ids):
            full_ids = torch.cat([start, sequence.detach().to(dtype=torch.long)]).detach().cpu().tolist()
            branch_positions = [i for i, token in enumerate(full_ids) if int(token) == branch]
            if not branch_positions:
                continue
            last_branch = branch_positions[-1]
            parent_prefix = [int(x) for x in full_ids[last_branch + 1 :]]
            if len(parent_prefix) >= 3:
                continue
            if not all(0 <= x < num_discrete for x in parent_prefix):
                continue

            previous_joints = _parse_completed_discrete_joints(self.tokenizer, full_ids[:last_branch])
            if not previous_joints:
                continue
            dim = len(parent_prefix)
            candidates = [
                joint
                for joint in previous_joints
                if all(int(joint[d]) == int(parent_prefix[d]) for d in range(dim))
            ]
            if not candidates:
                candidates = previous_joints
            allowed = sorted({int(joint[dim]) for joint in candidates})
            if not allowed:
                continue
            mask = torch.full_like(scores[batch_idx], -float("inf"))
            mask[allowed] = 0.0
            scores[batch_idx] = scores[batch_idx] + mask
        return scores


def _nn_chamfer(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size == 0 or b.size == 0:
        return {"mean": float("inf"), "p95": float("inf")}
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    a2b = d.min(axis=1)
    b2a = d.min(axis=0)
    both = np.concatenate([a2b, b2a], axis=0)
    return {"mean": float(both.mean()), "p95": float(np.percentile(both, 95))}


def _continuous_range(tokenizer: Any) -> tuple[float, float]:
    value = getattr(tokenizer, "continuous_range", (-1.0, 1.0))
    if callable(value):
        value = value()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi
    return -1.0, 1.0


def _bones_from_parents(joints: np.ndarray, parents: list[int | None]) -> np.ndarray:
    bones: list[np.ndarray] = []
    for child, parent in enumerate(parents):
        if parent is None or parent < 0 or parent >= len(joints) or child >= len(joints):
            continue
        bones.append(np.concatenate([joints[int(parent)], joints[int(child)]], axis=0))
    if not bones:
        return np.zeros((0, 6), dtype=np.float32)
    return np.stack(bones, axis=0).astype(np.float32)


def _sample_bones_np(bones: np.ndarray, num: int = 100) -> np.ndarray:
    if bones.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    samples = []
    head = bones[:, :3]
    tail = bones[:, 3:]
    for i in range(num):
        alpha = i / num
        samples.append(head * alpha + tail * (1.0 - alpha))
    return np.concatenate(samples, axis=0).astype(np.float32)


def _official_skeleton_metrics(
    pred_joints: np.ndarray,
    pred_bones: np.ndarray,
    target_joints: np.ndarray,
    target_bones: np.ndarray,
    continuous_range: tuple[float, float],
) -> dict[str, float]:
    """UniRig/RigNet-style set metrics for skeleton geometry.

    These metrics deliberately do not assume aligned joint IDs. They match the
    definitions in `external/UniRig/src/system/metrics.py`: J2J is bidirectional
    nearest-joint distance, J2B compares joints to sampled bones, and B2B is
    sampled-bone Chamfer. All values are normalized by the tokenizer range.
    """

    if pred_joints.size == 0 or target_joints.size == 0 or pred_bones.size == 0 or target_bones.size == 0:
        return {
            "j2j": float("inf"),
            "j2b": float("inf"),
            "b2b": float("inf"),
            "bone_cd": float("inf"),
        }

    scale = max(float(continuous_range[1] - continuous_range[0]), 1.0e-12)

    def mean_min(a: np.ndarray, b: np.ndarray) -> float:
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return float(d.min(axis=1).mean())

    sample_pred = _sample_bones_np(pred_bones)
    sample_target = _sample_bones_np(target_bones)
    j2j = 0.5 * (mean_min(pred_joints, target_joints) + mean_min(target_joints, pred_joints)) / scale
    j2b = 0.5 * (mean_min(pred_joints, sample_target) + mean_min(target_joints, sample_pred)) / scale
    b2b = 0.5 * (mean_min(sample_pred, sample_target) + mean_min(sample_target, sample_pred)) / scale
    return {
        "j2j": float(j2j),
        "j2b": float(j2b),
        "b2b": float(b2b),
        "bone_cd": float(b2b),
    }


def _edge_set(parents: list[int | None]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for child, parent in enumerate(parents):
        if parent is None or parent < 0 or parent >= len(parents) or child == parent:
            continue
        edges.add((int(parent), int(child)))
    return edges


def _edge_overlap(
    source_joints: np.ndarray,
    source_parents: list[int | None],
    target_joints: np.ndarray,
    target_parents: list[int | None],
) -> dict[str, float | int]:
    source_edges = _edge_set(source_parents)
    target_edges = _edge_set(target_parents)
    if not source_edges:
        return {"edge_count": 0, "matched_edges": 0, "score": 1.0 if not target_edges else 0.0}
    if target_joints.size == 0:
        return {"edge_count": len(source_edges), "matched_edges": 0, "score": 0.0}
    d = np.linalg.norm(source_joints[:, None, :] - target_joints[None, :, :], axis=-1)
    mapping = d.argmin(axis=1)
    matched = 0
    for parent, child in source_edges:
        mapped = (int(mapping[parent]), int(mapping[child]))
        if mapped in target_edges:
            matched += 1
    return {
        "edge_count": len(source_edges),
        "matched_edges": int(matched),
        "score": float(matched / max(len(source_edges), 1)),
    }


def _topology_metrics(
    pred_joints: np.ndarray,
    pred_parents: list[int | None],
    target_joints: np.ndarray,
    target_parents: list[int | None],
) -> dict[str, float | int]:
    precision = _edge_overlap(pred_joints, pred_parents, target_joints, target_parents)
    recall = _edge_overlap(target_joints, target_parents, pred_joints, pred_parents)
    p = float(precision["score"])
    r = float(recall["score"])
    f1 = 0.0 if p + r <= 1.0e-12 else 2.0 * p * r / (p + r)
    return {
        "edge_precision": p,
        "edge_recall": r,
        "edge_f1": float(f1),
        "pred_edge_count": int(precision["edge_count"]),
        "target_edge_count": int(recall["edge_count"]),
        "pred_matched_edges": int(precision["matched_edges"]),
        "target_matched_edges": int(recall["matched_edges"]),
    }


def _bone_features(joints: np.ndarray, parents: list[int | None]) -> np.ndarray:
    bones: list[np.ndarray] = []
    for child, parent in enumerate(parents):
        if parent is None or parent < 0 or parent >= len(joints) or child >= len(joints):
            continue
        p = joints[int(parent)]
        c = joints[int(child)]
        mid = (p + c) * 0.5
        vec = c - p
        length = np.linalg.norm(vec)
        # Midpoint + direction/length gives a stable set metric without assuming joint ids match.
        bones.append(np.concatenate([mid, vec, np.asarray([length], dtype=np.float32)], axis=0))
    if not bones:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack(bones, axis=0).astype(np.float32)


def _set_chamfer(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size == 0 or b.size == 0:
        return {"mean": float("inf"), "p95": float("inf")}
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    both = np.concatenate([d.min(axis=1), d.min(axis=0)], axis=0)
    return {"mean": float(both.mean()), "p95": float(np.percentile(both, 95))}


def _output_metrics(pred: Any | None, target: Any, continuous_range: tuple[float, float]) -> dict[str, Any]:
    target_joints = np.asarray(target.joints, dtype=np.float32)
    target_parents = _parents_from_output(target)
    target_bones = _bones_from_parents(target_joints, target_parents)
    out: dict[str, Any] = {
        "target_joint_count": int(target_joints.shape[0]),
        "target_bone_count": int(target_bones.shape[0]),
        "detokenize_ok": pred is not None,
    }
    if pred is None:
        return out

    pred_joints = np.asarray(pred.joints, dtype=np.float32)
    pred_parents = _parents_from_output(pred)
    pred_bones = _bones_from_parents(pred_joints, pred_parents)
    out["pred_joint_count"] = int(pred_joints.shape[0])
    out["pred_bone_count"] = int(pred_bones.shape[0])
    out["joint_count_error"] = int(pred_joints.shape[0] - target_joints.shape[0])
    out["joint_count_abs_error"] = int(abs(pred_joints.shape[0] - target_joints.shape[0]))
    if pred_joints.shape[0] >= 1 and target_joints.shape[0] >= 1:
        out["root_l2"] = float(np.linalg.norm(pred_joints[0] - target_joints[0]))
    if pred_joints.shape[0] >= 2 and target_joints.shape[0] >= 2:
        # Rootless DFS serialization makes joint 1 the first generated child.
        out["joint1_l2"] = float(np.linalg.norm(pred_joints[1] - target_joints[1]))
    out["joint_chamfer"] = _nn_chamfer(pred_joints, target_joints)
    out["bone_chamfer"] = _set_chamfer(
        _bone_features(pred_joints, pred_parents),
        _bone_features(target_joints, target_parents),
    )
    out["official"] = _official_skeleton_metrics(
        pred_joints,
        pred_bones,
        target_joints,
        target_bones,
        continuous_range,
    )
    out["topology"] = _topology_metrics(pred_joints, pred_parents, target_joints, target_parents)
    return out


def _puppeteer_metric_range(tokenizer: Any) -> tuple[float, float]:
    low, high = _continuous_range(tokenizer)
    scale = max(float(getattr(tokenizer, "target_coord_scale", 1.0)), 1.0e-12)
    return low / scale, high / scale


def _apply_puppeteer_checkpoint_defaults(args: argparse.Namespace) -> dict[str, Any]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}) or {})
    defaults: dict[str, Any] = {
        "puppeteer_root": "third_party_references/Puppeteer",
        "puppeteer_checkpoint": None,
        "puppeteer_llm": "facebook/opt-350m",
        "n_discrete_size": 128,
        "n_max_joints": 128,
        "target_coord_scale": 0.25,
        "no_strict_target_range": False,
        "cond_length": 1024,
        "projector_heads": 8,
        "condition_projection": "identity",
        "attn_implementation": "flash_attention_2",
        "local_files_only": True,
        "allow_resize_positions": True,
        "random_init": False,
        "random_init_smoke": False,
        "tiny_random_decoder": False,
        "decoder_hidden_size": 256,
        "decoder_layers": 4,
        "decoder_heads": 8,
        "decoder_ffn_dim": 0,
        "decoder_dropout": 0.0,
        "decoder_attention_dropout": 0.0,
        "decoder_activation_dropout": 0.0,
        "decoder_layerdrop": 0.0,
        "decoder_norm_style": "pre",
        "motion_checkpointing": False,
        "no_joint_slot_embedding": False,
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, train_args.get(name, default))
    del ckpt
    return train_args


def _build_puppeteer_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from rigweave.dynamic_rig import (
        AnchorWiseAlternatingMotionEncoder,
        DynamicRigConditioner,
        FixedQuerySurfaceTokenizer,
        PuppeteerDynamicRigModel,
        PuppeteerJointTokenizer,
        import_puppeteer_decoder,
    )
    from train_puppeteer_dynamic_rig import build_decoder_config

    unirig_tokenizer = build_tokenizer(args.tokenizer_config)
    unirig = load_unirig(unirig_tokenizer, args.model_config, args.unirig_checkpoint)
    surface_tokenizer = FixedQuerySurfaceTokenizer(unirig.mesh_encoder, unirig.output_proj)
    motion_encoder = AnchorWiseAlternatingMotionEncoder(
        dim=unirig.hidden_size,
        depth=args.motion_depth,
        heads=args.motion_heads,
        register_tokens=args.register_tokens,
        max_frames=max(args.frames, 48),
        use_motion_features=args.use_motion_features,
        use_time_embedding=args.use_time_embedding,
        gradient_checkpointing=False,
    )
    conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
    SkeletonOPTConfig, SkeletonOPT = import_puppeteer_decoder(args.puppeteer_root)
    decoder = SkeletonOPT(build_decoder_config(args, SkeletonOPTConfig))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    target_aware_pos_embed = state.get("target_aware_pos_embed")
    joint_tokenizer = PuppeteerJointTokenizer(
        n_discrete_size=args.n_discrete_size,
        target_coord_scale=args.target_coord_scale,
        strict_range=not args.no_strict_target_range,
    )
    model = PuppeteerDynamicRigModel(
        conditioner=conditioner,
        decoder=decoder,
        tokenizer=joint_tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        cond_length=args.cond_length,
        projector_heads=args.projector_heads,
        condition_projection=args.condition_projection,
        max_joints=args.n_max_joints,
        use_joint_slot_embedding=not args.no_joint_slot_embedding,
        target_aware_pos_embed=target_aware_pos_embed,
    )
    if "target_aware_pos_embed" in state:
        current_slots = model.state_dict()["target_aware_pos_embed"]
        loaded_slots = state["target_aware_pos_embed"]
        if tuple(loaded_slots.shape) != tuple(current_slots.shape):
            resized = current_slots.detach().clone()
            rows = min(int(loaded_slots.shape[1]), int(resized.shape[1]))
            resized[:, :rows].copy_(loaded_slots[:, :rows].to(dtype=resized.dtype))
            state = dict(state)
            state["target_aware_pos_embed"] = resized
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Puppeteer checkpoint state mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    del ckpt
    return model


@torch.no_grad()
def _dynamic_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    max_new_tokens: int,
    target_ids: np.ndarray | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    count_guidance: str = "none",
    count_guidance_kwargs: dict[str, float] | None = None,
    action_guidance: str = "none",
    branch_prior_guidance: str = "none",
    branch_prior_guidance_kwargs: dict[str, float] | None = None,
    branch_parent_snap: bool = False,
) -> np.ndarray:
    from src.model.unirig_ar import VocabSwitchingLogitsProcessor

    branch_prior = None
    if branch_prior_guidance != "none":
        cond, branch_prior = model.build_condition(batch, return_branch_prior=True)
    else:
        cond = model.build_condition(batch)
    cond = cond.to(dtype=model.transformer.dtype)
    start_tokens = torch.tensor(
        [tokenizer.bos, tokenizer.cls_name_to_token(batch["cls"][0])],
        device=cond.device,
        dtype=torch.long,
    )
    start_attention = torch.ones((1, start_tokens.numel()), device=cond.device, dtype=torch.long)
    start_embed = model.token_inputs_embeds(start_tokens.unsqueeze(0), start_attention)
    prompt = torch.cat([cond, start_embed], dim=1)
    if (
        action_guidance != "none"
        or bool(getattr(model, "use_grammar_state_embedding", False))
        or bool(getattr(model, "uses_action_group_bias", False))
    ):
        branch_prior_expected_count = None
        branch_prior_expected_count = None
        branch_prior_coords = None
        if branch_prior_guidance != "none":
            if branch_prior_guidance not in {"count", "coords", "count_coords"}:
                raise ValueError(f"unknown branch prior guidance mode {branch_prior_guidance!r}")
            if branch_prior is None:
                raise ValueError("branch prior guidance requires a checkpoint with branch_prior enabled")
            if branch_prior_guidance in {"count", "count_coords"}:
                branch_prior_expected_count = float(
                    branch_prior["exist_logits"].sigmoid().sum(dim=1).detach().float().cpu()[0]
                )
            if branch_prior_guidance in {"coords", "count_coords"}:
                branch_prior_coords = {
                    "root_xyz": branch_prior["root_xyz"][0],
                    "child_xyz": branch_prior["child_xyz"][0],
                    "exist_logits": branch_prior["exist_logits"][0],
                }
        return _dynamic_generate_action_guided(
            model=model,
            tokenizer=tokenizer,
            cond=cond,
            prompt=prompt,
            start_tokens=start_tokens,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
            action_guidance=action_guidance,
            target_ids=target_ids,
            branch_prior_expected_count=branch_prior_expected_count,
            branch_prior_coords=branch_prior_coords,
            branch_prior_guidance_kwargs=branch_prior_guidance_kwargs,
        )
    processors: list[LogitsProcessor] = [
        VocabSwitchingLogitsProcessor(tokenizer=tokenizer, start_tokens=start_tokens)
    ]
    if branch_parent_snap:
        processors.append(BranchParentSnapLogitsProcessor(tokenizer=tokenizer, start_tokens=start_tokens))
    if branch_prior_guidance != "none":
        if branch_prior_guidance not in {"count", "coords", "count_coords"}:
            raise ValueError(f"unknown branch prior guidance mode {branch_prior_guidance!r}")
        if branch_prior is None:
            raise ValueError("branch prior guidance requires a checkpoint with branch_prior enabled")
        kwargs = branch_prior_guidance_kwargs or {}
        if branch_prior_guidance in {"count", "count_coords"}:
            expected_branch_count = float(branch_prior["exist_logits"].sigmoid().sum(dim=1).detach().float().cpu()[0])
            processors.append(
                BranchPriorCountGuidanceLogitsProcessor(
                    tokenizer=tokenizer,
                    start_tokens=start_tokens,
                    expected_branch_count=expected_branch_count,
                    branch_margin=float(kwargs.get("branch_margin", 0.5)),
                    branch_under_bias=float(kwargs.get("branch_under_bias", 1.5)),
                    branch_over_penalty=float(kwargs.get("branch_over_penalty", 6.0)),
                    min_completed_joints=int(kwargs.get("min_completed_joints", 1)),
                )
            )
        if branch_prior_guidance in {"coords", "count_coords"}:
            processors.append(
                BranchPriorCoordinateGuidanceLogitsProcessor(
                    tokenizer=tokenizer,
                    start_tokens=start_tokens,
                    root_xyz=branch_prior["root_xyz"][0],
                    child_xyz=branch_prior["child_xyz"][0],
                    exist_logits=branch_prior["exist_logits"][0],
                    coord_bias=float(kwargs.get("coord_bias", 2.0)),
                    coord_sigma=float(kwargs.get("coord_sigma", 6.0)),
                    min_exist_prob=float(kwargs.get("coord_min_exist_prob", 0.2)),
                )
            )
    if count_guidance != "none":
        if count_guidance == "oracle":
            if target_ids is None:
                raise ValueError("oracle count guidance requires target_ids")
            joint_count, branch_count = _target_structure_counts_from_ids(tokenizer, target_ids)
        elif count_guidance == "predicted":
            counts = model.predict_structure_counts(cond).detach().float().cpu()[0]
            joint_count = float(counts[0])
            branch_count = float(counts[1])
        else:
            raise ValueError(f"unknown count guidance mode {count_guidance!r}")
        processors.append(
            StructureCountGuidanceLogitsProcessor(
                tokenizer=tokenizer,
                start_tokens=start_tokens,
                joint_count=joint_count,
                branch_count=branch_count,
                **(count_guidance_kwargs or {}),
            )
        )
    generated = model.transformer.generate(
        inputs_embeds=prompt,
        bos_token_id=tokenizer.bos,
        eos_token_id=tokenizer.eos,
        pad_token_id=tokenizer.pad,
        logits_processor=LogitsProcessorList(processors),
        max_new_tokens=max_new_tokens,
        **(generation_kwargs or {}),
    )[0]
    ids = generated.detach().cpu()
    for token in reversed(start_tokens.detach().cpu()):
        ids = pad(ids, (1, 0), value=int(token))
    ids_np = ids.numpy().astype(np.int64)
    return ids_np


@torch.no_grad()
def _dynamic_generate_explicit_tree(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    max_joints: int,
    min_joints: int,
    count_guidance: str = "none",
    count_guidance_kwargs: dict[str, float] | None = None,
) -> dict[str, Any]:
    return model.generate_explicit_tree(
        batch,
        max_joints=max_joints,
        min_joints=min_joints,
        count_guidance=count_guidance,
        count_guidance_kwargs=count_guidance_kwargs,
    )


@torch.no_grad()
def _dynamic_generate_action_guided(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    cond: torch.Tensor,
    prompt: torch.Tensor,
    start_tokens: torch.Tensor,
    max_new_tokens: int,
    generation_kwargs: dict[str, Any] | None,
    action_guidance: str,
    target_ids: np.ndarray | None = None,
    branch_prior_expected_count: float | None = None,
    branch_prior_coords: dict[str, torch.Tensor] | None = None,
    branch_prior_guidance_kwargs: dict[str, float] | None = None,
) -> np.ndarray:
    if action_guidance not in {"none", "head", "oracle-kind"}:
        raise ValueError(f"unknown action guidance mode {action_guidance!r}")
    if action_guidance == "oracle-kind" and target_ids is None:
        raise ValueError("oracle-kind action guidance requires target_ids")

    generated: list[int] = []
    attention_mask = torch.ones((1, prompt.shape[1]), device=prompt.device, dtype=torch.long)
    output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
    )
    past = output.past_key_values
    next_logits = output.logits[:, -1, :]
    next_hidden = output.hidden_states[-1][:, -1]
    next_logits = model.apply_action_group_bias_row(next_logits, next_hidden, cond)

    for _ in range(max_new_tokens):
        full = np.asarray(start_tokens.detach().cpu().tolist() + generated, dtype=np.int64)
        possible = set(int(x) for x in tokenizer.next_posible_token(ids=full))
        if branch_prior_expected_count is not None:
            kwargs = branch_prior_guidance_kwargs or {}
            next_logits = _apply_branch_prior_count_bias_to_scores(
                tokenizer=tokenizer,
                start_tokens=start_tokens,
                generated=generated,
                scores=next_logits,
                expected_branch_count=float(branch_prior_expected_count),
                branch_margin=float(kwargs.get("branch_margin", kwargs.get("margin", 0.5))),
                branch_under_bias=float(kwargs.get("branch_under_bias", 1.5)),
                branch_over_penalty=float(kwargs.get("branch_over_penalty", 6.0)),
                min_completed_joints=int(kwargs.get("min_completed_joints", kwargs.get("min_joints", 1))),
            )
        if branch_prior_coords is not None:
            kwargs = branch_prior_guidance_kwargs or {}
            next_logits = _apply_branch_prior_coordinate_bias_to_scores(
                tokenizer=tokenizer,
                start_tokens=start_tokens,
                generated=generated,
                scores=next_logits,
                root_xyz=branch_prior_coords["root_xyz"].detach().float().cpu().numpy(),
                child_xyz=branch_prior_coords["child_xyz"].detach().float().cpu().numpy(),
                exist_probs=branch_prior_coords["exist_logits"].detach().sigmoid().float().cpu().numpy(),
                coord_bias=float(kwargs.get("coord_bias", 2.0)),
                coord_sigma=float(kwargs.get("coord_sigma", 6.0)),
                min_exist_prob=float(kwargs.get("coord_min_exist_prob", 0.2)),
            )
        logits = _filter_logits_to_ids(next_logits, possible)
        groups = _possible_action_groups(tokenizer, possible)
        if action_guidance != "none" and len(groups) > 1:
            if action_guidance == "oracle-kind":
                assert target_ids is not None
                target_pos = len(start_tokens) + len(generated)
                if target_pos < int(target_ids.shape[0]):
                    action = model._token_action_group(int(target_ids[target_pos]))
                else:
                    action = 0
            else:
                action_logits = model.structure_action_head(next_hidden.to(dtype=torch.float32))
                action_mask = torch.full_like(action_logits, -float("inf"))
                action_mask[:, list(groups)] = 0.0
                action = int(torch.argmax(action_logits + action_mask, dim=-1).item())
            allowed = _tokens_for_action_group(tokenizer, possible, action)
            if allowed:
                logits = _filter_logits_to_ids(logits, allowed)

        next_token = _sample_or_argmax(logits, generation_kwargs)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == int(tokenizer.eos):
            break

        attention_mask = torch.ones((1, prompt.shape[1] + len(generated)), device=prompt.device, dtype=torch.long)
        if bool(getattr(model, "use_grammar_state_embedding", False)):
            full_prefix = start_tokens.detach().cpu().tolist() + generated
            next_embed = model.next_token_embed_with_state(full_prefix, prompt.device).to(dtype=model.transformer.dtype)
            output = model.transformer(
                inputs_embeds=next_embed,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
        else:
            output = model.transformer(
                input_ids=next_token.to(device=prompt.device, dtype=torch.long),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
        past = output.past_key_values
        next_logits = output.logits[:, -1, :]
        next_hidden = output.hidden_states[-1][:, -1]
        next_logits = model.apply_action_group_bias_row(next_logits, next_hidden, cond)

    return np.asarray(start_tokens.detach().cpu().tolist() + generated, dtype=np.int64)


@torch.no_grad()
def _dynamic_generate_logit_prior(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    max_new_tokens: int,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    prior_weight: float = 0.5,
    prior_scope: str = "all",
    static_unirig: torch.nn.Module | None = None,
) -> np.ndarray:
    """Greedy/product-of-experts generation with a static UniRig logit prior.

    The dynamic condition is still the primary model.  At every AR step, the
    same prefix is also evaluated under a static-mesh condition and its logits
    are added as a stabilizing prior before the UniRig grammar mask is applied.
    This tests whether preserving the static generator prior fixes free-running
    loops without falling back after a failed decode.
    """

    if generation_kwargs:
        if generation_kwargs.get("num_beams", 1) != 1 or generation_kwargs.get("do_sample", False):
            raise ValueError("logit-prior generation currently supports greedy num_beams=1 only")

    dyn_cond = model.build_condition(batch).to(dtype=model.transformer.dtype)
    if static_unirig is None:
        static_cond = model.build_static_condition(batch).to(device=dyn_cond.device, dtype=model.transformer.dtype)
        static_transformer = model.transformer
        static_embedding = model.transformer.get_input_embeddings()
    else:
        static_cond = static_unirig.encode_mesh_cond(
            vertices=batch["frame_vertices"][:, 0],
            normals=batch["vertex_normals"][:, 0],
        ).to(device=dyn_cond.device, dtype=static_unirig.transformer.dtype)
        static_transformer = static_unirig.transformer
        static_embedding = static_unirig.transformer.get_input_embeddings()
    dynamic_uses_state_embed = bool(getattr(model, "use_grammar_state_embedding", False))
    dynamic_uses_action_bias = bool(getattr(model, "uses_action_group_bias", False))

    start_tokens = torch.tensor(
        [tokenizer.bos, tokenizer.cls_name_to_token(batch["cls"][0])],
        device=dyn_cond.device,
        dtype=torch.long,
    )
    start_attention = torch.ones((1, start_tokens.numel()), device=dyn_cond.device, dtype=torch.long)
    dyn_start_embed = model.token_inputs_embeds(start_tokens.unsqueeze(0), start_attention)
    static_start_embed = static_embedding(start_tokens.unsqueeze(0)).to(dtype=static_cond.dtype)
    dyn_prompt = torch.cat([dyn_cond, dyn_start_embed], dim=1)
    static_prompt = torch.cat([static_cond, static_start_embed], dim=1)

    dyn_attention = torch.ones((1, dyn_prompt.shape[1]), device=dyn_cond.device, dtype=torch.long)
    static_attention = torch.ones((1, static_prompt.shape[1]), device=dyn_cond.device, dtype=torch.long)
    dyn_out = model.transformer(
        inputs_embeds=dyn_prompt,
        attention_mask=dyn_attention,
        use_cache=True,
        output_hidden_states=dynamic_uses_action_bias,
    )
    static_out = static_transformer(inputs_embeds=static_prompt, attention_mask=static_attention, use_cache=True)
    dyn_past = dyn_out.past_key_values
    static_past = static_out.past_key_values
    dyn_logits = dyn_out.logits[:, -1, :]
    if dynamic_uses_action_bias:
        dyn_logits = model.apply_action_group_bias_row(dyn_logits, dyn_out.hidden_states[-1][:, -1], dyn_cond)
    static_logits = static_out.logits[:, -1, :].to(device=dyn_logits.device, dtype=dyn_logits.dtype)

    generated: list[int] = []
    prior_weight = float(prior_weight)
    for _ in range(int(max_new_tokens)):
        full = np.asarray(start_tokens.detach().cpu().tolist() + generated, dtype=np.int64)
        possible = set(int(x) for x in tokenizer.next_posible_token(ids=full))
        fused = _fuse_dynamic_static_logits(
            tokenizer=tokenizer,
            possible=possible,
            dynamic_logits=dyn_logits,
            static_logits=static_logits,
            prior_weight=prior_weight,
            prior_scope=prior_scope,
        )
        next_token = _sample_or_argmax(fused, generation_kwargs)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == int(tokenizer.eos):
            break

        dyn_attention = torch.ones((1, dyn_prompt.shape[1] + len(generated)), device=dyn_cond.device, dtype=torch.long)
        static_attention = torch.ones(
            (1, static_prompt.shape[1] + len(generated)), device=dyn_cond.device, dtype=torch.long
        )
        if dynamic_uses_state_embed:
            full_prefix = start_tokens.detach().cpu().tolist() + generated
            dyn_next_embed = model.next_token_embed_with_state(full_prefix, dyn_cond.device).to(dtype=model.transformer.dtype)
            dyn_out = model.transformer(
                inputs_embeds=dyn_next_embed,
                attention_mask=dyn_attention,
                past_key_values=dyn_past,
                use_cache=True,
                output_hidden_states=dynamic_uses_action_bias,
            )
        else:
            dyn_out = model.transformer(
                input_ids=next_token.to(device=dyn_cond.device, dtype=torch.long),
                attention_mask=dyn_attention,
                past_key_values=dyn_past,
                use_cache=True,
                output_hidden_states=dynamic_uses_action_bias,
            )
        if static_unirig is None and dynamic_uses_state_embed:
            static_next_embed = model.next_token_embed_with_state(full_prefix, dyn_cond.device).to(dtype=model.transformer.dtype)
            static_out = static_transformer(
                inputs_embeds=static_next_embed,
                attention_mask=static_attention,
                past_key_values=static_past,
                use_cache=True,
            )
        else:
            static_out = static_transformer(
                input_ids=next_token.to(device=dyn_cond.device, dtype=torch.long),
                attention_mask=static_attention,
                past_key_values=static_past,
                use_cache=True,
            )
        dyn_past = dyn_out.past_key_values
        static_past = static_out.past_key_values
        dyn_logits = dyn_out.logits[:, -1, :]
        if dynamic_uses_action_bias:
            dyn_logits = model.apply_action_group_bias_row(dyn_logits, dyn_out.hidden_states[-1][:, -1], dyn_cond)
        static_logits = static_out.logits[:, -1, :].to(device=dyn_logits.device, dtype=dyn_logits.dtype)

    return np.asarray(start_tokens.detach().cpu().tolist() + generated, dtype=np.int64)


@torch.no_grad()
def _static_generate(
    unirig: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    max_new_tokens: int,
    generation_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    from src.model.unirig_ar import VocabSwitchingLogitsProcessor

    cond = unirig.encode_mesh_cond(
        vertices=batch["frame_vertices"][:, 0],
        normals=batch["vertex_normals"][:, 0],
    ).to(dtype=unirig.transformer.dtype)
    start_tokens = torch.tensor(
        [tokenizer.bos, tokenizer.cls_name_to_token(batch["cls"][0])],
        device=cond.device,
        dtype=torch.long,
    )
    start_embed = unirig.transformer.get_input_embeddings()(start_tokens.unsqueeze(0)).to(dtype=unirig.transformer.dtype)
    prompt = torch.cat([cond, start_embed], dim=1)
    processor = VocabSwitchingLogitsProcessor(tokenizer=tokenizer, start_tokens=start_tokens)
    generated = unirig.transformer.generate(
        inputs_embeds=prompt,
        bos_token_id=tokenizer.bos,
        eos_token_id=tokenizer.eos,
        pad_token_id=tokenizer.pad,
        logits_processor=LogitsProcessorList([processor]),
        max_new_tokens=max_new_tokens,
        **(generation_kwargs or {}),
    )[0]
    ids = generated.detach().cpu()
    for token in reversed(start_tokens.detach().cpu()):
        ids = pad(ids, (1, 0), value=int(token))
    ids_np = ids.numpy().astype(np.int64)
    return ids_np


def _summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    vals: dict[str, list[float]] = {
        "joint_count_abs_error": [],
        "joint_count_error": [],
        "root_l2": [],
        "joint1_l2": [],
        "joint_chamfer_mean": [],
        "joint_chamfer_p95": [],
        "bone_chamfer_mean": [],
        "bone_chamfer_p95": [],
        "j2j": [],
        "j2b": [],
        "b2b": [],
        "bone_cd": [],
        "topology_edge_precision": [],
        "topology_edge_recall": [],
        "topology_edge_f1": [],
        "generated_token_count": [],
        "target_token_count": [],
        "eos_new_token_index": [],
        "token_count_delta": [],
    }
    detok_ok = 0
    has_eos = 0
    hit_max = 0
    for row in rows:
        block = row[prefix]
        detok_ok += int(block.get("detokenize_ok", False))
        ids = block.get("generated_ids", [])
        block_has_eos = bool(block.get("has_eos", False)) or row["eos"] in ids
        has_eos += int(block_has_eos)
        hit_max += int(block.get("hit_max_without_eos", False))
        metrics = block.get("metrics", {})
        for key in ("joint_count_abs_error", "joint_count_error", "root_l2", "joint1_l2"):
            if key in metrics:
                vals[key].append(float(metrics[key]))
        if "joint_chamfer" in metrics:
            vals["joint_chamfer_mean"].append(float(metrics["joint_chamfer"]["mean"]))
            vals["joint_chamfer_p95"].append(float(metrics["joint_chamfer"]["p95"]))
        if "bone_chamfer" in metrics:
            vals["bone_chamfer_mean"].append(float(metrics["bone_chamfer"]["mean"]))
            vals["bone_chamfer_p95"].append(float(metrics["bone_chamfer"]["p95"]))
        official = metrics.get("official", {})
        if isinstance(official, dict):
            for key in ("j2j", "j2b", "b2b", "bone_cd"):
                value = official.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    vals[key].append(float(value))
        topology = metrics.get("topology", {})
        if isinstance(topology, dict):
            for out_key, in_key in (
                ("topology_edge_precision", "edge_precision"),
                ("topology_edge_recall", "edge_recall"),
                ("topology_edge_f1", "edge_f1"),
            ):
                value = topology.get(in_key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    vals[out_key].append(float(value))
        if "generated_new_tokens" in block:
            vals["generated_token_count"].append(float(block["generated_new_tokens"]))
        if "target_token_count" in row:
            vals["target_token_count"].append(float(row["target_token_count"]))
        eos_new_token_index = block.get("eos_new_token_index")
        if isinstance(eos_new_token_index, (int, float)) and math.isfinite(float(eos_new_token_index)):
            vals["eos_new_token_index"].append(float(eos_new_token_index))
        if "target_token_count" in row and "generated_new_tokens" in block:
            vals["token_count_delta"].append(float(block["generated_new_tokens"] - max(int(row["target_token_count"]) - 2, 0)))

    def agg(xs: list[float]) -> dict[str, float | int | None]:
        if not xs:
            return {"count": 0, "mean": None, "median": None}
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "count": len(rows),
        "detokenize_ok": detok_ok,
        "has_eos": has_eos,
        "hit_max_without_eos": hit_max,
        **{key: agg(value) for key, value in vals.items()},
    }


def _run_model(
    name: str,
    generate_fn: Any,
    rows: list[dict[str, Any]],
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
    seed: int,
) -> None:
    continuous_range = _continuous_range(tokenizer)
    for idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        target = _detokenize_target(tokenizer, batch)
        row = rows[idx]
        if "selected_frames" in batch:
            row["selected_frames"] = batch["selected_frames"][0].detach().cpu().numpy().astype(int).tolist()
            row["query_frame"] = int(row["selected_frames"][0])
        if "query_center" in batch:
            row["query_center"] = batch["query_center"][0].detach().cpu().numpy().astype(float).tolist()
        if "query_scale" in batch:
            row["query_scale"] = float(batch["query_scale"][0].detach().cpu())
        target_ids = batch["input_ids"][0][batch["attention_mask"][0].bool()].detach().cpu().numpy().astype(np.int64)
        row["target_token_count"] = int(target_ids.shape[0])
        row["target_ids"] = target_ids.astype(int).tolist()
        target_eos_hits = np.flatnonzero(target_ids == int(tokenizer.eos))
        row["target_eos_index"] = int(target_eos_hits[0]) if target_eos_hits.size else None
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + idx)
        torch.manual_seed(seed + idx)
        block: dict[str, Any] = {"detokenize_ok": False}
        try:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                generated = generate_fn(batch, target_ids)
            if isinstance(generated, np.ndarray):
                ids = generated
                block["generated_ids"] = ids.astype(int).tolist()
                block["generated_new_tokens"] = int(ids.shape[0] - 2)
                eos_hits = np.flatnonzero(ids == int(tokenizer.eos))
                block["has_eos"] = bool(eos_hits.size)
                block["eos_index"] = int(eos_hits[0]) if eos_hits.size else None
                block["eos_new_token_index"] = int(eos_hits[0] - 2) if eos_hits.size and int(eos_hits[0]) >= 2 else None
                block["hit_max_without_eos"] = bool(block["generated_new_tokens"] >= max_new_tokens and not eos_hits.size)
                pred = tokenizer.detokenize(ids)
                block["detokenize_ok"] = True
            elif isinstance(generated, dict) and "joints" in generated and "parents" in generated:
                joints = np.asarray(generated["joints"], dtype=np.float32)
                parents = [None if int(p) < 0 else int(p) for p in generated["parents"]]
                block["generated_ids"] = []
                block["generated_new_tokens"] = int(generated.get("steps", joints.shape[0]))
                block["has_eos"] = bool(generated.get("has_eos", False))
                block["eos_index"] = None
                block["eos_new_token_index"] = int(generated.get("steps", 0)) if block["has_eos"] else None
                block["hit_max_without_eos"] = bool(
                    block["generated_new_tokens"] >= max_new_tokens and not block["has_eos"]
                )
                pred = SimpleNamespace(joints=joints, parents=parents)
                block["detokenize_ok"] = True
            else:
                raise TypeError(f"unsupported generation result type: {type(generated)!r}")
            block["metrics"] = _output_metrics(pred, target, continuous_range)
        except Exception as exc:
            block["error"] = repr(exc)
            block["metrics"] = _output_metrics(None, target, continuous_range)
        row[name] = block
        row["target_joint_count"] = int(target.joints.shape[0])
        row["max_new_tokens"] = int(max_new_tokens)
        row["eos"] = int(tokenizer.eos)
        print(
            json.dumps(
                {
                    "model": name,
                    "index": idx,
                    "target_joint_count": row["target_joint_count"],
                    "detokenize_ok": block.get("detokenize_ok", False),
                    "pred_joint_count": block.get("metrics", {}).get("pred_joint_count"),
                    "joint_count_error": block.get("metrics", {}).get("joint_count_error"),
                    "joint_chamfer_mean": block.get("metrics", {}).get("joint_chamfer", {}).get("mean"),
                    "j2j": block.get("metrics", {}).get("official", {}).get("j2j"),
                    "b2b": block.get("metrics", {}).get("official", {}).get("b2b"),
                    "topology_f1": block.get("metrics", {}).get("topology", {}).get("edge_f1"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def _run_puppeteer_model(
    name: str,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
    max_joints: int,
    seed: int,
    generation_kwargs: dict[str, Any],
) -> None:
    tokenizer = model.tokenizer
    continuous_range = _puppeteer_metric_range(tokenizer)
    for idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        row = rows[idx]
        joint_count = int(batch["joint_count"][0].detach().cpu())
        target_joints = batch["target_joints"][0, :joint_count].detach().cpu().numpy().astype(np.float32)
        target_parent_values = batch["target_parents"][0, :joint_count].detach().cpu().numpy().astype(np.int64)
        target_parents = [None if int(p) < 0 else int(p) for p in target_parent_values.tolist()]
        target = SimpleNamespace(joints=target_joints, parents=target_parents)
        if "selected_frames" in batch:
            row["selected_frames"] = batch["selected_frames"][0].detach().cpu().numpy().astype(int).tolist()
            row["query_frame"] = int(row["selected_frames"][0])
        if "query_center" in batch:
            row["query_center"] = batch["query_center"][0].detach().cpu().numpy().astype(float).tolist()
        if "query_scale" in batch:
            row["query_scale"] = float(batch["query_scale"][0].detach().cpu())
        target_ids, _roles = tokenizer.encode_one(
            batch["target_joints"][0].to(device),
            batch["target_parents"][0].to(device),
            joint_count,
            path=str(row["path"]),
        )
        target_ids = torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id], device=target_ids.device, dtype=torch.long),
                target_ids,
                torch.tensor([tokenizer.eos_token_id], device=target_ids.device, dtype=torch.long),
            ],
            dim=0,
        )
        row["target_token_count"] = int(target_ids.numel())
        row["target_ids"] = target_ids.detach().cpu().numpy().astype(int).tolist()
        row["target_eos_index"] = int(row["target_token_count"] - 1)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + idx)
        torch.manual_seed(seed + idx)
        block: dict[str, Any] = {"detokenize_ok": False}
        try:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                generated = model.generate_skeleton(
                    batch,
                    max_new_tokens=max_new_tokens,
                    max_joints=max_joints,
                    generation_kwargs=generation_kwargs,
                )
            joints = np.asarray(generated["joints"], dtype=np.float32)
            parents = [None if int(p) < 0 else int(p) for p in np.asarray(generated["parents"], dtype=np.int64).tolist()]
            generated_ids = np.asarray(generated.get("generated_ids", []), dtype=np.int64)
            block["generated_ids"] = generated_ids.astype(int).tolist()
            block["generated_new_tokens"] = int(generated.get("steps", max(int(generated_ids.shape[0]) - 1, 0)))
            block["has_eos"] = bool(generated.get("has_eos", False))
            eos_hits = np.flatnonzero(generated_ids == int(tokenizer.eos_token_id))
            block["eos_index"] = int(eos_hits[0]) if eos_hits.size else None
            block["eos_new_token_index"] = int(eos_hits[0] - 1) if eos_hits.size and int(eos_hits[0]) >= 1 else None
            block["hit_max_without_eos"] = bool(block["generated_new_tokens"] >= max_new_tokens and not block["has_eos"])
            pred = SimpleNamespace(joints=joints, parents=parents)
            block["detokenize_ok"] = True
            block["metrics"] = _output_metrics(pred, target, continuous_range)
        except Exception as exc:
            block["error"] = repr(exc)
            block["metrics"] = _output_metrics(None, target, continuous_range)
        row[name] = block
        row["target_joint_count"] = int(target_joints.shape[0])
        row["max_new_tokens"] = int(max_new_tokens)
        row["eos"] = int(tokenizer.eos_token_id)
        print(
            json.dumps(
                {
                    "model": name,
                    "index": idx,
                    "target_joint_count": row["target_joint_count"],
                    "detokenize_ok": block.get("detokenize_ok", False),
                    "has_eos": block.get("has_eos", False),
                    "hit_max_without_eos": block.get("hit_max_without_eos", False),
                    "pred_joint_count": block.get("metrics", {}).get("pred_joint_count"),
                    "joint_count_error": block.get("metrics", {}).get("joint_count_error"),
                    "joint_chamfer_mean": block.get("metrics", {}).get("joint_chamfer", {}).get("mean"),
                    "j2j": block.get("metrics", {}).get("official", {}).get("j2j"),
                    "b2b": block.get("metrics", {}).get("official", {}).get("b2b"),
                    "topology_f1": block.get("metrics", {}).get("topology", {}).get("edge_f1"),
                    "error": block.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"do_sample": bool(args.do_sample)}
    if args.num_beams is not None:
        kwargs["num_beams"] = int(args.num_beams)
    if args.num_return_sequences is not None:
        kwargs["num_return_sequences"] = int(args.num_return_sequences)
    if args.temperature is not None:
        kwargs["temperature"] = float(args.temperature)
    if args.top_k is not None:
        kwargs["top_k"] = int(args.top_k)
    if args.top_p is not None:
        kwargs["top_p"] = float(args.top_p)
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(args.repetition_penalty)
    if args.no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate skeletons and compare dynamic checkpoints with static UniRig.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("EVOWEAVE_TEST_MANIFEST", "rigweave/configs/MISSING_TEST_MANIFEST.jsonl")),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--count-guidance", choices=["none", "oracle", "predicted"], default="none")
    parser.add_argument("--count-guidance-joint-margin", type=float, default=1.0)
    parser.add_argument("--count-guidance-early-eos-penalty", type=float, default=6.0)
    parser.add_argument("--count-guidance-eos-bias", type=float, default=6.0)
    parser.add_argument("--count-guidance-branch-over-penalty", type=float, default=4.0)
    parser.add_argument("--count-guidance-branch-under-bias", type=float, default=0.5)
    parser.add_argument("--action-guidance", choices=["none", "head", "oracle-kind"], default="none")
    parser.add_argument("--branch-prior-guidance", choices=["none", "count", "coords", "count_coords"], default="none")
    parser.add_argument("--branch-prior-guidance-margin", type=float, default=0.5)
    parser.add_argument("--branch-prior-guidance-under-bias", type=float, default=1.5)
    parser.add_argument("--branch-prior-guidance-over-penalty", type=float, default=6.0)
    parser.add_argument("--branch-prior-guidance-min-joints", type=int, default=1)
    parser.add_argument("--branch-prior-guidance-coord-bias", type=float, default=2.0)
    parser.add_argument("--branch-prior-guidance-coord-sigma", type=float, default=6.0)
    parser.add_argument("--branch-prior-guidance-coord-min-exist-prob", type=float, default=0.2)
    parser.add_argument("--branch-parent-snap", action="store_true")
    parser.add_argument("--dynamic-decode-mode", choices=["flat", "explicit_tree", "both", "puppeteer"], default="flat")
    parser.add_argument("--explicit-tree-max-joints", type=int, default=192)
    parser.add_argument("--explicit-tree-min-joints", type=int, default=1)
    parser.add_argument(
        "--logit-prior",
        choices=["none", "self-static", "official-static"],
        default="none",
        help="Fuse dynamic decoder logits with static-mesh logits at every autoregressive step.",
    )
    parser.add_argument("--logit-prior-weight", type=float, default=0.5)
    parser.add_argument(
        "--logit-prior-scope",
        choices=["all", "structure", "eos"],
        default="all",
        help="Which token groups receive the static logit prior.",
    )
    parser.add_argument("--tokenizer-config", type=Path, default=Path(os.environ.get("EVOWEAVE_TOKENIZER_CONFIG", "external/UniRig/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml")))
    parser.add_argument("--model-config", type=Path, default=Path(os.environ.get("EVOWEAVE_MODEL_CONFIG", "external/UniRig/configs/model/unirig_ar_350m_1024_81920_float32.yaml")))
    parser.add_argument("--unirig-checkpoint", type=Path, default=Path(os.environ.get("EVOWEAVE_UNIRIG_CKPT", "external/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt")))
    parser.add_argument("--puppeteer-root", type=Path, default=None)
    parser.add_argument("--puppeteer-checkpoint", type=Path, default=None)
    parser.add_argument("--puppeteer-llm", type=str, default=None)
    parser.add_argument("--n-discrete-size", type=int, default=None)
    parser.add_argument("--n-max-joints", type=int, default=None)
    parser.add_argument("--target-coord-scale", type=float, default=None)
    parser.add_argument("--no-strict-target-range", action="store_true", default=None)
    parser.add_argument("--cond-length", type=int, default=None)
    parser.add_argument("--projector-heads", type=int, default=None)
    parser.add_argument(
        "--condition-projection",
        choices=["cross_attention", "identity"],
        default=None,
    )
    parser.add_argument("--attn-implementation", choices=["flash_attention_2"], default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-resize-positions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--random-init-smoke", action="store_true", default=None)
    parser.add_argument("--tiny-random-decoder", action="store_true", default=None)
    parser.add_argument("--decoder-hidden-size", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--decoder-ffn-dim", type=int, default=None)
    parser.add_argument("--decoder-dropout", type=float, default=None)
    parser.add_argument("--decoder-attention-dropout", type=float, default=None)
    parser.add_argument("--decoder-activation-dropout", type=float, default=None)
    parser.add_argument("--decoder-layerdrop", type=float, default=None)
    parser.add_argument("--decoder-norm-style", choices=["config", "pre", "post"], default=None)
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
    parser.add_argument("--target-active-skin-only", action="store_true", default=None)
    parser.add_argument("--active-skin-threshold", type=float, default=None)
    parser.add_argument("--target-start-policy", choices=["joint0"], default=None)
    parser.add_argument("--target-root-policy", choices=["legacy"], default=None)
    parser.add_argument(
        "--input-space-policy",
        choices=["mesh_query_bbox"],
        default=None,
        help=(
            "How to construct the dynamic mesh input coordinate system. "
            "Strict rootless training uses query mesh bbox normalization only."
        ),
    )
    parser.add_argument(
        "--condition-fusion",
        choices=[
            "dynamic",
            "static_blend",
            "static_cross_attn",
            "static_cross_attn_zero",
            "anchor_motion_residual_zero",
        ],
        default=None,
    )
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
    parser.add_argument(
        "--explicit-tree-topology-mode",
        choices=["geometry", "topology", "hybrid", "split", "planner", "topomlp"],
        default=None,
    )
    parser.add_argument("--explicit-tree-coordinate-mode", choices=["absolute", "parent_delta"], default=None)
    parser.add_argument("--explicit-tree-action-eos-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-action-child-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-action-branch-loss-weight", type=float, default=None)
    parser.add_argument("--explicit-tree-xyz-loss-weight", type=float, default=None)
    parser.add_argument("--use-grammar-state-embedding", action="store_true", default=None)
    parser.add_argument("--use-action-group-bias", action="store_true", default=None)
    parser.add_argument("--use-condition-action-group-bias", action="store_true", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--skip-dynamic-baseline", action="store_true")
    parser.add_argument("--skip-static-base", action="store_true")
    parser.add_argument("--deterministic-query", action="store_true", help="Deprecated: deterministic query is the default.")
    parser.add_argument("--random-query", action="store_true")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()
    apply_checkpoint_eval_defaults(args)

    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    generation_kwargs = _generation_kwargs(args)
    if args.dynamic_decode_mode == "puppeteer":
        _apply_puppeteer_checkpoint_defaults(args)
        from rigweave.dynamic_rig import PuppeteerDynamicRigDataset, puppeteer_dynamic_collate

        dataset = PuppeteerDynamicRigDataset(
            args.manifest,
            frame_count=args.frames,
            limit=args.limit,
            random_query=args.random_query,
            seed=args.seed,
            motion_fps_ratio=args.motion_fps_ratio,
            motion_vertex_samples=args.motion_vertex_samples,
            max_joints=args.n_max_joints,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=puppeteer_dynamic_collate,
        )
        base_rows = [{"index": idx, "path": str(path)} for idx, path in enumerate(dataset.paths)]
        model = _build_puppeteer_model(args, device)
        _run_puppeteer_model(
            "dynamic_puppeteer",
            model,
            base_rows,
            loader,
            device,
            amp_dtype,
            args.max_new_tokens,
            args.n_max_joints,
            args.seed,
            generation_kwargs,
        )
        summary: dict[str, Any] = {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "limit": int(args.limit),
            "eval_contract": {
                "frames": args.frames,
                "surface_samples": args.surface_samples,
                "vertex_samples": args.vertex_samples,
                "query_tokens": args.query_tokens,
                "register_tokens": args.register_tokens,
                "motion_depth": args.motion_depth,
                "motion_heads": args.motion_heads,
                "use_motion_features": args.use_motion_features,
                "use_time_embedding": args.use_time_embedding,
                "motion_fps_ratio": args.motion_fps_ratio,
                "motion_vertex_samples": args.motion_vertex_samples,
                "input_space_policy": args.input_space_policy,
                "random_query": args.random_query,
                "generation_kwargs": generation_kwargs,
                "dynamic_decode_mode": args.dynamic_decode_mode,
                "n_discrete_size": args.n_discrete_size,
                "n_max_joints": args.n_max_joints,
                "target_coord_scale": args.target_coord_scale,
                "puppeteer_root": str(args.puppeteer_root),
                "puppeteer_llm": args.puppeteer_llm,
                "cond_length": args.cond_length,
            },
            "dynamic_puppeteer": _summarize(base_rows, "dynamic_puppeteer"),
        }
        result = {"summary": summary, "rows": base_rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    tokenizer = build_tokenizer(args.tokenizer_config)
    count_guidance_kwargs = {
        "joint_margin": args.count_guidance_joint_margin,
        "early_eos_penalty": args.count_guidance_early_eos_penalty,
        "eos_bias": args.count_guidance_eos_bias,
        "branch_over_penalty": args.count_guidance_branch_over_penalty,
        "branch_under_bias": args.count_guidance_branch_under_bias,
    }
    branch_prior_guidance_kwargs = {
        "branch_margin": args.branch_prior_guidance_margin,
        "branch_under_bias": args.branch_prior_guidance_under_bias,
        "branch_over_penalty": args.branch_prior_guidance_over_penalty,
        "min_completed_joints": args.branch_prior_guidance_min_joints,
        "coord_bias": args.branch_prior_guidance_coord_bias,
        "coord_sigma": args.branch_prior_guidance_coord_sigma,
        "coord_min_exist_prob": args.branch_prior_guidance_coord_min_exist_prob,
    }
    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=args.frames,
        limit=args.limit,
        random_query=args.random_query,
        seed=args.seed,
        motion_fps_ratio=args.motion_fps_ratio,
        motion_vertex_samples=args.motion_vertex_samples,
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
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )
    base_rows = [{"index": idx, "path": str(path)} for idx, path in enumerate(dataset.paths)]

    dynamic_model = None
    if not args.skip_dynamic:
        dynamic_model = _build_dynamic_model(args, tokenizer, device)
    needs_static_unirig = (not args.skip_static_base) or args.logit_prior == "official-static"
    static_unirig = None
    if needs_static_unirig:
        static_unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint).to(device).eval()

    if not args.skip_dynamic and not args.skip_dynamic_baseline and args.dynamic_decode_mode in {"flat", "both"}:
        assert dynamic_model is not None
        _run_model(
            "dynamic",
            lambda batch, target_ids: _dynamic_generate(
                dynamic_model,
                tokenizer,
                batch,
                args.max_new_tokens,
                target_ids,
                generation_kwargs,
                count_guidance=args.count_guidance,
                count_guidance_kwargs=count_guidance_kwargs,
                action_guidance=args.action_guidance,
                branch_prior_guidance=args.branch_prior_guidance,
                branch_prior_guidance_kwargs=branch_prior_guidance_kwargs,
                branch_parent_snap=args.branch_parent_snap,
            ),
            base_rows,
            loader,
            tokenizer,
            device,
            amp_dtype,
            args.max_new_tokens,
            args.seed,
        ),

    if not args.skip_dynamic and args.dynamic_decode_mode in {"explicit_tree", "both"}:
        assert dynamic_model is not None
        _run_model(
            "dynamic_explicit_tree",
            lambda batch, target_ids: _dynamic_generate_explicit_tree(
                dynamic_model,
                batch,
                max_joints=args.explicit_tree_max_joints,
                min_joints=args.explicit_tree_min_joints,
                count_guidance=args.count_guidance,
                count_guidance_kwargs=count_guidance_kwargs,
            ),
            base_rows,
            loader,
            tokenizer,
            device,
            amp_dtype,
            args.explicit_tree_max_joints,
            args.seed,
        )

    if args.logit_prior != "none" and not args.skip_dynamic:
        assert dynamic_model is not None
        prior_static = static_unirig if args.logit_prior == "official-static" else None
        _run_model(
            "dynamic_logit_prior",
            lambda batch, target_ids: _dynamic_generate_logit_prior(
                dynamic_model,
                tokenizer,
                batch,
                args.max_new_tokens,
                generation_kwargs,
                prior_weight=args.logit_prior_weight,
                prior_scope=args.logit_prior_scope,
                static_unirig=prior_static,
            ),
            base_rows,
            loader,
            tokenizer,
            device,
            amp_dtype,
            args.max_new_tokens,
            args.seed,
        )

    if not args.skip_static_base:
        assert static_unirig is not None
        _run_model(
            "static_base",
            lambda batch, target_ids: _static_generate(static_unirig, tokenizer, batch, args.max_new_tokens, generation_kwargs),
            base_rows,
            loader,
            tokenizer,
            device,
            amp_dtype,
            args.max_new_tokens,
            args.seed,
        )

    summary: dict[str, Any] = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "limit": int(args.limit),
        "eval_contract": {
            "frames": args.frames,
            "surface_samples": args.surface_samples,
            "vertex_samples": args.vertex_samples,
            "query_tokens": args.query_tokens,
            "register_tokens": args.register_tokens,
            "motion_depth": args.motion_depth,
            "motion_heads": args.motion_heads,
            "use_motion_features": args.use_motion_features,
            "use_time_embedding": args.use_time_embedding,
            "motion_fps_ratio": args.motion_fps_ratio,
            "motion_vertex_samples": args.motion_vertex_samples,
            "target_active_skin_only": args.target_active_skin_only,
            "target_root_policy": args.target_root_policy,
            "active_skin_threshold": args.active_skin_threshold,
            "condition_fusion": args.condition_fusion,
            "condition_fusion_heads": args.condition_fusion_heads,
            "condition_fusion_gate_init": args.condition_fusion_gate_init,
            "condition_fusion_depth": args.condition_fusion_depth,
            "condition_static_blend_weight": args.condition_static_blend_weight,
            "branch_prior_proposals": args.branch_prior_proposals,
            "branch_prior_heads": args.branch_prior_heads,
            "branch_prior_loss_weight": args.branch_prior_loss_weight,
            "branch_prior_coord_loss_weight": args.branch_prior_coord_loss_weight,
            "explicit_tree_loss_weight": args.explicit_tree_loss_weight,
            "explicit_tree_generated_prefix_weight": args.explicit_tree_generated_prefix_weight,
            "explicit_tree_oracle_prefix_weight": args.explicit_tree_oracle_prefix_weight,
            "explicit_tree_prefix_jitter_weight": args.explicit_tree_prefix_jitter_weight,
            "explicit_tree_prefix_jitter_std": args.explicit_tree_prefix_jitter_std,
            "explicit_tree_depth": args.explicit_tree_depth,
            "explicit_tree_heads": args.explicit_tree_heads,
            "explicit_tree_topology_mode": args.explicit_tree_topology_mode,
            "explicit_tree_coordinate_mode": args.explicit_tree_coordinate_mode,
            "explicit_tree_action_weights": [
                args.explicit_tree_action_eos_loss_weight,
                args.explicit_tree_action_child_loss_weight,
                args.explicit_tree_action_branch_loss_weight,
            ],
            "explicit_tree_xyz_loss_weight": args.explicit_tree_xyz_loss_weight,
            "use_grammar_state_embedding": args.use_grammar_state_embedding,
            "use_action_group_bias": args.use_action_group_bias,
            "use_condition_action_group_bias": args.use_condition_action_group_bias,
            "random_query": args.random_query,
            "generation_kwargs": generation_kwargs,
            "target_start_policy": args.target_start_policy,
            "input_space_policy": args.input_space_policy,
            "count_guidance": args.count_guidance,
            "count_guidance_kwargs": count_guidance_kwargs,
            "action_guidance": args.action_guidance,
            "branch_prior_guidance": args.branch_prior_guidance,
            "branch_prior_guidance_kwargs": branch_prior_guidance_kwargs,
            "branch_parent_snap": args.branch_parent_snap,
            "dynamic_decode_mode": args.dynamic_decode_mode,
            "explicit_tree_max_joints": args.explicit_tree_max_joints,
            "explicit_tree_min_joints": args.explicit_tree_min_joints,
            "logit_prior": args.logit_prior,
            "logit_prior_weight": args.logit_prior_weight,
            "logit_prior_scope": args.logit_prior_scope,
        },
    }
    if not args.skip_dynamic and not args.skip_dynamic_baseline and args.dynamic_decode_mode in {"flat", "both"}:
        summary["dynamic"] = _summarize(base_rows, "dynamic")
    if not args.skip_dynamic and args.dynamic_decode_mode in {"explicit_tree", "both"}:
        summary["dynamic_explicit_tree"] = _summarize(base_rows, "dynamic_explicit_tree")
    if args.logit_prior != "none" and not args.skip_dynamic:
        summary["dynamic_logit_prior"] = _summarize(base_rows, "dynamic_logit_prior")
    if not args.skip_static_base:
        summary["static_base"] = _summarize(base_rows, "static_base")

    result = {"summary": summary, "rows": base_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
