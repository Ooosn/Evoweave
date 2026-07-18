#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _decode_generated_ids(tokenizer, ids: list[int]) -> SimpleNamespace:
    decoded = tokenizer.decode_ids(ids, require_eos=False)
    parents = [None if int(parent) < 0 else int(parent) for parent in decoded["parents"].tolist()]
    return SimpleNamespace(joints=np.asarray(decoded["joints"], dtype=np.float32), parents=parents)


def _target_namespace(batch: dict) -> SimpleNamespace:
    joint_count = int(batch["joint_count"][0].detach().cpu())
    joints = batch["target_joints"][0, :joint_count].detach().cpu().numpy().astype(np.float32)
    parents_raw = batch["target_parents"][0, :joint_count].detach().cpu().numpy().astype(np.int64)
    parents = [None if int(parent) < 0 else int(parent) for parent in parents_raw.tolist()]
    return SimpleNamespace(joints=joints, parents=parents)


def _forced_logits(model, batch: dict, *, cond: torch.Tensor | None = None):
    logits, token_batch, _loss = model.teacher_forced_logits(batch, cond=cond)
    return logits, token_batch


def _forced_argmax_ids(
    model,
    batch: dict,
    *,
    cond: torch.Tensor | None = None,
) -> tuple[list[int], dict[str, object]]:
    tokenizer = model.tokenizer
    logits, token_batch = _forced_logits(model, batch, cond=cond)
    labels = token_batch.labels[0].to(logits.device)
    roles = token_batch.token_role[0].to(logits.device)
    pred = logits[0].argmax(dim=-1)
    valid = labels != -100
    pred_ids = [tokenizer.bos_token_id] + pred[valid].detach().cpu().numpy().astype(int).tolist()
    if tokenizer.eos_token_id not in pred_ids:
        pred_ids.append(tokenizer.eos_token_id)

    eos_pos = int((labels == tokenizer.eos_token_id).nonzero(as_tuple=False)[0].item())
    eos_logits = logits[0, eos_pos]
    eos_prob = torch.softmax(eos_logits.float(), dim=-1)[tokenizer.eos_token_id]
    eos_rank = int((eos_logits > eos_logits[tokenizer.eos_token_id]).sum().item() + 1)
    coord_mask = valid & (roles >= tokenizer.offset) & ((roles - tokenizer.offset) < 3)
    parent_mask = valid & (roles == tokenizer.offset + 3)
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    position_reports = []
    payload_position = 0
    role_names = ("x", "y", "z", "parent")
    for tensor_position in valid_indices.tolist():
        role_value = int(roles[tensor_position].item())
        target_id = int(labels[tensor_position].item())
        pred_id = int(pred[tensor_position].item())
        target_logit = logits[0, tensor_position, target_id]
        target_nll = torch.logsumexp(logits[0, tensor_position].float(), dim=-1) - target_logit.float()
        target_rank = int(
            (logits[0, tensor_position] > target_logit).sum().item() + 1
        )
        if tokenizer.offset <= role_value < tokenizer.offset + len(role_names):
            role = role_names[role_value - tokenizer.offset]
            joint_index = payload_position // tokenizer.bone_per_token
            payload_position += 1
        elif target_id == tokenizer.eos_token_id:
            role = "eos"
            joint_index = None
        else:
            role = "other"
            joint_index = None
        position_reports.append(
            {
                "tensor_position": tensor_position,
                "joint_index": joint_index,
                "role": role,
                "target_id": target_id,
                "pred_id": pred_id,
                "correct": int(pred_id == target_id),
                "target_nll": float(target_nll.detach().cpu()),
                "target_rank": target_rank,
            }
        )
    stats = {
        "forced_token_acc": float((pred[valid] == labels[valid]).float().mean().detach().cpu()),
        "forced_coord_acc": float((pred[coord_mask] == labels[coord_mask]).float().mean().detach().cpu()),
        "forced_parent_acc": float((pred[parent_mask] == labels[parent_mask]).float().mean().detach().cpu()),
        "forced_eos_top1": int(pred[eos_pos].item() == tokenizer.eos_token_id),
        "forced_eos_rank": eos_rank,
        "forced_eos_prob": float(eos_prob.detach().cpu()),
        "forced_positions": position_reports,
    }
    return pred_ids, stats


def _manifest_metadata(manifest: Path) -> dict[str, dict[str, object]]:
    metadata = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            path = str(Path(row["path"]))
            metadata[path] = {
                "eval_stratum": row.get("_eval_stratum"),
                "split": row.get("split"),
            }
    return metadata


def _aggregate_forced_positions(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows["all"].append(row)
        stratum = row.get("eval_stratum")
        split = row.get("split")
        if stratum:
            grouped_rows[str(stratum)].append(row)
        if split:
            grouped_rows[f"split:{split}"].append(row)
        if split and stratum:
            grouped_rows[f"split:{split}/stratum:{stratum}"].append(row)

    report = {}
    for group_name, group_rows in grouped_rows.items():
        by_role: dict[str, list[dict[str, object]]] = defaultdict(list)
        by_joint: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in group_rows:
            for position in row["forced_positions"]:
                role = str(position["role"])
                by_role[role].append(position)
                joint_index = position["joint_index"]
                if joint_index is not None:
                    by_joint[int(joint_index)].append(position)

        def summarize(positions: list[dict[str, object]]) -> dict[str, float | int]:
            return {
                "count": len(positions),
                "accuracy": float(
                    sum(int(position["correct"]) for position in positions) / len(positions)
                ),
                "target_nll": float(
                    sum(float(position["target_nll"]) for position in positions) / len(positions)
                ),
                "target_rank": float(
                    sum(int(position["target_rank"]) for position in positions) / len(positions)
                ),
            }

        report[group_name] = {
            "row_count": len(group_rows),
            "by_role": {
                role: summarize(positions)
                for role, positions in sorted(by_role.items())
            },
            "by_joint_index": {
                str(joint_index): {
                    "all": summarize(positions),
                    "coord": summarize(
                        [
                            position
                            for position in positions
                            if position["role"] in {"x", "y", "z"}
                        ]
                    ),
                    "parent": summarize(
                        [
                            position
                            for position in positions
                            if position["role"] == "parent"
                        ]
                    ),
                }
                for joint_index, positions in sorted(by_joint.items())
            },
        }
    return report


def _condition_sensitivity(
    model,
    batch: dict,
    *,
    cond: torch.Tensor | None = None,
) -> dict[str, object]:
    tokenizer = model.tokenizer
    if cond is None:
        cond = model._condition_embeds(batch)
    zero = torch.zeros_like(cond)
    prefix = torch.tensor([[tokenizer.bos_token_id]], device=cond.device, dtype=torch.long)
    real_logits = model._next_token_logits(cond, prefix).float()
    zero_logits = model._next_token_logits(zero, prefix).float()
    regular = slice(tokenizer.offset, tokenizer.offset + tokenizer.n_discrete_size)
    real_regular = real_logits[:, regular]
    zero_regular = zero_logits[:, regular]
    diff = real_regular - zero_regular
    return {
        "first_regular_l2_real_vs_zero": float(torch.linalg.vector_norm(diff).detach().cpu()),
        "first_regular_cos_real_zero": float(torch.nn.functional.cosine_similarity(real_regular, zero_regular, dim=-1)[0].detach().cpu()),
        "first_real_top5": (real_regular[0].topk(5).indices + tokenizer.offset).detach().cpu().numpy().astype(int).tolist(),
        "first_zero_top5": (zero_regular[0].topk(5).indices + tokenizer.offset).detach().cpu().numpy().astype(int).tolist(),
    }


def _target_likelihood_under_condition_swap(
    model,
    batch: dict,
    *,
    correct_cond: torch.Tensor,
    swapped_cond: torch.Tensor,
    swapped_path: str,
    scope: str,
) -> dict[str, object]:
    """Measure whether the correct asset condition helps predict the fixed GT target."""

    tokenizer = model.tokenizer
    original_joint_count = batch["joint_count"]
    if scope == "joint0":
        teacher_batch = dict(batch)
        teacher_batch["joint_count"] = torch.ones_like(original_joint_count)
    elif scope == "first10":
        teacher_batch = dict(batch)
        teacher_batch["joint_count"] = torch.clamp(original_joint_count, max=10)
    elif scope == "full":
        teacher_batch = batch
    else:
        raise ValueError(f"unknown condition-swap scope {scope!r}")

    zero_cond = torch.zeros_like(correct_cond)
    condition_logits: dict[str, torch.Tensor] = {}
    token_batch = None
    for name, cond in (
        ("correct", correct_cond),
        ("swapped", swapped_cond),
        ("zero", zero_cond),
    ):
        logits, current_token_batch, _loss = model.teacher_forced_logits(
            teacher_batch,
            cond=cond,
        )
        condition_logits[name] = logits.float()
        if token_batch is None:
            token_batch = current_token_batch
        elif not torch.equal(current_token_batch.labels, token_batch.labels):
            raise ValueError("condition swap changed teacher-forcing labels")

    assert token_batch is not None
    labels = token_batch.labels.to(correct_cond.device)
    roles = token_batch.token_role.to(correct_cond.device)
    valid = labels != -100
    coord = valid & (roles >= tokenizer.offset) & ((roles - tokenizer.offset) < 3)
    joint0_coord = torch.zeros_like(valid)
    joint0_positions = coord[0].nonzero(as_tuple=False).flatten()[:3]
    if int(joint0_positions.numel()) != 3:
        raise ValueError(
            "Puppeteer condition swap expects exactly three joint-0 coordinate positions"
        )
    joint0_coord[0, joint0_positions] = True
    role_masks = {
        "all": valid,
        "coord": coord,
        "joint0_coord": joint0_coord,
        "x": valid & (roles == tokenizer.offset),
        "y": valid & (roles == tokenizer.offset + 1),
        "z": valid & (roles == tokenizer.offset + 2),
        "parent": valid & (roles == tokenizer.offset + 3),
        "eos": valid & (labels == tokenizer.eos_token_id),
    }

    role_reports: dict[str, object] = {}
    for role_name, mask in role_masks.items():
        if not bool(mask.any()):
            continue
        role_labels = labels[mask]
        per_condition: dict[str, dict[str, float]] = {}
        for condition_name, logits in condition_logits.items():
            role_logits = logits[mask]
            nll = torch.nn.functional.cross_entropy(
                role_logits,
                role_labels,
                reduction="none",
            )
            target_logits = role_logits.gather(1, role_labels[:, None]).squeeze(1)
            target_ranks = (role_logits > target_logits[:, None]).sum(dim=1) + 1
            per_condition[condition_name] = {
                "nll": float(nll.mean().item()),
                "target_probability": float(torch.exp(-nll).mean().item()),
                "target_rank": float(target_ranks.float().mean().item()),
                "top1_accuracy": float(
                    (role_logits.argmax(dim=-1) == role_labels).float().mean().item()
                ),
            }
        role_reports[role_name] = {
            "positions": int(mask.sum().item()),
            **per_condition,
            "swapped_minus_correct_nll": float(
                per_condition["swapped"]["nll"] - per_condition["correct"]["nll"]
            ),
            "zero_minus_correct_nll": float(
                per_condition["zero"]["nll"] - per_condition["correct"]["nll"]
            ),
            "correct_vs_swapped_argmax_change_rate": float(
                (
                    condition_logits["correct"][mask].argmax(dim=-1)
                    != condition_logits["swapped"][mask].argmax(dim=-1)
                )
                .float()
                .mean()
                .item()
            ),
        }

    return {
        "path": batch["path"][0],
        "swapped_path": swapped_path,
        "scope": scope,
        "original_joint_count": int(original_joint_count[0].item()),
        "evaluated_joint_count": int(teacher_batch["joint_count"][0].item()),
        "roles": role_reports,
    }


def _condition_with_seed(model, batch: dict, seed: int) -> torch.Tensor:
    device = batch["frame_vertices"].device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    raw = model.build_condition(batch, generator=generator)
    return model._condition_embeds_from_raw(raw)


def _tensor_pair_report(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | list[int]]:
    left_f = left.float()
    right_f = right.float()
    if left_f.shape != right_f.shape:
        raise ValueError(f"stage tensor shapes differ: {tuple(left_f.shape)} vs {tuple(right_f.shape)}")
    diff = left_f - right_f
    left_norm = torch.linalg.vector_norm(left_f)
    right_norm = torch.linalg.vector_norm(right_f)
    scale = 0.5 * (left_norm + right_norm)
    report: dict[str, float | list[int]] = {
        "shape": list(left_f.shape),
        "left_rms": float(torch.sqrt(torch.mean(left_f.square())).item()),
        "right_rms": float(torch.sqrt(torch.mean(right_f.square())).item()),
        "diff_rms": float(torch.sqrt(torch.mean(diff.square())).item()),
        "diff_max_abs": float(diff.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / scale.clamp_min(1.0e-12)).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                left_f.reshape(1, -1), right_f.reshape(1, -1), dim=-1
            )[0].item()
        ),
    }
    if left_f.ndim >= 3 and int(left_f.shape[-2]) > 1:
        left_slot_mean = left_f.mean(dim=-2, keepdim=True)
        right_slot_mean = right_f.mean(dim=-2, keepdim=True)
        left_centered = left_f - left_slot_mean
        right_centered = right_f - right_slot_mean
        left_rms = torch.sqrt(torch.mean(left_f.square())).clamp_min(1.0e-12)
        right_rms = torch.sqrt(torch.mean(right_f.square())).clamp_min(1.0e-12)
        report["slot_cosine_mean"] = float(
            torch.nn.functional.cosine_similarity(left_f, right_f, dim=-1).mean().item()
        )
        report["left_slot_mean_rms"] = float(
            torch.sqrt(torch.mean(left_slot_mean.square())).item()
        )
        report["right_slot_mean_rms"] = float(
            torch.sqrt(torch.mean(right_slot_mean.square())).item()
        )
        report["left_slot_centered_rms"] = float(
            torch.sqrt(torch.mean(left_centered.square())).item()
        )
        report["right_slot_centered_rms"] = float(
            torch.sqrt(torch.mean(right_centered.square())).item()
        )
        report["left_slot_mean_rms_fraction"] = float(
            (torch.sqrt(torch.mean(left_slot_mean.square())) / left_rms).item()
        )
        report["right_slot_mean_rms_fraction"] = float(
            (torch.sqrt(torch.mean(right_slot_mean.square())) / right_rms).item()
        )
    return report


def _parameter_report(module: torch.nn.Module) -> dict[str, dict[str, float | int | list[int] | bool]]:
    report = {}
    for name, parameter in module.named_parameters():
        value = parameter.detach().float()
        report[name] = {
            "shape": list(value.shape),
            "count": int(value.numel()),
            "trainable": bool(parameter.requires_grad),
            "mean": float(value.mean().item()),
            "rms": float(torch.sqrt(torch.mean(value.square())).item()),
            "std": float(value.std(unbiased=False).item()),
            "max_abs": float(value.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(value).item()),
        }
    return report


def _trace_pre_norm_encoder_layer(
    layer: torch.nn.TransformerEncoderLayer,
    tokens: torch.Tensor,
    *,
    attention_scale: float = 1.0,
    mlp_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not layer.norm_first:
        raise ValueError("internal stage trace currently requires norm_first=True")

    attention_input = layer.norm1(tokens)
    attention_output = layer.self_attn(
        attention_input,
        attention_input,
        attention_input,
        need_weights=False,
    )[0]
    attention_delta_raw = layer.dropout1(attention_output)
    attention_delta = attention_delta_raw * float(attention_scale)
    after_attention = tokens + attention_delta

    mlp_input = layer.norm2(after_attention)
    mlp_pre_activation = layer.linear1(mlp_input)
    mlp_activation = layer.activation(mlp_pre_activation)
    mlp_output = layer.linear2(layer.dropout(mlp_activation))
    mlp_delta_raw = layer.dropout2(mlp_output)
    mlp_delta = mlp_delta_raw * float(mlp_scale)
    output = after_attention + mlp_delta
    return output, {
        "attention_input": attention_input,
        "attention_delta_raw": attention_delta_raw,
        "attention_delta": attention_delta,
        "after_attention": after_attention,
        "mlp_input": mlp_input,
        "mlp_pre_activation": mlp_pre_activation,
        "mlp_activation": mlp_activation,
        "mlp_delta_raw": mlp_delta_raw,
        "mlp_delta": mlp_delta,
        "output": output,
    }


@torch.no_grad()
def _slot_structure(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.float()
    slot_mean = value.mean(dim=-2, keepdim=True)
    centered = value - slot_mean
    rms = torch.sqrt(torch.mean(value.square())).clamp_min(1.0e-12)
    mean_rms = torch.sqrt(torch.mean(slot_mean.square()))
    return {
        "rms": float(rms.item()),
        "slot_mean_rms": float(mean_rms.item()),
        "slot_centered_rms": float(torch.sqrt(torch.mean(centered.square())).item()),
        "slot_mean_rms_fraction": float((mean_rms / rms).item()),
    }


@torch.no_grad()
def _pose_attention_routing(
    layer: torch.nn.TransformerEncoderLayer,
    attention_input: torch.Tensor,
    *,
    batch_size: int,
    frame_count: int,
    slot_count: int,
    anchor_start: int,
) -> dict[str, float | int]:
    frame0 = attention_input.reshape(batch_size, frame_count, slot_count, -1)[:, 0]
    _output, weights = layer.self_attn(
        frame0,
        frame0,
        frame0,
        need_weights=True,
        average_attn_weights=False,
    )
    anchor_rows = weights.float()[:, :, anchor_start:, :]
    probabilities = anchor_rows.clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    normalized_entropy = entropy / np.log(float(slot_count))
    mean_row = anchor_rows.mean(dim=-2, keepdim=True)
    top_keys = anchor_rows.argmax(dim=-1)
    anchor_square = anchor_rows[..., anchor_start:]
    self_mass = anchor_square.diagonal(dim1=-2, dim2=-1)
    embed_dim = int(frame0.shape[-1])
    head_count = int(weights.shape[1])
    head_dim = embed_dim // head_count
    in_proj_bias = layer.self_attn.in_proj_bias
    value_bias = None if in_proj_bias is None else in_proj_bias[2 * embed_dim :]
    value_vectors = torch.nn.functional.linear(
        frame0,
        layer.self_attn.in_proj_weight[2 * embed_dim :],
        value_bias,
    )
    value_heads = value_vectors.reshape(
        int(frame0.shape[0]), slot_count, head_count, head_dim
    ).permute(0, 2, 1, 3)
    attended_heads = torch.matmul(weights.float(), value_heads.float())
    attended_pre_out = attended_heads.permute(0, 2, 1, 3).reshape(
        int(frame0.shape[0]), slot_count, embed_dim
    )
    attention_output = torch.nn.functional.linear(
        attended_pre_out,
        layer.self_attn.out_proj.weight,
        layer.self_attn.out_proj.bias,
    )
    report: dict[str, float | int] = {
        "role_key_mass": float(anchor_rows[..., 0].mean().item()),
        "register_key_mass": float(anchor_rows[..., 1:anchor_start].sum(dim=-1).mean().item()),
        "anchor_key_mass": float(anchor_rows[..., anchor_start:].sum(dim=-1).mean().item()),
        "self_anchor_mass": float(self_mass.mean().item()),
        "normalized_entropy": float(normalized_entropy.mean().item()),
        "row_l1_to_mean": float((anchor_rows - mean_row).abs().sum(dim=-1).mean().item()),
        "max_key_mass": float(anchor_rows.max(dim=-1).values.mean().item()),
        "top_key_role_rate": float((top_keys == 0).float().mean().item()),
        "top_key_register_rate": float(
            ((top_keys > 0) & (top_keys < anchor_start)).float().mean().item()
        ),
        "top_key_anchor_rate": float((top_keys >= anchor_start).float().mean().item()),
        "top_key_unique_count": int(torch.unique(top_keys).numel()),
        "attention_output_reconstruction_max_abs": float(
            (attention_output[:, anchor_start:] - _output[:, anchor_start:].float()).abs().max().item()
        ),
    }
    for stage_name, stage_value in (
        ("value_projection", value_vectors[:, anchor_start:]),
        ("attended_pre_out", attended_pre_out[:, anchor_start:]),
        ("attention_output", attention_output[:, anchor_start:]),
    ):
        for metric_name, metric_value in _slot_structure(stage_value).items():
            report[f"{stage_name}_{metric_name}"] = metric_value
    return report


@torch.no_grad()
def _temporal_attention_routing(
    layer: torch.nn.TransformerEncoderLayer,
    attention_input: torch.Tensor,
    *,
    batch_size: int,
    frame_count: int,
    slot_count: int,
    anchor_start: int,
) -> dict[str, float | int]:
    anchors = attention_input.reshape(batch_size, slot_count, frame_count, -1)[:, anchor_start:]
    anchors = anchors.reshape(batch_size * (slot_count - anchor_start), frame_count, -1)
    _output, weights = layer.self_attn(
        anchors,
        anchors,
        anchors,
        need_weights=True,
        average_attn_weights=False,
    )
    query_frame0 = weights.float()[:, :, 0, :]
    probabilities = query_frame0.clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    normalized_entropy = entropy / np.log(float(frame_count))
    top_frames = query_frame0.argmax(dim=-1)
    return {
        "query_frame_key_mass": float(query_frame0[..., 0].mean().item()),
        "evidence_frame_key_mass": float(query_frame0[..., 1:].sum(dim=-1).mean().item()),
        "normalized_entropy": float(normalized_entropy.mean().item()),
        "max_frame_mass": float(query_frame0.max(dim=-1).values.mean().item()),
        "top_key_is_query_rate": float((top_frames == 0).float().mean().item()),
        "top_key_unique_frame_count": int(torch.unique(top_frames).numel()),
    }


@torch.no_grad()
def _condition_stage_trace(
    model,
    batch: dict,
    seed: int,
    *,
    refs=None,
    pose_attention_scale: float = 1.0,
    pose_mlp_scale: float = 1.0,
    temporal_attention_scale: float = 1.0,
    temporal_mlp_scale: float = 1.0,
    trace_attention_routing: bool = False,
) -> dict[str, object]:
    from rigweave.dynamic_rig.sampling import materialize_trackable_surface

    device = batch["frame_vertices"].device
    if refs is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        refs = model.sample_references(batch, generator=generator)

    frame_tokens = []
    frame_query_points = []
    for frame_index in range(int(batch["frame_vertices"].shape[1])):
        vertex_normals = batch.get("vertex_normals")
        face_normals = batch.get("face_normals")
        samples = materialize_trackable_surface(
            batch["frame_vertices"][:, frame_index],
            batch["faces"],
            refs,
            vertex_normals=None if vertex_normals is None else vertex_normals[:, frame_index],
            face_normals=None if face_normals is None else face_normals[:, frame_index],
        )
        frame_tokens.append(
            model.conditioner.surface_tokenizer(
                samples.dense_points,
                samples.dense_normals,
                samples.query_points,
                samples.query_normals,
            )
        )
        frame_query_points.append(samples.query_points)

    surface_sequence = torch.stack(frame_tokens, dim=1)
    query_sequence = torch.stack(frame_query_points, dim=1)
    motion_encoder = model.conditioner.motion_encoder
    anchor_start = 1 + int(motion_encoder.register_tokens)
    if motion_encoder.use_motion_features:
        motion = motion_encoder._motion_features(query_sequence).to(dtype=surface_sequence.dtype)
        z = surface_sequence + motion_encoder.motion_feature_mlp(motion)
    else:
        z = surface_sequence

    batch_size, frame_count, _, dim = z.shape
    if motion_encoder.register_tokens > 0:
        regs = motion_encoder.register.view(1, 1, motion_encoder.register_tokens, dim).expand(
            batch_size, frame_count, -1, -1
        )
    else:
        regs = z.new_empty((batch_size, frame_count, 0, dim))
    canonical_role = motion_encoder.role_token[:, 0:1].expand(batch_size, 1, -1, -1)
    motion_roles = motion_encoder.role_token[:, 1:2].expand(
        batch_size, max(0, frame_count - 1), -1, -1
    )
    role = torch.cat([canonical_role, motion_roles], dim=1)
    z = torch.cat([role, regs, z], dim=2)
    if motion_encoder.use_time_embedding:
        z = z + motion_encoder.time_embed[:frame_count].view(1, frame_count, 1, dim)

    stages: dict[str, torch.Tensor] = {
        "query_points_frame0": query_sequence[:, 0],
        "surface_tokens_frame0": surface_sequence[:, 0],
        "motion_input_frame0": z[:, 0, anchor_start:],
    }
    slot_count = int(z.shape[2])
    attention_routing: dict[str, dict[str, float | int]] = {}
    for block_index, block in enumerate(motion_encoder.blocks):
        pose_input = z.reshape(batch_size * frame_count, slot_count, dim)
        pose_output, pose_trace = _trace_pre_norm_encoder_layer(
            block.pose_inner,
            pose_input,
            attention_scale=pose_attention_scale,
            mlp_scale=pose_mlp_scale,
        )
        pose_output_4d = pose_output.reshape(batch_size, frame_count, slot_count, dim)

        temporal_input = pose_output_4d.permute(0, 2, 1, 3).reshape(
            batch_size * slot_count, frame_count, dim
        )
        temporal_output, temporal_trace = _trace_pre_norm_encoder_layer(
            block.anchor_temporal,
            temporal_input,
            attention_scale=temporal_attention_scale,
            mlp_scale=temporal_mlp_scale,
        )
        z = temporal_output.reshape(batch_size, slot_count, frame_count, dim).permute(0, 2, 1, 3)

        if block_index == 0:
            if trace_attention_routing:
                attention_routing["block_00_pose_inner"] = _pose_attention_routing(
                    block.pose_inner,
                    pose_trace["attention_input"],
                    batch_size=batch_size,
                    frame_count=frame_count,
                    slot_count=slot_count,
                    anchor_start=anchor_start,
                )
                attention_routing["block_00_anchor_temporal"] = _temporal_attention_routing(
                    block.anchor_temporal,
                    temporal_trace["attention_input"],
                    batch_size=batch_size,
                    frame_count=frame_count,
                    slot_count=slot_count,
                    anchor_start=anchor_start,
                )
            for stage_name, value in pose_trace.items():
                value_4d = value.reshape(
                    batch_size,
                    frame_count,
                    slot_count,
                    int(value.shape[-1]),
                )
                stages[f"motion_block_00_pose_{stage_name}_frame0"] = value_4d[
                    :, 0, anchor_start:
                ]
            for stage_name, value in temporal_trace.items():
                value_4d = value.reshape(
                    batch_size,
                    slot_count,
                    frame_count,
                    int(value.shape[-1]),
                ).permute(0, 2, 1, 3)
                stages[f"motion_block_00_temporal_{stage_name}_frame0"] = value_4d[
                    :, 0, anchor_start:
                ]
        stages[f"motion_block_{block_index:02d}_frame0"] = z[:, 0, anchor_start:]

    traced_motion_output = motion_encoder.norm(z)[:, 0, anchor_start:]
    unmodified_motion_output = motion_encoder(surface_sequence, query_points=query_sequence)
    trace_validation = _tensor_pair_report(traced_motion_output, unmodified_motion_output)
    intervention_active = any(
        float(scale) != 1.0
        for scale in (
            pose_attention_scale,
            pose_mlp_scale,
            temporal_attention_scale,
            temporal_mlp_scale,
        )
    )
    motion_output = traced_motion_output if intervention_active else unmodified_motion_output

    projected = model.prefix_projector(motion_output)
    decoder_condition = model._condition_embeds_from_raw(projected)
    stages["motion_output_frame0"] = motion_output
    stages["projected_condition"] = projected
    stages["decoder_condition"] = decoder_condition
    return {
        "refs": refs,
        "stages": stages,
        "condition": decoder_condition,
        "trace_validation": trace_validation,
        "intervention_active": intervention_active,
        "attention_routing": attention_routing,
    }


@torch.no_grad()
def _greedy_prefix_ids(model, cond: torch.Tensor, token_count: int) -> list[int]:
    prefix = torch.tensor(
        [[model.tokenizer.bos_token_id]],
        device=cond.device,
        dtype=torch.long,
    )
    generated = []
    for _ in range(max(0, int(token_count))):
        logits = model._next_token_logits(cond, prefix)
        logits = model._apply_generation_mask(
            logits,
            len(generated),
            max_joints=model.max_joints,
        )
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        prefix = torch.cat(
            [prefix, torch.tensor([[token]], device=prefix.device, dtype=torch.long)],
            dim=1,
        )
        if token == model.tokenizer.eos_token_id:
            break
    return generated


def _reference_pair_report(left, right) -> dict[str, float]:
    return {
        "vertex_index_match_rate": float((left.vertex_indices == right.vertex_indices).float().mean().item()),
        "face_index_match_rate": float((left.face_indices == right.face_indices).float().mean().item()),
        "query_index_match_rate": float((left.query_indices == right.query_indices).float().mean().item()),
        "barycentric_rms": float(torch.sqrt(torch.mean((left.barycentric - right.barycentric).float().square())).item()),
    }


def _stage_pair_report(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, object]:
    if left.keys() != right.keys():
        raise ValueError(f"stage trace keys differ: {left.keys()} vs {right.keys()}")
    return {name: _tensor_pair_report(left[name], right[name]) for name in left}


def _target_ids(model, batch: dict) -> torch.Tensor:
    token_batch = model._make_token_batch(batch, batch["target_joints"].device)
    labels = token_batch.labels[0]
    return labels[labels != -100]


def _paired_pose_target_change(
    model,
    batch_a: dict,
    batch_b: dict,
    *,
    prefix_joint_count: int = 10,
) -> dict[str, object]:
    """Describe how much the fixed asset target changes between two sampled poses."""

    if batch_a["path"] != batch_b["path"]:
        raise ValueError(f"paired pose paths differ: {batch_a['path']} vs {batch_b['path']}")
    joint_count_a = int(batch_a["joint_count"][0].item())
    joint_count_b = int(batch_b["joint_count"][0].item())
    if joint_count_a != joint_count_b:
        raise ValueError(
            f"paired pose joint counts differ: {joint_count_a} vs {joint_count_b}"
        )

    joints_a = batch_a["target_joints"][0, :joint_count_a].float()
    joints_b = batch_b["target_joints"][0, :joint_count_b].float()
    prefix_count = min(int(prefix_joint_count), joint_count_a)
    delta = joints_a - joints_b
    prefix_delta = delta[:prefix_count]
    ids_a = _target_ids(model, batch_a)
    ids_b = _target_ids(model, batch_b)
    if ids_a.shape != ids_b.shape:
        raise ValueError(
            f"paired target token shapes differ: {tuple(ids_a.shape)} vs {tuple(ids_b.shape)}"
        )

    payload_a = ids_a[:-1].reshape(-1, model.tokenizer.bone_per_token)
    payload_b = ids_b[:-1].reshape(-1, model.tokenizer.bone_per_token)
    payload_changed = payload_a != payload_b
    query_a = batch_a["frame_vertices"][:, 0].float()
    query_b = batch_b["frame_vertices"][:, 0].float()
    if query_a.shape != query_b.shape:
        raise ValueError(
            f"paired query mesh shapes differ: {tuple(query_a.shape)} vs {tuple(query_b.shape)}"
        )

    return {
        "query_frame_a": int(batch_a["selected_frames"][0, 0].item()),
        "query_frame_b": int(batch_b["selected_frames"][0, 0].item()),
        "same_query_frame": bool(
            batch_a["selected_frames"][0, 0].item()
            == batch_b["selected_frames"][0, 0].item()
        ),
        "joint_count": joint_count_a,
        "evaluated_prefix_joint_count": prefix_count,
        "root_joint_l2": float(torch.linalg.vector_norm(delta[0]).item()),
        "prefix_joint_rms": float(torch.sqrt(torch.mean(prefix_delta.square())).item()),
        "all_joint_rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        "query_mesh_rms": float(torch.sqrt(torch.mean((query_a - query_b).square())).item()),
        "target_token_change_count": int((ids_a != ids_b).sum().item()),
        "target_token_change_rate": float((ids_a != ids_b).float().mean().item()),
        "prefix_coordinate_token_change_count": int(
            payload_changed[:prefix_count, :3].sum().item()
        ),
        "prefix_coordinate_token_count": int(prefix_count * 3),
    }


def _paired_pose_response(
    model,
    batch_a: dict,
    batch_b: dict,
    *,
    cond_a: torch.Tensor,
    cond_b: torch.Tensor,
) -> dict[str, object]:
    tokenizer = model.tokenizer
    if batch_a["path"] != batch_b["path"]:
        raise ValueError(f"paired pose paths differ: {batch_a['path']} vs {batch_b['path']}")

    ids_a = _target_ids(model, batch_a)
    ids_b = _target_ids(model, batch_b)
    if ids_a.shape != ids_b.shape:
        raise ValueError(f"paired target token shapes differ: {tuple(ids_a.shape)} vs {tuple(ids_b.shape)}")
    payload_a = ids_a[:-1].reshape(-1, model.tokenizer.bone_per_token)
    payload_b = ids_b[:-1].reshape(-1, model.tokenizer.bone_per_token)
    payload_changed = payload_a != payload_b
    joint_changed = payload_changed.any(dim=1)
    changed_joint_indices = joint_changed.nonzero(as_tuple=False).flatten()

    cond_a_f = cond_a.float()
    cond_b_f = cond_b.float()
    cond_diff = cond_a_f - cond_b_f
    cond_scale = 0.5 * (
        torch.linalg.vector_norm(cond_a_f) + torch.linalg.vector_norm(cond_b_f)
    )

    logits_aa, token_batch_a, _loss = model.teacher_forced_logits(batch_a, cond=cond_a)
    logits_ba, _token_batch_ba, _loss = model.teacher_forced_logits(batch_a, cond=cond_b)
    logits_ab, token_batch_b, _loss = model.teacher_forced_logits(batch_b, cond=cond_a)
    valid = token_batch_a.labels.to(logits_aa.device) != -100
    if not torch.equal(valid, token_batch_b.labels.to(logits_aa.device) != -100):
        raise ValueError("paired pose token masks differ")
    shared_a = logits_aa[valid].float()
    shared_b = logits_ba[valid].float()
    shared_diff = shared_a - shared_b
    shared_scale = 0.5 * (
        torch.linalg.vector_norm(shared_a) + torch.linalg.vector_norm(shared_b)
    )
    probs_a = torch.softmax(shared_a, dim=-1)
    probs_b = torch.softmax(shared_b, dim=-1)
    condition_position_rel_l2 = torch.linalg.vector_norm(shared_diff, dim=-1) / (
        0.5
        * (
            torch.linalg.vector_norm(shared_a, dim=-1)
            + torch.linalg.vector_norm(shared_b, dim=-1)
        )
    ).clamp_min(1.0e-12)
    condition_position_probability_l1 = (probs_a - probs_b).abs().sum(dim=-1)
    valid_labels = token_batch_a.labels.to(logits_aa.device)[valid]
    valid_roles = token_batch_a.token_role.to(logits_aa.device)[valid]
    role_names = ("x", "y", "z", "parent")
    condition_swap_by_role = {}
    for role_offset, role_name in enumerate(role_names):
        role_mask = valid_roles == tokenizer.offset + role_offset
        if bool(role_mask.any()):
            condition_swap_by_role[role_name] = {
                "positions": int(role_mask.sum().item()),
                "relative_l2_mean": float(condition_position_rel_l2[role_mask].mean().item()),
                "probability_l1_mean": float(
                    condition_position_probability_l1[role_mask].mean().item()
                ),
                "argmax_change_rate": float(
                    (
                        shared_a[role_mask].argmax(dim=-1)
                        != shared_b[role_mask].argmax(dim=-1)
                    )
                    .float()
                    .mean()
                    .item()
                ),
            }
    eos_mask = valid_labels == tokenizer.eos_token_id
    if bool(eos_mask.any()):
        condition_swap_by_role["eos"] = {
            "positions": int(eos_mask.sum().item()),
            "relative_l2_mean": float(condition_position_rel_l2[eos_mask].mean().item()),
            "probability_l1_mean": float(
                condition_position_probability_l1[eos_mask].mean().item()
            ),
            "argmax_change_rate": float(
                (shared_a[eos_mask].argmax(dim=-1) != shared_b[eos_mask].argmax(dim=-1))
                .float()
                .mean()
                .item()
            ),
        }
    first_condition_swap_positions = []
    for position in range(min(12, int(shared_a.shape[0]))):
        role_value = int(valid_roles[position].item())
        if tokenizer.offset <= role_value < tokenizer.offset + len(role_names):
            role_name = role_names[role_value - tokenizer.offset]
        elif int(valid_labels[position].item()) == tokenizer.eos_token_id:
            role_name = "eos"
        else:
            role_name = "other"
        first_condition_swap_positions.append(
            {
                "position": position,
                "joint_index": position // tokenizer.bone_per_token,
                "role": role_name,
                "target_id": int(valid_labels[position].item()),
                "relative_l2": float(condition_position_rel_l2[position].item()),
                "probability_l1": float(condition_position_probability_l1[position].item()),
                "argmax_changed": bool(
                    shared_a[position].argmax().item() != shared_b[position].argmax().item()
                ),
            }
        )

    input_a = token_batch_a.input_ids.to(logits_aa.device)
    input_b = token_batch_b.input_ids.to(logits_aa.device)
    input_diff = input_a != input_b
    prefix_diverged_before = torch.cat(
        [
            torch.zeros_like(input_diff[:, :1]),
            input_diff.cumsum(dim=1)[:, :-1] > 0,
        ],
        dim=1,
    )
    prefix_mask = valid & prefix_diverged_before
    prefix_ref = logits_aa[prefix_mask].float()
    prefix_changed_logits = logits_ab[prefix_mask].float()
    condition_changed_logits = logits_ba[prefix_mask].float()
    prefix_delta = prefix_ref - prefix_changed_logits
    condition_delta = prefix_ref - condition_changed_logits
    prefix_scale = 0.5 * (
        torch.linalg.vector_norm(prefix_ref) + torch.linalg.vector_norm(prefix_changed_logits)
    )
    prefix_probs = torch.softmax(prefix_ref, dim=-1)
    prefix_changed_probs = torch.softmax(prefix_changed_logits, dim=-1)

    joints_a = batch_a["target_joints"][0, : int(batch_a["joint_count"][0].item())].float()
    joints_b = batch_b["target_joints"][0, : int(batch_b["joint_count"][0].item())].float()
    if joints_a.shape != joints_b.shape:
        raise ValueError(f"paired target joint shapes differ: {tuple(joints_a.shape)} vs {tuple(joints_b.shape)}")

    query_a = batch_a["frame_vertices"][:, 0].float()
    query_b = batch_b["frame_vertices"][:, 0].float()
    if query_a.shape != query_b.shape:
        raise ValueError(f"paired query mesh shapes differ: {tuple(query_a.shape)} vs {tuple(query_b.shape)}")

    return {
        "query_frame_a": int(batch_a["selected_frames"][0, 0].item()),
        "query_frame_b": int(batch_b["selected_frames"][0, 0].item()),
        "target_token_count": int(ids_a.numel()),
        "target_token_changes": int((ids_a != ids_b).sum().item()),
        "target_token_change_rate": float((ids_a != ids_b).float().mean().item()),
        "target_changed_joint_count": int(joint_changed.sum().item()),
        "target_first_changed_joint": (
            int(changed_joint_indices[0].item()) if changed_joint_indices.numel() else -1
        ),
        "target_first_joint_token_changes": int(payload_changed[0].sum().item()),
        "target_role_change_counts": {
            "x": int(payload_changed[:, 0].sum().item()),
            "y": int(payload_changed[:, 1].sum().item()),
            "z": int(payload_changed[:, 2].sum().item()),
            "parent": int(payload_changed[:, 3].sum().item()),
        },
        "target_joint_rms": float(torch.sqrt(torch.mean((joints_a - joints_b) ** 2)).item()),
        "query_mesh_rms": float(torch.sqrt(torch.mean((query_a - query_b) ** 2)).item()),
        "condition_rel_l2": float(
            (torch.linalg.vector_norm(cond_diff) / cond_scale.clamp_min(1.0e-12)).item()
        ),
        "condition_cosine": float(
            torch.nn.functional.cosine_similarity(
                cond_a_f.reshape(1, -1), cond_b_f.reshape(1, -1), dim=-1
            )[0].item()
        ),
        "condition_token_cosine_mean": float(
            torch.nn.functional.cosine_similarity(cond_a_f, cond_b_f, dim=-1).mean().item()
        ),
        "shared_prefix_logit_rel_l2": float(
            (torch.linalg.vector_norm(shared_diff) / shared_scale.clamp_min(1.0e-12)).item()
        ),
        "shared_prefix_logit_max_abs": float(shared_diff.abs().max().item()),
        "shared_prefix_argmax_change_rate": float(
            (shared_a.argmax(dim=-1) != shared_b.argmax(dim=-1)).float().mean().item()
        ),
        "shared_prefix_probability_l1_mean": float((probs_a - probs_b).abs().sum(dim=-1).mean().item()),
        "condition_swap_by_target_role": condition_swap_by_role,
        "first_condition_swap_positions": first_condition_swap_positions,
        "prefix_diverged_prediction_positions": int(prefix_mask.sum().item()),
        "same_condition_prefix_logit_rel_l2": float(
            (torch.linalg.vector_norm(prefix_delta) / prefix_scale.clamp_min(1.0e-12)).item()
        ),
        "same_condition_prefix_argmax_change_rate": float(
            (prefix_ref.argmax(dim=-1) != prefix_changed_logits.argmax(dim=-1)).float().mean().item()
        ),
        "same_condition_prefix_probability_l1_mean": float(
            (prefix_probs - prefix_changed_probs).abs().sum(dim=-1).mean().item()
        ),
        "condition_to_prefix_logit_l2_ratio": float(
            (
                torch.linalg.vector_norm(condition_delta)
                / torch.linalg.vector_norm(prefix_delta).clamp_min(1.0e-12)
            ).item()
        ),
        "target_tokens_identical": bool(torch.equal(ids_a, ids_b)),
        "selected_frames_a": batch_a["selected_frames"][0].detach().cpu().numpy().astype(int).tolist(),
        "selected_frames_b": batch_b["selected_frames"][0].detach().cpu().numpy().astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--puppeteer-llm", type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--pose-seed-a", type=int, default=None)
    parser.add_argument("--pose-seed-b", type=int, default=None)
    parser.add_argument("--surface-seed", type=int, default=20260715)
    parser.add_argument("--stage-trace", action="store_true")
    parser.add_argument("--pose-attention-scale", type=float, default=1.0)
    parser.add_argument("--pose-mlp-scale", type=float, default=1.0)
    parser.add_argument("--temporal-attention-scale", type=float, default=1.0)
    parser.add_argument("--temporal-mlp-scale", type=float, default=1.0)
    parser.add_argument("--attention-routing", action="store_true")
    parser.add_argument("--greedy-prefix-tokens", type=int, default=0)
    parser.add_argument("--cross-asset-condition-swap", action="store_true")
    parser.add_argument(
        "--same-asset-pose-condition-swap",
        action="store_true",
        help=(
            "Keep pose-A targets fixed and compare their likelihood under pose-A "
            "versus pose-B conditions from the same asset."
        ),
    )
    parser.add_argument(
        "--condition-swap-scope",
        choices=("full", "joint0", "first10"),
        default="full",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intervention_scales = (
        args.pose_attention_scale,
        args.pose_mlp_scale,
        args.temporal_attention_scale,
        args.temporal_mlp_scale,
    )
    if any(float(scale) != 1.0 for scale in intervention_scales) and not args.stage_trace:
        raise ValueError("motion residual interventions require --stage-trace")
    if args.attention_routing and not args.stage_trace:
        raise ValueError("--attention-routing requires --stage-trace")
    if args.greedy_prefix_tokens < 0:
        raise ValueError("--greedy-prefix-tokens must be non-negative")
    if args.same_asset_pose_condition_swap and args.pose_seed_b is None:
        raise ValueError("--same-asset-pose-condition-swap requires --pose-seed-b")

    import sys

    sys.path.insert(0, str(args.model_root / "rigweave" / "src"))
    sys.path.insert(0, str(args.model_root / "rigweave" / "scripts"))
    sys.path.insert(0, str(args.model_root / "third_party_references" / "Puppeteer" / "skeleton"))

    from eval_dynamic_rig_ce import apply_checkpoint_eval_defaults
    from eval_dynamic_rig_generation import _apply_puppeteer_checkpoint_defaults, _build_puppeteer_model, _output_metrics, _puppeteer_metric_range
    from rigweave.dynamic_rig import PuppeteerDynamicRigDataset, puppeteer_dynamic_collate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}) or {})
    del ckpt
    ns = argparse.Namespace(checkpoint=args.checkpoint)
    for name in (
        "tokenizer_config",
        "model_config",
        "unirig_checkpoint",
        "puppeteer_root",
        "puppeteer_checkpoint",
        "puppeteer_llm",
    ):
        setattr(ns, name, train_args.get(name))
    if args.puppeteer_llm is not None:
        ns.puppeteer_llm = str(args.puppeteer_llm)
    apply_checkpoint_eval_defaults(ns)
    _apply_puppeteer_checkpoint_defaults(ns)
    model = _build_puppeteer_model(ns, device)
    tokenizer = model.tokenizer
    default_pose_seed = ns.seed if hasattr(ns, "seed") else 20260527
    pose_seed_a = default_pose_seed if args.pose_seed_a is None else int(args.pose_seed_a)
    dataset = PuppeteerDynamicRigDataset(
        args.manifest,
        frame_count=ns.frames,
        limit=args.limit,
        random_query=False,
        seed=pose_seed_a,
        motion_fps_ratio=ns.motion_fps_ratio,
        motion_vertex_samples=ns.motion_vertex_samples,
        max_joints=ns.n_max_joints,
    )
    manifest_metadata = _manifest_metadata(args.manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=puppeteer_dynamic_collate)
    paired_loader = None
    if args.pose_seed_b is not None:
        paired_dataset = PuppeteerDynamicRigDataset(
            args.manifest,
            frame_count=ns.frames,
            limit=args.limit,
            random_query=False,
            seed=int(args.pose_seed_b),
            motion_fps_ratio=ns.motion_fps_ratio,
            motion_vertex_samples=ns.motion_vertex_samples,
            max_joints=ns.n_max_joints,
        )
        if dataset.paths != paired_dataset.paths:
            raise RuntimeError("paired pose datasets resolved different manifest rows")
        paired_loader = DataLoader(
            paired_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=puppeteer_dynamic_collate,
        )
    continuous_range = _puppeteer_metric_range(tokenizer)
    rows = []
    condition_a_cache: list[tuple[str, torch.Tensor]] = []
    condition_b_cache: list[tuple[str, torch.Tensor]] = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        paired_iter = iter(paired_loader) if paired_loader is not None else None
        for idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            paired_batch = None if paired_iter is None else _move_batch(next(paired_iter), device)
            target = _target_namespace(batch)
            trace_a = None
            if args.stage_trace:
                trace_a = _condition_stage_trace(
                    model,
                    batch,
                    args.surface_seed + idx,
                    pose_attention_scale=args.pose_attention_scale,
                    pose_mlp_scale=args.pose_mlp_scale,
                    temporal_attention_scale=args.temporal_attention_scale,
                    temporal_mlp_scale=args.temporal_mlp_scale,
                    trace_attention_routing=args.attention_routing,
                )
                cond = trace_a["condition"]
            else:
                cond = _condition_with_seed(model, batch, args.surface_seed + idx)
            condition_a_cache.append((batch["path"][0], cond.detach().cpu()))
            forced_ids, forced_stats = _forced_argmax_ids(model, batch, cond=cond)
            try:
                forced_pred = _decode_generated_ids(tokenizer, forced_ids)
                forced_metrics = _output_metrics(forced_pred, target, continuous_range)
            except Exception as exc:
                forced_metrics = {"detokenize_error": repr(exc)}
            sens = _condition_sensitivity(model, batch, cond=cond)
            row = {
                    "index": idx,
                    "path": batch["path"][0],
                    **manifest_metadata.get(batch["path"][0], {}),
                    "pose_seed_a": int(pose_seed_a),
                    "target_joint_count": int(target.joints.shape[0]),
                    **forced_stats,
                    "forced_pred_joint_count": forced_metrics.get("pred_joint_count"),
                    "forced_topology_f1": forced_metrics.get("topology", {}).get("edge_f1"),
                    "forced_joint_chamfer_mean": forced_metrics.get("joint_chamfer", {}).get("mean"),
                    **sens,
            }
            greedy_a = None
            if args.greedy_prefix_tokens > 0:
                greedy_a = _greedy_prefix_ids(model, cond, args.greedy_prefix_tokens)
                row["greedy_pose_a_ids"] = greedy_a
            if paired_batch is not None:
                trace_b = None
                if args.stage_trace:
                    assert trace_a is not None
                    trace_b = _condition_stage_trace(
                        model,
                        paired_batch,
                        args.surface_seed + idx,
                        pose_attention_scale=args.pose_attention_scale,
                        pose_mlp_scale=args.pose_mlp_scale,
                        temporal_attention_scale=args.temporal_attention_scale,
                        temporal_mlp_scale=args.temporal_mlp_scale,
                        trace_attention_routing=args.attention_routing,
                    )
                    paired_cond = trace_b["condition"]
                else:
                    paired_cond = _condition_with_seed(model, paired_batch, args.surface_seed + idx)
                condition_b_cache.append((paired_batch["path"][0], paired_cond.detach().cpu()))
                row["pose_seed_b"] = int(args.pose_seed_b)
                if args.same_asset_pose_condition_swap:
                    row["paired_pose_target_change"] = _paired_pose_target_change(
                        model,
                        batch,
                        paired_batch,
                    )
                    row["same_asset_pose_condition_swap"] = (
                        _target_likelihood_under_condition_swap(
                            model,
                            batch,
                            correct_cond=cond,
                            swapped_cond=paired_cond,
                            swapped_path=(
                                f"{paired_batch['path'][0]}::pose_seed={int(args.pose_seed_b)}"
                            ),
                            scope=args.condition_swap_scope,
                        )
                    )
                else:
                    row["paired_pose_response"] = _paired_pose_response(
                        model,
                        batch,
                        paired_batch,
                        cond_a=cond,
                        cond_b=paired_cond,
                    )
                if args.stage_trace:
                    assert trace_a is not None and trace_b is not None
                    shared_trace_b = _condition_stage_trace(
                        model,
                        paired_batch,
                        args.surface_seed + idx,
                        refs=trace_a["refs"],
                        pose_attention_scale=args.pose_attention_scale,
                        pose_mlp_scale=args.pose_mlp_scale,
                        temporal_attention_scale=args.temporal_attention_scale,
                        temporal_mlp_scale=args.temporal_mlp_scale,
                        trace_attention_routing=args.attention_routing,
                    )
                    row["condition_stage_trace"] = {
                        "manual_forward_validation": {
                            "pose_a": trace_a["trace_validation"],
                            "pose_b": trace_b["trace_validation"],
                            "pose_b_shared_references": shared_trace_b["trace_validation"],
                        },
                        "attention_routing": {
                            "pose_a": trace_a["attention_routing"],
                            "pose_b": trace_b["attention_routing"],
                            "pose_b_shared_references": shared_trace_b["attention_routing"],
                        },
                        "independent_references": {
                            "reference_match": _reference_pair_report(trace_a["refs"], trace_b["refs"]),
                            "stages": _stage_pair_report(trace_a["stages"], trace_b["stages"]),
                        },
                        "shared_pose_a_references": {
                            "stages": _stage_pair_report(trace_a["stages"], shared_trace_b["stages"]),
                        },
                    }
                if args.greedy_prefix_tokens > 0:
                    assert greedy_a is not None
                    greedy_b = _greedy_prefix_ids(model, paired_cond, args.greedy_prefix_tokens)
                    compare_count = min(len(greedy_a), len(greedy_b))
                    changed = [
                        position
                        for position in range(compare_count)
                        if greedy_a[position] != greedy_b[position]
                    ]
                    row["greedy_pose_b_ids"] = greedy_b
                    row["greedy_pose_pair"] = {
                        "compared_tokens": compare_count,
                        "changed_tokens": len(changed),
                        "first_changed_position": changed[0] if changed else -1,
                        "length_a": len(greedy_a),
                        "length_b": len(greedy_b),
                    }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        cross_asset_condition_swap = []
        if args.cross_asset_condition_swap:
            if len(condition_a_cache) < 2:
                raise ValueError("--cross-asset-condition-swap requires at least two rows")
            for idx, batch in enumerate(loader):
                batch = _move_batch(batch, device)
                correct_path, correct_cond_cpu = condition_a_cache[idx]
                if batch["path"][0] != correct_path:
                    raise ValueError(
                        f"condition cache path differs on second pass: "
                        f"{correct_path} vs {batch['path'][0]}"
                    )
                swapped_index = (idx + 1) % len(condition_a_cache)
                swapped_path, swapped_cond_cpu = condition_a_cache[swapped_index]
                swap_report = _target_likelihood_under_condition_swap(
                    model,
                    batch,
                    correct_cond=correct_cond_cpu.to(device),
                    swapped_cond=swapped_cond_cpu.to(device),
                    swapped_path=swapped_path,
                    scope=args.condition_swap_scope,
                )
                cross_asset_condition_swap.append(swap_report)
                print(
                    json.dumps(
                        {"event": "cross_asset_condition_swap", "index": idx, **swap_report},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    same_asset_pose_pairs = []
    for (path_a, cond_a), (path_b, cond_b) in zip(condition_a_cache, condition_b_cache):
        if path_a != path_b:
            raise ValueError(f"condition cache paths differ: {path_a} vs {path_b}")
        same_asset_pose_pairs.append({"path": path_a, **_tensor_pair_report(cond_a, cond_b)})
    different_asset_pairs = []
    for left_index in range(len(condition_a_cache)):
        for right_index in range(left_index + 1, len(condition_a_cache)):
            left_path, left_cond = condition_a_cache[left_index]
            right_path, right_cond = condition_a_cache[right_index]
            different_asset_pairs.append(
                {
                    "left_path": left_path,
                    "right_path": right_path,
                    **_tensor_pair_report(left_cond, right_cond),
                }
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "motion_intervention": {
            "pose_attention_scale": float(args.pose_attention_scale),
            "pose_mlp_scale": float(args.pose_mlp_scale),
            "temporal_attention_scale": float(args.temporal_attention_scale),
            "temporal_mlp_scale": float(args.temporal_mlp_scale),
            "attention_routing": bool(args.attention_routing),
            "greedy_prefix_tokens": int(args.greedy_prefix_tokens),
            "cross_asset_condition_swap": bool(args.cross_asset_condition_swap),
            "same_asset_pose_condition_swap": bool(
                args.same_asset_pose_condition_swap
            ),
            "condition_swap_scope": str(args.condition_swap_scope),
        },
        "motion_encoder_parameters": _parameter_report(model.conditioner.motion_encoder),
        "condition_separation": {
            "same_asset_different_pose": same_asset_pose_pairs,
            "different_asset_pose_a": different_asset_pairs,
        },
        "forced_position_accuracy": _aggregate_forced_positions(rows),
        "cross_asset_condition_swap": cross_asset_condition_swap,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
