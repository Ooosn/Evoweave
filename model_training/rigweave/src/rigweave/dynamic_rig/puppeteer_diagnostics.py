from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .puppeteer_dynamic import PuppeteerDynamicRigModel


def _tensor_delta_report(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f = left.detach().float()
    right_f = right.detach().float()
    diff = left_f - right_f
    left_flat = left_f.reshape(int(left_f.shape[0]), -1)
    right_flat = right_f.reshape(int(right_f.shape[0]), -1)
    cosine = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)
    return {
        "l2": float(diff.norm().cpu()),
        "relative_l2": float((diff.norm() / left_f.norm().clamp_min(1.0e-12)).cpu()),
        "mean_abs": float(diff.abs().mean().cpu()),
        "max_abs": float(diff.abs().max().cpu()),
        "cosine": float(cosine.mean().cpu()),
    }


def _masked_logit_delta_report(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    selected_left = left.detach().float()[mask]
    selected_right = right.detach().float()[mask]
    if selected_left.numel() == 0:
        raise ValueError("logit delta audit received an empty token mask")
    return {
        "positions": int(selected_left.shape[0]),
        **_tensor_delta_report(selected_left, selected_right),
    }


def pose_target_contract_audit(batch: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item_index, path_text in enumerate(batch["path"]):
        path = Path(path_text)
        query_frame = int(batch["selected_frames"][item_index, 0].detach().cpu())
        joint_count = int(batch["joint_count"][item_index].detach().cpu())
        vertex_count = int(batch["vertex_count"][item_index].detach().cpu())
        query_center = batch["query_center"][item_index].detach().cpu().numpy().astype(np.float32)
        query_scale = float(batch["query_scale"][item_index].detach().cpu())
        with np.load(path, allow_pickle=True) as raw:
            raw_frames = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)
            raw_targets = np.asarray(raw["target_joints_rootspace"], dtype=np.float32)
        expected_mesh = (raw_frames[query_frame] - query_center) / query_scale
        expected_target = (raw_targets[query_frame] - query_center) / query_scale
        frame0_target = (raw_targets[0] - query_center) / query_scale
        actual_mesh = batch["frame_vertices"][item_index, 0, :vertex_count].detach().cpu().numpy()
        actual_target = batch["target_joints"][item_index, :joint_count].detach().cpu().numpy()
        target_delta = actual_target - expected_target[:joint_count]
        mesh_delta = actual_mesh - expected_mesh[:vertex_count]
        query_vs_frame0 = expected_target[:joint_count] - frame0_target[:joint_count]
        rows.append(
            {
                "path": str(path),
                "query_frame": query_frame,
                "frame_count": int(raw_frames.shape[0]),
                "joint_count": joint_count,
                "query_is_frame0": query_frame == 0,
                "target_query_match_max_abs": float(np.max(np.abs(target_delta))),
                "target_query_match_rms": float(np.sqrt(np.mean(np.square(target_delta)))),
                "mesh_query_match_max_abs": float(np.max(np.abs(mesh_delta))),
                "mesh_query_match_rms": float(np.sqrt(np.mean(np.square(mesh_delta)))),
                "query_target_vs_frame0_rms": float(np.sqrt(np.mean(np.square(query_vs_frame0)))),
                "sequence_target_motion_rms_max": float(
                    max(
                        np.sqrt(np.mean(np.square(raw_targets[frame] - raw_targets[query_frame])))
                        for frame in range(int(raw_targets.shape[0]))
                    )
                ),
            }
        )
    return {
        "rows": rows,
        "max_target_query_match_abs": max(row["target_query_match_max_abs"] for row in rows),
        "max_mesh_query_match_abs": max(row["mesh_query_match_max_abs"] for row in rows),
    }


def _sequence_control_batch(batch: dict[str, Any], control: str) -> dict[str, Any]:
    controlled = dict(batch)
    for key in ("frame_vertices", "vertex_normals", "face_normals"):
        value = batch.get(key)
        if value is None:
            continue
        if control == "static_query":
            controlled[key] = value[:, :1].expand_as(value)
        elif control == "rotate_query":
            controlled[key] = torch.roll(value, shifts=-1, dims=1)
        else:
            raise ValueError(f"unknown sequence control {control!r}")
    return controlled


@torch.no_grad()
def condition_path_audit(model: PuppeteerDynamicRigModel, batch: dict[str, Any]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        device = batch["frame_vertices"].device
        generator = torch.Generator(device=device)
        generator.manual_seed(2026071401)
        refs = model.sample_references(batch, generator=generator)

        def encode(source: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
            raw = model.conditioner(
                source["frame_vertices"],
                source["faces"],
                refs,
                vertex_normals=source.get("vertex_normals"),
                face_normals=source.get("face_normals"),
            )
            return raw, model.prefix_projector(raw)

        raw_real, projected_real = encode(batch)
        raw_static, projected_static = encode(_sequence_control_batch(batch, "static_query"))
        raw_rotated, projected_rotated = encode(_sequence_control_batch(batch, "rotate_query"))

        cond_real = model._condition_embeds_from_raw(projected_real)
        cond_static = model._condition_embeds_from_raw(projected_static)
        cond_rotated = model._condition_embeds_from_raw(projected_rotated)
        cond_zero = model._condition_embeds_from_raw(torch.zeros_like(projected_real))
        bos = torch.full(
            (int(cond_real.shape[0]), 1),
            model.tokenizer.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        logits_real = model._next_token_logits(cond_real, bos)
        logits_static = model._next_token_logits(cond_static, bos)
        logits_rotated = model._next_token_logits(cond_rotated, bos)
        logits_zero = model._next_token_logits(cond_zero, bos)

        forced_real, token_batch, loss_real = model.teacher_forced_logits(batch, cond=cond_real)
        forced_static, _static_tokens, loss_static = model.teacher_forced_logits(batch, cond=cond_static)
        forced_rotated, _rotated_tokens, loss_rotated = model.teacher_forced_logits(batch, cond=cond_rotated)
        forced_zero, _zero_tokens, loss_zero = model.teacher_forced_logits(batch, cond=cond_zero)
        valid = token_batch.labels.to(device) != -100
        role = token_batch.token_role.to(device)
        coord = valid & (role >= model.tokenizer.offset) & ((role - model.tokenizer.offset) < 3)
        parent = valid & (role == model.tokenizer.offset + 3)
        eos = valid & (token_batch.labels.to(device) == model.tokenizer.eos_token_id)

        forced_reports: dict[str, Any] = {}
        for name, candidate in (
            ("static", forced_static),
            ("rotated_query", forced_rotated),
            ("zero_condition", forced_zero),
        ):
            forced_reports[name] = {
                "all_valid": _masked_logit_delta_report(forced_real, candidate, valid),
                "coordinate": _masked_logit_delta_report(forced_real, candidate, coord),
                "parent": _masked_logit_delta_report(forced_real, candidate, parent),
                "eos": _masked_logit_delta_report(forced_real, candidate, eos),
            }

        forced_losses = {
            "real": float(loss_real.detach().cpu()),
            "static": float(loss_static.detach().cpu()),
            "rotated_query": float(loss_rotated.detach().cpu()),
            "zero_condition": float(loss_zero.detach().cpu()),
            "static_minus_real": float((loss_static - loss_real).detach().cpu()),
            "rotated_query_minus_real": float((loss_rotated - loss_real).detach().cpu()),
            "zero_condition_minus_real": float((loss_zero - loss_real).detach().cpu()),
        }
        if int(cond_real.shape[0]) > 1:
            cond_swapped = torch.roll(cond_real, shifts=1, dims=0)
            forced_swapped, _swapped_tokens, loss_swapped = model.teacher_forced_logits(batch, cond=cond_swapped)
            forced_reports["batch_swapped"] = {
                "all_valid": _masked_logit_delta_report(forced_real, forced_swapped, valid),
                "coordinate": _masked_logit_delta_report(forced_real, forced_swapped, coord),
                "parent": _masked_logit_delta_report(forced_real, forced_swapped, parent),
                "eos": _masked_logit_delta_report(forced_real, forced_swapped, eos),
            }
            forced_losses["batch_swapped"] = float(loss_swapped.detach().cpu())
            forced_losses["batch_swapped_minus_real"] = float((loss_swapped - loss_real).detach().cpu())
        return {
            "raw_condition_shape": list(raw_real.shape),
            "projected_condition_shape": list(projected_real.shape),
            "raw_real_vs_static": _tensor_delta_report(raw_real, raw_static),
            "raw_real_vs_rotated_query": _tensor_delta_report(raw_real, raw_rotated),
            "projected_real_vs_static": _tensor_delta_report(projected_real, projected_static),
            "projected_real_vs_rotated_query": _tensor_delta_report(projected_real, projected_rotated),
            "first_token_logits_real_vs_static": _tensor_delta_report(logits_real, logits_static),
            "first_token_logits_real_vs_rotated_query": _tensor_delta_report(logits_real, logits_rotated),
            "first_token_logits_real_vs_zero_condition": _tensor_delta_report(logits_real, logits_zero),
            "first_token_top1": {
                "real": logits_real.argmax(dim=-1).detach().cpu().tolist(),
                "static": logits_static.argmax(dim=-1).detach().cpu().tolist(),
                "rotated_query": logits_rotated.argmax(dim=-1).detach().cpu().tolist(),
                "zero_condition": logits_zero.argmax(dim=-1).detach().cpu().tolist(),
            },
            "teacher_forced_logit_deltas": forced_reports,
            "teacher_forced_losses": forced_losses,
            "raw_condition_std": float(raw_real.float().std().cpu()),
            "projected_condition_std": float(projected_real.float().std().cpu()),
            "projected_tokenwise_std_mean": float(projected_real.float().std(dim=1).mean().cpu()),
        }
    finally:
        model.train(was_training)


def _decoder_block_parameters(model: PuppeteerDynamicRigModel) -> Iterable[torch.nn.Parameter]:
    decoder = model.decoder.model.decoder
    yield from decoder.layers.parameters()
    final_layer_norm = getattr(decoder, "final_layer_norm", None)
    if final_layer_norm is not None:
        yield from final_layer_norm.parameters()


def _append_unique_parameters(
    output: list[torch.nn.Parameter],
    parameters: Iterable[torch.nn.Parameter],
    seen: set[int],
) -> None:
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        output.append(parameter)


def gradient_path_audit(model: PuppeteerDynamicRigModel, loss: torch.Tensor) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    loss.backward()

    seen: set[int] = set()
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, parameters in (
        ("surface", model.conditioner.surface_tokenizer.parameters()),
        ("motion", model.conditioner.motion_encoder.parameters()),
        ("prefix_projector", model.prefix_projector.parameters()),
        ("joint_slot_embedding", [] if model.target_aware_pos_embed is None else [model.target_aware_pos_embed]),
        ("decoder_blocks", _decoder_block_parameters(model)),
        ("decoder_token_path", model.decoder.parameters()),
    ):
        group: list[torch.nn.Parameter] = []
        _append_unique_parameters(group, parameters, seen)
        groups[name] = group

    reports: dict[str, Any] = {}
    global_sq = 0.0
    for name, parameters in groups.items():
        group_sq = 0.0
        group_max = 0.0
        grad_parameters = 0
        grad_elements = 0
        nonfinite = 0
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            grad_f = grad.detach().float()
            if not bool(torch.isfinite(grad_f).all().item()):
                nonfinite += 1
            norm = float(grad_f.norm().cpu())
            group_sq += norm * norm
            group_max = max(group_max, float(grad_f.abs().max().cpu()))
            grad_parameters += 1
            grad_elements += int(parameter.numel())
        group_norm = math.sqrt(group_sq)
        global_sq += group_sq
        reports[name] = {
            "parameter_elements": int(sum(parameter.numel() for parameter in parameters)),
            "gradient_parameters": grad_parameters,
            "gradient_elements": grad_elements,
            "gradient_norm": group_norm,
            "gradient_abs_max": group_max,
            "nonfinite_gradient_parameters": nonfinite,
        }

    global_norm = math.sqrt(global_sq)
    clip_coefficient = min(1.0, 1.0 / max(global_norm, 1.0e-12))
    for report in reports.values():
        report["global_gradient_energy_fraction"] = (
            float(report["gradient_norm"] ** 2 / global_sq) if global_sq > 0.0 else 0.0
        )
        report["gradient_norm_after_global_clip_1"] = float(report["gradient_norm"] * clip_coefficient)
    result = {
        "loss": float(loss.detach().cpu()),
        "global_gradient_norm": global_norm,
        "global_clip_1_coefficient": clip_coefficient,
        "groups": reports,
        "unassigned_trainable_parameter_elements": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in seen)
        ),
    }
    model.zero_grad(set_to_none=True)
    return result
