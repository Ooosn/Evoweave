#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
) -> tuple[list[int], dict[str, float | int]]:
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
    stats = {
        "forced_token_acc": float((pred[valid] == labels[valid]).float().mean().detach().cpu()),
        "forced_coord_acc": float((pred[coord_mask] == labels[coord_mask]).float().mean().detach().cpu()),
        "forced_parent_acc": float((pred[parent_mask] == labels[parent_mask]).float().mean().detach().cpu()),
        "forced_eos_top1": int(pred[eos_pos].item() == tokenizer.eos_token_id),
        "forced_eos_rank": eos_rank,
        "forced_eos_prob": float(eos_prob.detach().cpu()),
    }
    return pred_ids, stats


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
        report["slot_cosine_mean"] = float(
            torch.nn.functional.cosine_similarity(left_f, right_f, dim=-1).mean().item()
        )
    return report


@torch.no_grad()
def _condition_stage_trace(
    model,
    batch: dict,
    seed: int,
    *,
    refs=None,
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
    block_outputs: list[torch.Tensor] = []
    handles = []

    def capture_block(_module, _inputs, output):
        block_outputs.append(output[:, 0, anchor_start:].detach())

    for block in motion_encoder.blocks:
        handles.append(block.register_forward_hook(capture_block))
    try:
        motion_output = motion_encoder(surface_sequence, query_points=query_sequence)
    finally:
        for handle in handles:
            handle.remove()

    projected = model.prefix_projector(motion_output)
    decoder_condition = model._condition_embeds_from_raw(projected)
    stages: dict[str, torch.Tensor] = {
        "query_points_frame0": query_sequence[:, 0],
        "surface_tokens_frame0": surface_sequence[:, 0],
    }
    for block_index, output in enumerate(block_outputs):
        stages[f"motion_block_{block_index:02d}_frame0"] = output
    stages["motion_output_frame0"] = motion_output
    stages["projected_condition"] = projected
    stages["decoder_condition"] = decoder_condition
    return {"refs": refs, "stages": stages, "condition": decoder_condition}


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


def _paired_pose_response(
    model,
    batch_a: dict,
    batch_b: dict,
    *,
    cond_a: torch.Tensor,
    cond_b: torch.Tensor,
) -> dict[str, object]:
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
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--pose-seed-a", type=int, default=None)
    parser.add_argument("--pose-seed-b", type=int, default=None)
    parser.add_argument("--surface-seed", type=int, default=20260715)
    parser.add_argument("--stage-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        paired_iter = iter(paired_loader) if paired_loader is not None else None
        for idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            paired_batch = None if paired_iter is None else _move_batch(next(paired_iter), device)
            target = _target_namespace(batch)
            trace_a = None
            if args.stage_trace:
                trace_a = _condition_stage_trace(model, batch, args.surface_seed + idx)
                cond = trace_a["condition"]
            else:
                cond = _condition_with_seed(model, batch, args.surface_seed + idx)
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
                    "pose_seed_a": int(pose_seed_a),
                    "target_joint_count": int(target.joints.shape[0]),
                    **forced_stats,
                    "forced_pred_joint_count": forced_metrics.get("pred_joint_count"),
                    "forced_topology_f1": forced_metrics.get("topology", {}).get("edge_f1"),
                    "forced_joint_chamfer_mean": forced_metrics.get("joint_chamfer", {}).get("mean"),
                    **sens,
            }
            if paired_batch is not None:
                trace_b = None
                if args.stage_trace:
                    assert trace_a is not None
                    trace_b = _condition_stage_trace(model, paired_batch, args.surface_seed + idx)
                    paired_cond = trace_b["condition"]
                else:
                    paired_cond = _condition_with_seed(model, paired_batch, args.surface_seed + idx)
                row["pose_seed_b"] = int(args.pose_seed_b)
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
                    )
                    row["condition_stage_trace"] = {
                        "independent_references": {
                            "reference_match": _reference_pair_report(trace_a["refs"], trace_b["refs"]),
                            "stages": _stage_pair_report(trace_a["stages"], trace_b["stages"]),
                        },
                        "shared_pose_a_references": {
                            "stages": _stage_pair_report(trace_a["stages"], shared_trace_b["stages"]),
                        },
                    }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
