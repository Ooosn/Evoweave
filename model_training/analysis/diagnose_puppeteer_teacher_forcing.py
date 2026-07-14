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
) -> dict[str, float | int | bool | list[int]]:
    if batch_a["path"] != batch_b["path"]:
        raise ValueError(f"paired pose paths differ: {batch_a['path']} vs {batch_b['path']}")

    ids_a = _target_ids(model, batch_a)
    ids_b = _target_ids(model, batch_b)
    if ids_a.shape != ids_b.shape:
        raise ValueError(f"paired target token shapes differ: {tuple(ids_a.shape)} vs {tuple(ids_b.shape)}")

    cond_a_f = cond_a.float()
    cond_b_f = cond_b.float()
    cond_diff = cond_a_f - cond_b_f
    cond_scale = 0.5 * (
        torch.linalg.vector_norm(cond_a_f) + torch.linalg.vector_norm(cond_b_f)
    )

    logits_aa, token_batch_a, _loss = model.teacher_forced_logits(batch_a, cond=cond_a)
    logits_ba, _token_batch_ba, _loss = model.teacher_forced_logits(batch_a, cond=cond_b)
    valid = token_batch_a.labels.to(logits_aa.device) != -100
    shared_a = logits_aa[valid].float()
    shared_b = logits_ba[valid].float()
    shared_diff = shared_a - shared_b
    shared_scale = 0.5 * (
        torch.linalg.vector_norm(shared_a) + torch.linalg.vector_norm(shared_b)
    )
    probs_a = torch.softmax(shared_a, dim=-1)
    probs_b = torch.softmax(shared_b, dim=-1)

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
                paired_cond = _condition_with_seed(model, paired_batch, args.surface_seed + idx)
                row["pose_seed_b"] = int(args.pose_seed_b)
                row["paired_pose_response"] = _paired_pose_response(
                    model,
                    batch,
                    paired_batch,
                    cond_a=cond,
                    cond_b=paired_cond,
                )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
