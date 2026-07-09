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


def _forced_logits(model, batch: dict):
    logits, token_batch, _loss = model.teacher_forced_logits(batch)
    return logits, token_batch


def _forced_argmax_ids(model, batch: dict) -> tuple[list[int], dict[str, float | int]]:
    tokenizer = model.tokenizer
    logits, token_batch = _forced_logits(model, batch)
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


def _condition_sensitivity(model, batch: dict) -> dict[str, object]:
    tokenizer = model.tokenizer
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
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
    dataset = PuppeteerDynamicRigDataset(
        args.manifest,
        frame_count=ns.frames,
        limit=args.limit,
        random_query=False,
        seed=ns.seed if hasattr(ns, "seed") else 20260527,
        motion_fps_ratio=ns.motion_fps_ratio,
        motion_vertex_samples=ns.motion_vertex_samples,
        max_joints=ns.n_max_joints,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=puppeteer_dynamic_collate)
    continuous_range = _puppeteer_metric_range(tokenizer)
    rows = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            target = _target_namespace(batch)
            forced_ids, forced_stats = _forced_argmax_ids(model, batch)
            try:
                forced_pred = _decode_generated_ids(tokenizer, forced_ids)
                forced_metrics = _output_metrics(forced_pred, target, continuous_range)
            except Exception as exc:
                forced_metrics = {"detokenize_error": repr(exc)}
            sens = _condition_sensitivity(model, batch)
            rows.append(
                {
                    "index": idx,
                    "path": batch["path"][0],
                    "target_joint_count": int(target.joints.shape[0]),
                    **forced_stats,
                    "forced_pred_joint_count": forced_metrics.get("pred_joint_count"),
                    "forced_topology_f1": forced_metrics.get("topology", {}).get("edge_f1"),
                    "forced_joint_chamfer_mean": forced_metrics.get("joint_chamfer", {}).get("mean"),
                    **sens,
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
