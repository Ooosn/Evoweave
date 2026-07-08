#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def target_namespace(batch: dict) -> SimpleNamespace:
    joint_count = int(batch["joint_count"][0].detach().cpu())
    joints = batch["target_joints"][0, :joint_count].detach().cpu().numpy().astype(np.float32)
    parents_raw = batch["target_parents"][0, :joint_count].detach().cpu().numpy().astype(np.int64)
    parents = [None if int(parent) < 0 else int(parent) for parent in parents_raw.tolist()]
    return SimpleNamespace(joints=joints, parents=parents)


def decode(tokenizer, ids: list[int]) -> SimpleNamespace:
    decoded = tokenizer.decode_ids(ids, require_eos=False)
    parents = [None if int(parent) < 0 else int(parent) for parent in decoded["parents"].tolist()]
    return SimpleNamespace(joints=np.asarray(decoded["joints"], dtype=np.float32), parents=parents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.model_root / "rigweave" / "src"))
    sys.path.insert(0, str(args.model_root / "rigweave" / "scripts"))
    sys.path.insert(0, str(args.model_root / "third_party_references" / "Puppeteer" / "skeleton"))

    from eval_dynamic_rig_ce import apply_checkpoint_eval_defaults
    from eval_dynamic_rig_generation import _apply_puppeteer_checkpoint_defaults, _build_puppeteer_model, _output_metrics, _puppeteer_metric_range
    from rigweave.dynamic_rig import PuppeteerDynamicRigDataset, puppeteer_dynamic_collate

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _build_puppeteer_model(ns, device)
    model.decoder.config.use_cache = False
    model.decoder.generation_config.use_cache = False
    tokenizer = model.tokenizer
    dataset = PuppeteerDynamicRigDataset(
        args.manifest,
        frame_count=ns.frames,
        limit=args.limit,
        random_query=False,
        seed=20260527,
        motion_fps_ratio=ns.motion_fps_ratio,
        motion_vertex_samples=ns.motion_vertex_samples,
        max_joints=ns.n_max_joints,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=puppeteer_dynamic_collate)
    rows = []
    continuous_range = _puppeteer_metric_range(tokenizer)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            target = target_namespace(batch)
            generated = model.generate_skeleton(
                batch,
                max_new_tokens=int(ns.n_max_joints) * 4 + 1,
                max_joints=int(ns.n_max_joints),
                generation_kwargs={"do_sample": False},
            )
            used_ids = generated["generated_ids"].astype(int).tolist()
            pred = SimpleNamespace(
                joints=np.asarray(generated["joints"], dtype=np.float32),
                parents=[None if int(parent) < 0 else int(parent) for parent in generated["parents"].tolist()],
            )
            metrics = _output_metrics(pred, target, continuous_range)
            row = {
                "index": idx,
                "target_joint_count": int(target.joints.shape[0]),
                "raw_first_tokens": used_ids[:24],
                "used_first_tokens": used_ids[:24],
                "generated_len": len(used_ids),
                "has_eos": int(tokenizer.eos_token_id in used_ids),
                "pred_joint_count": metrics.get("pred_joint_count"),
                "topology_f1": metrics.get("topology", {}).get("edge_f1") if isinstance(metrics.get("topology"), dict) else None,
                "joint_chamfer_mean": metrics.get("joint_chamfer", {}).get("mean") if isinstance(metrics.get("joint_chamfer"), dict) else None,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
