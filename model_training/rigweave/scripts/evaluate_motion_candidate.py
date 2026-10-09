"""Paired multi-frame and static checks for a recorded screening checkpoint."""
from __future__ import annotations

import argparse
from functools import partial
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model, _control_batch
from eval_dynamic_rig_generation import _continuous_range, _dynamic_generate, _output_metrics
from motion_experiment_runtime import load_config, source_metadata, verify_manifests, write_json
from train_dynamic_rig import build_tokenizer, move_batch
from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate


def summarize(rows):
    result = {}
    for frames in sorted({row["frames"] for row in rows}):
        for control in ("normal", "zero"):
            selected = [row for row in rows if row["frames"] == frames and row["control"] == control]
            if not selected:
                continue
            generated = [row["generation"] for row in selected if "generation" in row]
            result[f"t{frames}_{control}"] = {
                "count": len(selected), "ce": float(np.mean([row["ce"] for row in selected])),
                "eos_accuracy": float(np.mean([row["eos_accuracy"] for row in selected])),
                "median_forward_ms": float(np.median([row["forward_ms"] for row in selected])),
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in selected),
                "generation_count": len(generated),
                "generation_success": sum(row["success"] for row in generated),
                "hit_max_without_eos": sum(row["hit_max_without_eos"] for row in generated),
                "topology_f1_all_rows": float(np.mean([row["topology_f1_or_zero"] for row in generated])) if generated else None,
            }
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    screen = config["screening"]
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.93)
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    settings = {**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": args.checkpoint}
    checkpoint_step = payload["step"]
    del payload
    model_args = SimpleNamespace(**settings)
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    model = _build_dynamic_model(model_args, tokenizer, device)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint_step,
              "source": source_metadata(config), "manifests": verify_manifests(config),
              "fusion": settings["motion_evidence_fusion"], "biased_heads": settings["motion_evidence_heads"],
              "evaluation_note": "All variants use the same query pose, frame selector, surface seed and full continuous GT. No EOS is forced; non-terminating or invalid outputs receive zero topology F1.",
              "rows": [], "summary": {}}
    data_seed = config["baseline"]["args"]["seed"] + 17
    collate = partial(dynamic_rig_collate, pad_token=tokenizer.pad)
    for frames in screen["validation_frames"]:
        dataset = DynamicRigManifestDataset(
            model_args.val_manifest, tokenizer, frame_count=frames, limit=screen["validation_rows"],
            random_query=False, seed=data_seed, motion_fps_ratio=0.75, minimum_random_frames=1,
            motion_vertex_samples=config["baseline"]["args"]["motion_vertex_samples"],
        )
        for index in range(len(dataset)):
            original = move_batch(collate([dataset[index]]), device)
            controls = ["normal"]
            if frames == 8 and index < screen["static_validation_rows"]:
                controls.append("zero")
            for control in controls:
                batch = _control_batch(original, control, seed=data_seed + index)
                torch.manual_seed(data_seed + index)
                refs = model.sample_references(batch)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cond = model.build_condition(batch, refs=refs)
                    result = model._ar_losses(cond, batch)
                torch.cuda.synchronize()
                ce = float(result["ce_loss"])
                if not np.isfinite(ce):
                    raise RuntimeError(f"non-finite validation loss: {batch['path'][0]}")
                row = {"index": index, "path": batch["path"][0], "frames": frames, "control": control,
                       "selected_frames": batch["selected_frames"][0].tolist(),
                       "query_center": batch["query_center"][0].tolist(), "query_scale": float(batch["query_scale"][0]),
                       "ce": ce, "eos_accuracy": float(result["eos_acc"]),
                       "forward_ms": 1000 * (time.perf_counter() - start),
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                if control == "normal" and frames in screen["generation_frames"] and index < screen["generation_rows"]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = _dynamic_generate(
                            model, tokenizer, batch, screen["max_new_tokens"],
                            generation_kwargs={"do_sample": False, "num_beams": 1}, precomputed_cond=cond,
                        )
                    has_eos = bool(np.any(generated == int(tokenizer.eos)))
                    hit_max = len(generated) - 2 >= screen["max_new_tokens"] and not has_eos
                    prediction, error = None, None
                    try:
                        prediction = tokenizer.detokenize(generated)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    success = has_eos and prediction is not None and not hit_max
                    generation = {"generated_ids": generated.tolist(), "has_eos": has_eos,
                                  "hit_max_without_eos": hit_max, "detokenize_ok": prediction is not None,
                                  "success": success, "error": error, "topology_f1_or_zero": 0.0}
                    if success:
                        count = int(batch["joint_count"][0])
                        parents = batch["target_parents"][0, :count].cpu().tolist()
                        target = SimpleNamespace(joints=batch["target_joints"][0, :count].cpu().numpy(),
                                                 parents=[None if value < 0 else int(value) for value in parents])
                        metrics = _output_metrics(prediction, target, _continuous_range(tokenizer))
                        generation["metrics"] = metrics
                        generation["topology_f1_or_zero"] = float(metrics["topology"]["edge_f1"])
                    row["generation"] = generation
                report["rows"].append(row)
                del cond, result, refs
            del original, batch
            if (index + 1) % 8 == 0:
                print(json.dumps({"checkpoint": args.checkpoint.parent.name, "frames": frames,
                                  "completed": index + 1, "total": len(dataset)}), flush=True)
                report["summary"] = summarize(report["rows"])
                write_json(args.output, report)
        report["summary"] = summarize(report["rows"])
        write_json(args.output, report)
    normal_scores = [value["ce"] for key, value in report["summary"].items() if key.endswith("_normal")]
    report["mean_multiframe_ce"] = float(np.mean(normal_scores))
    report["complete"] = True
    write_json(args.output, report)
    print("COMPLETE", json.dumps({"checkpoint": str(args.checkpoint), "score": report["mean_multiframe_ce"],
                                  "summary": report["summary"]}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
