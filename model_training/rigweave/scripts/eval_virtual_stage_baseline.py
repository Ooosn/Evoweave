"""Read-only native-checkpoint evaluation on the virtual-frame trial inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_experiment_runtime import guard, load_config, source_metadata, write_json
from train_virtual_frame import dataset_for, evaluation_rows, geometry_scores, load_entries
from virtual_frame_runtime import file_sha256


def verify_saved_input(batch, saved):
    for field, actual in (("target_joints_common", batch["frame_target_joints"][0].numpy()),
                          ("target_parents", batch["target_parents"][0].numpy()),
                          ("selected_frames", batch["selected_frames"].numpy()),
                          ("query_center", batch["query_center"].numpy()),
                          ("query_scale", batch["query_scale"].numpy())):
        if not np.array_equal(actual, np.asarray(saved[field])):
            raise ValueError(f"native baseline input differs from saved new-model evaluation: {field}")
    if batch["path"] != saved["path"]:
        raise ValueError("native baseline backing NPZ differs")


def prepare_paired_surface(batch, config, device):
    from rigweave.dynamic_rig import virtual_frame_data
    captured = []
    sample = virtual_frame_data.sample_trackable_surface
    def capture(*args, **kwargs):
        refs = sample(*args, **kwargs)
        captured.append(refs)
        return refs
    with patch.object(virtual_frame_data, "sample_trackable_surface", capture):
        surface, targets = virtual_frame_data.prepare_virtual_surface([batch], device, **config["surface_sampling"])
    if len(captured) != 1:
        raise ValueError("paired preparation must draw exactly one surface correspondence")
    return surface, targets, captured[0]


def joint_coordinate_offsets(ids, parents, tokenizer):
    """Map joint XYZ tokens without inferring identity from quantized positions."""
    offsets, cursor = [], 2
    for index, parent in enumerate(parents):
        if index and parent != index - 1:
            if ids[cursor] != tokenizer.token_id_branch:
                raise ValueError("unexpected native branch serialization")
            cursor += 4
        offsets.append([cursor, cursor + 1, cursor + 2])
        if any(not 0 <= int(value) < tokenizer.num_discrete for value in ids[cursor:cursor + 3]):
            raise ValueError("joint XYZ positions are not coordinate tokens")
        cursor += 3
    if cursor != len(ids) - 1 or ids[cursor] != tokenizer.eos:
        raise ValueError("unexpected native target serialization length")
    return np.asarray(offsets, dtype=np.int64)


def teacher_result(model, tokenizer, condition, batch):
    from eval_dynamic_rig_generation import _topology_metrics
    from src.tokenizer.tokenizer_part import undiscretize
    ids, mask = batch["input_ids"], batch["attention_mask"]
    prompt = torch.cat((condition.to(model.transformer.dtype), model.token_inputs_embeds(ids, mask)), dim=1)
    attention = torch.nn.functional.pad(mask, (condition.shape[1], 0), value=1)
    logits = model.transformer(inputs_embeds=prompt, attention_mask=attention, use_cache=False).logits
    logits = logits[:, condition.shape[1]:-1].float()
    labels = ids[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    coordinate = (labels >= 0) & (labels < tokenizer.num_discrete)
    parents = batch["target_parents"][0].tolist()
    offsets = joint_coordinate_offsets(ids[0].cpu().tolist(), parents, tokenizer)
    bins = logits[0, torch.as_tensor(offsets - 1, device=logits.device), :tokenizer.num_discrete].argmax(-1).cpu().numpy()
    xyz = undiscretize(bins, tokenizer.continuous_range, tokenizer.num_discrete).astype(np.float32)
    target = batch["target_joints"][0].cpu().numpy()
    return {"full_gt_prefix_ce": float(losses.mean()), "coordinate_ce": float(losses[coordinate].mean()),
            "coordinate_accuracy": float((logits.argmax(-1)[coordinate] == labels[coordinate]).float().mean()),
            "full_gt_prefix_joint_xyz_l1": float(np.abs(xyz - target).mean()),
            "full_gt_prefix_edge_f1": _topology_metrics(xyz, parents, target, parents)["edge_f1"],
            "joint_xyz": xyz}


def summarize(rows):
    result = {}
    for split in ("train", "valid"):
        group = [row for row in rows if row["split"] == split]
        result[split] = {"assets": len(group), "natural_eos": sum(row["success"] for row in group)}
        for key in ("edge_f1", "edge_precision", "edge_recall", "exact_count", "full_gt_prefix_ce",
                    "coordinate_ce", "coordinate_accuracy", "full_gt_prefix_joint_xyz_l1", "full_gt_prefix_edge_f1"):
            result[split][key] = float(np.mean([row[key] for row in group])) if group else None
        successful = [row for row in group if row["geometry"]["j2j"] is not None]
        result[split]["geometry_successful_assets"] = len(successful)
        result[split]["geometry"] = {key: float(np.mean([row["geometry"][key] for row in successful])) if successful else None
                                      for key in ("j2j", "j2b", "b2b")}
    return result


@torch.no_grad()
def run(args):
    config = load_config(args.config)
    reference = config["reference"]
    virtual = load_config(config["virtual_config"])
    guard(config, "matched_eval")
    source = source_metadata(config)
    state = json.loads((Path(config["runtime"]["repo"]) / "model_training/state/current.json").read_text())
    operation = state["active_operation"]
    if operation["label"] != config["label"] or operation["config_sha256"] != file_sha256(args.config):
        raise ValueError("baseline evaluation differs from the recorded operation")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.9, device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / f"rank_{rank}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    if file_sha256(reference["path"]) != reference["sha256"]:
        raise ValueError("native checkpoint checksum differs")
    payload = torch.load(reference["path"], map_location="cpu", mmap=True, weights_only=False)
    if payload["step"] != reference["step"] or payload["sample_seen"] != reference["sample_seen"]:
        raise ValueError("native checkpoint training progress differs")
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from eval_dynamic_rig_generation import _topology_metrics
    from probe_common_range import generation_metrics
    from rigweave.dynamic_rig.data import _make_token_input
    from train_dynamic_rig import build_tokenizer, move_batch
    model_args = SimpleNamespace(**{**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": Path(reference["path"])})
    del payload
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    original_load = torch.nn.Module.load_state_dict
    def strict_load(module, *values, **kwargs):
        result = original_load(module, *values, **kwargs)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"native model load mismatch: {result}")
        return result
    with patch.object(torch.nn.Module, "load_state_dict", strict_load):
        model = _build_dynamic_model(model_args, tokenizer, device)
    model.eval().requires_grad_(False)
    if model.condition_fusion != "dynamic" or model.branch_prior is not None:
        raise ValueError("paired input replay requires the original dynamic/no-prior method")
    report = {"complete": False, "rank": rank, "world": world, "source": source, "reference": reference,
              "started_at": datetime.now(timezone.utc).isoformat(), "rows": [], "no_parameter_updates": True,
              "comparability": config["comparability"]}
    write_json(report_path, report)
    saved = {}
    for r in range(config["new_world_size"]):
        path = Path(config["new_prediction_root"]) / f"predictions_rank_{r}.pt"
        expected = config["new_prediction_sha256"][r]
        if file_sha256(path) != expected:
            raise ValueError("saved new-model predictions changed")
        saved.update(torch.load(path, map_location="cpu", weights_only=False))
    predictions = {}
    selected = evaluation_rows(load_entries(virtual), virtual)
    for split, entries in selected.items():
        dataset = dataset_for(entries, virtual, mode="seen_eval")
        for index in range(rank, len(entries), world):
            started = time.monotonic()
            entry = entries[index]
            key = f"{split}_{entry['manifest_index']}"
            cpu_batch = dataset[index]
            verify_saved_input(cpu_batch, saved[key])
            parents = cpu_batch["target_parents"][0].tolist()
            ids = _make_token_input(cpu_batch["target_joints"][0].numpy(),
                                   [None if p < 0 else p for p in parents], cpu_batch["cls"][0], tokenizer, None)
            cpu_batch["input_ids"] = torch.as_tensor(ids, dtype=torch.long)[None]
            cpu_batch["attention_mask"] = torch.ones_like(cpu_batch["input_ids"])
            batch = move_batch(cpu_batch, device)
            surface, _, refs = prepare_paired_surface(cpu_batch, virtual, device)
            observed = 0
            def observe(_module, inputs):
                nonlocal observed
                for name, actual in zip(("dense_points", "dense_normals", "query_points", "query_normals"), inputs):
                    if not torch.equal(actual, surface[name][:, observed]):
                        delta = (actual.float() - surface[name][:, observed].float()).abs()
                        raise ValueError(f"old/new sampled encoder input mismatch: {name}, frame {observed}, "
                                         f"maxabs={float(delta.max())}, values={int((delta > 0).sum())}")
                observed += 1
            hook = model.conditioner.surface_tokenizer.register_forward_pre_hook(observe)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    condition = model.build_condition(batch, refs=refs)
            finally:
                hook.remove()
            if observed != virtual["evaluation"]["frames"] or not bool(torch.isfinite(condition).all()):
                raise ValueError("encoder input parity or condition finiteness failed")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher = teacher_result(model, tokenizer, condition, batch)
                if not report["rows"]:
                    native_ce = float(model._ar_losses(condition, batch, include_loop_recovery=False,
                                                       include_generated_prefix_recovery=False)["ce_loss"])
                    if abs(native_ce - teacher["full_gt_prefix_ce"]) > 2e-5:
                        raise ValueError("teacher forcing differs from native training CE")
                    report["teacher_ce_parity"] = {"native_ce": native_ce, "difference": abs(native_ce - teacher["full_gt_prefix_ce"])}
                generated = generation_metrics(model, tokenizer, batch, condition, config["max_new_tokens"])
            target = cpu_batch["target_joints"][0].numpy()
            row = {"split": split, "asset_id": entry["asset_id"], "manifest_index": entry["manifest_index"],
                   "success": generated["success"], "gt_count": len(parents), "pred_count": None,
                   "edge_f1": 0., "edge_precision": 0., "edge_recall": 0., "exact_count": False,
                   "geometry": {name: None for name in ("j2j", "j2b", "b2b")},
                   "selected_frames": cpu_batch["selected_frames"].tolist(), "input_and_label_parity": True,
                   "query_target_maxabs": float(np.abs(target).max()),
                   **{k: v for k, v in teacher.items() if k != "joint_xyz"}}
            if generated["success"]:
                xyz = np.asarray(generated["predicted_joints"], dtype=np.float32)
                pp = [-1 if p is None else p for p in generated["predicted_parents"]]
                row.update({k: v for k, v in _topology_metrics(xyz, pp, target, parents).items() if k.startswith("edge_")})
                row.update(pred_count=len(pp), exact_count=len(pp) == len(parents),
                           geometry=geometry_scores(xyz, pp, target, parents, float(cpu_batch["frame_mesh_bbox_side"][0, 0])))
            row["seconds"] = time.monotonic() - started
            predictions[key] = {"row": row, "generation": generated, "teacher_joints": teacher["joint_xyz"],
                                "target_joints": target, "target_parents": parents, "target_ids": ids}
            report["rows"].append(row)
            write_json(report_path, report)
            print("PAIRED_BASELINE " + json.dumps({"rank": rank, **row}), flush=True)
            del batch, cpu_batch, refs, surface, condition, teacher, generated
    artifact = root / f"predictions_rank_{rank}.pt"
    torch.save(predictions, artifact)
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(), summary=summarize(report["rows"]),
                  prediction_sha256=file_sha256(artifact))
    write_json(report_path, report)
    if world > 1:
        dist.barrier()
    if rank == 0:
        parts = [json.loads((root / f"rank_{r}.json").read_text()) for r in range(world)]
        rows = [row for part in parts for row in part["rows"]]
        expected = {(s, e["asset_id"]) for s, es in selected.items() for e in es}
        if not all(p["complete"] for p in parts) or {(r["split"], r["asset_id"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("paired evaluation incomplete or duplicate")
        write_json(root / "summary.json", {"complete": True, "reference": reference, "source": source,
                   "rows": rows, "summary": summarize(rows), "comparability": config["comparability"],
                   "geometry_units": "query_mesh_bbox_side", "failed_generations_score_zero": True})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args())
