"""Two-rank full-manifest training with matched periodic generation evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from motion_experiment_runtime import load_config, source_metadata, verify_manifests, write_json
from probe_shared_structure import _finite_gradients, _load_original, _new_model, _optimizer
from probe_shared_structure_medium import (
    asset_seed, generation_metrics, move_batch, online_entries, peak_cpu_gib,
    prepare_online_batches, select_online_input,
)
from relation_probe_metrics import file_hash
from run_shared_structure_experiment import _checkpoint_identity, build_stage_plan, require_full_inventory
from shared_structure_online_batch import select_online_batch


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class AssetStream:
    """Identical rank-local planners partition one shuffled global asset stream."""
    def __init__(self, count, seed):
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.order = self.rng.permutation(count).tolist()
        self.cursor = 0
        self.exposures = np.zeros(count, dtype=np.int64)

    def take(self, size, frames_min, frames_max):
        selected = []
        while len(selected) < size:
            if self.cursor == self.count:
                self.order = self.rng.permutation(self.count).tolist()
                self.cursor = 0
            n = min(size - len(selected), self.count - self.cursor)
            selected.extend(self.order[self.cursor:self.cursor+n])
            self.cursor += n
        np.add.at(self.exposures, selected, 1)
        frames = int(self.rng.integers(frames_min, frames_max + 1))
        banks = ["train_a" if value == 0 else "train_b" for value in self.rng.integers(0, 2, size=size)]
        seeds = self.rng.integers(0, 2**31, size=size).tolist()
        return selected, frames, banks, seeds

    def state(self):
        return {"order": self.order, "cursor": self.cursor, "numpy_rng": self.rng.bit_generator.state,
                "asset_exposure_counts": self.exposures.copy()}


def warmstart_skeleton(model, payload, config):
    old = payload["configuration"]["architecture"]
    new = config["architecture"]
    if any(old[k] != new[k] for k in old if k not in {"xyz_bins", "xyz_min", "xyz_max"}):
        raise ValueError("warm start may only extend the fixed coordinate vocabulary")
    if payload["source_checkpoint"] != _checkpoint_identity(config) or not payload["feature_encoder_frozen"]:
        raise ValueError("warm start has a different feature encoder")
    if payload["training_accounting"]["heldout_frame_exposures"] != 0:
        raise ValueError("warm start contains held-out supervision")
    if payload["step"] != config["full"]["warmstart"]["step"]:
        raise ValueError("warm start step differs from the recorded checkpoint")
    state = dict(payload["skeleton"])
    old_step = (old["xyz_max"] - old["xyz_min"]) / old["xyz_bins"]
    new_step = (new["xyz_max"] - new["xyz_min"]) / new["xyz_bins"]
    offset_float = (old["xyz_min"] - new["xyz_min"]) / new_step
    offset = round(offset_float)
    if abs(new_step-old_step) > 1e-12 or abs(offset-offset_float) > 1e-8:
        raise ValueError("coordinate extension must preserve existing bin centers exactly")
    if offset < 0 or offset + old["xyz_bins"] > new["xyz_bins"]:
        raise ValueError("new vocabulary does not contain the old vocabulary")
    if new["xyz_bins"] != old["xyz_bins"]:
        for suffix in ("weight", "bias"):
            key = f"positions.xyz_head.{suffix}"
            target = model.skeleton.state_dict()[key].detach().cpu().clone()
            target.fill_(0 if suffix == "weight" else -8)
            old_value = state[key].reshape(3, old["xyz_bins"], *state[key].shape[1:])
            target.reshape(3, new["xyz_bins"], *target.shape[1:])[:, offset:offset+old["xyz_bins"]].copy_(old_value)
            state[key] = target
    model.skeleton.load_state_dict(state, strict=True)
    return {"source_step": payload["step"], "prior_head_asset_exposures": payload["training_accounting"]["assets"],
            "old_xyz_bins": old["xyz_bins"], "xyz_bin_offset": offset,
            "new_xyz_bins": new["xyz_bins"], "bin_width_unchanged": old_step,
            "optimizer_reused": False}


def core_inputs(model, source, target, batch):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        anchors, registers = model.encode_surface(source)
    return (anchors, registers,
            torch.ones(source.tokens.shape[:2], dtype=torch.bool, device=anchors.device),
            target.tokens, target.query_points, batch["target_parents"], batch["joint_mask"],
            batch["frame_mask"], batch["frame_target_joints"])


def geometry_metrics(pred, parents, target, target_parents, padding):
    """Exact NN definition of the legacy geometry scores, with bounded memory."""
    from scipy.spatial import cKDTree
    from eval_dynamic_rig_generation import _bones_from_parents, _sample_bones_np
    pb, tb = _bones_from_parents(pred, parents), _bones_from_parents(target, target_parents)
    if not len(pb) or not len(tb):
        return {key: None for key in ("j2j", "j2b", "b2b")}
    ps, ts = _sample_bones_np(pb), _sample_bones_np(tb)

    def distance(a, b):
        return float(cKDTree(b).query(a, workers=1)[0].mean())

    # Convert padded mesh half-extent units into original mesh bbox-side units.
    scale = padding / 2
    return {"j2j": .5 * (distance(pred, target) + distance(target, pred)) * scale,
            "j2b": .5 * (distance(pred, ts) + distance(target, ps)) * scale,
            "b2b": .5 * (distance(ps, ts) + distance(ts, ps)) * scale}


def summarize(rows):
    result = {}
    for split in sorted({row["split"] for row in rows}):
        group = [row for row in rows if row["split"] == split]
        block = {"assets": len(group), "natural_eos": sum(r["success"] for r in group),
                 "exact_tree": sum(r.get("exact_tree", False) for r in group),
                 "exact_count": sum(r.get("count_correct", False) for r in group)}
        for key in ("query", "heldout_mean", "gt_prefix_query", "gt_prefix_heldout_mean", "geometry"):
            if not any(key in r for r in group):
                continue
            names = sorted({n for r in group for n in r.get(key, {})})
            block[key] = {}
            for name in names:
                values = [r[key][name] for r in group if r.get(key, {}).get(name) is not None]
                block[key][name] = float(np.mean(values)) if values else None
            if key == "geometry":
                block["geometry_successful_assets"] = sum(r.get("geometry", {}).get("j2j") is not None for r in group)
        result[split] = block
    return result


def eval_entries(inventory, config):
    reference = config["full"]["evaluation_selection"]
    if file_hash(reference["path"]) != reference["sha256"]:
        raise ValueError("evaluation selection changed")
    chosen = json.loads(Path(reference["path"]).read_text())
    result = {}
    for split, count in (("train", 32), ("valid", 64)):
        by_id = {r["asset_id"]: r for r in inventory[split]}
        result[split] = []
        for row in chosen["splits"][split]["selected"][:count]:
            current = by_id[row["asset_id"]]
            if any(current[k] != row[k] for k in ("path", "index", "frame_count", "joint_count")):
                raise ValueError("evaluation asset differs from the pinned prior selection")
            result[split].append({**current, "integration_source": row["integration_source"]})
        if len(result[split]) != count:
            raise ValueError("incomplete fixed evaluation subset")
    return result


@torch.no_grad()
def evaluate(model, original, entries, config, root, step, rank):
    from eval_dynamic_rig_generation import _topology_metrics
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
    from shared_structure_legacy_f1 import score_view, teacher_control
    model.eval()
    rows, predictions = [], {}
    directory = root / f"evaluation_step_{step:05d}"
    directory.mkdir(exist_ok=True)
    for split, values in entries.items():
        for entry in values[rank::2]:
            source, target, batch, selection = select_online_input(
                original, entry, config, "train_a", 8, training=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inputs = core_inputs(model, source, target, batch)
                output = model.skeleton(*inputs, xyz_weight=config["probe"]["xyz_weight"])
                teacher = model.skeleton.quantizer.decode(output["xyz_logits"].argmax(-1))
                generated = model.skeleton.generate_structure(*inputs[:3], joint_limit=config["probe"]["generation_joint_limit"])
                record = None
                if generated["success"]:
                    record = generated["skeleton_sequence"].cpu_record()
                    generated.update(model.skeleton.localize(generated["skeleton_sequence"], inputs[3], inputs[4]))
            identity = generation_metrics(generated, batch, selection["heldout_target_positions"],
                                          padding=config["data"]["target_mesh_padding"])
            saved = {"selection": selection, "generation": identity, "skeleton_sequence": record,
                     "gt_parents": batch["target_parents"].cpu(),
                     "gt_joints_mesh_local": batch["frame_target_joints"].cpu(),
                     "pred_joints_mesh_local": generated["joints"].float().cpu() if generated["success"] else None,
                     "teacher_joints_mesh_local": teacher.float().cpu(),
                     "target_source_centers": batch["target_source_centers"].cpu(),
                     "target_source_scales": batch["target_source_scales"].cpu()}
            scores, control = score_view(saved, _topology_metrics), teacher_control(saved, _topology_metrics)
            row = {"split": split, "asset_id": entry["asset_id"], "index": entry["index"],
                   "integration_source": entry["integration_source"],
                   "success": generated["success"], "exact_tree": identity["exact_tree"],
                   "count_correct": identity["count_correct"], "structure_ce": float(output["structure_loss"]),
                   "xyz_ce": float(output["xyz_loss"]), "identity": identity,
                   **{key: {n: scores[key][n] for n in ("edge_precision", "edge_recall", "edge_f1")}
                      for key in ("query", "heldout_mean")},
                   **{f"gt_prefix_{key}": {n: control[key][n] for n in ("edge_precision", "edge_recall", "edge_f1")}
                      for key in ("query", "heldout_mean")}}
            row["geometry"] = {key: None for key in ("j2j", "j2b", "b2b")}
            if generated["success"]:
                row["geometry"] = geometry_metrics(saved["pred_joints_mesh_local"][0, 0].numpy(), generated["parents"],
                    saved["gt_joints_mesh_local"][0, 0].numpy(), batch["target_parents"][0].tolist(),
                    config["data"]["target_mesh_padding"])
            key = f"{split}_{entry['index']}"
            predictions[key] = saved
            row["artifact"] = {"path": str(directory / f"predictions_rank_{rank}.pt"), "key": key}
            rows.append(row)
            write_json(directory / f"rank_{rank}.json", {"complete": False, "step": step, "rows": rows, "summary": summarize(rows)})
            print(f"EVAL step={step} rank={rank} {split}/{entry['index']} F1={row['query']['edge_f1']:.4f}", flush=True)
            del source, target, batch, inputs, output, teacher, generated, saved
    artifact = directory / f"predictions_rank_{rank}.pt"
    torch.save(predictions, artifact)
    write_json(directory / f"rank_{rank}.json", {"complete": True, "step": step, "rows": rows, "summary": summarize(rows),
                                                "artifact": {"path": str(artifact), "sha256": file_hash(artifact)}})
    model.train()
    return rows


@torch.no_grad()
def evaluate_legacy(entries, config, root, milestone, rank):
    from eval_dynamic_rig_generation import _dynamic_generate, _parents_from_output, _topology_metrics
    reference = config["full"]["legacy_checkpoints"][str(milestone)]
    before = file_hash(reference["path"])
    if before != reference["sha256"]:
        raise ValueError("native legacy checkpoint checksum changed")
    native_config = deepcopy(config)
    native_config["baseline"]["checkpoint"] = reference["path"]
    native, tokenizer, _ = _load_original(native_config)
    rows, predictions = [], {}
    directory = root / f"legacy_sample_{milestone:05d}"
    directory.mkdir(exist_ok=True)
    for split, values in entries.items():
        for entry in values[rank::2]:
            source, target, _, selection = prepare_online_batches(entry, config, "train_a", 8, training=False)
            batch = move_batch(source, torch.device("cuda:0"))
            with torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(asset_seed(config, entry["asset_id"]))
                ids = _dynamic_generate(native, tokenizer, batch, 1400, generation_kwargs={"do_sample": False})
            prediction = None
            if tokenizer.eos in ids[2:]:
                try:
                    prediction = tokenizer.detokenize(ids)
                except (ValueError, AssertionError, IndexError):
                    pass
            gt = source["frame_target_joints"][0, 0].numpy()
            parents = source["target_parents"][0].tolist()
            row = {"split": split, "asset_id": entry["asset_id"], "index": entry["index"],
                   "integration_source": entry["integration_source"],
                   "success": prediction is not None, "query": {n: 0. for n in ("edge_precision", "edge_recall", "edge_f1")},
                   "geometry": {n: None for n in ("j2j", "j2b", "b2b")}}
            if prediction is not None:
                pred = np.asarray(prediction.joints, dtype=np.float32)
                pp = _parents_from_output(prediction)
                row["query"] = {n: float(v) for n,v in _topology_metrics(pred, pp, gt, parents).items() if n.startswith("edge_")}
                row["geometry"] = geometry_metrics(pred, pp, gt, parents, 1.)
                row.update(exact_tree=pp == parents, count_correct=len(pp) == len(parents))
            predictions[f"{split}_{entry['index']}"] = {"ids": ids,
                "prediction": None if prediction is None else {"joints": pred, "parents": pp},
                "gt_joints": gt, "gt_parents": parents, "selection": selection}
            rows.append(row)
            write_json(directory / f"rank_{rank}.json", {"complete": False, "checkpoint": reference, "rows": rows, "summary": summarize(rows)})
            print(f"LEGACY sample={milestone} rank={rank} {split}/{entry['index']} F1={row['query']['edge_f1']:.4f}", flush=True)
            del batch, source, target
    artifact = directory / f"predictions_rank_{rank}.pt"
    torch.save(predictions, artifact)
    write_json(directory / f"rank_{rank}.json", {"complete": True, "checkpoint": reference, "rows": rows, "summary": summarize(rows),
                                                "artifact": {"path": str(artifact), "sha256": file_hash(artifact)}})
    del native
    torch.cuda.empty_cache()
    return rows


def merged_evaluation(root, folder, entries):
    parts = [json.loads((root / folder / f"rank_{r}.json").read_text()) for r in range(2)]
    if not all(p["complete"] for p in parts):
        raise RuntimeError("incomplete periodic evaluation")
    rows = [row for part in parts for row in part["rows"]]
    keys = [(row["split"], row["asset_id"]) for row in rows]
    expected = {(split, entry["asset_id"]) for split, values in entries.items() for entry in values}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("periodic evaluation omitted or duplicated assets")
    result = {"complete": True, "rows": rows, "summary": summarize(rows),
              "artifacts": [p["artifact"] for p in parts],
              "summary_by_integration": {source: summarize([r for r in rows if r["integration_source"] == source])
                                          for source in sorted({r["integration_source"] for r in rows})}}
    write_json(root / folder / "summary.json", result)
    return {"path": str(root / folder / "summary.json"), "summary": result["summary"]}


def worker(config, output, rank):
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each worker must see exactly its one recorded physical GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["memory_fraction"])
    torch.manual_seed(config["data"]["model_seed"])
    random.seed(config["data"]["seed"] + rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(hours=2))
    root = Path(config["runtime"]["output_root"]) / "full"
    inventory = json.loads(Path(config["runtime"]["inventory"]["path"]).read_text())
    entries = online_entries({key: block["selected"] for key,block in inventory["splits"].items()}, config)
    evaluation = eval_entries(entries, config)
    original, _, _ = _load_original(config)
    attention = {"implementation": original.transformer.model.config._attn_implementation,
                 "class": type(original.transformer.model.decoder.layers[0].self_attn).__name__}
    if attention["implementation"] != config["full"]["backbone_attention"]:
        raise RuntimeError("source attention implementation differs from the recorded BF16 runtime")
    model = _new_model(original, {**config, "data": {**config["data"], "seed": config["data"]["model_seed"]}})
    payload = torch.load(config["full"]["warmstart"]["path"], map_location="cpu", mmap=True, weights_only=False)
    reuse = warmstart_skeleton(model, payload, config)
    del payload
    plan = config["probe"]
    ddp = DistributedDataParallel(model.skeleton, device_ids=[0], broadcast_buffers=False,
                                  gradient_as_bucket_view=True, find_unused_parameters=False)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=[plan["lr_new"], plan["lr_backbone"]], total_steps=plan["steps"],
        pct_start=.1, div_factor=5, final_div_factor=10)
    stream = AssetStream(len(entries["train"]), config["data"]["seed"])
    frozen_versions = {name: p._version for name,p in model.conditioner.named_parameters()}
    watched = {"memory.queries", "structure.embedding.weight", "structure.head.weight", "positions.xyz_head.weight",
               "structure.backbone.decoder.layers.0.self_attn.q_proj.weight"}
    snapshots = {name:p.detach().clone() for name,p in model.skeleton.named_parameters() if name in watched}
    report = {"rank": rank, "pid": os.getpid(), "complete": False, "warmstart": reuse,
              "backbone_attention": attention,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(0),
              "trace": [], "checkpoints": [], "evaluations": [], "legacy_evaluations": []}
    report_path = root / f"rank_{rank}.json"
    write_json(report_path, report)
    started = time.monotonic()
    physical = plan["batch_size"]
    source_frames_total = 0
    model.train()
    for step in range(1, plan["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device="cuda:0")
        update_started = time.monotonic()
        for micro in range(plan["accumulation"]):
            indices, frames, banks, seeds = stream.take(physical*2, config["data"]["frames_min"], config["data"]["frames_max"])
            source_frames_total += len(indices) * frames
            shard = slice(rank*physical, (rank+1)*physical)
            selected = [entries["train"][i] for i in indices[shard]]
            source, target, batch, selections = select_online_batch(original, selected, config, banks[shard], frames,
                training=True, seeds=seeds[shard])
            inputs = core_inputs(model, source, target, batch)
            if step == 1 and micro == 0:
                report["observed_first_batch"] = {"source_shape": list(source.tokens.shape),
                    "target_shape": list(target.tokens.shape), "supervision_shape": list(batch["frame_target_joints"].shape),
                    "core_input_dtype": str(inputs[0].dtype), "frozen_token_dtype": str(source.tokens.dtype)}
            context = ddp.no_sync() if micro+1 < plan["accumulation"] else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = ddp(*inputs, xyz_weight=plan["xyz_weight"])
                if step == 1 and micro == 0:
                    report["observed_first_batch"].update(
                        structure_logit_dtype=str(result["structure_logits"].dtype),
                        xyz_logit_dtype=str(result["xyz_logits"].dtype), loss_dtype=str(result["loss"].dtype))
                if not torch.isfinite(result["loss"]):
                    raise RuntimeError(f"nonfinite loss at step {step}")
                (result["loss"] / plan["accumulation"]).backward()
            totals += torch.stack([result[key].detach() for key in ("loss", "structure_loss", "xyz_loss")])
            del source, target, batch, inputs, result
        dist.all_reduce(totals)
        totals /= 2 * plan["accumulation"]
        if step == 1 or step % 64 == 0:
            report["gradient_groups"] = _finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.skeleton.parameters(), plan["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1:
            changed = {name: not torch.equal(value, dict(model.skeleton.named_parameters())[name]) for name,value in snapshots.items()}
            if len(changed) != 5 or not all(changed.values()):
                raise RuntimeError("first real training update did not change all five trainable groups")
            report["first_update"] = {"component_updates": changed, "physical_batch_per_rank": physical,
                                      "world_size": 2, "accumulation": plan["accumulation"], "global_batch": 48}
            del snapshots
        if any(p.grad is not None for p in model.conditioner.parameters()) or frozen_versions != {n:p._version for n,p in model.conditioner.named_parameters()}:
            raise RuntimeError("frozen encoder was modified")
        item = {"step": step, "new_asset_exposures": step*48, "loss": float(totals[0]),
                "structure_ce": float(totals[1]), "xyz_ce": float(totals[2]), "gradient_norm": float(norm),
                "lr": [g["lr"] for g in optimizer.param_groups], "update_seconds": time.monotonic()-update_started,
                "elapsed_seconds": time.monotonic()-started, "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_cpu_gib": peak_cpu_gib()}
        report["trace"].append(item)
        if item["peak_cpu_gib"] >= .45 * config["runtime"]["host_memory_budget_gib"]:
            raise RuntimeError("rank exceeded its host-memory safety budget")
        write_json(report_path, report)
        print("TRAIN " + json.dumps({"rank": rank, **item}), flush=True)
        if step in plan["checkpoint_steps"]:
            dist.barrier()
            if rank == 0:
                checkpoint = root / f"checkpoint_step_{step:05d}.pt"
                if checkpoint.exists():
                    raise FileExistsError(checkpoint)
                torch.save({"schema_version": 1, "skeleton": model.skeleton.state_dict(), "step": step,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "sampler": stream.state(), "configuration": config, "source": source_metadata(config),
                    "source_checkpoint": _checkpoint_identity(config), "warmstart": config["full"]["warmstart"],
                    "feature_encoder_frozen": True, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
                    "training_accounting": {"assets": step*48, "source_frame_exposures": source_frames_total,
                        "target_frame_exposures": step*48*config["data"]["target_frames_per_exposure"],
                        "heldout_frame_exposures": 0, "prior_head_asset_exposures": reuse["prior_head_asset_exposures"]}}, checkpoint)
                report["checkpoints"].append({"step": step, "path": str(checkpoint), "sha256": file_hash(checkpoint)})
                write_json(report_path, report)
            dist.barrier()
            evaluate(model, original, evaluation, config, root, step, rank)
            dist.barrier()
            if rank == 0:
                report["evaluations"].append({"step": step, "new_asset_exposures": step*48,
                    **merged_evaluation(root, f"evaluation_step_{step:05d}", evaluation)})
                write_json(report_path, report)
            milestone = config["full"]["legacy_at_step"].get(str(step))
            if milestone is not None:
                evaluate_legacy(evaluation, config, root, milestone, rank)
                dist.barrier()
                if rank == 0:
                    report["legacy_evaluations"].append({"sample_milestone": milestone,
                        **merged_evaluation(root, f"legacy_sample_{milestone:05d}", evaluation)})
                    write_json(report_path, report)
    report.update(complete=True, finished_at=timestamp())
    write_json(report_path, report)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=[0, 1])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.worker is not None:
        worker(config, args.output, args.worker)
        return
    plan = build_stage_plan(config, "full", args.config)
    root = Path(plan["output_dir"])
    root.mkdir(exist_ok=False)
    report = {"experiment": config["experiment"], "stage": "full", "complete": False, "accepted": False,
              "started_at": timestamp(), "source": source_metadata(config),
              "source_checkpoint": _checkpoint_identity(config), "manifests": verify_manifests(config),
              "configuration": config, "workers": []}
    children, logs = [], []
    try:
        require_full_inventory(config)
        if file_hash(config["baseline"]["checkpoint"]) != config["baseline"]["checkpoint_sha256"]:
            raise RuntimeError("source checkpoint changed")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        for rank, physical in enumerate(config["runtime"]["physical_gpus"]):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical), RANK=str(rank), LOCAL_RANK="0",
                               WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
            handle = (root / f"rank_{rank}.log").open("x", buffering=1)
            logs.append(handle)
            child = subprocess.Popen([sys.executable, "-B", __file__, "--config", str(args.config),
                "--stage", "full", "--output", str(args.output), "--worker", str(rank)],
                env=environment, stdout=handle, stderr=subprocess.STDOUT)
            children.append(child)
            report["workers"].append({"rank": rank, "physical_gpu": physical, "pid": child.pid,
                                       "path": str(root / f"rank_{rank}.json"), "log": str(root / f"rank_{rank}.log")})
        write_json(args.output, report)
        while any(p.poll() is None for p in children):
            failed = [p.returncode for p in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"training worker failed: {failed}")
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("training worker exited unsuccessfully")
        result = json.loads((root / "rank_0.json").read_text())
        report.update(complete=True, accepted=True, checkpoints=result["checkpoints"],
                      evaluations=result["evaluations"], legacy_evaluations=result["legacy_evaluations"])
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        for handle in logs:
            handle.close()
        report["finished_at"] = timestamp()
        write_json(args.output, report)


if __name__ == "__main__":
    main()
