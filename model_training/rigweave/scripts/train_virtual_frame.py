#!/usr/bin/env python3
"""Capacity measurements and exposure-matched virtual-frame training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_experiment_runtime import guard, load_config, source_metadata, write_json
from virtual_frame_runtime import (
    FrameBudgetStream, build_virtual_model, file_sha256, finite_gradient_groups,
    normalize_asset_gradients, set_exposure_lr,
)
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_frame_data import VirtualFrameDataset, prepare_virtual_surface, read_virtual_manifest


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_entries(config):
    return {split: read_virtual_manifest(value["path"], expected_count=value["rows"], expected_sha256=value["sha256"])
            for split, value in config["data"]["manifests"].items()}


def dataset_for(entries, config, *, mode="train", frames=None):
    data = config["data"]
    return VirtualFrameDataset(
        entries, frame_seed=data["frame_seed"], sampling_seed=data["seed"],
        frames=frames or config["evaluation"]["frames"], heldout_frames=data["heldout_frames"], mode=mode,
        motion_fps_ratio=data["motion_fps_ratio"], minimum_random_frames=data["minimum_random_frames"],
        motion_vertex_samples=data["motion_vertex_samples"],
    )


def prepare(dataset, requests, config, device):
    batches = [dataset[request] for request in requests]
    return prepare_virtual_surface(batches, device, **config["surface_sampling"])


def optimizer_for(model, config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config["optimization"]["peak_lr"] / config["optimization"]["div_factor"],
                            betas=(.95, .999), weight_decay=config["optimization"]["weight_decay"])


def _stress_indices(entries):
    def measure(row, key):
        return int(row.get("rootless_target_strict_metrics", {}).get(key,
                   row.get("canonical_metrics", {}).get("target_joint_count" if key == "target_joint_count" else "filtered_vertex_count", 0)))
    largest_joints = sorted(range(len(entries)), key=lambda i: measure(entries[i], "target_joint_count"), reverse=True)
    largest_meshes = sorted(range(len(entries)), key=lambda i: measure(entries[i], "vertex_count"), reverse=True)
    return list(dict.fromkeys(largest_joints[:3] + largest_meshes[:1]))


def profile_case(model, optimizer, dataset, config, device, frames, batch_size, indices, xyz_weight):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    requests = [FrameSampleRequest(indices[i % len(indices)], frames, 77000 + i) for i in range(batch_size)]
    surface, targets = prepare(dataset, requests, config, device)
    losses = []
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(surface, targets, xyz_weight=xyz_weight)
        if not bool(torch.isfinite(result["loss"])):
            raise RuntimeError("nonfinite loss during capacity measurement")
        losses.append(float(result["loss"].detach()))
        (result["loss"] / 2).backward()
        del result
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimization"]["clip_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    return {"frames": frames, "batch_per_rank": batch_size, "real_frames_per_rank": frames * batch_size,
            "max_joints": int(targets["joint_count"].max()), "selected_assets": targets["asset_id"],
            "input_maxabs": float(surface["dense_points"].abs().max()),
            "target_maxabs": float(targets["frame_target_joints"].abs().max()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "seconds": time.monotonic() - started, "losses": losses, "gradient_norm": float(norm), "finite": True}


def run_profile(model, initialization, entries, config, args, device, rank, world):
    plan = config[args.plan]
    dataset = dataset_for(entries["train"], config)
    optimizer = optimizer_for(model, config)
    wrapped = DistributedDataParallel(
        model, device_ids=[device.index], find_unused_parameters=True,
        gradient_as_bucket_view=config["runtime"]["ddp_gradient_as_bucket_view"],
    ) if world > 1 else model
    indices = _stress_indices(entries["train"])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "started_at": now(), "rank": rank, "world_size": world,
              "initialization": initialization, "source": source_metadata(config), "plan": plan, "cases": [],
              "capacity_only_no_weights_reused": True, "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30}
    path = root / f"rank_{rank}.json"
    write_json(path, report)
    # Materialize Adam states before recording steady-state capacity.
    model.train()
    report["warmup"] = profile_case(wrapped, optimizer, dataset, config, device, 2, 1, indices[:1], 1.)
    report["gradient_groups"] = finite_gradient_groups(model)
    write_json(path, report)
    for frames, batch in plan["cases"]:
        try:
            case = profile_case(wrapped, optimizer, dataset, config, device, frames, batch, indices, 1.)
            case["approved"] = case["peak_allocated_gib"] <= plan["allocated_fraction_limit"] * report["gpu_total_gib"]
            report["cases"].append(case)
            print("CAPACITY " + json.dumps({"rank": rank, **case}), flush=True)
        except torch.OutOfMemoryError as exc:
            report["cases"].append({"frames": frames, "batch_per_rank": batch, "approved": False,
                                     "oom": True, "error": str(exc)[:800],
                                     "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                                     "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                                     "traceback": traceback.format_exc()})
            write_json(path, report)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if world > 1:
                raise
            break
        write_json(path, report)
        if not case["approved"] and plan.get("stop_at_limit", True):
            break
    report.update(complete=True, finished_at=now())
    write_json(path, report)


def run_balance(model, initialization, entries, config, args, device):
    plan = config[args.plan]
    dataset = dataset_for(entries["train"], config)
    report = {"complete": False, "initialization": initialization, "batches": [], "source": source_metadata(config)}
    rng = np.random.default_rng(config["data"]["seed"] + 982)
    model.train()
    parameter_names = ("motion_encoder.norm.weight", "structure.embedding.weight")
    parameters = [dict(model.named_parameters())[name] for name in parameter_names]
    for number, frames in enumerate(plan["frames"]):
        batch = min(plan["batch_cap"], plan["frame_budget"] // frames)
        indices = rng.choice(len(dataset), batch, replace=False).tolist()
        requests = [FrameSampleRequest(i, frames, 91000 + number * 100 + j) for j, i in enumerate(indices)]
        surface, targets = prepare(dataset, requests, config, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(surface, targets, xyz_weight=1.)
        structure = torch.autograd.grad(result["structure_loss"], parameters, retain_graph=True)
        xyz = torch.autograd.grad(result["xyz_loss"], parameters)
        values = {}
        for name, left, right in zip(parameter_names, structure, xyz):
            left_norm, right_norm = float(left.float().norm()), float(right.float().norm())
            values[name] = {"structure_gradient_norm": left_norm, "xyz_gradient_norm": right_norm,
                            "ratio": left_norm / max(right_norm, 1e-12)}
        report["batches"].append({"frames": frames, "batch_size": batch, "gradients": values,
                                   "structure_loss": float(result["structure_loss"]), "xyz_loss": float(result["xyz_loss"])})
        del result, surface, targets, structure, xyz
        write_json(Path(args.output) / "balance.json", report)
    report["median_ratios"] = {name: float(np.median([x["gradients"][name]["ratio"] for x in report["batches"]]))
                               for name in parameter_names}
    report["complete"] = True
    write_json(Path(args.output) / "balance.json", report)


def geometry_scores(prediction, parents, target, target_parents, bbox_side):
    from scipy.spatial import cKDTree
    from eval_dynamic_rig_generation import _bones_from_parents, _sample_bones_np
    pb, tb = _bones_from_parents(prediction, parents), _bones_from_parents(target, target_parents)
    if not len(pb) or not len(tb):
        return {"j2j": None, "j2b": None, "b2b": None}
    p, t = _sample_bones_np(pb), _sample_bones_np(tb)
    def distance(a, b):
        return float(cKDTree(b).query(a)[0].mean())
    return {"j2j": .5 * (distance(prediction, target) + distance(target, prediction)) / bbox_side,
            "j2b": .5 * (distance(prediction, t) + distance(target, p)) / bbox_side,
            "b2b": .5 * (distance(p, t) + distance(t, p)) / bbox_side}


def evaluation_rows(entries, config):
    selected = {}
    rng = np.random.default_rng(config["evaluation"]["selection_seed"])
    for split in ("train", "valid"):
        count = config["evaluation"][f"{split}_assets"]
        order = rng.permutation(len(entries[split]))[:count]
        selected[split] = [{**entries[split][i], "manifest_index": int(i)} for i in order]
    return selected


@torch.no_grad()
def evaluate(model, entries, config, root, samples, device, rank, world):
    from eval_dynamic_rig_generation import _topology_metrics
    model.eval()
    directory = root / f"evaluation_sample_{samples:06d}"
    directory.mkdir(exist_ok=True)
    rows, predictions = [], {}
    for split, values in evaluation_rows(entries, config).items():
        dataset = dataset_for(values, config, mode="seen_eval")
        for index in range(rank, len(values), world):
            surface, targets = prepare(dataset, [index], config, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                anchors, memory = model.encode(surface)
                teacher = model.forward_encoded(anchors, memory, surface["query_points"], targets,
                                                xyz_weight=config["optimization"]["xyz_weight"])
                generated = model.generate_encoded(anchors, memory, surface["query_points"])
            count = int(targets["joint_count"][0])
            target = targets["frame_target_joints"][0, :, :count].cpu().numpy()
            parents = targets["target_parents"][0, :count].tolist()
            posed_teacher = teacher["joints"][0, :, :count].float().cpu().numpy()
            control = _topology_metrics(posed_teacher[0], parents, target[0], parents)
            result = {"split": split, "asset_id": values[index]["asset_id"], "manifest_index": values[index]["manifest_index"],
                      "success": generated["success"], "status": generated["status"], "gt_count": count,
                      "pred_count": len(generated["parents"]) if generated["success"] else None,
                      "structure_ce": float(teacher["structure_loss"]), "xyz_l1": float(teacher["xyz_loss"]),
                      "gt_prefix_query_f1": control["edge_f1"],
                      "edge_f1": 0., "edge_precision": 0., "edge_recall": 0.,
                      "exact_count": False, "exact_tree": False, "geometry": {"j2j": None, "j2b": None, "b2b": None}}
            posed = None
            if generated["success"]:
                posed = generated["joints"][0].float().cpu().numpy()
                scores = _topology_metrics(posed[0], generated["parents"], target[0], parents)
                result.update({key: scores[key] for key in ("edge_f1", "edge_precision", "edge_recall")})
                result["exact_count"] = len(generated["parents"]) == count
                result["exact_tree"] = generated["parents"] == parents
                result["geometry"] = geometry_scores(posed[0], generated["parents"], target[0], parents,
                                                     float(targets["frame_mesh_bbox_side"][0, 0]))
                result["frame_mean_f1"] = float(np.mean([
                    _topology_metrics(pose, generated["parents"], gt, parents)["edge_f1"]
                    for pose, gt in zip(posed, target)]))
            else:
                result["frame_mean_f1"] = 0.
            key = f"{split}_{values[index]['manifest_index']}"
            predictions[key] = {"row": result, "target_joints_common": target, "target_parents": parents,
                                "prediction_joints_common": posed, "teacher_joints_common": posed_teacher,
                                "generated_tokens": generated["tokens"],
                                "prediction_parents": generated.get("parents"),
                                "skeleton_sequence": generated["skeleton_sequence"].cpu_record() if generated["success"] else None,
                                "query_center": targets["query_center"].cpu(), "query_scale": targets["query_scale"].cpu(),
                                "selected_frames": targets["selected_frames"].cpu(), "path": targets["path"]}
            rows.append(result)
            write_json(directory / f"rank_{rank}.json", {"complete": False, "samples": samples, "rows": rows})
            print("EVAL " + json.dumps({"rank": rank, "samples": samples, **result}), flush=True)
            del surface, targets, anchors, memory, teacher, generated
    torch.save(predictions, directory / f"predictions_rank_{rank}.pt")
    write_json(directory / f"rank_{rank}.json", {"complete": True, "samples": samples, "rows": rows})
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_rows = [row for r in range(world) for row in json.loads((directory / f"rank_{r}.json").read_text())["rows"]]
        summary = {}
        for split in ("train", "valid"):
            group = [row for row in all_rows if row["split"] == split]
            stats = {"assets": len(group), "natural_eos": sum(row["success"] for row in group)}
            for key in ("edge_f1", "edge_precision", "edge_recall", "gt_prefix_query_f1", "frame_mean_f1",
                        "structure_ce", "xyz_l1", "exact_count", "exact_tree"):
                stats[key] = float(np.mean([row[key] for row in group]))
            stats["geometry_successful_assets"] = sum(row["geometry"]["j2j"] is not None for row in group)
            stats["geometry"] = {key: (float(np.mean(values)) if values else None)
                                  for key in ("j2j", "j2b", "b2b")
                                  for values in [[row["geometry"][key] for row in group if row["geometry"][key] is not None]]}
            summary[split] = stats
        write_json(directory / "summary.json", {"complete": True, "samples": samples, "summary": summary,
                                                "rows": all_rows, "failed_generations_score_zero": True,
                                                "geometry_units": "query_mesh_bbox_side"})
    if world > 1:
        dist.barrier()
    model.train()


def save_checkpoint(model, optimizer, stream, config, root, samples, frames, step, initialization, source, rank, world):
    local = {"stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        payload = {"schema": "virtual_frame_rig_v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "samples": samples, "input_frames": frames, "supervised_frames": frames, "step": step,
                   "rank_states": states, "config": config, "architecture": model.configuration(),
                   "initialization": initialization, "source": source, "precision": "bfloat16_autocast"}
        target = root / f"checkpoint_sample_{samples:06d}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        write_json(root / "latest_checkpoint.json", {"path": str(target), "samples": samples, "step": step, "bytes": target.stat().st_size})
    if world > 1:
        dist.barrier()


def run_training(model, initialization, entries, config, args, device, rank, world):
    plan = config[args.plan]
    if not plan.get("approved", False):
        raise RuntimeError("training plan has not been approved after capacity and configuration review")
    if plan["target_samples"] % world or plan["target_samples"] > config["optimization"]["schedule_samples"]:
        raise ValueError("exposure endpoint must divide evenly across ranks and fit the learning-rate schedule")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source = source_metadata(config)
    dataset = dataset_for(entries["train"], config)
    optimizer = optimizer_for(model, config)
    resume = None
    samples, total_frames, step = 0, 0, 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("schema") != "virtual_frame_rig_v1" or resume["architecture"] != model.configuration():
            raise ValueError("resume checkpoint is not this virtual-frame architecture")
        if resume["config"]["data"] != config["data"] or resume["config"]["optimization"] != config["optimization"]:
            raise ValueError("resume data or optimizer settings changed")
        if resume["initialization"] != initialization or len(resume["rank_states"]) != world:
            raise ValueError("resume initialization or world size differs")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        samples, total_frames, step = resume["samples"], resume["input_frames"], resume["step"]
    stream = FrameBudgetStream(
        dataset, rank=rank, world_size=world, seed=config["data"]["seed"],
        frame_budget=plan["frame_budget"], batch_cap=plan["batch_cap"],
        min_frames=config["data"]["frames_min"], max_frames=config["data"]["frames_max"],
        state=resume["rank_states"][rank]["stream"] if resume else None,
    )
    if resume:
        states = resume["rank_states"][rank]
        torch.set_rng_state(states["torch_rng"])
        torch.cuda.set_rng_state(states["cuda_rng"])
        np.random.set_state(states["numpy_rng"])
        random.setstate(states["python_rng"])
        del resume
    wrapped = DistributedDataParallel(
        model, device_ids=[device.index], find_unused_parameters=True,
        gradient_as_bucket_view=config["runtime"]["ddp_gradient_as_bucket_view"],
    ) if world > 1 else model
    started = time.monotonic()
    report = {"complete": False, "started_at": now(), "rank": rank, "world_size": world, "plan": plan,
              "candidate": args.candidate, "initialization": initialization, "source": source,
              "samples": samples, "input_frames": total_frames, "step": step, "evaluations": []}
    write_json(root / f"rank_{rank}.json", report)
    milestones = sorted(set(plan["evaluation_samples"] + [plan["target_samples"]]))
    model.train()
    with (root / f"trace_rank_{rank}.jsonl").open("a", buffering=1) as trace:
        while samples < plan["target_samples"]:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            local_assets = 0
            schedule = []
            next_boundary = min(x for x in milestones if x > samples)
            lr, beta1 = set_exposure_lr(optimizer, samples, config)
            for micro in range(plan["accumulation"]):
                available = (next_boundary - samples) // world - local_assets
                requests = stream.take(available)
                surface, targets = prepare(dataset, requests, config, device)
                size, frames = len(requests), requests[0].frames
                last = micro + 1 == plan["accumulation"] or size == available
                context = wrapped.no_sync() if world > 1 and not last else nullcontext()
                with context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = wrapped(surface, targets, xyz_weight=config["optimization"]["xyz_weight"])
                    if not bool(torch.isfinite(output["loss"])):
                        raise RuntimeError(f"nonfinite loss at update {step + 1}")
                    output["loss_per_asset"].sum().backward()
                totals += torch.stack((output["loss"].detach() * size, output["structure_loss"].detach() * size,
                                       output["xyz_loss"].detach() * size, totals.new_tensor(size), totals.new_tensor(size * frames)))
                local_assets += size
                schedule.append({"assets_per_rank": size, "frames_per_asset": frames})
                del output, surface, targets
                if last:
                    break
            if world > 1:
                dist.all_reduce(totals)
            update_assets, update_frames = int(totals[3].item()), int(totals[4].item())
            normalize_asset_gradients(model.parameters(), global_assets=update_assets, world_size=world)
            if step == 0:
                report["gradient_groups"] = finite_gradient_groups(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimization"]["clip_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            samples += update_assets
            total_frames += update_frames
            step += 1
            item = {"step": step, "samples": samples, "input_frames": total_frames, "supervised_frames": total_frames,
                    "assets_in_update": update_assets, "microbatches": schedule,
                    "loss": float(totals[0] / update_assets), "structure_ce": float(totals[1] / update_assets),
                    "xyz_l1": float(totals[2] / update_assets), "gradient_norm": float(norm), "lr": lr, "beta1": beta1,
                    "update_seconds": time.monotonic() - tick, "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
            trace.write(json.dumps(item) + "\n")
            report.update(samples=samples, input_frames=total_frames, supervised_frames=total_frames, step=step, latest=item)
            write_json(root / f"rank_{rank}.json", report)
            print("TRAIN " + json.dumps({"rank": rank, **item}), flush=True)
            if samples in milestones:
                save_checkpoint(model, optimizer, stream, config, root, samples, total_frames, step, initialization, source, rank, world)
                evaluate(model, entries, config, root, samples, device, rank, world)
                report["evaluations"].append(samples)
    report.update(complete=True, finished_at=now(), elapsed_seconds=time.monotonic() - started)
    write_json(root / f"rank_{rank}.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "balance", "train"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("this entrypoint requires CUDA; CPU tests exercise the model separately")
    if rank == 0:
        guard(config, "train" if args.mode == "train" else "preflight")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        entries = load_entries(config)
        model, initialization = build_virtual_model(config, args.candidate, device)
        if args.mode == "profile":
            run_profile(model, initialization, entries, config, args, device, rank, world)
        elif args.mode == "balance":
            if world != 1:
                raise ValueError("loss-balance measurement uses one non-DDP model")
            run_balance(model, initialization, entries, config, args, device)
        else:
            run_training(model, initialization, entries, config, args, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
