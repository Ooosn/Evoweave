"""Bounded, single-GPU memory calibration on an expendable in-memory model copy."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from preflight_motion_evidence import difference
from train_dynamic_rig import build_tokenizer, move_batch
from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate


def validate_plan(raw):
    plan = dict(raw)
    for name, lower, upper in (("validation_rows", 1, 32), ("frame_budget", 24, None),
                               ("max_batch_cap", 1, None)):
        value = plan[name]
        if type(value) is not int or value < lower or (upper is not None and value > upper):
            raise ValueError(f"invalid {name}: {value!r}")
    memory, acceptance = plan["memory_fraction"], plan["acceptance_fraction"]
    if (type(memory) not in (int, float) or type(acceptance) not in (int, float)
            or not 0 < acceptance <= 0.90 or not acceptance < memory <= 0.93):
        raise ValueError("require 0 < acceptance_fraction <= 0.90 and acceptance < memory_fraction <= 0.93")
    if not isinstance(plan["checkpoint"], str) or not plan["checkpoint"].strip():
        raise ValueError("checkpoint must be an explicit path")
    if Path(plan["checkpoint"]).name != "checkpoint_sample_80000.pt":
        raise ValueError("calibration requires the final checkpoint_sample_80000.pt reference")
    cases = plan["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list of [frames, batch] pairs")
    seen, previous_batch = set(), 0
    for case in cases:
        if not isinstance(case, (list, tuple)) or len(case) != 2 or any(type(v) is not int for v in case):
            raise ValueError(f"invalid case: {case!r}")
        frames, batch = case
        if not 2 <= frames <= 24 or not 1 <= batch <= plan["max_batch_cap"]:
            raise ValueError(f"case exceeds frame or batch bounds: {case}")
        if frames * batch > plan["frame_budget"]:
            raise ValueError(f"case exceeds the frame budget: {case}")
        if batch < previous_batch or (frames, batch) in seen:
            raise ValueError("cases must be unique and ordered by nondecreasing batch size")
        seen.add((frames, batch))
        previous_batch = batch
    plan["cases"] = [list(case) for case in cases]
    return plan


def validate_output(output, inputs):
    if output.suffix.lower() != ".json":
        raise ValueError("profiling output must be a JSON report, never a checkpoint")
    protected = {Path(path).resolve() for path in inputs}
    for path in (output, output.with_suffix(output.suffix + ".tmp")):
        if path.resolve() in protected:
            raise ValueError(f"report would overwrite an input: {path}")


def checkpoint_recipe(payload, config):
    saved = dict(payload["args"])
    if payload.get("step") != 1667 or payload.get("sample_seen") != 80016:
        raise ValueError("expected the completed step-1667 / nominal-sample-80016 checkpoint")
    required = {"motion_evidence_fusion": "bias", "motion_evidence_heads": 8,
                "motion_depth": 12, "frames": 24, "motion_checkpointing": True,
                "use_time_embedding": False, "use_motion_features": False,
                "freeze_ar": False, "freeze_conditioner": False, "train_surface_tokenizer": True,
                "target_active_skin_only": False, "input_space_policy": "mesh_query_bbox"}
    settings = {**CHECKPOINT_DEFAULTS, **saved}
    for name, expected in required.items():
        if settings.get(name) != expected:
            raise ValueError(f"checkpoint recipe mismatch: {name}={settings.get(name)!r}, expected {expected!r}")
    for name in ("train_manifest", "val_manifest"):
        if Path(saved[name]).resolve() != Path(config["baseline"]["args"][name]).resolve():
            raise ValueError(f"checkpoint {name} is not the verified historical manifest")
    groups = payload["optimizer"]["param_groups"]
    if [len(group["params"]) for group in groups] != [304, 388, 196]:
        raise ValueError("checkpoint optimizer must contain 304/388/196 parameter tensors")
    if len(payload["optimizer"]["state"]) != 888:
        raise ValueError("final checkpoint does not contain all 888 original Adam states")
    recipe = []
    for name, group in zip(("motion", "ar", "surface"), groups):
        lr, decay = saved[f"lr_{name}"], saved["weight_decay"]
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(decay) or decay < 0:
            raise ValueError("invalid saved AdamW learning rate or weight decay")
        if group.get("name") != name or group["weight_decay"] != decay:
            raise ValueError(f"checkpoint optimizer group disagrees with the saved recipe: {name}")
        recipe.append({"name": name, "lr": lr, "weight_decay": decay,
                       "checkpoint_final_lr_not_used": group["lr"]})
    return settings, recipe


def sample_summary(sample, index):
    if (tuple(sample.frame_vertices.shape[:1]) != (24,) or sample.frame_vertices.ndim != 3
            or sample.frame_vertices.shape[-1] != 3 or sample.frame_vertices.shape[1] < 1):
        raise ValueError("stress selection requires actual [24, vertices, 3] samples")
    if sample.frame_vertices.device.type != "cpu" or sample.input_ids.device.type != "cpu":
        raise ValueError("stress selection must remain on CPU")
    if (sample.input_ids.ndim != 1 or sample.input_ids.numel() == 0
            or sample.attention_mask.shape != sample.input_ids.shape
            or not bool((sample.attention_mask == 1).all())):
        raise ValueError("stress selection requires the complete unpadded input_ids sequence")
    if sample.target_joints.shape != (sample.joint_count, 3) or sample.target_parents.shape != (sample.joint_count,):
        raise ValueError("sample target dimensions do not contain the complete declared GT")
    return {"index": index, "path": sample.path, "input_tokens": int(sample.input_ids.numel()),
            "vertices": int(sample.frame_vertices.shape[1]), "faces": int(sample.faces.shape[0]),
            "joint_count": sample.joint_count, "source_joint_count": sample.source_joint_count,
            "selected_frames": sample.selected_frames.tolist(), "query_center": sample.query_center.tolist(),
            "query_scale": float(sample.query_scale),
            "input_ids_sha256": hashlib.sha256(sample.input_ids.numpy().tobytes()).hexdigest()}


def select_stress_samples(dataset, limit):
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("stress selection is limited to 1..32 actual validation rows")
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    selected, descriptors, inspected = {}, {}, []
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        summary = sample_summary(sample, index)
        inspected.append(summary)
        for role, metric in (("longest_input", "input_tokens"), ("largest_mesh", "vertices")):
            if role not in descriptors or summary[metric] > descriptors[role][metric]:
                selected[role], descriptors[role] = sample, summary
        del sample
    return [selected[role] for role in ("longest_input", "largest_mesh")], {
        "inspected_rows": len(inspected), "inspected": inspected, "selected": descriptors,
        "unique_retained_samples": len({row["index"] for row in descriptors.values()}),
        "policy": "First fixed validation rows at T24; first index wins ties. Retain at most two samples, not the scanned pool.",
    }


def slice_sample(sample, frames):
    if type(frames) is not int or not 2 <= frames <= sample.frame_vertices.shape[0]:
        raise ValueError("frame slice must preserve the query and at least one actual evidence pose")
    return replace(sample, frame_vertices=sample.frame_vertices[:frames],
                   vertex_normals=sample.vertex_normals[:frames], face_normals=sample.face_normals[:frames],
                   selected_frames=sample.selected_frames[:frames])


def stress_batch(samples, frames, batch_size, pass_index, collate):
    # Alternating roles makes even B1 cover both stress samples across the two passes.
    return collate([slice_sample(samples[(index + pass_index) % 2], frames) for index in range(batch_size)])


def classify_candidate(row, total_memory, acceptance_fraction):
    peak = row.get("peak_allocated_bytes")
    finite = all(row.get(name) is True for name in ("finite_loss", "finite_gradients", "finite_updates"))
    approved = (row.get("status") == "measured" and row.get("backward_passes", 0) >= 2
                and row.get("optimizer_steps") == 1
                and finite and type(peak) is int and 0 <= peak <= total_memory * acceptance_fraction)
    decision = "approved" if approved else "invalid_measurement"
    if row.get("status") == "oom":
        decision = "unsafe_oom"
    elif type(peak) is int and peak > total_memory * acceptance_fraction:
        decision = "unsafe_memory"
    return {**row, "approved": bool(approved), "decision": decision,
            "peak_allocated_fraction": peak / total_memory if type(peak) is int else None,
            "acceptance_limit_bytes": int(total_memory * acceptance_fraction)}


def summarize_cases(plan, rows):
    approved = [row for row in rows if row.get("approved") is True]
    candidates = sorted({row["batch"] for row in approved}, reverse=True)
    configured_frames = {frames for frames, _ in plan["cases"]}

    def supported_cap(frames):
        for cap in candidates:
            if all(any(row["frames"] == count and row["batch"] == min(cap, plan["frame_budget"] // count)
                       for row in approved) for count in frames):
                return cap
        return None

    return {
        "largest_approved_measured_batch": max(candidates, default=None),
        "measured_global_cap": supported_cap(configured_frames),
        "measured_global_cap_scope": "Exact measured B=min(cap, floor(frame_budget/T)) pairs at configured T only; these stress samples only.",
        "full_t2_t24_global_cap": supported_cap(set(range(2, 25))),
        "unmeasured_frames": sorted(set(range(2, 25)) - {row["frames"] for row in rows if row["status"] == "measured"}),
        "all_candidates_approved": len(rows) == len(plan["cases"]) and all(row["approved"] for row in rows),
        "unattempted_cases": plan["cases"][len(rows):],
    }


def trainable_groups(model, recipe):
    encoder = model.conditioner.motion_encoder
    if encoder.use_time_embedding or encoder.use_motion_features or encoder.time_embed.requires_grad:
        raise RuntimeError("disabled time/motion-feature branches must stay disabled and frozen")
    if any(parameter.requires_grad for parameter in encoder.motion_feature_mlp.parameters()):
        raise RuntimeError("disabled motion-feature MLP was unfrozen")
    encoder.gradient_checkpointing = True
    modules = (encoder, model.transformer, model.conditioner.surface_tokenizer)
    groups = [{"params": [parameter for parameter in module.parameters() if parameter.requires_grad],
               "name": entry["name"], "lr": entry["lr"], "weight_decay": entry["weight_decay"]}
              for module, entry in zip(modules, recipe)]
    if [len(group["params"]) for group in groups] != [304, 388, 196]:
        raise RuntimeError("live trainable groups must contain 304/388/196 parameter tensors")
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(set(ids)) != 888 or set(ids) != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise RuntimeError("optimizer groups do not cover every original trainable parameter exactly once")
    model.train()
    return groups


def finite_tensors(tensors):
    # Scalar norms avoid allocating full-size boolean masks for the large AR matrices.
    norms = torch.stack([tensor.detach().float().norm() for tensor in tensors])
    return bool(torch.isfinite(norms).all())


def memory_snapshot():
    memory = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    return {**memory, **{name.replace("_bytes", "_gib"): value / 2**30 for name, value in memory.items()}}


def measure_step(model, optimizer, samples, collate, frames, batch_size, *, warmup=False):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    row = {"frames": frames, "batch": batch_size, "status": "measured", "backward_passes": 0,
           "optimizer_steps": 0, "losses": [], "before_pass": [], "baseline_memory": memory_snapshot()}
    for pass_index in range(2):
        gradients_present = sum(parameter.grad is not None for parameter in parameters)
        if gradients_present != (888 if pass_index else 0):
            raise RuntimeError("second forward must see all 888 resident accumulated gradients")
        if not warmup and len(optimizer.state) != 888:
            raise RuntimeError("candidate is missing resident Adam states from the warmup")
        row["before_pass"].append({"gradients_present": gradients_present,
                                   "adam_states_present": len(optimizer.state), **memory_snapshot()})
        batch = move_batch(stress_batch(samples, frames, batch_size, pass_index, collate), torch.device("cuda:0"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = out["loss"]
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss at T{frames}/B{batch_size}, pass {pass_index}")
        row["losses"].append(value)
        (loss / 2).backward()
        row["backward_passes"] += 1
        del out, loss, batch
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("not all original trainable routes received gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    row.update(finite_loss=True, finite_gradients=True, gradient_norm=float(norm))
    probes = {group["name"]: torch.cat([p.detach().reshape(-1)[:16] for p in group["params"]]).clone()
              for group in optimizer.param_groups}
    optimizer.step()
    row["optimizer_steps"] = 1
    row["group_updates"] = {}
    for group in optimizer.param_groups:
        current = torch.cat([p.detach().reshape(-1)[:16] for p in group["params"]])
        update = difference(probes[group["name"]], current)
        if not all(math.isfinite(value) for value in update.values()) or update["max_abs"] <= 0:
            raise FloatingPointError(f"missing or non-finite probed parameter update: {group['name']}")
        row["group_updates"][group["name"]] = {"parameter_tensors": len(group["params"]), **update}
    if len(optimizer.state) != 888 or any("exp_avg" not in state or "exp_avg_sq" not in state for state in optimizer.state.values()):
        raise RuntimeError("Adam did not initialize all 888 original trainable parameter states")
    moments = [state[name] for state in optimizer.state.values() for name in ("exp_avg", "exp_avg_sq")]
    if not finite_tensors(parameters) or not finite_tensors(moments):
        raise FloatingPointError("non-finite updated parameter or Adam moment")
    row["finite_updates"] = True
    row["update_check"] = "All updated parameter/moment norms finite; sampled parameter deltas finite and nonzero in each group."
    torch.cuda.synchronize()
    row.update(memory_snapshot(), seconds=time.perf_counter() - start)
    return row


def run_candidates(plan, measure, cleanup, total_memory, record):
    rows = []
    stop_reason = "all_candidates_measured"
    for frames, batch in plan["cases"]:
        start = time.perf_counter()
        try:
            row = measure(frames, batch)
        except torch.OutOfMemoryError as exc:
            row = {"frames": frames, "batch": batch, "status": "oom", "error": str(exc),
                   "seconds": time.perf_counter() - start, **memory_snapshot()}
        # Leave the exception block before cleanup so traceback-held activations can be freed.
        cleanup()
        row = classify_candidate(row, total_memory, plan["acceptance_fraction"])
        rows.append(row)
        record(rows)
        if row["status"] == "oom":
            stop_reason = "first_oom_no_retry"
            break
    return rows, stop_reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = validate_plan(config["frame_budget_profile"])
    checkpoint = Path(plan["checkpoint"])
    validate_output(args.output, [args.config, checkpoint, *(row["path"] for row in config["baseline"]["manifests"])])
    if Path(config["runtime"]["repo"]).resolve() != Path(__file__).resolve().parents[3]:
        raise RuntimeError("the profiler must execute from the recorded source checkout")
    guard(config, "preflight")
    guard(config, "train")
    source, manifests = source_metadata(config), verify_manifests(config)
    checkpoint_before = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    settings, recipe = checkpoint_recipe(payload, config)
    del payload
    gc.collect()
    settings["checkpoint"] = checkpoint
    model_args = SimpleNamespace(**settings)
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    dataset = DynamicRigManifestDataset(
        model_args.val_manifest, tokenizer, frame_count=24, limit=plan["validation_rows"], random_query=False,
        seed=model_args.seed + 17, motion_fps_ratio=model_args.motion_fps_ratio,
        minimum_random_frames=model_args.minimum_random_frames, motion_vertex_samples=model_args.motion_vertex_samples,
        target_active_skin_only=model_args.target_active_skin_only, active_skin_threshold=model_args.active_skin_threshold,
        target_start_policy=model_args.target_start_policy, target_root_policy=model_args.target_root_policy,
        input_space_policy=model_args.input_space_policy,
    )
    samples, selection = select_stress_samples(dataset, plan["validation_rows"])
    del dataset
    report = {"complete": False, "source": source, "manifests": manifests, "plan": plan,
              "checkpoint": str(checkpoint), "checkpoint_step": 1667, "checkpoint_nominal_sample_seen": 80016,
              "checkpoint_size_bytes": checkpoint_before.st_size, "selection": selection,
              "optimizer_recipe": recipe, "cases": [], "torch": torch.__version__,
              "scope": "Single-GPU calibration only. In-memory parameter updates are discarded; no checkpoint writes or training continuation.",
              "limits": ["At most 32 fixed validation rows, with at most two retained stress samples; not dataset-wide safety.",
                         "Repeated real assets stress padding dimensions, not independent training samples; no GT, token, vertex, or face pruning.",
                         "Other T values slice the normalized T24 poses, retaining the query and complete GT; this is not a fresh per-T FPS selection.",
                         "Two accumulated backward passes per candidate, not a full training run. No DDP communication/buckets are measured.",
                         "Uses saved nominal LR/weight-decay recipe and the trainer's AdamW defaults, not final scheduler LR or saved moments.",
                         "Cases mutate the same disposable model sequentially. Norm/update auditing is included in measured peaks and wall time."]}
    write_json(args.output, report)
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("frame-budget profiling requires exactly one allocated visible GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan["memory_fraction"], device=0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the profiling recipe requires CUDA BF16 support")
    torch.manual_seed(model_args.seed)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report.update(gpu=properties.name, total_memory_bytes=properties.total_memory, amp="bfloat16", motion_checkpointing=True)
    model = _build_dynamic_model(model_args, tokenizer, device)
    for name, value in settings.items():
        if name.endswith("_weight") and hasattr(model, name) and getattr(model, name) != value:
            raise RuntimeError(f"model builder changed the saved loss recipe: {name}")
    groups = trainable_groups(model, recipe)
    optimizer = torch.optim.AdamW(groups, weight_decay=model_args.weight_decay)
    report["optimizer_defaults"] = optimizer.defaults
    collate = partial(dynamic_rig_collate, pad_token=tokenizer.pad)

    def cleanup():
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    report["warmup"] = measure_step(model, optimizer, samples, collate, 2, 1, warmup=True)
    cleanup()
    write_json(args.output, report)

    def record(rows):
        report["cases"] = list(rows)
        report["summary"] = summarize_cases(plan, rows)
        write_json(args.output, report)
        print(json.dumps(rows[-1]), flush=True)

    rows, stop_reason = run_candidates(
        plan, lambda frames, batch: measure_step(model, optimizer, samples, collate, frames, batch),
        cleanup, properties.total_memory, record,
    )
    checkpoint_after = checkpoint.stat()
    if (checkpoint_after.st_size, checkpoint_after.st_mtime_ns) != (checkpoint_before.st_size, checkpoint_before.st_mtime_ns):
        raise RuntimeError("reference checkpoint metadata changed during profiling")
    report.update(complete=True, stop_reason=stop_reason, cases=rows, summary=summarize_cases(plan, rows),
                  checkpoint_file_unchanged=True,
                  completion_meaning="The bounded calibration protocol ended; complete=true does not approve unsafe or unmeasured cases.")
    write_json(args.output, report)
    print("COMPLETE", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
