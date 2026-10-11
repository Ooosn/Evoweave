"""Evaluate a preserved skeleton checkpoint and measure real physical batches."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from probe_shared_structure import _finite_gradients, _load_original, _new_model, _optimizer
from probe_shared_structure_medium import (
    check_memory_budget, evaluate, evaluation_views, online_entries, peak_cpu_gib,
    selected_rows, select_online_input, summarize_rows,
)
from relation_probe_metrics import file_hash
from run_shared_structure_experiment import _checkpoint_identity, build_stage_plan, require_online_preflight
from shared_structure_online_batch import select_online_batch


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def worker_entries(entries, config, rank):
    if rank not in (0, 1):
        raise ValueError("assessment has exactly two workers")
    count, detailed = config["probe"]["evaluate_train_assets"], config["probe"]["detailed_eval_assets"]
    if count % 2 or detailed % 2:
        raise ValueError("assessment diagnostic subsets must split evenly across two workers")
    split = {"train": entries["train"][:count][rank::2], "valid": entries["valid"][rank::2]}
    view_config = deepcopy(config)
    view_config["probe"].update(evaluate_train_assets=count // 2, detailed_eval_assets=detailed // 2)
    view_config["runtime"]["host_memory_budget_gib"] /= 2
    return split, view_config


def expected_row_keys(entries, config, label):
    keys = set()
    for split, values in entries.items():
        if split == "train":
            values = values[:config["probe"]["evaluate_train_assets"]]
        for ordinal, entry in enumerate(values):
            for view, _, _ in evaluation_views(config, ordinal, label):
                keys.add((split, entry["index"], view))
    return keys


def merge_evaluations(parts, entries, config, label):
    rows = [row for part in parts for row in part["rows"]]
    keys = [(row["split"], row["index"], row["view"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_row_keys(entries, config, label):
        raise RuntimeError("assessment workers duplicated or omitted an expected asset/view")
    rows.sort(key=lambda row: (row["split"], row["index"], row["view"]))
    return {"rows": rows, "summary": summarize_rows(rows),
            "artifacts": [item for part in parts for item in part["artifacts"]],
            "cross_pose_sequence_comparisons": [item for part in parts for item in part["cross_pose_sequence_comparisons"]],
            "summary_by_integration": {name: summarize_rows([row for row in rows if row["integration_source"] == name])
                                       for name in sorted({row["integration_source"] for row in rows})}}


def validate_candidate(payload, config):
    expected = config["assessment"]["checkpoint"]
    if payload.get("step") != expected["step"] or payload.get("source_checkpoint") != _checkpoint_identity(config):
        raise RuntimeError("preserved checkpoint step or original source does not match the assessment")
    previous = payload["configuration"]
    for key in ("architecture", "data", "features"):
        if previous[key] != config[key]:
            raise RuntimeError(f"assessment changed the preserved model contract: {key}")
    if payload.get("feature_encoder_frozen") is not True:
        raise RuntimeError("candidate is not from the recorded frozen-feature run")
    accounting = payload["training_accounting"]
    if accounting["heldout_frame_exposures"] != 0:
        raise RuntimeError("candidate checkpoint trained on held-out frames")
    if accounting["assets"] != payload["step"] * previous["probe"]["batch_size"] * previous["probe"]["accumulation"]:
        raise RuntimeError("preserved asset-exposure accounting is inconsistent")
    return {"step": payload["step"], "asset_exposures": accounting["assets"],
            "epochs": accounting["assets"] / previous["data"]["train_assets"],
            "heldout_exposures": 0, "optimizer_and_sampler_preserved": all(k in payload for k in ("optimizer", "scheduler", "sampler"))}


WATCHED = ("memory.queries", "structure.embedding.weight", "structure.head.weight",
           "positions.xyz_head.weight", "structure.backbone.decoder.layers.0.self_attn.q_proj.weight")


def gradient_snapshot(model):
    params = dict(model.skeleton.named_parameters())
    return {name: params[name].grad.detach().float().cpu().clone() for name in WATCHED}


def compare_gradients(left, right):
    result = {}
    for name in WATCHED:
        a, b = left[name].flatten().double(), right[name].flatten().double()
        result[name] = {"relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
                        "cosine": float(torch.nn.functional.cosine_similarity(a[None], b[None]))}
    return result


def isolate_batch_numerics(original, model, entries, config, rank):
    frames = (8, 24)[rank]
    selected = entries[:4] if rank == 0 else sorted(entries, key=lambda item: item["joint_count"], reverse=True)[:4]
    source, target, batch, selections = select_online_batch(original, selected, config,
        ["train_a", "train_b", "train_a", "train_b"], frames, training=True,
        seeds=[117, 118, 119, 120])
    model.train()
    versions = {name: p._version for name, p in model.named_parameters()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, r = model.encode_surface(source)
        anchors, registers = a.detach().clone(), r.detach().clone()
        del a, r
    source_mask = torch.ones(anchors.shape[:2], dtype=torch.bool, device=anchors.device)
    inputs = [anchors, registers, target.tokens, target.query_points, batch["frame_target_joints"], batch["target_parents"]]
    input_versions = [value._version for value in inputs]

    def run(size, *, a=anchors, r=registers):
        model.zero_grad(set_to_none=True)
        loss = 0.0
        for start in range(0, 4, size):
            index = slice(start, start + size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.skeleton(a[index], r[index], source_mask[index], target.tokens[index],
                    target.query_points[index], batch["target_parents"][index], batch["joint_mask"][index],
                    batch["frame_mask"][index], batch["frame_target_joints"][index])
            (out["loss"] * size / 4).backward()
            loss += float(out["loss"].detach()) * size / 4
            del out
        _finite_gradients(model)
        return loss, gradient_snapshot(model)

    full_loss, full_grads = run(4)
    repeat_loss, repeat_grads = run(4)
    serial_loss, serial_grads = run(1)
    pair_loss, pair_grads = run(2)
    limits = config["assessment"]["profile"]

    def comparison(loss, gradients, reference_loss, reference_gradients):
        differences = compare_gradients(gradients, reference_gradients)
        passed = (abs(loss - reference_loss) <= limits["loss_atol"] + limits["loss_rtol"] * abs(reference_loss)
                  and all(d["relative_l2"] <= limits["gradient_relative_l2"] and d["cosine"] >= limits["gradient_cosine"]
                          for d in differences.values()))
        return {"accepted": passed, "loss": loss, "reference_loss": reference_loss, "gradients": differences}

    comparisons = {"B4_repeat": comparison(repeat_loss, repeat_grads, full_loss, full_grads),
                   "B4_vs_B1_acc4": comparison(full_loss, full_grads, serial_loss, serial_grads),
                   "B2_acc2_vs_B1_acc4": comparison(pair_loss, pair_grads, serial_loss, serial_grads)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        again_a, again_r = model.encode_surface(source)
    frozen_difference = {}
    for name, before, after in (("anchors", anchors, again_a), ("registers", registers, again_r)):
        delta = after.float() - before.float()
        frozen_difference[name] = {"exact": bool(torch.equal(before, after)), "maxabs": float(delta.abs().max()),
                                  "relative_l2": float(delta.norm() / before.float().norm().clamp_min(1e-12))}
    reencoded_loss, reencoded_grads = run(4, a=again_a, r=again_r)
    reencoded = comparison(reencoded_loss, reencoded_grads, full_loss, full_grads)
    model.zero_grad(set_to_none=True)
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise RuntimeError("batch isolation changed model parameters")
    if input_versions != [value._version for value in inputs]:
        raise RuntimeError("batch isolation modified its fixed inputs")
    return {"accepted": all(item["accepted"] for item in comparisons.values()), "source_frames": frames,
            "comparisons": comparisons, "frozen_motion_recompute": frozen_difference,
            "reencoded_B4_vs_fixed_B4": reencoded, "thresholds": limits,
            "input_contract": "Capture one batch and its frozen motion outputs once in GPU memory, split the exact same tensors for B4/B2/B1; retain identical parent padding. No disk feature cache or resampling between core comparisons.",
            "assets": [{key: entry[key] for key in ("index", "asset_id", "joint_count")} for entry in selected],
            "selections": selections, "model_parameters_unchanged": True, "optimizer_steps": 0,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}


def profile_physical_batches(original, model, entries, config, rank, report, output):
    frames = (8, 24)[rank]
    profile = config["assessment"]["profile"]
    selected = entries[:4] if rank == 0 else sorted(entries, key=lambda item: item["joint_count"], reverse=True)[:4]
    report["profile_assets"] = [{key: entry[key] for key in ("index", "asset_id", "joint_count")} for entry in selected]
    banks, seeds = ["train_a", "train_b", "train_a", "train_b"], [117, 118, 119, 120]
    versions = {name: p._version for name, p in model.named_parameters()}
    results = []
    model.train()

    def run_group(size, serial=False):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss = 0.0
        if serial:
            for index in range(size):
                source, target, batch, _ = select_online_input(original, selected[index], config, banks[index], frames,
                    training=True, seed=seeds[index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.forward_surface(source, target, batch)
                (out["loss"] / size).backward()
                loss += float(out["loss"].detach()) / size
                del source, target, batch, out
        else:
            source, target, batch, selections = select_online_batch(original, selected[:size], config,
                banks[:size], frames, training=True, seeds=seeds[:size])
            if any(s["heldout_target_positions"] for s in selections):
                raise RuntimeError("physical batching included held-out targets")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_surface(source, target, batch)
            out["loss"].backward()
            loss = float(out["loss"].detach())
            del source, target, batch, selections, out
        torch.cuda.synchronize()
        return {"loss": loss, "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "physical_batch": 1 if serial else size, "accumulation": size if serial else 1,
                "assets": size, "source_frames": frames, "target_frames": 32}

    try:
        run_group(1)
        for size in (1, 2, 4):
            measurements = [run_group(size) for _ in range(profile["repeats"])]
            check_memory_budget(config)
            _finite_gradients(model)
            item = {"status": "passed", "physical_batch": size, "source_frames": frames,
                    "seconds_per_asset": sum(r["seconds"] for r in measurements) / len(measurements) / size,
                    "peak_allocated_gib": max(r["peak_allocated_gib"] for r in measurements),
                    "measurements": measurements}
            results.append(item)
            report["physical_batch_profile"] = results
            write_json(output, report)
            print(f"PROFILE rank={rank} T={frames} B={size} seconds_per_asset={item['seconds_per_asset']:.4f} peakGiB={item['peak_allocated_gib']:.2f}", flush=True)
        batched = run_group(4)
        batched_gradients = gradient_snapshot(model)
        serial = run_group(4, serial=True)
        serial_gradients = gradient_snapshot(model)
        differences = compare_gradients(batched_gradients, serial_gradients)
        equivalent = (abs(batched["loss"] - serial["loss"]) <= profile["loss_atol"] + profile["loss_rtol"] * abs(serial["loss"])
                      and all(d["relative_l2"] <= profile["gradient_relative_l2"] and d["cosine"] >= profile["gradient_cosine"] for d in differences.values()))
        report["physical_batch_equivalence"] = {"accepted": equivalent, "batched": batched, "serial_accumulation": serial,
            "gradient_comparison": differences, "thresholds": profile,
            "meaning": "Same four assets, refs, frame IDs, GT and equal-asset loss; BF16 numerical tolerance, not bitwise identity."}
    except Exception as error:
        report["physical_batch_profile_error"] = f"{type(error).__name__}: {error}"
        report["physical_batch_equivalence"] = {"accepted": False}
        traceback.print_exc()
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if versions != {name: p._version for name, p in model.named_parameters()}:
            raise RuntimeError("assessment profile modified model parameters")
        report["profile_optimizer_steps"] = 0
        report["profile_parameters_unchanged"] = True
        write_json(output, report)
    return results


def worker(config, plan, output, rank):
    physical = config["runtime"]["physical_gpus"][rank]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical) or torch.cuda.device_count() != 1:
        raise RuntimeError("assessment worker did not receive exactly its recorded physical GPU")
    guard(config, "matched_eval")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["memory_fraction"])
    torch.backends.mha.set_fastpath_enabled(False)
    report = {"rank": rank, "physical_gpu": physical, "started_at": timestamp(), "complete": False, "accepted": False}
    write_json(output, report)
    try:
        entries = online_entries(selected_rows(config), config)
        subset, view_config = worker_entries(entries, config, rank)
        root = Path(plan["output_dir"]) / f"worker_{rank}"
        root.mkdir(exist_ok=False)
        checkpoint = config["assessment"]["checkpoint"]
        payload = torch.load(checkpoint["path"], map_location="cpu", mmap=True, weights_only=False)
        report["candidate_accounting"] = validate_candidate(payload, config)
        original, _, _ = _load_original(config)
        initialization = {**config, "data": {**config["data"], "seed": config["data"]["model_seed"]}}
        model = _new_model(original, initialization)
        model.skeleton.load_state_dict(payload["skeleton"], strict=True)
        optimizer = _optimizer(model, config)
        optimizer.load_state_dict(payload["optimizer"])
        report["profile_optimizer_state_resident"] = True
        del payload
        if config["assessment"].get("batch_isolation_only", False):
            report["physical_batch_equivalence"] = isolate_batch_numerics(original, model, entries["train"], view_config, rank)
        else:
            report["trained"] = evaluate(model, subset, view_config, root, "trained", report, output, original=original)
            write_json(output, report)
            profile_physical_batches(original, model, entries["train"], view_config, rank, report, output)
            del optimizer, model, original
            torch.cuda.empty_cache()
            original, _, _ = _load_original(config)
            model = _new_model(original, initialization)
            report["initialized"] = evaluate(model, subset, view_config, root, "initialized", report, output, original=original)
        check_memory_budget(view_config)
        report.update(complete=True, accepted=True, finished_at=timestamp(), peak_cpu_gib=peak_cpu_gib())
        write_json(output, report)
        print(f"ASSESSMENT_WORKER_COMPLETE rank={rank}", flush=True)
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", complete=False, accepted=False, finished_at=timestamp())
        write_json(output, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["assessment"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=[0, 1])
    args = parser.parse_args()
    config = load_config(args.config)
    plan = build_stage_plan(config, "assessment", args.config)
    root = Path(plan["output_dir"])
    expected = Path(plan["evaluation_path"]) if args.worker is None else root / f"worker_{args.worker}.json"
    if args.output.resolve() != expected.resolve() or args.output.exists():
        raise RuntimeError("assessment output is not its unused recorded path")
    if args.worker is not None:
        worker(config, plan, args.output, args.worker)
        return
    source = source_metadata(config)
    guard(config, "matched_eval")
    reference = require_online_preflight(config)
    candidate = config["assessment"]["checkpoint"]
    preserved = config["assessment"]["preserved_receipt"]
    if file_hash(preserved["path"]) != preserved["sha256"]:
        raise RuntimeError("checkpoint-preservation receipt changed")
    receipt = json.loads(Path(preserved["path"]).read_text())
    recorded = receipt["checkpoint"]
    if (Path(recorded["path"]).resolve() != Path(candidate["path"]).resolve()
            or recorded["sha256"] != candidate["sha256"] or recorded["step"] != candidate["step"]):
        raise RuntimeError("assessment candidate is not the verified preserved checkpoint")
    quality_reference = config["assessment"].get("quality_reference")
    if quality_reference and file_hash(quality_reference["path"]) != quality_reference["sha256"]:
        raise RuntimeError("previous quality assessment checksum changed")
    if file_hash(candidate["path"]) != candidate["sha256"]:
        raise RuntimeError("preserved checkpoint checksum changed")
    identity = _checkpoint_identity(config)
    if file_hash(identity["path"]) != identity["sha256"]:
        raise RuntimeError("initialization checkpoint checksum changed")
    root.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "experiment": config["experiment"], "stage": "assessment",
        "source": source, "source_checkpoint": identity, "manifests": verify_manifests(config),
        "candidate_checkpoint": candidate, "preflight_reference": reference, "configuration": config,
        "started_at": timestamp(), "complete": False, "accepted": False, "workers": []}
    write_json(args.output, report)
    children, handles = [], []
    try:
        for rank, physical in enumerate(config["runtime"]["physical_gpus"]):
            worker_output = root / f"worker_{rank}.json"
            log = root / f"worker_{rank}.log"
            handle = log.open("x", buffering=1)
            handles.append(handle)
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(physical)}
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                       "--stage", "assessment", "--output", str(worker_output), "--worker", str(rank)]
            child = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=config["runtime"]["repo"])
            children.append(child)
            report["workers"].append({"rank": rank, "physical_gpu": physical, "pid": child.pid,
                                      "path": str(worker_output), "log": str(log)})
        write_json(args.output, report)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("an assessment worker failed; preserved worker logs contain the cause")
            time.sleep(3)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("assessment worker returned nonzero")
        parts = [json.loads(Path(item["path"]).read_text()) for item in report["workers"]]
        if not all(part.get("complete") and part.get("accepted") for part in parts):
            raise RuntimeError("assessment worker results are incomplete")
        entries = online_entries(selected_rows(config), config)
        if not config["assessment"].get("batch_isolation_only", False):
            for label in ("trained", "initialized"):
                report[label] = merge_evaluations([part[label] for part in parts], entries, config, label)
        report["physical_batch_results"] = [{key: part.get(key) for key in (
            "rank", "physical_gpu", "physical_batch_profile", "physical_batch_equivalence",
            "physical_batch_profile_error", "profile_parameters_unchanged", "profile_optimizer_steps",
            "profile_optimizer_state_resident", "profile_assets", "peak_cpu_gib")}
            for part in parts]
        report["physical_batching_accepted"] = all(part.get("physical_batch_equivalence", {}).get("accepted") for part in parts)
        if config["assessment"].get("batch_isolation_only", False):
            report["batching_acceptance_scope"] = "Fixed frozen-encoder outputs; independent frozen-motion recomputation drift is reported separately. No quality evaluation is repeated."
            report["quality_reference"] = quality_reference
        if file_hash(candidate["path"]) != candidate["sha256"] or file_hash(identity["path"]) != identity["sha256"]:
            raise RuntimeError("assessment modified a preserved checkpoint")
        if source_metadata(config) != source:
            raise RuntimeError("source checkout changed during assessment")
        report.update(complete=True, accepted=True, method_quality_accepted=False,
            source_checkpoint_unchanged=True, candidate_checkpoint_unchanged=True,
            candidate_accounting=parts[0]["candidate_accounting"], finished_at=timestamp(),
            next_action="Primary review of generation/reuse and throughput; no automatic training continuation.")
        write_json(args.output, report)
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        report.update(error=f"{type(error).__name__}: {error}", complete=False, accepted=False, finished_at=timestamp())
        write_json(args.output, report)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
