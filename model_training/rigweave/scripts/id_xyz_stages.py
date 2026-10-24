"""Evidence-producing audit, bounded tests, capacity and matched training stages."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_frame_data import collate_virtual_targets
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets
from id_xyz_runtime import (
    build_model, build_targets, fixed_dataset, gradient_groups, make_stream,
    optimizer_for, paired_contract_hash, parameter_fingerprint, prepare_batch,
    set_learning_rate, sha256, source_hashes, training_contract, write_json,
)
from id_xyz_evaluation import evaluate_item, evaluate_panel, summarize
from structure_xyz_fill_runtime import data_source_hashes, load_small_reference_panel


def gather(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def read_reviewed(spec):
    if not isinstance(spec, dict) or sha256(spec["path"]) != spec["sha256"]:
        raise ValueError("missing or changed evidence receipt")
    result = json.loads(Path(spec["path"]).read_text())
    if result.get("passed") is not True:
        raise ValueError("evidence receipt did not pass")
    for key in ("code_commit", "config_sha256"):
        if not spec.get(key) or result.get(key) != spec[key]:
            raise ValueError(f"evidence producer {key} changed")
    return result


def validate_receipt(config, receipt, *, mode=None):
    if mode is not None and receipt.get("mode") != mode:
        raise ValueError("evidence belongs to the wrong execution stage")
    if (receipt["paired_contract_sha256"] == paired_contract_hash(config)
            and receipt["source_sha256"] == source_hashes()):
        return
    if config.get("continuation") and mode in ("audit", "preflight", "capacity"):
        from id_xyz_resume import validate_prior_receipt
        validate_prior_receipt(config, receipt)
        return
    if (receipt["paired_contract_sha256"] != paired_contract_hash(config)
            or receipt["source_sha256"] != source_hashes()):
        raise ValueError("evidence does not cover the current source/data/model contract")


def evidence(config, source, config_path):
    return {"code_commit": source, "config_sha256": sha256(config_path),
            "source_sha256": source_hashes(), "paired_contract_sha256": paired_contract_hash(config)}


def check_targets(geometry, targets, tokenizer, codec):
    old = build_xyz_aux_targets(geometry, tokenizer, AuxSkeletonTokens(), max_main_tokens=2048)
    rows = []
    for index, count in enumerate(geometry["joint_count"].tolist()):
        source_parents = geometry["target_parents"][index, :count].cpu().numpy()
        order = np.asarray(targets["dfs_orders"][index])
        inverse = np.argsort(order)
        expected = np.asarray([-1 if source_parents[s] < 0 else inverse[source_parents[s]] for s in order])
        for frame in range(targets["input_ids"].shape[1]):
            ids = targets["input_ids"][index, frame][targets["attention_mask"][index, frame]].cpu().numpy()
            parsed = codec.decode(ids)
            old_ids = old["main_input_ids"][index, frame][old["main_attention_mask"][index, frame]].cpu().numpy()
            old_parsed = tokenizer.detokenize(old_ids)
            if (not np.array_equal(parsed["parents"], expected)
                    or not np.array_equal(parsed["joints"], np.asarray(old_parsed.joints, dtype=np.float32)[order])):
                raise ValueError("new IDs changed a parent or official XYZ coordinate bin")
        selected = geometry["selected_frames"][index].tolist()
        if len(selected) != len(set(selected)) or set(selected) & set(geometry["heldout_raw_ids"][index]):
            raise ValueError("training loader duplicates or leaks heldout frames")
        rows.append({"asset_id": geometry["asset_id"][index], "joints": count,
                     "frames": len(selected), "raw_frames": selected,
                     "tokens": targets["lengths"][index], "parent_identity_exact": True,
                     "official_xyz_bins_exact": True})
    return rows


def audit(args, config, entries, source):
    if config.get("continuation"):
        from id_xyz_resume import audit_continuation
        return audit_continuation(args, config, source)
    previous = read_reviewed(config["audit"]["prior_data_certificate"])
    if (previous["manifests"] != config["data"]["manifests"]
            or previous["target_source_sha256"] != data_source_hashes()):
        raise ValueError("prior static-data certificate no longer covers unchanged source data")
    static = previous["new_static_target_contract"]
    for split, rows in static.items():
        if len(rows) != len(entries[split]):
            raise ValueError("static certificate row count changed")
        for index, row in enumerate(rows):
            if (row["manifest_index"] != index or row["asset_id"] != entries[split][index]["asset_id"]
                    or row["joints"] > config["codec"]["max_joints"]
                    or row["fill_tokens"] + 1 > config["codec"]["max_sequence_tokens"]):
                raise ValueError("static source identity or full-sequence capacity changed")
    model, tokenizer, codec, initial = build_model(config, torch.device("cpu"))
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("CPU initialization created device weights")
    if config["architecture"]["query_tokens"] + codec.max_encoded_length > model.xyz_decoder.config.max_position_embeddings:
        raise ValueError("decoder cannot represent the maximum grammar target")
    del model
    groups = deepcopy(previous["stress_groups"])
    groups["longest_tokens"] = groups["longest_fill"]
    checks = []
    for split, rows in static.items():
        longest = sorted(rows, key=lambda row: -row["fill_tokens"])[:3]
        smallest = min(rows, key=lambda row: row["joints"])
        random_index = int(np.random.default_rng(config["audit"]["selection_seed"]).integers(len(rows)))
        indices = sorted({row["manifest_index"] for row in longest + [smallest, rows[random_index]]})
        dataset = fixed_dataset(entries[split], config, 2, mode="train")
        for index in indices:
            with np.load(entries[split][index]["path"], allow_pickle=False) as raw:
                parents = np.asarray(raw["target_parents"], dtype="<i8")
            if hashlib.sha256(parents.tobytes()).hexdigest() != rows[index]["source_parents_sha256"]:
                raise ValueError("sampled source parents changed since the full static audit")
            for frames in (2, 24):
                request = FrameSampleRequest(index, frames, (config["data"]["stream_seed"] << 32) + index + frames)
                geometry = collate_virtual_targets([dataset[request]], "cpu")
                targets = build_targets(geometry, codec, "cpu")
                checks.extend({"split": split, **row} for row in check_targets(geometry, targets, tokenizer, codec))
    # Reference checks compare the old token protocol before introducing new IDs.
    panel, reference, manifest = load_small_reference_panel(config, entries)
    reference_views = 0
    for group, rows in panel.items():
        dataset = fixed_dataset(rows, config, 8, mode="heldout_eval" if group == "train_heldout" else "seen_eval")
        for index in range(len(dataset)):
            geometry = collate_virtual_targets([dataset[index]], "cpu")
            targets = build_xyz_aux_targets(geometry, tokenizer, AuxSkeletonTokens(), max_main_tokens=2048)
            for frame in config["evaluation"]["frame_indices"]:
                identity = (group, geometry["asset_id"][0], frame)
                old = reference[identity]
                ids = targets["main_input_ids"][0, frame][targets["main_attention_mask"][0, frame]].tolist()
                if (old["raw_frame_index"] != int(geometry["selected_frames"][0, frame])
                        or old["target_tokens"] != ids):
                    raise ValueError("best-B comparison frame or original GT changed")
                reference_views += 1
    streams = [make_stream(config, rank, 2) for rank in range(2)]
    if any(int(stream.global_ends[-1]) != 20000 for stream in streams):
        raise ValueError("sampling plan does not end at the exact exposure budget")
    report = {"passed": True, "mode": "audit", "no_gpu_or_training": True,
              "initialization": initial, "manifests": config["data"]["manifests"],
              "prior_static_certificate": config["audit"]["prior_data_certificate"],
              "new_static_target_contract": static, "stress_groups": groups,
              "real_loader_checks": checks, "matched_reference_views": reference_views,
              "evaluation_panel": manifest, "sampling_plan": streams[0].plan_report(),
              "max_full_tokens": {split: max(row["fill_tokens"] + 1 for row in rows)
                                  for split, rows in static.items()},
              **evidence(config, source, args.config)}
    write_json(args.output / "audit.json", report)
    return report


def requests_for(config, indices, *, frames, batch, rank, number):
    return [FrameSampleRequest(indices[(rank * batch + i) % len(indices)], frames,
            (config["data"]["stream_seed"] << 32) + 100000 + number * 1000 + rank * 100 + i)
            for i in range(batch)]


def memory_check(config, device):
    total = torch.cuda.get_device_properties(device).total_memory
    peak = torch.cuda.max_memory_allocated(device)
    if peak > total * config["capacity"]["allocated_fraction_limit"]:
        raise RuntimeError("allocation exceeded the reviewed VRAM headroom")
    return {"peak_allocated_bytes": peak, "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": total}


def wrap(model, device, world):
    return DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
        gradient_as_bucket_view=True) if world > 1 else model


def metrics(output):
    return {key: float(output[key].detach()) for key in (
        "loss", "main_loss", "consistency_ce", "consistency_kl", "structure_ce", "coordinate_ce",
        "structure_accuracy", "coordinate_accuracy")}


def preflight(model, tokenizer, codec, initial, entries, config, args, device, rank, world, reviewed):
    dataset = fixed_dataset(entries["train"], config, 2, mode="train")
    routes = []
    model.train()
    for frames in config["preflight"]["test_frames"]:
        group = "longest_tokens" if frames == 2 else f"geometry_t{frames}"
        request = requests_for(config, reviewed["stress_groups"][group], frames=frames, batch=1, rank=rank, number=frames)
        surface, targets, geometry = prepare_batch(dataset, request, config, codec, device)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            conditions = model.encode(surface)
            conditions.retain_grad()
            output = model.forward_encoded(conditions, targets, consistency_weight=config["consistency"]["weight"])
        output["loss"].backward()
        norms = conditions.grad.float().square().sum((-1, -2)).sqrt()
        if not bool(torch.isfinite(norms).all()) or not bool((norms > 0).all()):
            raise RuntimeError("not every real frame receives finite nonzero condition gradients")
        routes.append({"frames": frames, "assets": geometry["asset_id"], "condition_gradient_norms": norms.tolist(),
                       "gradient_groups": gradient_groups(model), **metrics(output), **memory_check(config, device)})
        del conditions, output, surface, targets, geometry
    model.zero_grad(set_to_none=True)
    if parameter_fingerprint(model) != initial["initial_model_sha256"]:
        raise RuntimeError("forward/backward contract checks modified weights")
    candidates = [row["manifest_index"] for row in reviewed["new_static_target_contract"]["train"]
                  if 20 <= row["joints"] <= 60 and row["joints"] != 52]
    if len(candidates) < config["preflight"]["fit_assets"] or config["preflight"]["fit_assets"] != world:
        raise ValueError("the fixed fit requires one distinct non52 asset per rank")
    requests = requests_for(config, candidates[:world], frames=config["preflight"]["frames"], batch=1, rank=rank, number=500)
    surface, targets, geometry = prepare_batch(dataset, requests, config, codec, device)
    optimizer = optimizer_for(model, config)
    wrapped = wrap(model, device, world)
    history, evaluations = [], []
    for step in range(config["preflight"]["updates"] + 1):
        if step in config["preflight"]["evaluation_updates"]:
            model.eval()
            rows = evaluate_item(model, tokenizer, codec, surface, targets, geometry,
                frame_indices=(0, config["preflight"]["frames"] - 1), group="fixed_fit")
            evaluations.append({"step": step, "rows": rows, "summary": summarize(rows)})
            model.train()
        if step == config["preflight"]["updates"]:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                final_metrics = metrics(model(surface, targets, consistency_weight=config["consistency"]["weight"]))
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(surface, targets, consistency_weight=config["consistency"]["weight"])
        values = metrics(output)
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("nonfinite fixed-fit loss")
        output["loss"].backward()
        groups = gradient_groups(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        history.append({"step": step + 1, **values, "gradient_norm": norm, "gradient_groups": groups})
        write_json(args.output / f"fit_rank_{rank}.json", {"history": history, "evaluations": evaluations})
        print("FIT " + json.dumps({"rank": rank, **history[-1]}), flush=True)
        del output
    ratios = {key: final_metrics[key] / history[0][key] for key in ("main_loss", "structure_ce", "coordinate_ce")}
    natural = sum(row["success"] and not row["eos_only_legal_choice"] for row in evaluations[-1]["rows"])
    thresholds = config["preflight"]["maximum_final_initial_ratio"]
    passed = all(ratios[key] <= thresholds[key] for key in ratios) and natural > 0
    local = {"rank": rank, "passed": passed, "gradient_routes": routes, "fit_asset": geometry["asset_id"],
             "initial": history[0], "final": final_metrics, "ratios": ratios,
             "natural_final_generations": natural, "evaluations": evaluations, **memory_check(config, device)}
    write_json(args.output / f"preflight_rank_{rank}.json", local)
    reports = gather(local, world)
    result = {"passed": all(row["passed"] for row in reports), "mode": "preflight", "ranks": reports,
              "acceptance_thresholds": thresholds, "training_contract": training_contract(config),
              **evidence(config, initial["code_commit"], args.config)}
    if rank == 0:
        write_json(args.output / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("fixed-fit loss improvement or natural-termination gate did not pass")
    return result


def capacity_cases(config, reviewed):
    cases = []
    for frames in (6, 12, 24):
        geometry = reviewed["stress_groups"][f"geometry_t{frames}"]
        longest = reviewed["stress_groups"]["longest_tokens"]
        for name, indices in (("longest_tokens", longest), ("geometry", geometry), ("mixed", longest + geometry)):
            cases.append({"frames": frames, "batch_per_rank": config["training"]["frame_policy"][str(frames)],
                          "stress_group": name, "indices": indices})
    return cases


def capacity(model, tokenizer, codec, initial, entries, config, args, device, rank, world, reviewed):
    pre = read_reviewed(config["safety"]["preflight"])
    validate_receipt(config, pre, mode="preflight")
    if config["safety"]["primary_accepted"] is not True:
        raise ValueError("preflight evidence requires primary review before capacity")
    if config.get("batch_transition"):
        return batch_capacity(model, codec, initial, entries, config, args, device, rank, world, reviewed)
    dataset = fixed_dataset(entries["train"], config, 2, mode="train")
    optimizer = optimizer_for(model, config)
    wrapped = wrap(model, device, world)
    model.train()
    results = []
    for number, case in enumerate(capacity_cases(config, reviewed)):
        requests = requests_for(config, case["indices"], frames=case["frames"], batch=case["batch_per_rank"], rank=rank, number=number)
        measurements = []
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            losses = []
            for micro in range(2):
                surface, targets, geometry = prepare_batch(dataset, requests, config, codec, device)
                with (wrapped.no_sync() if world > 1 and micro == 0 else nullcontext()):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = wrapped(surface, targets, consistency_weight=config["consistency"]["weight"])
                    (output["loss"] / 2).backward()
                losses.append(float(output["loss"].detach()))
                max_tokens, identities = max(targets["lengths"]), geometry["asset_id"]
                del surface, targets, geometry, output
            groups = gradient_groups(model)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize(device)
            measurements.append({"iteration": iteration, "loss": float(np.mean(losses)), "gradient_norm": norm,
                                 "gradient_groups": groups, "maximum_tokens": max_tokens, "assets": identities,
                                 "seconds": time.monotonic() - tick, **memory_check(config, device)})
        results.append({**case, "passed": True, "measurements": measurements})
        write_json(args.output / f"capacity_rank_{rank}.json", {"passed": False, "rank": rank, "cases": results})
        print("CAPACITY " + json.dumps({"rank": rank, "case": case, "measurements": measurements}), flush=True)
    reports = gather({"rank": rank, "cases": results, "passed": True}, world)
    result = {"passed": True, "mode": "capacity", "ranks": reports,
              "frame_policy": config["training"]["frame_policy"], "cases": capacity_cases(config, reviewed),
              "profile_weights_are_never_training_initialization": True,
              **evidence(config, initial["code_commit"], args.config)}
    if rank == 0:
        write_json(args.output / "capacity.json", result)
    return result


class TimedCapacityForward(torch.nn.Module):
    """Split forward timing without changing the production model or DDP graph."""
    def __init__(self, model, device):
        super().__init__()
        self.model, self.device = model, device
        self.seconds = {}

    def forward(self, surface, targets, *, consistency_weight):
        tick = time.monotonic()
        conditions = self.model.encode(surface)
        torch.cuda.synchronize(self.device)
        encoded = time.monotonic()
        output = self.model.forward_encoded(conditions, targets, consistency_weight=consistency_weight)
        torch.cuda.synchronize(self.device)
        self.seconds = {"encode": encoded - tick, "decode_and_loss": time.monotonic() - encoded}
        return output


def batch_capacity(model, codec, initial, entries, config, args, device, rank, world, reviewed):
    from id_xyz_resume import batch_source_configuration
    source = batch_source_configuration(config)
    if world != 2:
        raise ValueError("matched batch profiling requires both approved GPUs")
    dataset = fixed_dataset(entries["train"], config, 2, mode="train")
    optimizer = optimizer_for(model, config)
    # Keep model tensors fixed while allocating and exercising real Adam state.
    for group in optimizer.param_groups:
        group["lr"] = 0.
    timed = TimedCapacityForward(model, device)
    wrapped = wrap(timed, device, world)
    model.train()
    results = []
    for number, case in enumerate(capacity_cases(config, reviewed)):
        requests = requests_for(config, case["indices"], frames=case["frames"], batch=24, rank=rank, number=number)
        variants = [("candidate", case["batch_per_rank"])]
        if case["stress_group"] == "mixed":
            variants.append(("baseline", source["training"]["frame_policy"][str(case["frames"])]))
        profiles, reference_inputs = {}, None
        for label, batch in variants:
            measurements = []
            for iteration in range(2):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.monotonic()
                sections = dict.fromkeys(("prepare", "encode", "decode_and_loss", "backward", "optimizer"), 0.)
                losses, identities, widths = [], [], []
                inputs = hashlib.sha256()
                for offset in range(0, 24, batch):
                    chunk = requests[offset:offset + batch]
                    tick = time.monotonic()
                    surface, targets, geometry = prepare_batch(dataset, chunk, config, codec, device)
                    torch.cuda.synchronize(device)
                    sections["prepare"] += time.monotonic() - tick
                    for index, request in enumerate(chunk):
                        width = targets["lengths"][index]
                        inputs.update(json.dumps([request.item_index, request.frames, str(request.sampling_seed),
                            geometry["selected_frames"][index].tolist(), width], separators=(",", ":")).encode())
                        inputs.update(targets["input_ids"][index, :, :width].detach().cpu().numpy().tobytes())
                    with (wrapped.no_sync() if offset + len(chunk) < 24 else nullcontext()):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = wrapped(surface, targets, consistency_weight=config["consistency"]["weight"])
                        if not bool(torch.isfinite(output["loss_per_asset"]).all()):
                            raise RuntimeError("nonfinite physical-batch profile loss")
                        sections["encode"] += timed.seconds["encode"]
                        sections["decode_and_loss"] += timed.seconds["decode_and_loss"]
                        tick = time.monotonic()
                        (output["loss_per_asset"].sum() / 24).backward()
                        torch.cuda.synchronize(device)
                        sections["backward"] += time.monotonic() - tick
                    losses.append(float(output["loss"].detach()) * len(chunk))
                    identities.extend(geometry["asset_id"])
                    widths.extend(targets["lengths"])
                    del surface, targets, geometry, output
                digest = inputs.hexdigest()
                if reference_inputs is None:
                    reference_inputs = digest
                if digest != reference_inputs:
                    raise ValueError("batch comparison changed asset/frame/seed/GT identity")
                tick = time.monotonic()
                groups = gradient_groups(model)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                optimizer.step()
                torch.cuda.synchronize(device)
                sections["optimizer"] = time.monotonic() - tick
                measurement = {"iteration": iteration, "loss": sum(losses) / 24,
                    "gradient_norm": norm, "gradient_groups": groups, "maximum_tokens": max(widths),
                    "assets": identities, "input_identity_sha256": digest,
                    "global_assets": 48, "microbatches_per_rank": 24 // batch,
                    "seconds": time.monotonic() - started, "section_seconds": sections,
                    **memory_check(config, device)}
                measurements.append(measurement)
                print("BATCH_PROFILE " + json.dumps({"rank": rank, "case": number, "variant": label,
                    "frames": case["frames"], "batch_per_rank": batch, **measurement}), flush=True)
            profiles[label] = {"batch_per_rank": batch, "measurements": measurements}
        results.append({**case, "passed": True, "measurements": profiles["candidate"]["measurements"],
                        "profiles": profiles})
        write_json(args.output / f"capacity_rank_{rank}.json", {"passed": False, "rank": rank, "cases": results})
    reports = gather({"rank": rank, "cases": results, "passed": True}, world)
    comparisons = []
    for number, case in enumerate(capacity_cases(config, reviewed)):
        if case["stress_group"] != "mixed":
            continue
        times = {label: max(report["cases"][number]["profiles"][label]["measurements"][1]["seconds"]
                           for report in reports) for label in ("baseline", "candidate")}
        comparisons.append({"frames": case["frames"], **times,
                            "speedup": times["baseline"] / times["candidate"]})
    result = {"passed": True, "mode": "capacity", "ranks": reports,
        "frame_policy": config["training"]["frame_policy"], "cases": capacity_cases(config, reviewed),
        "profile_weights_are_never_training_initialization": True, "profile_learning_rate": 0.,
        "comparison_global_assets_per_update": 48, "comparison_inputs_identical": True,
        "performance_comparisons": comparisons,
        "aggregate_matched_speedup": sum(row["baseline"] for row in comparisons) / sum(row["candidate"] for row in comparisons),
        "timing_scope": "Second Adam-resident update, synchronized section timing; same asset/frame/seed/GT and fixed model weights. Not a full-run ETA.",
        **evidence(config, initial["code_commit"], args.config)}
    if rank == 0:
        write_json(args.output / "capacity.json", result)
    return result


def random_state():
    return {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}


@contextmanager
def preserve_random():
    state = random_state()
    try:
        yield
    finally:
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])


def save_checkpoint(model, optimizer, stream, config, args, initial, source, samples, frames, step, rank, world):
    ranks = gather({"rank": rank, "stream": stream.state_dict(), **random_state()}, world)
    if rank == 0:
        names = {id(p): name for name, p in model.named_parameters()}
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rank_states": ranks,
                   "optimizer_parameter_names": [{"group": g["name"], "parameters": [names[id(p)] for p in g["params"]]}
                                                 for g in optimizer.param_groups],
                   "asset_exposures": samples, "input_frame_exposures": frames, "supervised_frame_exposures": frames,
                   "step": step, "initialization": initial, "training_contract": training_contract(config),
                   "configuration": config, "code_commit": source, "source_sha256": source_hashes()}
        path = args.output / f"checkpoint_{samples:06d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(path.with_suffix(".json"), {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                   "asset_exposures": samples, "step": step, "complete": True})
    if world > 1:
        dist.barrier()


def train(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source, reviewed):
    for key in ("preflight", "capacity"):
        validate_receipt(config, read_reviewed(config["safety"][key]), mode=key)
    cap = read_reviewed(config["safety"]["capacity"])
    if (config["safety"]["primary_accepted"] is not True
            or cap["frame_policy"] != config["training"]["frame_policy"]
            or cap["cases"] != capacity_cases(config, reviewed)):
        raise ValueError("formal batches differ from the primary-reviewed measured capacity")
    dataset = fixed_dataset(entries["train"], config, 2, mode="train")
    stream = make_stream(config, rank, world)
    optimizer = optimizer_for(model, config)
    samples, frames_seen, step = 0, 0, 0
    restored, rng_state = None, None
    if config["training"].get("resume_checkpoint"):
        from id_xyz_resume import restore_checkpoint
        checked = read_reviewed(config["safety"]["continuation_audit"])
        validate_receipt(config, checked, mode="resume_audit")
        stream, rng_state, initial, restored = restore_checkpoint(
            model, optimizer, config, rank=rank, world=world, device=device)
        if restored != checked["restored"]:
            raise ValueError("GPU checkpoint restore differs from the reviewed CPU continuation audit")
        samples, frames_seen, step = (restored[k] for k in ("asset_exposures", "supervised_frame_exposures", "step"))
        write_json(args.output / f"restore_rank_{rank}.json", restored)
    wrapped = wrap(model, device, world)
    if rng_state is not None:
        from id_xyz_resume import apply_random_state
        apply_random_state(rng_state, device)
    model.train()
    report = {"complete": False, "rank": rank, "asset_exposures": samples, "input_frame_exposures": frames_seen,
              "supervised_frame_exposures": frames_seen, "step": step, "evaluations": [], "initialization": initial,
              "resume_checkpoint": config["training"].get("resume_checkpoint"),
              "paired_contract_sha256": paired_contract_hash(config)}
    write_json(args.output / f"rank_{rank}.json", report)
    if rank == 0:
        write_json(args.output / "sampling_plan.json", stream.plan_report())
    with preserve_random():
        evaluate_panel(model, tokenizer, codec, entries, config, args.output, samples=samples, rank=rank, world=world)
    report["evaluations"].append(samples)
    first_step = step
    last_saved_samples = None
    started = time.monotonic()
    names = ("loss", "main_loss", "consistency_ce", "consistency_kl", "structure_ce", "coordinate_ce",
             "structure_accuracy", "coordinate_accuracy")
    with (args.output / f"trace_rank_{rank}.jsonl").open("x", buffering=1) as trace:
        while samples < config["training"]["target_assets"]:
            from id_xyz_resume import allocation_handoff_due
            due = torch.tensor(int(allocation_handoff_due(config)), device=device)
            if world > 1:
                dist.all_reduce(due, op=dist.ReduceOp.MAX)
            if bool(due):
                if last_saved_samples != samples:
                    save_checkpoint(model, optimizer, stream, config, args, initial, source, samples, frames_seen, step, rank, world)
                report.update(stage_complete=True, allocation_handoff_required=True)
                write_json(args.output / f"rank_{rank}.json", report)
                return {"passed": True, "complete": False, "stage_complete": True, "mode": "train",
                        "allocation_handoff_required": True, "asset_exposures": samples, "step": step,
                        "input_frame_exposures": frames_seen, "supervised_frame_exposures": frames_seen,
                        **evidence(config, source, args.config)}
            update = int(np.searchsorted(stream.global_ends, samples, side="right"))
            global_goal = int(stream.update_sizes[update])
            local_goal = global_goal // world
            optimizer.zero_grad(set_to_none=True)
            lr, beta1 = set_learning_rate(optimizer, samples, config)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            totals = torch.zeros(len(names) + 2, dtype=torch.float64, device=device)
            count, schedule = 0, []
            while count < local_goal:
                requests = stream.take(local_goal - count)
                write_json(args.output / f"current_batch_rank_{rank}.json", {"step": step + 1,
                    "requests": [{"index": r.item_index, "frames": r.frames, "seed": r.sampling_seed} for r in requests]})
                surface, targets, geometry = prepare_batch(dataset, requests, config, codec, device)
                size = len(requests)
                with (wrapped.no_sync() if world > 1 and count + size < local_goal else nullcontext()):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = wrapped(surface, targets, consistency_weight=config["consistency"]["weight"])
                    if not bool(torch.isfinite(output["loss"])):
                        raise RuntimeError("nonfinite training objective")
                    output["loss_per_asset"].sum().backward()
                totals += torch.stack([output[key].detach() * size for key in names] +
                                      [totals.new_tensor(size), totals.new_tensor(size * requests[0].frames)])
                selected = geometry["selected_frames"].tolist()
                for values, heldout, request in zip(selected, geometry["heldout_raw_ids"], requests):
                    if len(set(values)) != request.frames or set(values) & set(heldout):
                        raise ValueError("training frame selection duplicates or leaks heldout poses")
                schedule.append({"assets": geometry["asset_id"], "indices": [r.item_index for r in requests],
                    "seeds": [r.sampling_seed for r in requests], "raw_frames": selected, "frames": requests[0].frames,
                    "tokens": targets["lengths"], "target_sha256": hashlib.sha256(
                        targets["input_ids"].detach().cpu().numpy().tobytes()).hexdigest()})
                count += size
                del surface, targets, geometry, output
            if world > 1:
                dist.all_reduce(totals)
            actual_assets, actual_frames = int(totals[-2]), int(totals[-1])
            if actual_assets != global_goal:
                raise RuntimeError("actual asset count differs from the sampling plan")
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(world / actual_assets)
            groups = gradient_groups(model) if step == first_step else None
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            samples, frames_seen, step = samples + actual_assets, frames_seen + actual_frames, step + 1
            torch.cuda.synchronize(device)
            item = {"step": step, "asset_exposures": samples, "input_frame_exposures": frames_seen,
                    "supervised_frame_exposures": frames_seen, "assets_in_update": actual_assets,
                    **{key: float(totals[i] / actual_assets) for i, key in enumerate(names)},
                    "gradient_norm": norm, "first_gradient_groups": groups, "learning_rate": lr, "beta1": beta1,
                    "update_seconds": time.monotonic() - tick, "elapsed_seconds": time.monotonic() - started,
                    **memory_check(config, device)}
            trace.write(json.dumps({**item, "microbatches": schedule}, allow_nan=False) + "\n")
            report.update({key: item[key] for key in ("step", "asset_exposures", "input_frame_exposures", "supervised_frame_exposures")})
            report["latest"] = item
            write_json(args.output / f"rank_{rank}.json", report)
            print("TRAIN " + json.dumps({"rank": rank, **item}, allow_nan=False), flush=True)
            if samples in config["training"]["checkpoint_assets"]:
                save_checkpoint(model, optimizer, stream, config, args, initial, source, samples, frames_seen, step, rank, world)
                last_saved_samples = samples
            if samples in config["training"]["evaluation_assets"]:
                with preserve_random():
                    evaluate_panel(model, tokenizer, codec, entries, config, args.output, samples=samples, rank=rank, world=world)
                report["evaluations"].append(samples)
    with preserve_random():
        evaluate_panel(model, tokenizer, codec, entries, config, args.output, samples=samples, rank=rank, world=world, final=True)
    fingerprint = parameter_fingerprint(model)
    if len(set(gather(fingerprint, world))) != 1:
        raise RuntimeError("rank parameters diverged")
    report.update(complete=True, final_model_sha256=fingerprint, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / f"rank_{rank}.json", report)
    return {"passed": True, "complete": True, "mode": "train", "asset_exposures": samples, "step": step,
            "input_frame_exposures": frames_seen, "supervised_frame_exposures": frames_seen,
            "quality_accepted": False, "next_stage_requires_primary_analysis": True,
            **evidence(config, source, args.config)}


def run_stage(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source, reviewed):
    if args.mode == "train":
        return train(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source, reviewed)
    function = {"preflight": preflight, "capacity": capacity}[args.mode]
    return function(model, tokenizer, codec, initial, entries, config, args, device, rank, world, reviewed)
