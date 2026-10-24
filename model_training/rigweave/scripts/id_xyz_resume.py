"""Verified checkpoint continuation without changing the model or data recipe."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_frame_runtime import exposure_learning_rate


CONTINUATION_SOURCE_CHANGES = {
    "scripts/id_xyz_runtime.py", "scripts/id_xyz_stages.py",
    "scripts/train_id_xyz_consistency.py", "scripts/id_xyz_resume.py",
}
TEMPORAL_SOURCE = "src/rigweave/dynamic_rig/motion_encoder.py"


def temporal_execution(config):
    transition = config.get("batch_transition")
    if transition is None:
        return None
    required = {"kind", "source_configuration", "source_audit"}
    if not isinstance(transition, dict):
        raise ValueError("invalid physical-batch transition declaration")
    if transition.get("kind") == "physical_batch_only_v1" and set(transition) == required:
        return None
    if (transition.get("kind") != "physical_batch_temporal_chunks_v2"
            or set(transition) != required | {"temporal_execution"}):
        raise ValueError("invalid physical-batch transition declaration")
    spec = transition["temporal_execution"]
    if (not isinstance(spec, dict) or set(spec) != {
            "kind", "max_sequences", "source_before_sha256", "source_after_sha256"}
            or spec["kind"] != "independent_temporal_sequence_chunks_v1"
            or type(spec["max_sequences"]) is not int or spec["max_sequences"] != 13452
            or any(not isinstance(spec[key], str) or len(spec[key]) != 64
                   or any(c not in "0123456789abcdef" for c in spec[key])
                   for key in ("source_before_sha256", "source_after_sha256"))
            or spec["source_before_sha256"] == spec["source_after_sha256"]):
        raise ValueError("invalid reviewed temporal execution change")
    return spec


def approved_source_changes(config, previous, current):
    allowed = set(CONTINUATION_SOURCE_CHANGES)
    execution = temporal_execution(config)
    if execution is not None:
        if (previous.get(TEMPORAL_SOURCE) != execution["source_before_sha256"]
                or current.get(TEMPORAL_SOURCE) != execution["source_after_sha256"]):
            raise ValueError("temporal source does not match the exact reviewed before/after hashes")
        allowed.add(TEMPORAL_SOURCE)
    return allowed


def batch_source_configuration(config):
    from id_xyz_runtime import paired_contract_hash, sha256, source_hashes
    from id_xyz_stages import read_reviewed
    transition = config.get("batch_transition")
    if transition is None:
        return None
    temporal_execution(config)
    spec = transition["source_configuration"]
    if sha256(spec["path"]) != spec["sha256"]:
        raise ValueError("the immutable batch-source configuration changed")
    source = json.loads(Path(spec["path"]).read_text())
    if source.get("batch_transition") or source["training"]["target_assets"] != 100000:
        raise ValueError("batch transition must reference the reviewed original100k recipe")
    excluded = {"runtime", "training", "safety", "batch_transition"}
    if {k: v for k, v in config.items() if k not in excluded} != {
            k: v for k, v in source.items() if k not in excluded}:
        raise ValueError("batch transition changed model, data, loss, schedule or evaluation")
    if {k: v for k, v in config["runtime"].items() if k != "repo"} != {
            k: v for k, v in source["runtime"].items() if k != "repo"}:
        raise ValueError("batch transition changed runtime resources or dependencies")
    plan_excluded = {"frame_policy", "resume_checkpoint"}
    if {k: v for k, v in config["training"].items() if k not in plan_excluded} != {
            k: v for k, v in source["training"].items() if k not in plan_excluded}:
        raise ValueError("batch transition changed the global exposure/update plan")
    old, new = source["training"]["frame_policy"], config["training"]["frame_policy"]
    if set(old) != set(new) or set(new) != {str(t) for t in range(2, 25)}:
        raise ValueError("batch policies must cover exactly2..24 frames")
    if old == new:
        raise ValueError("batch transition must increase at least one physical tier")
    for frame in range(2, 25):
        before, after = old[str(frame)], new[str(frame)]
        ceiling = 6 if frame <= 6 else 12 if frame <= 12 else 24
        if (type(after) is not int or after not in (before, before * 2) or 24 % after
                or after != new[str(ceiling)]):
            raise ValueError("only an unchanged or doubled physical batch tier is allowed")
    audit = read_reviewed(transition["source_audit"])
    if audit.get("mode") != "resume_audit" or audit["paired_contract_sha256"] != paired_contract_hash(source):
        raise ValueError("batch-source audit does not cover its exact recipe")
    previous, current = audit["source_sha256"], source_hashes()
    if (set(previous) != set(current)
            or not {key for key in previous if previous[key] != current[key]}.issubset(
                approved_source_changes(config, previous, current))):
        raise ValueError("batch transition changed model, data, loss, generation or sampling source")
    return source


def parent_configuration(config):
    from id_xyz_runtime import sha256
    spec = config["continuation"]["parent_configuration"]
    if sha256(spec["path"]) != spec["sha256"]:
        raise ValueError("the immutable parent configuration changed")
    parent = json.loads(Path(spec["path"]).read_text())
    batch_source = batch_source_configuration(config)
    excluded = {"arm", "runtime", "training", "safety", "continuation", "batch_transition"}
    if {k: v for k, v in config.items() if k not in excluded} != {
            k: v for k, v in parent.items() if k not in excluded}:
        raise ValueError("continuation changed the model, data, objective or evaluation recipe")
    runtime_excluded = {"repo", "stop_before_utc"}
    if {k: v for k, v in config["runtime"].items() if k not in runtime_excluded} != {
            k: v for k, v in parent["runtime"].items() if k not in runtime_excluded}:
        raise ValueError("continuation changed runtime resources or dependencies")
    plan_excluded = {"target_assets", "checkpoint_assets", "evaluation_assets", "resume_checkpoint"}
    if batch_source is not None:
        plan_excluded.add("frame_policy")
    if {k: v for k, v in config["training"].items() if k not in plan_excluded} != {
            k: v for k, v in parent["training"].items() if k not in plan_excluded}:
        raise ValueError("continuation changed the effective or physical batch policy")
    if parent["training"]["target_assets"] != 20000 or parent.get("continuation"):
        raise ValueError("the budget extension must refer to the original published 20k recipe")
    return parent


def validate_prior_receipt(config, receipt):
    from id_xyz_runtime import paired_contract_hash, source_hashes
    parent = parent_configuration(config)
    if receipt["paired_contract_sha256"] != paired_contract_hash(parent):
        raise ValueError("prior evidence does not match the immutable parent recipe")
    old, new = receipt["source_sha256"], source_hashes()
    if not set(old).issubset(new) or set(new) - set(old) != {"scripts/id_xyz_resume.py"}:
        raise ValueError("unexpected source additions or removals at continuation")
    changes = {name for name in old if old[name] != new[name]}
    if not changes.issubset(approved_source_changes(config, old, new)):
        raise ValueError("model, data, loss, generation or sampling implementation changed")


def extended_learning_rate(samples, config):
    policy, optim = config["continuation"], config["optimization"]
    start, end = policy["handoff_assets"], config["training"]["target_assets"]
    if not 0 <= samples <= end or start != 5000 or end != 100000:
        raise ValueError("invalid exposure position for the authorized 100k extension")
    def original(position):
        return exposure_learning_rate(position, optim["schedule_samples"], peak=optim["peak_lr"],
            warmup_fraction=optim["warmup_fraction"], div_factor=optim["div_factor"],
            final_div_factor=optim["final_div_factor"])
    if samples <= start:
        return original(samples)
    lr, beta = original(start)
    final = optim["peak_lr"] / optim["div_factor"] / optim["final_div_factor"]
    fraction = .5 * (1 - math.cos(math.pi * (samples - start) / (end - start)))
    return lr + fraction * (final - lr), beta + fraction * (.95 - beta)


def restore_stream(config, state, *, samples, frames, step, rank, world):
    from id_xyz_runtime import make_stream
    contract = state["contract"]
    if contract["rank"] != rank or contract["world_size"] != world:
        raise ValueError("saved sampler belongs to a different rank/world")
    old = MatchedFrameStream(range(contract["dataset_rows"]), rank=rank, world_size=world,
        seed=contract["seed"], target_assets=contract["target_assets"],
        effective_assets=contract["effective_assets"], frame_policy=contract["frame_policy"],
        manifest_sha256=contract["manifest_sha256"], min_frames=contract["min_frames"],
        max_frames=contract["max_frames"], state=state, update_boundaries=contract["update_boundaries"])
    if (type(samples) is not int or type(step) is not int or samples != old.cursor * world
            or step < 1 or step > len(old.global_ends) or int(old.global_ends[step - 1]) != samples
            or int(np.dot(old.update_sizes[:step].astype(np.int64), old.frame_counts[:step])) != frames):
        raise ValueError("checkpoint counters do not identify a completed sampler update")
    new = make_stream(config, rank, world)
    for key in set(old.contract) - {"target_assets", "update_boundaries"}:
        if old.contract[key] != new.contract[key]:
            if key == "frame_policy" and config.get("batch_transition"):
                source = batch_source_configuration(config)
                if old.contract[key] == source["training"][key]:
                    continue
            raise ValueError(f"continuation changed sampler contract {key}")
    n = len(old.update_sizes)
    if (new.target_assets < old.target_assets
            or new.contract["update_boundaries"][:len(old.contract["update_boundaries"])] != old.contract["update_boundaries"]
            or not np.array_equal(old.order, new.order[:old.target_assets])
            or not np.array_equal(old.frame_counts, new.frame_counts[:n])
            or not np.array_equal(old.update_sizes, new.update_sizes[:n])):
        raise ValueError("extended sampler did not preserve the entire original plan prefix")
    new.cursor = old.cursor
    return new


def optimizer_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [{"group": g["name"], "parameters": [names[id(p)] for p in g["params"]]} for g in optimizer.param_groups]


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def optimizer_fingerprint(model, optimizer):
    state = {}
    for name, parameter in model.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                state[name + "/" + key] = value
    groups = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
    digest = hashlib.sha256(json.dumps(groups, sort_keys=True).encode())
    digest.update(state_fingerprint(state).encode())
    return digest.hexdigest()


def rng_fingerprint(state):
    numpy_state = state["numpy_rng"]
    payload = {"torch": state["torch_rng"].tolist(), "cuda": state["cuda_rng"].tolist(),
               "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
               "python": state["python_rng"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def apply_random_state(state, device):
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], device)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def validate_random_state(state):
    try:
        for key in ("torch_rng", "cuda_rng"):
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("RNG states must be byte vectors")
        torch.Generator(device="cpu").set_state(state["torch_rng"])
        # The pinned CUDA runtime uses a Philox seed and aligned 64-bit offset.
        cuda = state["cuda_rng"].cpu().numpy().tobytes()
        if len(cuda) != 16 or int.from_bytes(cuda[8:], "little") % 4:
            raise ValueError("invalid CUDA Philox state")
        np.random.RandomState(0).set_state(state["numpy_rng"])
        random.Random(0).setstate(state["python_rng"])
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("invalid saved RNG state") from error


def restore_checkpoint(model, optimizer, config, *, rank, world, device, saved=None):
    from id_xyz_runtime import make_stream, sha256, source_hashes, training_contract
    from id_xyz_stages import read_reviewed
    if type(rank) is not int or type(world) is not int or world != config["runtime"]["world_size"] or not 0 <= rank < world:
        raise ValueError("invalid continuation rank/world")
    spec = config["training"]["resume_checkpoint"]
    parent = parent_configuration(config)
    if saved is None:
        if Path(spec["path"]).stat().st_size != spec["bytes"] or sha256(spec["path"]) != spec["sha256"]:
            raise ValueError("continuation checkpoint identity changed")
        saved = torch.load(spec["path"], map_location="cpu", mmap=True, weights_only=False)
    batch_source = batch_source_configuration(config)
    from_batch_source = batch_source is not None and saved["training_contract"] == training_contract(batch_source)
    from_parent = saved["training_contract"] == training_contract(parent)
    if (batch_source is not None and from_parent
            or not (from_parent or from_batch_source) and saved["training_contract"] != training_contract(config)):
        raise ValueError("saved model/data/loss/schedule contract changed")
    expected_source = (read_reviewed(config["batch_transition"]["source_audit"])["source_sha256"] if from_batch_source
                       else read_reviewed(config["safety"]["data_audit"])["source_sha256"] if from_parent
                       else source_hashes())
    samples, step = saved["asset_exposures"], saved["step"]
    frames = saved["supervised_frame_exposures"]
    if (saved["code_commit"] != spec["producer_head"] or saved["source_sha256"] != expected_source
            or samples != spec["asset_exposures"] or step != spec["step"]
            or saved["input_frame_exposures"] != frames or len(saved["rank_states"]) != world
            or (from_parent and samples != config["continuation"]["handoff_assets"])):
        raise ValueError("checkpoint producer, counters or rank coverage changed")
    saved_recipe = batch_source if from_batch_source else parent if from_parent else config
    for index, state in enumerate(saved["rank_states"]):
        if state["stream"]["contract"] != make_stream(saved_recipe, index, world).contract:
            raise ValueError("saved sampler does not match its producer training contract")
    if saved["optimizer_parameter_names"] != optimizer_names(model, optimizer):
        raise ValueError("optimizer parameter names or order changed")
    ids = [index for group in saved["optimizer"]["param_groups"] for index in group["params"]]
    if (any(type(index) is not int for index in ids) or ids != list(range(len(ids)))
            or set(saved["optimizer"]["state"]) != set(ids)):
        raise ValueError("checkpoint must retain Adam state for every optimized parameter")
    for name, value in saved["model"].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite checkpoint parameter {name}")
    expected_model = state_fingerprint(saved["model"])
    model.load_state_dict(saved["model"], strict=True)
    if state_fingerprint(model.state_dict()) != expected_model:
        raise ValueError("checkpoint restoration changed a model tensor")
    optimizer.load_state_dict(saved["optimizer"])
    if not optimizer.state:
        raise ValueError("checkpoint has no trained Adam state")
    for parameter, state in optimizer.state.items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != step:
            raise ValueError("checkpoint Adam step/state is incomplete")
        for name in ("exp_avg", "exp_avg_sq"):
            if state[name].shape != parameter.shape or not bool(torch.isfinite(state[name]).all()):
                raise ValueError("invalid checkpoint Adam moment")
            state[name] = state[name].to(device)
    signatures, streams = [], []
    for index, state in enumerate(saved["rank_states"]):
        if state["rank"] != index or any(state[key].dtype != torch.uint8 or state[key].ndim != 1
                                        for key in ("torch_rng", "cuda_rng")):
            raise ValueError("saved rank RNG identity is invalid")
        validate_random_state(state)
        stream = restore_stream(config, state["stream"], samples=samples, frames=frames, step=step, rank=index, world=world)
        clone = deepcopy(stream)
        requests = clone.take() if clone.cursor < clone.local_total else []
        signatures.append({"rank": index, "rng_sha256": rng_fingerprint(state),
                           "sampler": stream.state_dict(), "next_requests": [
                               [r.item_index, r.frames, r.sampling_seed] for r in requests]})
        streams.append(stream)
    report = {"model_sha256": expected_model, "optimizer_sha256": optimizer_fingerprint(model, optimizer),
              "asset_exposures": samples, "step": step, "input_frame_exposures": frames,
              "supervised_frame_exposures": frames, "rank_signatures": signatures,
              "next_learning_rate_and_beta1": list(extended_learning_rate(samples, config)),
              "loaded_learning_rates": [g["lr"] for g in optimizer.param_groups]}
    if batch_source is not None:
        report["physical_batch_transition"] = {
            "source_policy": batch_source["training"]["frame_policy"],
            "resumed_policy": config["training"]["frame_policy"],
            "global_asset_frame_seed_order_preserved": True,
            "future_bitwise_trajectory_claimed": False,
        }
        if temporal_execution(config) is not None:
            report["physical_batch_transition"]["temporal_execution"] = temporal_execution(config)
    return streams[rank], saved["rank_states"][rank], saved["initialization"], report


def audit_continuation(args, config, source):
    from id_xyz_runtime import build_model, optimizer_for, parameter_fingerprint, write_json
    from id_xyz_stages import evidence, read_reviewed, validate_receipt
    prior = {}
    for name in ("data_audit", "preflight", "capacity"):
        prior[name] = read_reviewed(config["safety"][name])
        validate_receipt(config, prior[name], mode="audit" if name == "data_audit" else name)
    model, _, _, initial = build_model(config, torch.device("cpu"))
    if parameter_fingerprint(model) != prior["data_audit"]["initialization"]["initial_model_sha256"]:
        raise ValueError("continuation loader changed official initialization tensors")
    optimizer = optimizer_for(model, config)
    _, _, _, restored = restore_checkpoint(model, optimizer, config, rank=0,
        world=config["runtime"]["world_size"], device=torch.device("cpu"))
    report = {"passed": True, "mode": "resume_audit", "no_GPU_or_optimizer_update": True,
              "initialization": initial, "restored": restored,
              "reused_prior_engineering_evidence": config["safety"],
              "approved_training_source_changes": sorted(CONTINUATION_SOURCE_CHANGES | (
                  {TEMPORAL_SOURCE} if temporal_execution(config) else set())),
              **evidence(config, source, args.config)}
    write_json(args.output / "resume_audit.json", report)
    return report


def allocation_handoff_due(config, now=None):
    value = config["runtime"].get("stop_before_utc")
    if value is None:
        return False
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("allocation handoff deadline needs an explicit timezone")
    return (now or datetime.now(timezone.utc)) >= deadline
