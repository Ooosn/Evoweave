"""Read-only interventions on the original frozen bias_h8 evidence channels."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from bias_replay import restore_rng, tensor_delta
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from relation_probe_metrics import file_hash
from run_relation_probe import generate, summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from audit_inactive_joint_predictions import joint_matches


VIEWS = ["normal8", "weak2", "static8"]
CONTROLS = ["unknown", "permuted"]


def logged_hash(path, label):
    print(f"HASH_START {label} path={path} bytes={path.stat().st_size}", flush=True)
    started = time.monotonic()
    digest = file_hash(path)
    print(f"HASH_DONE {label} seconds={time.monotonic() - started:.2f} sha256={digest}", flush=True)
    return digest


def validate_plan(plan):
    for key in ("checkpoint", "source_cache", "source_evaluation"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError(f"missing input path: {key}")
    for key in ("source_cache_sha256", "source_evaluation_sha256"):
        value = plan.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid pinned hash: {key}")
    for key in ("seed", "permutation_seed"):
        if type(plan.get(key)) is not int or not 0 <= plan[key] < 2 ** 63:
            raise ValueError(f"invalid local seed: {key}")
    if (plan.get("views") != VIEWS or plan.get("controls") != CONTROLS
            or plan.get("max_new_tokens") != 1400 or plan.get("prefix_tokens") != 64):
        raise ValueError("requires three original views, two controls, 1400 full tokens and 64 prefix tokens")


def intervene_states(states, control, permutation_seed, asset_index):
    if (states.ndim != 4 or states.shape[1] == 0 or states.shape[1] != states.shape[2]
            or states.shape[-1] != 3 or not torch.is_floating_point(states)):
        raise ValueError("evidence must have shape (B,Q,Q,3)")
    if (not torch.isfinite(states).all() or bool((states < 0).any()) or bool((states > 1).any())
            or not torch.allclose(states.float().sum(-1), torch.ones_like(states[..., 0]).float(),
                                  atol=2e-6, rtol=0)):
        raise ValueError("evidence channels must be finite normalized probabilities")
    order = None
    if control == "base":
        changed = states.clone()
    elif control == "unknown":
        changed = torch.zeros_like(states)
        changed[..., 0] = 1
    elif control == "permuted":
        generator = torch.Generator(device="cpu").manual_seed(permutation_seed + asset_index)
        order = torch.randperm(states.shape[1], generator=generator)
        indices = order.to(states.device)
        changed = states.index_select(1, indices).index_select(2, indices)
    else:
        raise ValueError(f"unsupported original-evidence control: {control}")
    details = {"target": "MotionEvidence.states", "channels": ["u", "c", "d"],
               "local_seed": permutation_seed + asset_index if order is not None else None,
               "anchor_order": None if order is None else order.tolist(),
               "state_delta": tensor_delta(states, changed),
               "channel_sum_max_error": float((changed.float().sum(-1) - 1).abs().max())}
    return changed, details


def cached_forward(model, entry, control, permutation_seed):
    from rigweave.dynamic_rig.motion_evidence import MotionEvidence
    from train_dynamic_rig import move_batch
    cache = entry["input"]
    device = next(model.parameters()).device
    restore_rng(cache["rng"])
    tokens = cache["tokens"].to(device).detach().clone().requires_grad_(True)
    states, details = intervene_states(cache["states"].to(device), control, permutation_seed, entry["index"])
    evidence = MotionEvidence(states, cache["anchor_features"].to(device).clone(),
                              cache["valid_groups"].to(device).clone())
    # Input gradients select the recorded checkpointed execution path; never backward.
    with torch.enable_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        cond = model.conditioner.motion_encoder(tokens, query_points=cache["query_points"].to(device).clone(),
                                                 motion_evidence=evidence)
        batch = move_batch(entry["batch"], device)
        losses = model._ar_losses(cond, batch)
    return cond, losses, batch, details


def require_parity(ce_difference, condition_delta, *, same_prefix=True):
    if (not math.isfinite(ce_difference) or not 0 <= ce_difference <= 2e-5
            or condition_delta["max_abs"] != 0 or not same_prefix):
        raise RuntimeError("original baseline replay/parity failed; stop before interventions")


def prefix_matches(ids, original, limit):
    # The existing generation helper includes the two prompt tokens in generated_ids.
    return bool(ids) and ids == original[:min(len(original), limit + 2)]


def validate_inputs(entries, prior, plan):
    if prior.get("complete") is not True or prior.get("changed_base_parameters") != []:
        raise ValueError("source report must be completed with an unchanged original base")
    for key in ("checkpoint", "seed", "max_new_tokens", "views"):
        if prior["plan"][key] != plan[key]:
            raise ValueError(f"original input/generation contract changed: {key}")
    baselines = {(r["index"], r["view"]): r for r in prior["baseline"]["valid"]}
    observations = {(r["index"], r["view"]): r for r in prior["observations"]}
    keys = [(e["index"], e["view"]) for e in entries]
    assets = {e["index"] for e in entries}
    expected = {(i, v) for i in assets for v in VIEWS}
    if (len(assets) != 16 or len(entries) != 48 or len(set(keys)) != 48
            or set(keys) != expected or set(baselines) != expected or set(observations) != expected
            or len(prior["baseline"]["valid"]) != 48 or len(prior["observations"]) != 48):
        raise ValueError("requires exactly the original 16 assets x three views")
    selected = {r["index"]: r for r in prior["selected"]["valid"]}
    if set(selected) != assets:
        raise ValueError("selected validation assets changed")
    for entry in entries:
        key = entry["index"], entry["view"]
        observation = {k: v for k, v in observations[key].items() if k not in {"index", "view"}}
        if entry["baseline"] != baselines[key] or entry["observation"] != observation:
            raise ValueError(f"cached baseline or original stratum masks changed: {key}")
        count = int(entry["batch"]["joint_count"][0])
        if (count != selected[entry["index"]]["joint_count"] or count != entry["joint_count"]
                or entry["asset_id"] != selected[entry["index"]]["asset_id"]
                or len(observation["valid"]) != count
                or any(len(mask) != count for mask in observation["strata"].values())):
            raise ValueError(f"full target or asset identity changed: {key}")
    for index in assets:
        group = [e for e in entries if e["index"] == index]
        for entry in group[1:]:
            for key in ("input_ids", "target_joints", "target_parents", "query_center", "query_scale"):
                if not torch.equal(entry["batch"][key], group[0]["batch"][key]):
                    raise ValueError(f"paired full target/normalization changed: {index}, {key}")
            if not torch.equal(entry["input"]["query_points"][:, 0], group[0]["input"]["query_points"][:, 0]):
                raise ValueError(f"paired query anchors changed: {index}")
    return baselines


def prefix_entries(entries):
    records = {e["index"]: e for e in entries}
    order = sorted(records, key=lambda i: (records[i]["joint_count"], i))
    return [e for e in entries if e["index"] in {order[0], order[-1]} and e["view"] != "static8"]


def prediction_details(tokenizer, generation, batch):
    from eval_dynamic_rig_generation import _parents_from_output
    prediction = tokenizer.detokenize(np.asarray(generation["generated_ids"])) if generation["success"] else None
    joints = None if prediction is None else np.asarray(prediction.joints)
    parents = [] if prediction is None else _parents_from_output(prediction)
    count = int(batch["joint_count"][0])
    target = batch["target_joints"][0, :count].detach().cpu().numpy()
    target_parents = batch["target_parents"][0, :count].detach().cpu().numpy()
    return {"pred_joints": None if joints is None else joints.tolist(), "pred_parents": parents,
            "joint_matches": joint_matches(joints, parents, target, target_parents)}


def tensor_versions(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: (id(value), value._version)}
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else []
    return {key: version for name, child in children
            for key, version in tensor_versions(child, f"{prefix}/{name}").items()}


def reserve_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(path.suffix + ".tmp").exists():
        raise FileExistsError("existing incremental report temporary file")
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"complete": False, "status": "reserved"}, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["inactive_motion_use"], default="inactive_motion_use")
    args = parser.parse_args()
    config = load_config(args.config)
    plan = config["inactive_motion_use"]
    validate_plan(plan)
    protected = [Path(plan[k]).resolve() for k in ("checkpoint", "source_cache", "source_evaluation")]
    if args.output.resolve() in protected:
        raise ValueError("output must not be an immutable input")
    if args.output.exists():
        raise FileExistsError(args.output)
    print("GUARD_START matched_eval", flush=True)
    guard(config, "matched_eval")
    print("GUARD_DONE matched_eval", flush=True)
    hashes = {key: logged_hash(Path(plan[key]), key) for key in ("source_cache", "source_evaluation")}
    if any(value != plan[key + "_sha256"] for key, value in hashes.items()):
        raise RuntimeError("immutable input cache/report hash changed")
    prior = json.loads(Path(plan["source_evaluation"]).read_text())
    checkpoint = Path(plan["checkpoint"])
    identity = prior["source_checkpoint"]
    def verify_sources(*, recheck_inputs=True):
        if (str(checkpoint) != identity["path"] or checkpoint.stat().st_size != identity["bytes"]
                or checkpoint.stat().st_mtime_ns != identity["mtime_ns"]
                or logged_hash(checkpoint, "checkpoint") != identity["sha256"]
                or (recheck_inputs and any(logged_hash(Path(plan[key]), key) != value for key, value in hashes.items()))):
            raise RuntimeError("immutable checkpoint/cache/report identity changed")
    verify_sources(recheck_inputs=False)
    print("CACHE_LOAD_START", flush=True)
    cache = torch.load(plan["source_cache"], map_location="cpu", mmap=True, weights_only=False)
    print("CACHE_LOAD_DONE; INPUT_VALIDATION_START", flush=True)
    entries = cache["valid"]
    baselines = validate_inputs(entries, prior, plan)
    cache_versions = tensor_versions(entries)
    print("INPUT_VALIDATION_DONE rows=48; SOURCE_MANIFEST_CHECK_START", flush=True)
    source, manifests = source_metadata(config), verify_manifests(config)
    print("SOURCE_MANIFEST_CHECK_DONE", flush=True)
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("inactive-motion probe requires exactly one recorded GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(plan["seed"])
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer
    print("MODEL_LOAD_START", flush=True)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model_args = SimpleNamespace(**{**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": checkpoint})
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    model = _build_dynamic_model(model_args, tokenizer, torch.device("cuda:0"))
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.requires_grad_(False)
    model.eval()
    encoder = model.conditioner.motion_encoder
    encoder.gradient_checkpointing = True
    encoder.train()
    if (model.condition_fusion != "dynamic" or model.branch_prior is not None
            or encoder.motion_evidence_fusion != "bias" or encoder.motion_evidence_heads != 8
            or any(getattr(m, "relation_residual", None) is not None for m in encoder.modules())):
        raise ValueError("requires original dynamic bias_h8 with no relation adapter or prior")
    if any(m.p != 0 for m in encoder.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("frozen cached motion execution requires zero dropout")
    versions = {name: value._version for name, value in model.named_parameters()}
    print("MODEL_LOAD_DONE frozen=True relation_adapter=False", flush=True)
    report = {"source": source, "manifests": manifests, "source_checkpoint": identity,
              "source_input_hashes": hashes, "plan": plan, "stage": "inactive_motion_use",
              "started_at": datetime.now(timezone.utc).isoformat(), "complete": False,
              "selected": deepcopy(prior["selected"]), "observations": deepcopy(prior["observations"]),
              "baseline": {"valid": [dict(deepcopy(e["baseline"]), generation_source="original_saved_baseline") for e in entries]},
              "evaluation": {control: [] for control in CONTROLS}, "cachedparity": [], "prefix_checks": [],
              "controlsdetails": {"target": "MotionEvidence.states", "channels": ["u", "c", "d"],
                  "unknown": "every pair is [1,0,0]", "permuted": "same CPU-local permutation of both anchor axes; seed=permutation_seed+asset_index",
                  "preserved": ["surface tokens", "query points", "anchor features", "valid groups", "full GT", "base parameters"],
                  "static_generation": "reuse original only after exact condition parity"},
              "counts": {"base_replays": 0, "prefix_checks": 0, "new_full_generations": 0, "static_reuses": 0}}
    reserve_output(args.output)
    def save(message):
        write_json(args.output, report)
        print(message, flush=True)
    try:
        save("START inactive_motion_use original frozen bias_h8")
        for position, entry in enumerate(entries, 1):
            print(f"REPLAY_START {position}/48 index={entry['index']} view={entry['view']}", flush=True)
            cond, losses, batch, _ = cached_forward(model, entry, "base", plan["permutation_seed"])
            ce = float(losses["ce_loss"].detach())
            row = {"index": entry["index"], "view": entry["view"], "ce": ce,
                   "ce_abs_difference": abs(ce - entry["baseline"]["ce"]),
                   "condition_vs_base": tensor_delta(entry["baseline_condition"].to(cond.device), cond)}
            report["cachedparity"].append(row)
            report["counts"]["base_replays"] += 1
            save(f"REPLAY index={entry['index']} view={entry['view']} ce_delta={row['ce_abs_difference']:.8g}")
            require_parity(row["ce_abs_difference"], row["condition_vs_base"])
            del cond, losses, batch
        print("REPLAY_DONE rows=48 parity=passed", flush=True)
        for position, entry in enumerate(prefix_entries(entries), 1):
            print(f"PREFIX_START {position}/4 index={entry['index']} view={entry['view']}", flush=True)
            cond, losses, batch, _ = cached_forward(model, entry, "base", plan["permutation_seed"])
            cond = cond.detach()
            del losses
            generated = generate(model, tokenizer, entry, cond, batch, plan["prefix_tokens"])
            same = prefix_matches(generated["generated_ids"], entry["baseline"]["generation"]["generated_ids"], plan["prefix_tokens"])
            report["prefix_checks"].append({"index": entry["index"], "view": entry["view"],
                "same_prefix": same, "generated_ids": generated["generated_ids"]})
            report["counts"]["prefix_checks"] += 1
            save(f"PREFIX index={entry['index']} view={entry['view']} match={same}")
            require_parity(0, {"max_abs": 0}, same_prefix=same)
            del cond, batch
        print("PREFIX_DONE rows=4 parity=passed", flush=True)
        for entry in entries:
            base = baselines[entry["index"], entry["view"]]
            for control in CONTROLS:
                print(f"EVAL_START {control} index={entry['index']} view={entry['view']}", flush=True)
                cond, losses, batch, details = cached_forward(model, entry, control, plan["permutation_seed"])
                ce = float(losses["ce_loss"].detach())
                if not math.isfinite(ce):
                    raise RuntimeError("nonfinite intervention CE")
                delta = tensor_delta(entry["baseline_condition"].to(cond.device), cond)
                cond = cond.detach()
                del losses
                if entry["view"] == "static8":
                    require_parity(abs(ce - base["ce"]), delta)
                    generation = deepcopy(base["generation"])
                    provenance = "original_saved_baseline_exact_static_condition"
                    report["counts"]["static_reuses"] += 1
                else:
                    generation = generate(model, tokenizer, entry, cond, batch, plan["max_new_tokens"])
                    provenance = "new_natural_greedy_generation"
                    report["counts"]["new_full_generations"] += 1
                row = {"index": entry["index"], "asset_id": entry["asset_id"], "view": entry["view"],
                       "ce": ce, "token_weight": entry["token_weight"], "ce_difference_vs_base": ce - base["ce"],
                       "condition_vs_base": delta, "control_details": details,
                       "generation_source": provenance, "generation": generation,
                       **prediction_details(tokenizer, generation, batch)}
                report["evaluation"][control].append(row)
                save(f"EVAL {control} index={entry['index']} view={entry['view']} ce={ce:.6f} source={provenance}")
                del cond, batch
        report["changed_base_parameters"] = [n for n, p in model.named_parameters() if p._version != versions[n]]
        report["parameters_with_grad"] = [n for n, p in model.named_parameters() if p.grad is not None]
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["cache_tensor_versions_unchanged"] = tensor_versions(entries) == cache_versions
        if (report["changed_base_parameters"] or report["parameters_with_grad"] or report["trainable_parameters"]
                or not report["cache_tensor_versions_unchanged"]):
            raise RuntimeError("frozen parameters or immutable cached inputs changed")
        verify_sources()
        if (report["counts"] != {"base_replays": 48, "prefix_checks": 4, "new_full_generations": 64, "static_reuses": 32}
                or any(len(rows) != 48 for rows in report["evaluation"].values())):
            raise RuntimeError("planned probe results are incomplete")
        report["summary"] = {"base": summarize(report["baseline"]["valid"]),
                              **{control: summarize(rows) for control, rows in report["evaluation"].items()}}
        report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                      source_hashes_verified_after=True, parameter_versions_unchanged=True)
        save("COMPLETE inactive_motion_use")
    except BaseException as exc:
        report.update(complete=False, error=f"{type(exc).__name__}: {exc}",
                      stopped_at=datetime.now(timezone.utc).isoformat(),
                      changed_base_parameters=[n for n, p in model.named_parameters() if p._version != versions[n]],
                      parameters_with_grad=[n for n, p in model.named_parameters() if p.grad is not None],
                      cache_tensor_versions_unchanged=tensor_versions(entries) == cache_versions)
        save(f"STOP inactive_motion_use: {report['error']}")
        raise


if __name__ == "__main__":
    main()
