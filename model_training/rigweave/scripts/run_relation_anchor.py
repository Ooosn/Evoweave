"""Test motion-only corrections anchored to the allunknown evidence reference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bias_replay import tensor_delta
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from relation_probe_metrics import file_hash
from run_relation_probe import cached_forward, evaluate, summarize, train_arm


def extremes(entries):
    records = {entry["index"]: entry for entry in entries}
    selected = {min(records, key=lambda key: records[key]["joint_count"]),
                max(records, key=lambda key: records[key]["joint_count"])}
    return [entry for entry in entries if entry["index"] in selected]


def check_parity(rows):
    return bool(rows) and all(row["zero_ce_difference"] <= 2e-5
        and row["zero_condition"]["max_abs"] == 0 and row["same_prefix"] for row in rows)


def ce_control(model, entries, arm):
    rows = []
    for entry in entries:
        cond, losses, batch = cached_forward(model, entry, arm)
        ce = float(losses["ce_loss"].detach())
        if not np.isfinite(ce):
            raise RuntimeError("nonfinite CE in the anchored comparison")
        rows.append({"index": entry["index"], "asset_id": entry["asset_id"], "view": entry["view"],
                     "token_weight": entry["token_weight"], "ce": ce,
                     "base_ce_abs_difference": abs(ce - entry["baseline"]["ce"]),
                     "condition_vs_base": tensor_delta(entry["baseline_condition"].to(cond.device), cond)})
        del cond, losses, batch
    return rows


def main():
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["relation_anchor_preflight", "relation_anchor_screen"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = config["relation_anchor_probe"]
    if plan["reference_subtraction"] is not True or plan["arms"] != ["actual"]:
        raise ValueError("this experiment requires the explicit anchored single-arm plan")
    preflight = args.stage == "relation_anchor_preflight"
    guard(config, "train")
    source_report_path, cache_path = Path(plan["source_evaluation"]), Path(plan["source_cache"])
    if file_hash(source_report_path) != plan["source_evaluation_sha256"] or file_hash(cache_path) != plan["source_cache_sha256"]:
        raise RuntimeError("the paired input-cache provenance changed")
    prior = json.loads(source_report_path.read_text())
    if not prior["complete"] or prior["changed_base_parameters"]:
        raise RuntimeError("the reference comparison did not complete with an unchanged base")
    shared = ("checkpoint", "seed", "bottleneck_dim", "train_assets", "valid_assets", "steps", "accumulation",
              "lr", "weight_decay", "clip_grad_norm", "preflight_steps", "preflight_prefix_tokens", "max_new_tokens", "views")
    if any(plan[key] != prior["plan"][key] for key in shared):
        raise ValueError("anchoring must be the only training change in this follow-up")
    caches = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)
    if len(caches["train"]) != 3 * plan["train_assets"] or len(caches["valid"]) != 3 * plan["valid_assets"]:
        raise RuntimeError("paired cache count changed")
    if preflight:
        caches = {split: extremes(entries) for split, entries in caches.items()}
    selected = {split: [row for row in prior["selected"][split]
                if row["index"] in {entry["index"] for entry in entries}] for split, entries in caches.items()}
    for entries in caches.values():
        for entry in entries:
            entry["permutation_seed"] = plan["permutation_seed"] + entry["index"]
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("anchored probe requires exactly one recorded GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = Path(plan["checkpoint"])
    source_identity = prior["source_checkpoint"]
    if (checkpoint.stat().st_size != source_identity["bytes"]
            or checkpoint.stat().st_mtime_ns != source_identity["mtime_ns"]
            or file_hash(checkpoint) != source_identity["sha256"]):
        raise RuntimeError("immutable base checkpoint identity changed")
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
    if model.condition_fusion != "dynamic" or model.branch_prior is not None:
        raise ValueError("anchored probe requires the same dynamic/no-prior base")
    if any(m.p != 0 for m in encoder.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("frozen motion blocks must have zero dropout")
    versions = {name: value._version for name, value in model.named_parameters()}
    root = Path(plan["output_root"]) / args.stage
    root.mkdir(parents=True, exist_ok=False)
    report = {"source": source_metadata(config), "source_checkpoint": source_identity,
              "manifests": verify_manifests(config), "source_evaluation": str(source_report_path),
              "source_cache_sha256": plan["source_cache_sha256"], "plan": plan, "stage": args.stage,
              "selected": selected, "started_at": datetime.now(timezone.utc).isoformat(),
              "baseline": {split: [entry["baseline"] for entry in entries] for split, entries in caches.items()},
              "observations": prior["observations"], "training": {}, "evaluation": {}, "ce_controls": {},
              "complete": False, "baseline_generation_reused_from_verified_cache": True}
    write_json(args.output, report)
    base_replay = ce_control(model, caches["valid"], "base")
    report["base_cache_replay"] = base_replay
    if any(row["base_ce_abs_difference"] > 2e-5 or row["condition_vs_base"]["max_abs"] != 0 for row in base_replay):
        write_json(args.output, report)
        raise RuntimeError("reused cache no longer matches its baseline")
    torch.manual_seed(plan["seed"])
    encoder.enable_relation_residual(plan["bottleneck_dim"], reference_subtraction=True)
    adapters = torch.nn.ModuleList([block.relation_residual for block in encoder.blocks])
    initial = {key: value.detach().cpu().clone() for key, value in adapters.state_dict().items()}
    expected_ids = {id(p) for p in adapters.parameters()}
    if {id(p) for p in model.parameters() if p.requires_grad} != expected_ids:
        raise RuntimeError("only the anchored adapters may train")
    report["trainable_parameters"] = sum(p.numel() for p in adapters.parameters())
    report["zero_parity"] = evaluate(model, tokenizer, caches["valid"] if preflight else extremes(caches["valid"]),
                                    "actual", plan, preflight=True)
    write_json(args.output, report)
    if not check_parity(report["zero_parity"]):
        raise RuntimeError("anchored zero initialization changed the baseline")
    optimizer = train_arm(model, adapters, caches["train"], "actual", plan, report, args.output,
                          plan["preflight_steps"] if preflight else plan["steps"])
    report["updated_adapter_keys"] = [key for key, value in adapters.state_dict().items()
                                       if not torch.equal(value.cpu(), initial[key])]
    if not report["updated_adapter_keys"]:
        raise RuntimeError("anchored adapters did not learn")
    report["ce_controls"]["unknown"] = ce_control(model, caches["valid"], "unknown")
    if any(row["base_ce_abs_difference"] > 2e-5 or row["condition_vs_base"]["max_abs"] != 0
           for row in report["ce_controls"]["unknown"]):
        write_json(args.output, report)
        raise RuntimeError("trained allunknown correction no longer preserves the baseline")
    if preflight:
        report["trained_unknown_parity"] = evaluate(model, tokenizer, caches["valid"], "unknown", plan, preflight=True)
        if not check_parity(report["trained_unknown_parity"]):
            write_json(args.output, report)
            raise RuntimeError("trained allunknown branch changed the greedy prefix")
    else:
        adapter_path = root / "anchored_adapter.pt"
        torch.save({"format": "motion_relation_anchor_adapter_v1", "reference_subtraction": True,
                    "base_checkpoint": source_identity, "plan": plan, "modules": adapters.state_dict(),
                    "optimizer": optimizer.state_dict(), "steps": plan["steps"]}, adapter_path)
        report["adapter_checkpoint"] = {"path": str(adapter_path), "sha256": file_hash(adapter_path)}
        report["evaluation"]["actual"] = evaluate(model, tokenizer, caches["valid"], "actual", plan)
        report["ce_controls"]["permuted"] = ce_control(model, caches["valid"], "permuted")
    changed_base = [name for name, value in model.named_parameters() if name in versions
                    and (value._version != versions[name] or value.grad is not None)]
    report["changed_base_parameters"] = changed_base
    if changed_base or checkpoint.stat().st_size != source_identity["bytes"] or checkpoint.stat().st_mtime_ns != source_identity["mtime_ns"]:
        write_json(args.output, report)
        raise RuntimeError("the anchored run modified the base")
    report["summary"] = {"base": summarize(report["baseline"]["valid"]),
                         **{arm: summarize(rows) for arm, rows in report["evaluation"].items()}}
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    write_json(args.output, report)
    print(f"COMPLETE {args.stage}", flush=True)


if __name__ == "__main__":
    main()
