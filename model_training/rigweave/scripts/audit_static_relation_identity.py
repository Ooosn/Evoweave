"""Read-only end-to-end check of exact-static evidence and anchored identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bias_replay import tensor_delta
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from relation_probe_metrics import file_hash
from run_relation_probe import GradientCapture, cached_forward, evaluate, generate, pack_input, summarize


def main():
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model, _control_batch
    from train_dynamic_rig import build_tokenizer, move_batch
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = config["relation_static_identity"]
    guard(config, "matched_eval")
    previous = json.loads(Path(plan["anchor_report"]).read_text())
    if not previous["complete"] or previous["changed_base_parameters"]:
        raise RuntimeError("anchored source run is incomplete")
    if file_hash(plan["anchor_report"]) != plan["anchor_report_sha256"]:
        raise RuntimeError("anchored evaluation identity changed")
    cache_path = Path(previous["plan"]["source_cache"])
    if file_hash(cache_path) != previous["source_cache_sha256"]:
        raise RuntimeError("original cache changed")
    cached = torch.load(cache_path, mmap=True, map_location="cpu", weights_only=False)["valid"]
    static = [entry for entry in cached if entry["view"] == "static8"]
    if len(static) != 16:
        raise ValueError("expected the original sixteen static assets")
    moving = [entry for entry in cached if entry["view"] != "static8"]
    if not all(not torch.equal(entry["input"]["query_points"], entry["input"]["query_points"][:, :1].expand_as(entry["input"]["query_points"]))
               for entry in moving):
        raise RuntimeError("a nonstatic cache case also needs the exact-static correction")
    checkpoint = Path(previous["source_checkpoint"]["path"])
    if file_hash(checkpoint) != previous["source_checkpoint"]["sha256"]:
        raise RuntimeError("base checkpoint changed")
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("static identity audit requires one recorded GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.mha.set_fastpath_enabled(False)
    payload = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=False)
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
    versions = {name: value._version for name, value in model.named_parameters()}
    dataset = DynamicRigManifestDataset(model_args.val_manifest, tokenizer, frame_count=8, random_query=False,
        seed=previous["plan"]["seed"], motion_fps_ratio=0.75, minimum_random_frames=1,
        motion_vertex_samples=model_args.motion_vertex_samples)
    report = {"source": source_metadata(config), "manifests": verify_manifests(config), "plan": plan,
              "complete": False, "new_full_generations": 0, "rows": [], "static_rows": [],
              "nonstatic_cache_cases_proved_not_exactly_static": len(moving)}
    write_json(args.output, report)
    corrected = []
    for entry in static:
        batch = move_batch(dynamic_rig_collate([dataset[entry["index"]]], tokenizer.pad), torch.device("cuda:0"))
        batch = _control_batch(batch, "zero", previous["plan"]["seed"])
        for key in ("input_ids", "target_joints", "target_parents", "query_center", "query_scale"):
            if not torch.equal(batch[key].cpu(), entry["batch"][key]):
                raise RuntimeError(f"query target/normalization changed: {key}")
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16), GradientCapture(model) as capture:
            cond = model.build_condition(batch, refs=entry["surface_references"].to("cuda:0"))
            losses = model._ar_losses(cond, batch)
        fresh = pack_input(capture.motion_input)
        expected = torch.zeros_like(fresh["states"])
        expected[..., 0] = 1
        feature_delta = tensor_delta(entry["input"]["tokens"], fresh["tokens"])
        point_delta = tensor_delta(entry["input"]["query_points"], fresh["query_points"])
        if not torch.equal(fresh["states"], expected) or feature_delta["max_abs"] != 0 or point_delta["max_abs"] != 0:
            raise RuntimeError("static evidence is not exactly unknown or another input changed")
        current = {**entry, "input": fresh, "baseline_condition": cond.detach().cpu()}
        ce = float(losses["ce_loss"].detach())
        delta = tensor_delta(entry["baseline_condition"].to(cond.device), cond)
        row = {"index": entry["index"], "evidence_exact_unknown": True, "surface_feature_delta": feature_delta,
               "query_point_delta": point_delta, "base_condition_vs_previous": delta,
               "base_ce_abs_difference": abs(ce - entry["baseline"]["ce"])}
        del losses, capture
        cond = cond.detach()
        if delta["max_abs"] == 0 and row["base_ce_abs_difference"] <= 2e-5:
            generation = entry["baseline"]["generation"]
            row["generation_source"] = "previous_exact_condition"
        else:
            generation = generate(model, tokenizer, current, cond, batch, previous["plan"]["max_new_tokens"])
            row["generation_source"] = "new_corrected_base"
            report["new_full_generations"] += 1
        current["baseline"] = {**entry["baseline"], "ce": ce, "generation": generation}
        corrected.append(current)
        report["rows"].append(row)
        write_json(args.output, report)
        del cond, batch
        print(f"STATIC_CAPTURE {entry['index']} exact_unknown base_delta={delta['max_abs']}", flush=True)
    adapter_info = previous["adapter_checkpoint"]
    if file_hash(adapter_info["path"]) != adapter_info["sha256"]:
        raise RuntimeError("trained anchored checkpoint changed")
    encoder.enable_relation_residual(previous["plan"]["bottleneck_dim"], reference_subtraction=True)
    adapters = torch.nn.ModuleList([block.relation_residual for block in encoder.blocks])
    adapter = torch.load(adapter_info["path"], map_location="cpu", weights_only=False)
    if adapter["format"] != "motion_relation_anchor_adapter_v1" or adapter["reference_subtraction"] is not True:
        raise ValueError("not the explicit anchored checkpoint")
    adapters.load_state_dict(adapter["modules"], strict=True)
    adapters.requires_grad_(False)
    adapter_versions = {name: value._version for name, value in adapters.named_parameters()}
    del adapter
    for entry in corrected:
        cond, losses, batch = cached_forward(model, entry, "actual")
        delta = tensor_delta(entry["baseline_condition"].to(cond.device), cond)
        ce_difference = abs(float(losses["ce_loss"].detach()) - entry["baseline"]["ce"])
        if delta["max_abs"] != 0 or ce_difference > 2e-5:
            raise RuntimeError("actual corrected static evidence changes the anchored condition")
        report["static_rows"].append({**entry["baseline"], "condition_vs_corrected_base": delta,
                                     "ce_abs_vs_corrected_base": ce_difference,
                                     "generation_source": "verified_identical_corrected_base_condition"})
        del cond, losses, batch
    prefix_entries = [min(corrected, key=lambda e: e["joint_count"]), max(corrected, key=lambda e: e["joint_count"])]
    report["prefix_checks"] = evaluate(model, tokenizer, prefix_entries, "actual", previous["plan"], preflight=True)
    if not all(row["same_prefix"] for row in report["prefix_checks"]):
        raise RuntimeError("corrected static greedy prefix mismatch")
    changed = [name for name, value in model.named_parameters() if name in versions and value._version != versions[name]]
    changed_adapter = [name for name, value in adapters.named_parameters() if value._version != adapter_versions[name]]
    if changed or changed_adapter:
        raise RuntimeError("read-only audit changed model parameters")
    report.update(complete=True, changed_base_parameters=changed, changed_adapter_parameters=changed_adapter,
                  summary=summarize(report["static_rows"]), peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    write_json(args.output, report)
    print("COMPLETE exact-static identity audit", flush=True)


if __name__ == "__main__":
    main()
