"""Matched small-set adaptation of an explicit pair-aware value residual."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from bias_replay import MotionCapture, restore_rng, tensor_delta
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from relation_probe_metrics import file_hash, observability, select_rows, structure_by_stratum


class GradientCapture(MotionCapture):
    def _before(self, module, args, kwargs):
        super()._before(module, args, kwargs)
        return (args[0].detach().requires_grad_(True),), kwargs


def cpu_batch(batch):
    excluded = {"frame_vertices", "faces", "vertex_normals", "face_normals"}
    return {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key not in excluded}


def pack_input(value):
    return {"tokens": value.tokens.cpu(), "query_points": value.query_points.cpu(), "rng": value.rng,
            **{key: getattr(value.evidence, key).cpu() for key in ("states", "anchor_features", "valid_groups")}}


def cached_forward(model, entry, arm):
    from rigweave.dynamic_rig.motion_evidence import MotionEvidence
    from train_dynamic_rig import move_batch
    cache = entry["input"]
    device = next(model.parameters()).device
    restore_rng(cache["rng"])
    tokens = cache["tokens"].to(device).detach().requires_grad_(True)
    evidence = MotionEvidence(*(cache[key].to(device) for key in ("states", "anchor_features", "valid_groups")))
    relation_states = None
    if arm == "unknown":
        relation_states = torch.zeros_like(evidence.states)
        relation_states[..., 0] = 1
    elif arm == "permuted":
        generator = torch.Generator(device=device).manual_seed(entry["permutation_seed"])
        order = torch.randperm(evidence.states.shape[1], device=device, generator=generator)
        relation_states = evidence.states[:, order][:, :, order]
    elif arm not in {"base", "actual"}:
        raise ValueError(f"unknown arm: {arm}")
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cond = model.conditioner.motion_encoder(tokens, query_points=cache["query_points"].to(device),
            motion_evidence=evidence, relation_states=relation_states)
        batch = move_batch(entry["batch"], device)
        losses = model._ar_losses(cond, batch)
    return cond, losses, batch


def tree_checks(joints, parents):
    cycles = 0
    invalid = 0
    lengths = []
    for j, parent in enumerate(parents):
        if parent is None or parent < 0:
            continue
        if parent >= len(parents):
            invalid += 1
            continue
        lengths.append(float(np.linalg.norm(joints[j] - joints[parent])))
        seen, node = set(), j
        while node is not None and 0 <= node < len(parents):
            if node in seen:
                cycles += 1
                break
            seen.add(node)
            node = parents[node]
    return {"nodes_reaching_cycle": cycles, "invalid_parent_count": invalid,
            "near_zero_edge_fraction": float(np.mean(np.asarray(lengths) <= 0.001)) if lengths else 0.0}


@torch.no_grad()
def generate(model, tokenizer, entry, cond, batch, limit):
    from diagnose_motion_bias import generation_result
    from eval_dynamic_rig_generation import _parents_from_output
    result = generation_result(model, tokenizer, batch, cond.detach(), limit)
    pred = tokenizer.detokenize(np.asarray(result["generated_ids"])) if result["success"] else None
    parents = _parents_from_output(pred) if pred is not None else []
    if pred is not None:
        result["tree"] = tree_checks(np.asarray(pred.joints), parents)
    if entry.get("observation") is not None:
        count = int(batch["joint_count"][0])
        result["strata"] = structure_by_stratum(
            None if pred is None else np.asarray(pred.joints), parents,
            batch["target_joints"][0, :count].cpu().numpy(), batch["target_parents"][0, :count].cpu().numpy(),
            entry["observation"])
    return result


def summarize(rows):
    out = {}
    for view in sorted({row["view"] for row in rows}):
        group = [row for row in rows if row["view"] == view]
        entry = {"assets": len(group), "ce_macro": float(np.mean([row["ce"] for row in group])),
                 "ce_token_weighted": float(sum(row["ce"] * row["token_weight"] for row in group)
                                            / sum(row["token_weight"] for row in group))}
        generated = [row["generation"] for row in group if "generation" in row]
        if generated:
            entry.update(generated=len(generated), success=sum(row["success"] for row in generated),
                hitmax=sum(row["hit_max_without_eos"] for row in generated),
                topology_f1_all_rows=float(np.mean([row["topology_f1_or_zero"] for row in generated])))
            strata = {}
            for name in generated[0].get("strata", {}):
                values = [row["strata"][name] for row in generated]
                item = {"joints": sum(row["joints"] for row in values), "edges": sum(row["edges"] for row in values)}
                for metric, weight in (("joint_coverage_005", "joints"), ("edge_recall", "edges")):
                    item[metric] = (sum(row[metric] * row[weight] for row in values if row[metric] is not None)
                                    / item[weight]) if item[weight] else None
                strata[name] = item
            entry["strata"] = strata
        out[view] = entry
    return out


def build_cache(model, tokenizer, model_args, selected, plan, report, output, *, preflight):
    from eval_dynamic_rig_ce import _control_batch
    from train_dynamic_rig import move_batch
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
    from rigweave.dynamic_rig.frame_budget_accounting import target_token_weight
    device = next(model.parameters()).device
    caches = {}
    for split, records in selected.items():
        manifest = getattr(model_args, "train_manifest" if split == "train" else "val_manifest")
        datasets = {frames: DynamicRigManifestDataset(manifest, tokenizer, frame_count=frames,
            random_query=False, seed=plan["seed"], motion_fps_ratio=0.75, minimum_random_frames=1,
            motion_vertex_samples=model_args.motion_vertex_samples) for frames in (2, 8)}
        entries = []
        for record in records:
            reference_batch = None
            reference_refs = None
            reference_cpu = None
            for view, frames in (("normal8", 8), ("weak2", 2), ("static8", 8)):
                batch = move_batch(dynamic_rig_collate([datasets[frames][record["index"]]], tokenizer.pad), device)
                if batch["path"][0] != record["path"]:
                    raise RuntimeError("manifest selection and dataset disagree")
                if reference_batch is None:
                    reference_batch = cpu_batch(batch)
                for key in ("input_ids", "target_joints", "target_parents", "query_center", "query_scale"):
                    if not torch.equal(batch[key].cpu(), reference_batch[key]):
                        raise RuntimeError(f"paired query/GT/normalization changed: {key}")
                if view == "static8":
                    batch = _control_batch(batch, "zero", plan["seed"])
                if reference_refs is None:
                    torch.manual_seed(plan["seed"] + record["index"])
                    reference_refs = model.sample_references(batch)
                    reference_cpu = reference_refs.to("cpu")
                refs = reference_refs
                with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16), GradientCapture(model) as capture:
                    cond = model.build_condition(batch, refs=refs)
                    losses = model._ar_losses(cond, batch)
                first_ce = float(losses["ce_loss"].detach())
                entry = {**record, "view": view, "input": pack_input(capture.motion_input),
                         "batch": cpu_batch(batch), "baseline_condition": cond.detach().cpu(),
                         "surface_references": reference_cpu,
                         "token_weight": target_token_weight(batch, tokenizer.eos, model.eos_loss_weight)}
                selected_frames = batch["selected_frames"][0].cpu().tolist()
                effective_frames = [selected_frames[0]] if view == "static8" else selected_frames
                if split == "valid":
                    entry["observation"] = observability(record["path"], effective_frames, float(batch["query_scale"][0]))
                    report.setdefault("observations", []).append({"index": record["index"], "view": view,
                                                                    **entry["observation"]})
                del cond, losses, refs, capture
                cond, losses, compact = cached_forward(model, entry, "base")
                ce = float(losses["ce_loss"].detach())
                contract = {"split": split, "index": record["index"], "view": view,
                            "ce_abs_difference": abs(ce - first_ce),
                            "condition": tensor_delta(entry["baseline_condition"].to(device), cond),
                            "selected_frames": selected_frames, "effective_frames": effective_frames,
                            "state_means": entry["input"]["states"].float().mean(dim=(0, 1, 2)).tolist()}
                report["cache_contracts"].append(contract)
                if not np.isfinite(ce) or contract["ce_abs_difference"] > 2e-5:
                    write_json(output, report)
                    raise RuntimeError("cached boundary failed the fixed 2e-5 CE contract")
                baseline = {"index": record["index"], "asset_id": record["asset_id"], "view": view,
                            "ce": ce, "token_weight": entry["token_weight"]}
                cond = cond.detach()
                del losses
                if split == "valid":
                    baseline["generation"] = generate(model, tokenizer, entry, cond, compact,
                        plan["preflight_prefix_tokens"] if preflight else plan["max_new_tokens"])
                entry["baseline"] = baseline
                entries.append(entry)
                report["baseline"].setdefault(split, []).append(baseline)
                write_json(output, report)
                del cond, compact, batch
            print(f"CACHE {split} index={record['index']} views=3", flush=True)
        caches[split] = entries
    return caches


def evaluate(model, tokenizer, entries, arm, plan, *, preflight=False):
    rows = []
    for entry in entries:
        cond, losses, batch = cached_forward(model, entry, arm)
        ce = float(losses["ce_loss"].detach())
        cond = cond.detach()
        del losses
        row = {"index": entry["index"], "asset_id": entry["asset_id"], "view": entry["view"],
               "token_weight": entry["token_weight"], "ce": ce,
               "generation": generate(model, tokenizer, entry, cond, batch,
                    plan["preflight_prefix_tokens"] if preflight else plan["max_new_tokens"])}
        if preflight:
            row["zero_ce_difference"] = abs(row["ce"] - entry["baseline"]["ce"])
            row["zero_condition"] = tensor_delta(entry["baseline_condition"].to(cond.device), cond)
            ids = row["generation"]["generated_ids"]
            row["same_prefix"] = ids == entry["baseline"]["generation"]["generated_ids"][:len(ids)]
        rows.append(row)
        print(f"EVAL {arm} index={entry['index']} view={entry['view']} ce={ce:.6f}", flush=True)
        del cond, batch
    return rows


def train_arm(model, adapters, entries, arm, plan, report, output, steps):
    parameters = list(adapters.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=plan["lr"], weight_decay=plan["weight_decay"])
    generator = np.random.default_rng(plan["seed"] + 1)
    order = []
    while len(order) < steps * plan["accumulation"]:
        order.extend(generator.permutation(len(entries)).tolist())
    history, samples, frames, tokens = [], 0, 0, 0.0
    for step in range(1, steps + 1):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        chosen = [entries[i] for i in order[(step - 1) * plan["accumulation"]:step * plan["accumulation"]]]
        total_weight = sum(entry["token_weight"] for entry in chosen)
        loss_sum = 0.0
        for entry in chosen:
            cond, losses, batch = cached_forward(model, entry, arm)
            loss = losses["ce_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite adapter loss")
            (loss * (entry["token_weight"] / total_weight)).backward()
            loss_sum += float(loss.detach()) * entry["token_weight"]
            del cond, losses, batch, loss
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("missing/nonfinite adapter gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, plan["clip_grad_norm"], error_if_nonfinite=True)
        group_norms = {name: float(sum(p.grad.float().square().sum() for n, p in adapters.named_parameters()
                       if name in n).sqrt()) for name in ("in_proj", "mix_proj", "out_proj")}
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("nonfinite adapter update")
        samples += len(chosen)
        frames += sum(entry["input"]["tokens"].shape[1] for entry in chosen)
        tokens += total_weight
        row = {"step": step, "ce": loss_sum / total_weight, "gradient_norm": float(gradient_norm),
               "parameter_group_gradient_norms": group_norms, "samples": samples, "frames": frames,
               "target_tokens": tokens, "seconds": time.monotonic() - start,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        history.append(row)
        report["training"][arm] = history
        if step <= 2 or step % 10 == 0 or step == steps:
            write_json(output, report)
            print(f"TRAIN {arm} step={step}/{steps} ce={row['ce']:.6f} samples={samples}", flush=True)
    return optimizer


def main():
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["relation_preflight", "relation_screen"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = config["relation_probe"]
    preflight = args.stage == "relation_preflight"
    guard(config, "train")
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("relation probe requires exactly one recorded GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = Path(plan["checkpoint"])
    source_identity = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                       "mtime_ns": checkpoint.stat().st_mtime_ns, "sha256": file_hash(checkpoint)}
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
        raise ValueError("probe requires dynamic/no-prior base")
    if any(m.p != 0 for m in encoder.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("frozen motion encoder must have zero dropout")
    base_versions = {name: value._version for name, value in model.named_parameters()}
    selected = {split: select_rows(getattr(model_args, manifest), plan[split + "_assets"], plan["seed"])
                for split, manifest in (("train", "train_manifest"), ("valid", "val_manifest"))}
    if set(row["asset_id"] for row in selected["train"]) & set(row["asset_id"] for row in selected["valid"]):
        raise RuntimeError("selected train/validation asset IDs overlap")
    if preflight:
        if plan["preflight_assets_per_split"] != 2:
            raise ValueError("preflight must include exactly the selected min/max target-count assets")
        selected = {split: [min(values, key=lambda row: row["joint_count"]), max(values, key=lambda row: row["joint_count"])]
                    for split, values in selected.items()}
    root = Path(plan["output_root"]) / args.stage
    root.mkdir(parents=True, exist_ok=False)
    report = {"source": source_metadata(config), "manifests": verify_manifests(config), "plan": plan,
              "source_checkpoint": source_identity, "selected": selected, "stage": args.stage,
              "started_at": datetime.now(timezone.utc).isoformat(), "complete": False,
              "cache_contracts": [], "baseline": {}, "training": {}, "evaluation": {}}
    write_json(args.output, report)
    caches = build_cache(model, tokenizer, model_args, selected, plan, report, args.output, preflight=preflight)
    if not preflight:
        torch.save(caches, root / "cached_inputs.pt")
        report["cache_sha256"] = file_hash(root / "cached_inputs.pt")
    torch.manual_seed(plan["seed"])
    encoder.enable_relation_residual(plan["bottleneck_dim"])
    adapters = torch.nn.ModuleList([block.relation_residual for block in encoder.blocks])
    initial = {key: value.detach().cpu().clone() for key, value in adapters.state_dict().items()}
    report["trainable_parameters"] = sum(p.numel() for p in adapters.parameters())
    expected_ids = {id(p) for p in adapters.parameters()}
    if {id(p) for p in model.parameters() if p.requires_grad} != expected_ids:
        raise RuntimeError("only relation adapters may be trainable")
    for arm in ("actual", "unknown"):
        adapters.load_state_dict(initial, strict=True)
        zero = evaluate(model, tokenizer, caches["valid"] if preflight else caches["valid"][:1], arm, plan, preflight=True)
        if not all(row["zero_ce_difference"] <= 2e-5 and row["zero_condition"]["max_abs"] == 0
                   and row["same_prefix"] for row in zero):
            report["zero_failed"] = zero
            write_json(args.output, report)
            raise RuntimeError("zero-init base parity failed")
        report.setdefault("zero_parity", {})[arm] = zero
        optimizer = train_arm(model, adapters, caches["train"], arm, plan, report, args.output,
                              plan["preflight_steps"] if preflight else plan["steps"])
        changed = [key for key, value in adapters.state_dict().items() if not torch.equal(value.cpu(), initial[key])]
        if not changed:
            raise RuntimeError("adapter failed to update")
        report.setdefault("updated_adapter_keys", {})[arm] = changed
        if not preflight:
            adapter_path = root / (arm + "_adapter.pt")
            torch.save({"format": "motion_relation_adapter_v1", "base_checkpoint": source_identity,
                        "plan": plan, "arm": arm, "modules": adapters.state_dict(),
                        "optimizer": optimizer.state_dict(), "steps": plan["steps"]}, adapter_path)
            report.setdefault("adapter_checkpoints", {})[arm] = {"path": str(adapter_path), "sha256": file_hash(adapter_path)}
            report["evaluation"][arm] = evaluate(model, tokenizer, caches["valid"], arm, plan)
            if arm == "actual":
                report["evaluation"]["actual_evidence_ablated"] = evaluate(model, tokenizer, caches["valid"], "unknown", plan)
        del optimizer
        write_json(args.output, report)
    changed_base = [name for name, value in model.named_parameters() if name in base_versions
                    and (value._version != base_versions[name] or value.grad is not None)]
    report["changed_base_parameters"] = changed_base
    if changed_base or checkpoint.stat().st_size != source_identity["bytes"] or checkpoint.stat().st_mtime_ns != source_identity["mtime_ns"]:
        write_json(args.output, report)
        raise RuntimeError("base model/checkpoint was modified")
    report["summary"] = {"base": summarize(report["baseline"]["valid"]),
                         **{arm: summarize(rows) for arm, rows in report["evaluation"].items()}}
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    write_json(args.output, report)
    print(f"COMPLETE {args.stage}", flush=True)


if __name__ == "__main__":
    main()
