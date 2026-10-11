"""Bounded learnability/reuse test of hierarchical skeleton generation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from relation_probe_metrics import file_hash, select_rows
from run_shared_structure_experiment import (
    _checkpoint_identity, build_stage_plan, require_prior_stage, validate_cache_report,
)

from rigweave.dynamic_rig.shared_structure import SharedStructureConfig, SharedStructureRig, SkeletonSequence
from rigweave.dynamic_rig.shared_structure_coordinates import normalize_target_meshes, restore_target_coordinates
from rigweave.dynamic_rig.shared_structure_data import SharedStructureDataset, shared_structure_collate
from rigweave.dynamic_rig.shared_structure_dynamic import (
    SharedStructureDynamicRig, SharedSurfaceSequence, capture_surface_sequence, subset_frame_targets,
)


def validate_data_plan(data: dict[str, Any]) -> None:
    for name in ("pool_frames", "source_pool_frames", "reuse_frames", "frames_min", "frames_max",
                 "train_assets", "valid_assets"):
        if type(data[name]) is not int or data[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 2 <= data["frames_min"] <= data["frames_max"] <= data["source_pool_frames"]:
        raise ValueError("dynamic evidence-frame bounds are invalid")
    if data["source_pool_frames"] + data["reuse_frames"] != data["pool_frames"]:
        raise ValueError("source and excluded-target pools must partition the full pool")


def evidence_indices(data: dict[str, Any], frames: int, *, seed: int | None = None,
                     alternate: bool = False) -> torch.Tensor:
    validate_data_plan(data)
    if not data["frames_min"] <= frames <= data["frames_max"]:
        raise ValueError("evidence frame count is outside the configured dynamic range")
    remaining = torch.arange(1, data["source_pool_frames"], dtype=torch.long)
    if seed is not None:
        remaining = remaining[torch.randperm(len(remaining), generator=torch.Generator().manual_seed(seed))]
    elif alternate:
        remaining = remaining.flip(0)
    return torch.cat((torch.zeros(1, dtype=torch.long), remaining[:frames - 1]))


def target_indices(data: dict[str, Any], source_indices: torch.Tensor) -> torch.Tensor:
    if source_indices.ndim != 1 or source_indices.dtype != torch.long or not source_indices.numel():
        raise ValueError("source indices must be a non-empty long vector")
    if bool(((source_indices < 0) | (source_indices >= data["source_pool_frames"])).any()):
        raise ValueError("excluded target frames must not enter structure generation")
    if source_indices.unique().numel() != source_indices.numel():
        raise ValueError("source indices must be unique")
    return torch.cat((source_indices, torch.arange(data["source_pool_frames"], data["pool_frames"])))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split, view in sorted({(row["split"], row["view"]) for row in rows}):
        group = [row for row in rows if (row["split"], row["view"]) == (split, view)]
        completed = [row for row in group if row["generation"]["success"]]
        identity = [row["generation"]["reuse_identity_mpjpe"] for row in completed
                    if row["generation"].get("reuse_identity_mpjpe") is not None]
        result[f"{split}/{view}"] = {
            "assets": len(group), "natural_eos": len(completed),
            "max_joints_failures": sum(row["generation"]["status"] == "max_joints" for row in group),
            "structure_ce": float(np.mean([row["structure_ce"] for row in group])),
            "xyz_ce": float(np.mean([row["xyz_ce"] for row in group])),
            "teacher_forced_reuse_mpjpe": float(np.mean([row["teacher_forced_reuse_mpjpe"] for row in group])),
            "exact_generated_tree": sum(row["generation"].get("exact_tree", False) for row in group),
            "exact_generated_count": sum(row["generation"].get("count_correct", False) for row in group),
            "generated_reuse_identity_mpjpe_count_matched": None if not identity else float(np.mean(identity)),
            "generated_reuse_identity_assets": len(identity),
            "generated_reuse_unique_joint_coverage_all": float(np.mean([
                row["generation"].get("reuse_unique_coverage", 0.0) for row in group])),
        }
    return result


def _cpu_targets(batch: dict[str, Any]) -> dict[str, Any]:
    keys = ("path", "cls", "target_parents", "joint_mask", "joint_count", "frame_mask",
            "target_joints", "frame_target_joints", "selected_frames", "query_center", "query_scale",
            "target_mesh_centers", "target_mesh_scales", "target_source_centers", "target_source_scales")
    return {key: batch[key].detach().cpu().clone() if isinstance(batch[key], torch.Tensor) else batch[key]
            for key in keys}


def _selected(config: dict[str, Any], *, preflight: bool) -> dict[str, list[dict[str, Any]]]:
    manifests = {item["argument"]: item["path"] for item in config["baseline"]["manifests"]}
    selected = {split: select_rows(manifests[field], config["data"][f"{split}_assets"], config["data"]["seed"])
                for split, field in (("train", "train_manifest"), ("valid", "val_manifest"))}
    if {row["asset_id"] for row in selected["train"]} & {row["asset_id"] for row in selected["valid"]}:
        raise RuntimeError("selected train and validation assets overlap")
    if preflight:
        selected = {split: [min(rows, key=lambda row: row["joint_count"]),
                            max(rows, key=lambda row: row["joint_count"])] for split, rows in selected.items()}
    return selected


def _load_original(config: dict[str, Any]):
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer

    checkpoint = Path(config["baseline"]["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    args = SimpleNamespace(**{**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": checkpoint})
    tokenizer = build_tokenizer(args.tokenizer_config)
    model = _build_dynamic_model(args, tokenizer, torch.device("cuda:0"))
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.eval().requires_grad_(False)
    if model.condition_fusion != "dynamic" or model.branch_prior is not None:
        raise ValueError("source must be the completed dynamic/no-prior model")
    if model.conditioner.motion_encoder.motion_evidence_fusion != "bias":
        raise ValueError("source must retain original bias_h8 evidence")
    if any(block.relation_residual is not None for block in model.conditioner.motion_encoder.blocks):
        raise ValueError("experimental relation residuals are not part of this method test")
    return model, tokenizer, args


def _new_model(original, config: dict[str, Any]) -> SharedStructureDynamicRig:
    torch.manual_seed(config["data"]["seed"])
    backbone = original.transformer.model
    if backbone.get_input_embeddings().embedding_dim != config["architecture"]["dim"]:
        raise ValueError("inherited backbone embedding width differs from the skeleton-token width")
    backbone.requires_grad_(True)
    backbone.get_input_embeddings().requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for name in ("dropout", "attention_dropout", "activation_dropout", "layerdrop"):
            if isinstance(getattr(module, name, None), (int, float)):
                setattr(module, name, 0.0)
    backbone.gradient_checkpointing_disable()
    skeleton = SharedStructureRig(backbone, SharedStructureConfig(**config["architecture"]))
    return SharedStructureDynamicRig(
        original.conditioner, skeleton, register_temporal_mode=config["features"]["register_temporal_mode"],
        freeze_features=config["features"]["freeze_for_feasibility"],
    ).to("cuda:0")


def _optimizer(model: SharedStructureDynamicRig, config: dict[str, Any]):
    backbone_ids = {id(p) for p in model.skeleton.structure.backbone.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids],
         "lr": config["probe"]["lr_new"]},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids],
         "lr": config["probe"]["lr_backbone"]},
    ]
    return torch.optim.AdamW(groups, weight_decay=config["probe"]["weight_decay"])


@torch.no_grad()
def build_cache(original, tokenizer, config, selected, report, report_path):
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset
    from train_dynamic_rig import move_batch

    data = config["data"]
    manifests = {item["argument"]: item["path"] for item in config["baseline"]["manifests"]}
    entries = {"train": [], "valid": []}
    versions = {name: p._version for name, p in original.conditioner.named_parameters()}
    quantizer = SharedStructureConfig(**config["architecture"])
    from rigweave.dynamic_rig.shared_structure import StrictXYZQuantizer
    xyz = StrictXYZQuantizer(quantizer.xyz_bins, quantizer.xyz_min, quantizer.xyz_max)
    for split, field in (("train", "train_manifest"), ("valid", "val_manifest")):
        dataset = SharedStructureDataset(DynamicRigManifestDataset(
            manifests[field], tokenizer, frame_count=data["pool_frames"], random_query=False,
            seed=data["seed"], motion_fps_ratio=data["motion_fps_ratio"],
            motion_vertex_samples=data["motion_vertex_samples"], minimum_random_frames=data["minimum_random_frames"],
        ))
        for row in selected[split]:
            batch = move_batch(shared_structure_collate([dataset[row["index"]]], tokenizer.pad), torch.device("cuda:0"))
            if batch["path"] != [row["path"]] or int(batch["joint_count"][0]) != row["joint_count"]:
                raise RuntimeError("manifest selection changed target identity")
            target = normalize_target_meshes(batch, padding=data["target_mesh_padding"])
            valid = target["frame_mask"][:, :, None] & target["joint_mask"][:, None]
            xyz.encode(target["frame_target_joints"], valid)
            reconstructed = restore_target_coordinates(target["frame_target_joints"],
                target["target_source_centers"], target["target_source_scales"])
            expected = (batch["frame_target_joints"] * batch["query_scale"][:, None, None, None]
                        + batch["query_center"][:, None, None])
            torch.testing.assert_close(reconstructed[valid], expected[valid], atol=1e-5, rtol=1e-5)
            torch.manual_seed(data["seed"] + row["index"])
            refs = original.sample_references(batch)
            source_batch = dict(batch)
            for key in ("frame_vertices", "vertex_normals", "face_normals"):
                source_batch[key] = batch[key][:, :data["source_pool_frames"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source = capture_surface_sequence(original.conditioner, source_batch, refs)
                target_surface = capture_surface_sequence(original.conditioner, target, refs,
                                                          include_motion_geometry=False)
            cached_source = SharedSurfaceSequence.from_record(source.cpu_record()).to("cuda:0")
            for name in ("tokens", "query_points", "query_indices", "dense_points"):
                if not torch.equal(getattr(source, name), getattr(cached_source, name)):
                    raise RuntimeError(f"surface cache changed raw feature field {name}")
            entry = {**row, "split": split, "source": source.cpu_record(),
                     "target": target_surface.cpu_record(), "batch": _cpu_targets(target)}
            entries[split].append(entry)
            report.setdefault("cache_contracts", []).append({
                "split": split, "index": row["index"], "joints": row["joint_count"],
                "selected_frames": target["selected_frames"].cpu().tolist(),
                "source_shape": list(source.tokens.shape), "target_shape": list(target_surface.tokens.shape),
                "raw_surface_cache_exact": True,
                "target_range": [float(target["frame_target_joints"][valid].min()),
                                 float(target["frame_target_joints"][valid].max())],
                "source_coordinate_roundtrip_maxabs": float((reconstructed[valid] - expected[valid]).abs().max()),
            })
            write_json(report_path, report)
            print(f"CACHE {split} index={row['index']} J={row['joint_count']} "
                  f"source={data['source_pool_frames']} target={data['pool_frames']}", flush=True)
            del batch, target, source_batch, source, cached_source, target_surface, reconstructed, expected
    if versions != {name: p._version for name, p in original.conditioner.named_parameters()}:
        raise RuntimeError("feature-cache construction changed source feature parameters")
    return entries


def selected_input(entry, config, source_ids):
    from train_dynamic_rig import move_batch
    target_ids = target_indices(config["data"], source_ids)
    source = SharedSurfaceSequence.from_record(entry["source"]).select_frames(source_ids).to("cuda:0")
    target = SharedSurfaceSequence.from_record(entry["target"]).select_frames(target_ids).to("cuda:0")
    batch = move_batch(subset_frame_targets(entry["batch"], target_ids), torch.device("cuda:0"))
    return source, target, batch


def _finite_gradients(model) -> dict[str, float]:
    groups = {name: 0.0 for name in ("memory", "embedding", "head", "backbone", "positions")}
    for name, parameter in model.skeleton.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"missing or nonfinite gradient: {name}")
        group = ("memory" if name.startswith("memory.") else "positions" if name.startswith("positions.")
                 else "backbone" if name.startswith("structure.backbone.")
                 else "embedding" if name.startswith("structure.embedding.") else "head")
        groups[group] += float(parameter.grad.detach().float().abs().sum())
    if any(value <= 0 for value in groups.values()):
        raise RuntimeError(f"a trainable module received no gradient: {groups}")
    return groups


def _prefix_parity(model, output, batch, config):
    decoder = model.skeleton.structure
    memory = output["shared_memory"]
    forced = decoder.teacher_force(memory, batch["target_parents"], batch["joint_mask"])
    lengths = sorted({1, 2, max(2, forced["input_ids"].shape[1] // 2), forced["input_ids"].shape[1]})
    max_delta = 0.0
    for length in lengths:
        logits, hidden = decoder.forward_prefix(memory, forced["input_ids"][:, :length],
                                                  forced["attention_mask"][:, :length])
        expected = forced["logits"][:, length - 1]
        delta = float((logits[:, -1].float() - expected.float()).abs().max())
        max_delta = max(max_delta, delta)
        torch.testing.assert_close(logits[:, -1], expected, atol=config["preflight"]["bf16_prefix_atol"], rtol=0.01)
        if length > 1:
            torch.testing.assert_close(hidden[:, -1], forced["joint_features"][:, length - 2],
                                       atol=config["preflight"]["bf16_prefix_atol"], rtol=0.01)
    return max_delta


def preflight(model, entries, config, report, report_path):
    feature_versions = {name: p._version for name, p in model.conditioner.named_parameters()}
    model.eval()
    checks = []
    for entry in entries["train"] + entries["valid"]:
        for frames in config["preflight"]["frames"]:
            ids = evidence_indices(config["data"], frames)
            source, target, batch = selected_input(entry, config, ids)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward_surface(source, target, batch)
                if not bool(torch.isfinite(output["loss"])):
                    raise RuntimeError("nonfinite preflight loss")
                parity = _prefix_parity(model, output, batch, config)
                sequence = SkeletonSequence.from_record(output["skeleton_sequence"].cpu_record()).to("cuda:0")
                rebound = model.localize_surface(sequence, target)
                torch.testing.assert_close(output["xyz_logits"], rebound["xyz_logits"], atol=0, rtol=0)
                last_ids = torch.tensor([target.tokens.shape[1] - 1])
                alone = model.localize_surface(sequence, target.select_frames(last_ids))
                torch.testing.assert_close(rebound["xyz_logits"][:, -1:], alone["xyz_logits"],
                                           atol=config["preflight"]["bf16_prefix_atol"], rtol=0.01)
            checks.append({"split": entry["split"], "index": entry["index"], "frames": frames,
                           "target_frames": target.tokens.shape[1], "prefix_maxabs": parity,
                           "loss": float(output["loss"]), "sequence_roundtrip_rebind": True})
            del source, target, batch, output, rebound, alone, sequence
    report["preflight_cases"] = checks
    write_json(report_path, report)
    entry = max(entries["train"], key=lambda value: value["joint_count"])
    source, target, batch = selected_input(entry, config, evidence_indices(config["data"], config["data"]["frames_max"]))
    optimizer = _optimizer(model, config)
    model.train()
    updates = []
    snapshots = {name: p.detach().clone() for name, p in model.skeleton.named_parameters()
                 if name in {"memory.queries", "structure.embedding.weight", "structure.head.weight",
                             "positions.xyz_head.weight", "structure.backbone.decoder.layers.0.self_attn.q_proj.weight"}}
    if len(snapshots) != 5:
        raise RuntimeError("expected five explicit update-check parameters")
    for step in range(config["preflight"]["optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_surface(source, target, batch)
        if step == 0:
            watched = [model.skeleton.memory.queries, model.skeleton.structure.embedding.weight,
                       model.skeleton.structure.backbone.decoder.layers[0].self_attn.q_proj.weight]
            gradients = torch.autograd.grad(output["xyz_loss"], watched, retain_graph=True)
            report["xyz_only_structure_gradients"] = [float(value.detach().abs().sum()) for value in gradients]
            if any(not bool(torch.isfinite(value).all()) or float(value.abs().sum()) <= 0 for value in gradients):
                raise RuntimeError("localization loss does not train the structure representation")
        output["loss"].backward()
        groups = _finite_gradients(model)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["probe"]["clip_grad_norm"],
                                      error_if_nonfinite=True)
        optimizer.step()
        updates.append({"step": step + 1, "loss": float(output["loss"].detach()), "gradients": groups})
    changed = {name: bool((dict(model.skeleton.named_parameters())[name].detach() != value).any())
               for name, value in snapshots.items()}
    if not all(changed.values()):
        raise RuntimeError(f"preflight failed to update a required module: {changed}")
    del output, snapshots
    optimizer.zero_grad(set_to_none=True)
    stress = dict(batch)
    count = config["preflight"]["stress_joints"]
    stress["target_parents"] = torch.arange(-1, count - 1, device="cuda:0")[None]
    stress["joint_mask"] = torch.ones(1, count, dtype=torch.bool, device="cuda:0")
    stress["frame_target_joints"] = torch.zeros(1, target.tokens.shape[1], count, 3, device="cuda:0")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_surface(source, target, stress)
    output["loss"].backward()
    _finite_gradients(model)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if peak >= config["runtime"]["memory_fraction"] * total:
        raise RuntimeError(f"stress exceeded recorded memory fraction: {peak}/{total} GiB")
    if feature_versions != {name: p._version for name, p in model.conditioner.named_parameters()}:
        raise RuntimeError("frozen feature encoder changed")
    if any(p.grad is not None for p in model.conditioner.parameters()):
        raise RuntimeError("frozen feature encoder received gradients")
    report.update(preflight_updates=updates, update_checks=changed, frozen_features_unchanged=True,
                  stress={"source_frames": source.tokens.shape[1], "target_frames": target.tokens.shape[1],
                          "joints": count, "synthetic_targets": True, "allocated_peak_gib": peak,
                          "total_gpu_gib": total})
    del optimizer, output, stress, source, target, batch
    model.zero_grad(set_to_none=True)


def _generation_metrics(result, batch, reuse_frames):
    row = {"success": result["success"], "status": result["status"], "tokens": result["tokens"]}
    if not result["success"]:
        return row
    parents = result["parents"]
    actual = batch["target_parents"][0, batch["joint_mask"][0]].cpu().tolist()
    prediction = result["joints"][0].float().cpu()
    gt = batch["frame_target_joints"][0][:, batch["joint_mask"][0]].float().cpu()
    same_count = len(parents) == len(actual)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
    from report_inactive_joint_capability import unique_matches
    coverages = []
    for frame in range(prediction.shape[0] - reuse_frames, prediction.shape[0]):
        matches = unique_matches(gt[frame].numpy(), actual, prediction[frame].numpy(), parents)
        coverages.append(float(np.mean(matches["covered"])))
    raw = restore_target_coordinates(result["joints"], batch["target_source_centers"],
                                      batch["target_source_scales"])
    row.update(parents=parents, joint_count=len(parents), count_correct=same_count, exact_tree=parents == actual,
               reuse_identity_mpjpe=(float((prediction[-reuse_frames:] - gt[-reuse_frames:]).norm(dim=-1).mean())
                                     if same_count else None),
               reuse_unique_coverage=float(np.mean(coverages)), joints_mesh_local=prediction.tolist(),
               joints_source=raw[0].float().cpu().tolist())
    return row


@torch.no_grad()
def evaluate(model, entries, config, output_root, label, report, report_path):
    model.eval()
    rows, sequences = [], {}
    reuse = config["data"]["reuse_frames"]
    for split, values in entries.items():
        for entry in values:
            views = [(f"T{t}", t, False) for t in config["probe"]["evaluation_frames"]]
            if split == "valid":
                views.append(("T8_alternate", 8, True))
            for view, frames, alternate in views:
                ids = evidence_indices(config["data"], frames, alternate=alternate)
                source, target, batch = selected_input(entry, config, ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_surface(source, target, batch)
                    teacher = model.skeleton.quantizer.decode(output["xyz_logits"].argmax(-1))
                    generated = model.generate_structure_surface(source, joint_limit=config["probe"]["generation_joint_limit"])
                    if generated["success"]:
                        saved = generated["skeleton_sequence"].cpu_record()
                        restored = SkeletonSequence.from_record(saved).to("cuda:0")
                        localized = model.localize_surface(restored, target)
                        generated = {**generated, **localized}
                        sequences[f"{split}/{entry['index']}/{view}"] = saved
                    zero_memory = torch.zeros_like(output["shared_memory"])
                    zero_structure = model.skeleton.structure.teacher_force(zero_memory, batch["target_parents"], batch["joint_mask"])
                    zero_sequence = SkeletonSequence(batch["target_parents"], zero_structure["joint_features"], batch["joint_mask"])
                    zero_xyz = model.localize_surface(zero_sequence, target)["joints"]
                gt = batch["frame_target_joints"]
                valid = batch["joint_mask"][:, None].expand(*gt.shape[:3])
                errors = (teacher - gt).norm(dim=-1)
                zero_errors = (zero_xyz - gt).norm(dim=-1)
                row = {"split": split, "index": entry["index"], "asset_id": entry["asset_id"], "view": view,
                       "source_pool_indices": ids.tolist(), "target_frames": batch["selected_frames"].cpu().tolist(),
                       "gt_parents": batch["target_parents"][0].cpu().tolist(),
                       "gt_joints_mesh_local": gt[0].float().cpu().tolist(),
                       "structure_ce": float(output["structure_loss"]), "xyz_ce": float(output["xyz_loss"]),
                       "teacher_forced_mpjpe": float(errors[valid].mean()),
                       "teacher_forced_reuse_mpjpe": float(errors[:, -reuse:][valid[:, -reuse:]].mean()),
                       "zero_memory_structure_ce": float(zero_structure["loss_per_asset"].mean()),
                       "zero_memory_reuse_mpjpe": float(zero_errors[:, -reuse:][valid[:, -reuse:]].mean()),
                       "generation": _generation_metrics(generated, batch, reuse)}
                rows.append(row)
                print(f"EVAL {label} {split} {entry['index']} {view} structure_ce={row['structure_ce']:.5f} "
                      f"reuse_mpjpe={row['teacher_forced_reuse_mpjpe']:.5f} generation={generated['status']}", flush=True)
                del source, target, batch, output, teacher, generated, zero_memory, zero_structure, zero_sequence, zero_xyz
            report.setdefault("evaluation_progress", {})[label] = {"finished_asset": entry["index"], "rows": len(rows)}
            write_json(report_path, report)
    sequence_path = output_root / f"sequences_{label}.pt"
    torch.save(sequences, sequence_path)
    comparison = []
    for entry in entries["valid"]:
        pair = {row["view"]: row for row in rows if row["split"] == "valid" and row["index"] == entry["index"]}
        a, b = pair["T8"]["generation"], pair["T8_alternate"]["generation"]
        matching = a["success"] and b["success"] and a["parents"] == b["parents"]
        comparison.append({"index": entry["index"], "both_eos": a["success"] and b["success"],
                           "same_parent_sequence": bool(matching),
                           "rebind_identity_difference": (float(np.linalg.norm(
                               np.asarray(a["joints_mesh_local"])[-reuse:] - np.asarray(b["joints_mesh_local"])[-reuse:],
                               axis=-1).mean()) if matching else None)})
    return {"rows": rows, "summary": summarize_rows(rows), "cross_subset_reuse": comparison,
            "sequences": {"path": str(sequence_path), "sha256": file_hash(sequence_path), "count": len(sequences)},
            "coordinate_units": "padded target mesh maximum-half-extent units; not cross-frame-equal XYZ"}


def train_probe(model, entries, config, report, report_path):
    plan = config["probe"]
    if plan["batch_size"] != 1:
        raise ValueError("bounded probe implements exactly one asset per microbatch")
    model.train()
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[plan["lr_new"], plan["lr_backbone"]], total_steps=plan["steps"],
    )
    rng = np.random.default_rng(config["data"]["seed"])
    versions = {name: p._version for name, p in model.conditioner.named_parameters()}
    order = list(rng.permutation(len(entries)))
    cursor = 0
    assets, source_frames, target_frames = 0, 0, 0
    trace = []
    started = time.monotonic()
    for step in range(plan["steps"]):
        optimizer.zero_grad(set_to_none=True)
        losses, structure_losses, xyz_losses = [], [], []
        for microbatch in range(plan["accumulation"]):
            if cursor == len(order):
                order = list(rng.permutation(len(entries)))
                cursor = 0
            entry = entries[order[cursor]]
            cursor += 1
            frames = int(rng.integers(config["data"]["frames_min"], config["data"]["frames_max"] + 1))
            ids = evidence_indices(config["data"], frames, seed=int(rng.integers(0, 2 ** 31)))
            source, target, batch = selected_input(entry, config, ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward_surface(source, target, batch, xyz_weight=plan["xyz_weight"])
            if not bool(torch.isfinite(output["loss"])):
                raise RuntimeError(f"nonfinite loss at step {step} microbatch {microbatch}")
            (output["loss"] / plan["accumulation"]).backward()
            losses.append(float(output["loss"].detach()))
            structure_losses.append(float(output["structure_loss"].detach()))
            xyz_losses.append(float(output["xyz_loss"].detach()))
            assets += 1
            source_frames += frames
            target_frames += target.tokens.shape[1]
            del source, target, batch, output
        gradients = _finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], plan["clip_grad_norm"],
                                             error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        item = {"step": step + 1, "loss": float(np.mean(losses)), "structure_ce": float(np.mean(structure_losses)),
                "xyz_ce": float(np.mean(xyz_losses)), "gradient_norm": float(norm),
                "assets": assets, "source_frame_exposures": source_frames, "target_frame_exposures": target_frames,
                "elapsed_seconds": time.monotonic() - started}
        trace.append(item)
        if (step + 1) % 10 == 0 or step == 0:
            report["training"] = {"trace": trace, "latest_gradient_groups": gradients}
            write_json(report_path, report)
            print(f"TRAIN step={step + 1}/{plan['steps']} loss={item['loss']:.5f} "
                  f"structure={item['structure_ce']:.5f} xyz={item['xyz_ce']:.5f} assets={assets}", flush=True)
    if versions != {name: p._version for name, p in model.conditioner.named_parameters()}:
        raise RuntimeError("probe changed frozen features")
    report["training"] = {"trace": trace, "assets": assets, "source_frame_exposures": source_frames,
                           "target_frame_exposures": target_frames, "frozen_features_unchanged": True,
                           "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "cache", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_data_plan(config["data"])
    stage_plan = build_stage_plan(config, args.stage, args.config)
    if args.output.resolve() != Path(stage_plan["evaluation_path"]).resolve():
        raise ValueError("child output does not match the recorded stage plan")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a previous child evaluation")
    source = source_metadata(config)
    for field in ("range_audit", "coordinate_audit"):
        if file_hash(Path(config["data"][field])) != config["data"][field + "_sha256"]:
            raise RuntimeError(f"pinned coordinate evidence changed: {field}")
    for operation in stage_plan["guard_operations"]:
        guard(config, operation)
    if args.stage != "preflight":
        require_prior_stage(config, args.config, "preflight", source)
    if args.stage == "probe":
        require_prior_stage(config, args.config, "cache", source)
    output_root = Path(stage_plan["output_dir"])
    output_root.mkdir(parents=True, exist_ok=False)
    identity = _checkpoint_identity(config)
    checkpoint = Path(identity["path"])
    stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    if file_hash(checkpoint) != identity["sha256"]:
        raise RuntimeError("source checkpoint checksum changed")
    report = {"schema_version": 1, "experiment": config["experiment"], "stage": args.stage,
              "source": source, "source_checkpoint": identity, "manifests": verify_manifests(config),
              "started_at": datetime.now(timezone.utc).isoformat(), "complete": False, "accepted": False,
              "configuration": config, "scope": "bounded reusable-sequence feasibility; no formal model promotion"}
    write_json(args.output, report)
    try:
        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("requires exactly one recorded GPU")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["memory_fraction"])
        torch.backends.mha.set_fastpath_enabled(False)
        original, tokenizer, _ = _load_original(config)
        selected = _selected(config, preflight=args.stage == "preflight")
        report["selected"] = selected
        if args.stage in {"preflight", "cache"}:
            entries = build_cache(original, tokenizer, config, selected, report, args.output)
        else:
            cache_report = json.loads(Path(build_stage_plan(config, "cache", args.config)["evaluation_path"]).read_text())
            cache_meta = validate_cache_report(cache_report, config)
            payload = torch.load(cache_meta["path"], map_location="cpu", mmap=True, weights_only=False)
            if payload["source_checkpoint"] != identity or payload["selected"] != selected:
                raise RuntimeError("cache identity/selection differs from the configured probe")
            entries = payload["entries"]
        if args.stage == "cache":
            cache_path = output_root / "surface_cache.pt"
            torch.save({"schema_version": 1, "source_checkpoint": identity, "selected": selected,
                        "entries": entries}, cache_path)
            report["cache"] = {"path": str(cache_path), "sha256": file_hash(cache_path),
                               "train_assets": len(entries["train"]), "valid_assets": len(entries["valid"])}
        else:
            model = _new_model(original, config)
            del original
            report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if args.stage == "preflight":
                preflight(model, entries, config, report, args.output)
            else:
                report["initialized"] = evaluate(model, entries, config, output_root, "initialized", report, args.output)
                write_json(args.output, report)
                train_probe(model, entries["train"], config, report, args.output)
                report["trained"] = evaluate(model, entries, config, output_root, "trained", report, args.output)
                checkpoint_path = output_root / "checkpoint_probe.pt"
                torch.save({"schema_version": 1, "skeleton": model.skeleton.state_dict(),
                            "configuration": config, "source_checkpoint": identity,
                            "feature_encoder_frozen": True, "source": source}, checkpoint_path)
                report["checkpoint"] = {"path": str(checkpoint_path), "sha256": file_hash(checkpoint_path)}
                report["method_quality_accepted"] = False
                report["quality_review"] = "Primary review required; complete/accepted below means execution only."
        if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != stat or file_hash(checkpoint) != identity["sha256"]:
            raise RuntimeError("source checkpoint changed during the stage")
        if source_metadata(config) != source:
            raise RuntimeError("source checkout changed while the stage was running")
        report.update(complete=True, accepted=True, source_checkpoint_unchanged=True,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output, report)
        print(f"COMPLETE stage={args.stage} execution_accepted=true", flush=True)
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", complete=False, accepted=False,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    main()
