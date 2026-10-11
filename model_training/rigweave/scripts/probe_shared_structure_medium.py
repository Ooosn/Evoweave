"""Bounded data expansion with strictly held-out poses and reusable skeletons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from probe_shared_structure import _cpu_targets, _finite_gradients, _load_original, _new_model, _optimizer, _prefix_parity
from relation_probe_metrics import file_hash
from run_shared_structure_experiment import (
    _checkpoint_identity, build_stage_plan, require_prior_stage, require_online_preflight, validate_cache_report,
)
from rigweave.dynamic_rig.shared_structure import SkeletonSequence, StrictXYZQuantizer
from rigweave.dynamic_rig.shared_structure_coordinates import normalize_target_meshes, restore_target_coordinates
from rigweave.dynamic_rig.shared_structure_dynamic import (
    SharedSurfaceSequence, capture_surface_sequence, subset_frame_targets,
)
from rigweave.dynamic_rig.shared_structure_medium_data import (
    load_medium_batch, partition_raw_frames, select_source_view, source_bank_indices,
)


TRAIN_BANKS = ("train_a", "train_b")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def validate_medium_config(config: dict[str, Any]) -> None:
    data, plan = config["data"], config["probe"]
    if not config["features"]["freeze_for_feasibility"]:
        raise ValueError("this bounded experiment requires the recorded frozen features")
    if data["pool_frames"] != data["train_frames"] + data["heldout_frames"]:
        raise ValueError("training and held-out frames must partition the original frame pool")
    if not 2 <= data["frames_min"] <= data["frames_max"] <= data["train_frames"]:
        raise ValueError("invalid dynamic evidence-frame bounds")
    if data["heldout_frames"] != 8 or data["source_banks"] != 2:
        raise ValueError("this medium protocol requires two training queries and eight held-out frames")
    if plan["batch_size"] != 1 or plan["steps"] < 1 or plan["accumulation"] < 1:
        raise ValueError("invalid medium-probe optimization budget")
    if plan["steps"] * plan["accumulation"] % data["train_assets"]:
        raise ValueError("the declared exposure budget must contain complete training-asset passes")
    if not 1 <= plan["generation_joint_limit"] <= config["architecture"]["max_joints"]:
        raise ValueError("generation cap exceeds architecture capacity")
    if not 1 <= plan["evaluate_train_assets"] <= data["train_assets"]:
        raise ValueError("invalid training diagnostic subset")
    if not 1 <= plan["detailed_eval_assets"] <= min(plan["evaluate_train_assets"], data["valid_assets"]):
        raise ValueError("invalid detailed multi-view subset")
    if not set(plan["checkpoint_steps"]) <= set(range(1, plan["steps"] + 1)):
        raise ValueError("checkpoint milestones are outside the bounded training schedule")
    if config["runtime"]["probe_script"] != Path(__file__).name:
        raise ValueError("medium probe must be the explicitly recorded entry point")


def selected_rows(config: dict[str, Any], *, preflight: bool = False) -> dict[str, list[dict[str, Any]]]:
    data = config["data"]
    path = Path(data["selection_audit"])
    if file_hash(path) != data["selection_audit_sha256"]:
        raise RuntimeError("selected-asset audit checksum changed")
    audit = json.loads(path.read_text())
    if audit["seed"] != data["seed"]:
        raise RuntimeError("selection seed differs from the pinned audit")
    manifests = {item["argument"]: item for item in config["baseline"]["manifests"]}
    result = {}
    for split, argument in (("train", "train_manifest"), ("valid", "val_manifest")):
        recorded = audit["splits"][split]
        expected = manifests[argument]
        if any(recorded[key] != expected[key] for key in ("path", "rows", "sha256")):
            raise RuntimeError("selected assets are not from the recorded current manifest")
        rows = recorded["selected"]
        if len(rows) != data[f"{split}_assets"] or len({row["asset_id"] for row in rows}) != len(rows):
            raise RuntimeError("selected-asset count or uniqueness is invalid")
        if any(row["frame_count"] != data["pool_frames"] for row in rows):
            raise ValueError("the audited raw frame count differs from this protocol")
        if any(row["joint_count"] > config["probe"]["generation_joint_limit"] for row in rows):
            raise ValueError("a selected asset cannot be represented under the fixed generation cap")
        result[split] = rows
    if {row["asset_id"] for row in result["train"]} & {row["asset_id"] for row in result["valid"]}:
        raise RuntimeError("training and validation asset identities overlap")
    if preflight:
        result = {split: [min(rows, key=lambda row: row["joint_count"]),
                          max(rows, key=lambda row: row["joint_count"])] for split, rows in result.items()}
    return result


def asset_seed(config: dict[str, Any], asset_id: str) -> int:
    key = f"{config['data']['frame_seed']}:{asset_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def evidence_slots(count: int, frames: int, *, seed: int | None = None) -> torch.Tensor:
    if not 1 <= frames <= count:
        raise ValueError("evidence count exceeds the chosen source bank")
    other = torch.arange(1, count, dtype=torch.long)
    if seed is not None:
        other = other[torch.randperm(len(other), generator=torch.Generator().manual_seed(seed))]
    return torch.cat((torch.zeros(1, dtype=torch.long), other[:frames - 1]))


def raw_frame_slots(available: torch.Tensor, requested: list[int]) -> torch.Tensor:
    ids = available.detach().cpu().reshape(-1).tolist()
    if len(set(ids)) != len(ids) or len(set(requested)) != len(requested) or not requested:
        raise ValueError("raw frame IDs must be unique and the requested set nonempty")
    lookup = {int(value): index for index, value in enumerate(ids)}
    if any(int(value) not in lookup for value in requested):
        raise ValueError("requested raw frame is not present in the cache")
    return torch.tensor([lookup[int(value)] for value in requested], dtype=torch.long)


def validate_entry(entry: dict[str, Any]) -> None:
    partition = entry["partition"]
    train = set(partition["train_raw_ids"])
    heldout = set(partition["heldout_raw_ids"])
    all_ids = entry["batch"]["selected_frames"][0].tolist()
    if not train or not heldout or train & heldout or train | heldout != set(all_ids):
        raise ValueError("cache frame partition is not complete and disjoint")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("cache target frames contain duplicates")
    for bank, ids in entry["source_frame_ids"].items():
        expected = heldout if bank == "heldout" else train
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("source bank contains frames from the wrong partition")
        surface = entry["source"][bank]
        if surface["tokens"].shape[1] != len(ids) or surface["dense_points"] is None:
            raise ValueError("source geometry does not match its recorded frame IDs")
    if set(entry["source"]) != {"train_a", "train_b", "heldout"}:
        raise ValueError("cache must have exactly the three recorded source banks")
    if entry["source_frame_ids"]["train_a"][0] == entry["source_frame_ids"]["train_b"][0]:
        raise ValueError("the two training banks must use different query frames")
    if entry["target"]["tokens"].shape[1] != len(all_ids) or entry["target"]["dense_points"] is not None:
        raise ValueError("target cache must contain independent per-frame geometry only")


def select_cached_input(entry, bank: str, frames: int, *, training: bool,
                        seed: int | None = None, device: str = "cuda:0"):
    validate_entry(entry)
    if training and bank not in TRAIN_BANKS:
        raise ValueError("held-out source geometry is forbidden during optimization")
    source_ids = evidence_slots(len(entry["source_frame_ids"][bank]), frames, seed=seed)
    source_raw = [entry["source_frame_ids"][bank][i] for i in source_ids.tolist()]
    partition = entry["partition"]
    train_ids = partition["train_raw_ids"]
    requested = train_ids if training else train_ids[:8] + partition["heldout_raw_ids"]
    if training and (set(source_raw) | set(requested)) & set(partition["heldout_raw_ids"]):
        raise RuntimeError("held-out frames leaked into training input or supervision")
    target_ids = raw_frame_slots(entry["batch"]["selected_frames"], requested)
    source = SharedSurfaceSequence.from_record(entry["source"][bank]).select_frames(source_ids).to(device)
    target = SharedSurfaceSequence.from_record(entry["target"]).select_frames(target_ids).to(device)
    batch = move_batch(subset_frame_targets(entry["batch"], target_ids), torch.device(device))
    metadata = {"bank": bank, "query_raw_frame": source_raw[0], "source_raw_frames": source_raw,
                "target_raw_frames": requested,
                "heldout_target_positions": ([] if training else list(range(8, len(requested))))}
    return source, target, batch, metadata


def online_entries(selected, config):
    result = {}
    for split, rows in selected.items():
        result[split] = []
        for row in rows:
            train, heldout = partition_raw_frames(row.get("frame_count", config["data"]["pool_frames"]),
                asset_seed(config, row["asset_id"]), config["data"]["heldout_frames"])
            train, heldout = train.tolist(), heldout.tolist()
            result[split].append({**row, "split": split,
                "partition": {"train_raw_ids": train, "heldout_raw_ids": heldout},
                "source_frame_ids": {"train_a": train, "train_b": source_bank_indices(train, 1).tolist(),
                                     "heldout": heldout}})
    return result


def prepare_online_batches(entry, config, bank, frames, *, training, seed=None):
    if bank not in (*TRAIN_BANKS, "heldout") or (training and bank not in TRAIN_BANKS):
        raise ValueError("held-out source geometry is forbidden during optimization")
    train = entry["partition"]["train_raw_ids"]
    heldout = entry["partition"]["heldout_raw_ids"]
    ids = entry["source_frame_ids"][bank]
    source_raw = [ids[i] for i in evidence_slots(len(ids), frames, seed=seed).tolist()]
    requested = train if training else train[:8] + heldout
    target_count = config["data"].get("target_frames_per_exposure") if training else None
    if target_count is not None:
        if seed is None or not 1 <= target_count <= len(train):
            raise ValueError("target subsampling requires a seed and a valid frame count")
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed ^ 0x51A7))
        requested = [train[i] for i in order[:target_count].tolist()]
    if training and (set(source_raw) | set(requested)) & set(heldout):
        raise RuntimeError("held-out frames leaked into online training")
    # Only training-eligible frames are materialized during optimization.
    needed = list(dict.fromkeys([train[0], *source_raw, *requested]))
    base = load_medium_batch(entry["path"], needed)
    if int(base["joint_count"][0]) != entry["joint_count"]:
        raise RuntimeError("manifest and raw NPZ joint counts differ")
    source = (load_medium_batch(entry["path"], source_raw) if bank == "heldout"
              else select_source_view(base, source_raw))
    targets = select_source_view(base, requested)
    targets = normalize_target_meshes(targets, padding=config["data"]["target_mesh_padding"], precise_inverse=True)
    reference = select_source_view(base, [train[0]])
    selection = {"bank": bank, "query_raw_frame": source_raw[0], "source_raw_frames": source_raw,
                 "target_raw_frames": requested,
                 "heldout_target_positions": ([] if training else list(range(8, len(requested))))}
    return source, targets, reference, selection


@torch.no_grad()
def select_online_input(original, entry, config, bank, frames, *, training, seed=None, device="cuda:0"):
    source_cpu, target_cpu, reference_cpu, selection = prepare_online_batches(
        entry, config, bank, frames, training=training, seed=seed)
    device = torch.device(device)
    source_batch, target_batch, reference_batch = (
        move_batch(batch, device) for batch in (source_cpu, target_cpu, reference_cpu))
    surface_seed = asset_seed(config, entry["asset_id"])
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        torch.manual_seed(surface_seed)
        target_refs = original.sample_references(reference_batch)
        target = capture_surface_sequence(original.conditioner, target_batch, target_refs,
                                          include_motion_geometry=False)
        torch.manual_seed(surface_seed + {"train_a": 0, "train_b": 1, "heldout": 2}[bank])
        source_refs = original.sample_references(source_batch)
        source = capture_surface_sequence(original.conditioner, source_batch, source_refs)
    batch = move_batch(_cpu_targets(target_cpu), device)
    return source, target, batch, selection


def peak_cpu_gib() -> float | None:
    if not sys.platform.startswith("linux"):
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def check_memory_budget(config) -> None:
    peak = peak_cpu_gib()
    if peak is not None and peak >= .9 * config["runtime"]["host_memory_budget_gib"]:
        raise RuntimeError(f"CPU peak {peak:.2f} GiB exceeded the recorded safety budget")


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def coordinate_roundtrip_report(target_batch, source_batch, *, atol: float = 2e-6) -> dict[str, float]:
    restored = restore_target_coordinates(target_batch["frame_target_joints"],
        target_batch["target_source_centers"], target_batch["target_source_scales"])
    expected = (source_batch["frame_target_joints"].double() * source_batch["query_scale"].double()[:, None, None, None]
                + source_batch["query_center"].double()[:, None, None])
    difference = (restored.double() - expected).abs()
    normalized = difference / target_batch["target_source_scales"].double()[:, :, None, None]
    torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=atol, rtol=0)
    return {"source_coordinate_roundtrip_maxabs": float(difference.max()),
            "mesh_normalized_roundtrip_maxabs": float(normalized.max()),
            "mesh_normalized_roundtrip_atol": atol}


def load_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    path = Path(metadata["shard_path"])
    if path.stat().st_size != metadata["shard_bytes"]:
        raise RuntimeError("cache shard size changed")
    entry = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if any(entry[key] != metadata[key] for key in ("split", "index", "asset_id", "partition", "source_frame_ids")):
        raise RuntimeError("cache shard identity does not match its index")
    validate_entry(entry)
    return entry


def verify_shards(entries) -> dict[str, Any]:
    total = 0
    for rows in entries.values():
        for row in rows:
            path = Path(row["shard_path"])
            if path.stat().st_size != row["shard_bytes"] or file_hash(path) != row["shard_sha256"]:
                raise RuntimeError(f"cache shard changed: {path}")
            total += path.stat().st_size
    return {"shards": sum(map(len, entries.values())), "bytes": total, "all_sha256_verified": True}


@torch.no_grad()
def build_cache(original, config, selected, stage, report, report_path):
    data = config["data"]
    root = Path(config["runtime"]["cache_root"]) / stage
    root.mkdir(parents=True, exist_ok=False)
    quantizer = StrictXYZQuantizer(config["architecture"]["xyz_bins"], -1.0, 1.0)
    versions = {name: p._version for name, p in original.conditioner.named_parameters()}
    entries = {"train": [], "valid": []}
    for split, rows in selected.items():
        for row in rows:
            seed = asset_seed(config, row["asset_id"])
            train_ids, heldout_ids = partition_raw_frames(data["pool_frames"], seed, data["heldout_frames"])
            train_ids, heldout_ids = list(map(int, train_ids)), list(map(int, heldout_ids))
            all_ids = train_ids + heldout_ids
            cpu_batch = load_medium_batch(row["path"], all_ids)
            if int(cpu_batch["joint_count"][0]) != row["joint_count"]:
                raise RuntimeError("manifest and raw NPZ joint counts differ")
            target_cpu = normalize_target_meshes(cpu_batch, padding=data["target_mesh_padding"], precise_inverse=True)
            valid = target_cpu["frame_mask"][:, :, None] & target_cpu["joint_mask"][:, None]
            quantizer.encode(target_cpu["frame_target_joints"], valid)
            target_batch = move_batch(target_cpu, torch.device("cuda:0"))
            query_batch = move_batch(select_source_view(cpu_batch, train_ids), torch.device("cuda:0"))
            torch.manual_seed(seed)
            refs = original.sample_references(query_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target_surface = capture_surface_sequence(original.conditioner, target_batch, refs,
                                                          include_motion_geometry=False)
            entry = {**row, "split": split, "partition": {"train_raw_ids": train_ids, "heldout_raw_ids": heldout_ids},
                     "source_frame_ids": {}, "source": {}, "target": target_surface.cpu_record(),
                     "batch": _cpu_targets(target_cpu), "source_query_affines": {}}
            del target_surface, query_batch, target_batch
            for bank, ids in (("train_a", source_bank_indices(train_ids, 0)),
                              ("train_b", source_bank_indices(train_ids, 1)), ("heldout", heldout_ids)):
                ids = list(map(int, ids))
                source_cpu = (load_medium_batch(row["path"], ids) if bank == "heldout"
                              else select_source_view(cpu_batch, ids))
                source_batch = move_batch(source_cpu, torch.device("cuda:0"))
                torch.manual_seed(seed + (0 if bank == "train_a" else 1 if bank == "train_b" else 2))
                refs = original.sample_references(source_batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    surface = capture_surface_sequence(original.conditioner, source_batch, refs)
                entry["source"][bank] = surface.cpu_record()
                entry["source_frame_ids"][bank] = ids
                entry["source_query_affines"][bank] = {"center": source_cpu["query_center"].tolist(),
                                                        "scale": source_cpu["query_scale"].tolist()}
                del source_cpu, source_batch, surface
            validate_entry(entry)
            shard = root / f"{split}_{row['index']}.pt"
            if shard.exists():
                raise FileExistsError(shard)
            torch.save(entry, shard)
            restored = torch.load(shard, map_location="cpu", mmap=True, weights_only=False)
            for bank in entry["source"]:
                for field in ("tokens", "query_points", "query_indices", "dense_points"):
                    if not torch.equal(entry["source"][bank][field], restored["source"][bank][field]):
                        raise RuntimeError("raw source surface shard changed a feature tensor")
            for field in ("tokens", "query_points", "query_indices"):
                if not torch.equal(entry["target"][field], restored["target"][field]):
                    raise RuntimeError("raw target surface shard changed a feature tensor")
            roundtrip = coordinate_roundtrip_report(target_cpu, cpu_batch)
            meta = {**row, "split": split, "partition": entry["partition"],
                    "source_frame_ids": entry["source_frame_ids"], "shard_path": str(shard),
                    "shard_sha256": file_hash(shard), "shard_bytes": shard.stat().st_size,
                    "tensor_bytes": tensor_bytes(entry)}
            entries[split].append(meta)
            report.setdefault("cache_contracts", []).append({"split": split, "index": row["index"],
                "joint_count": row["joint_count"], "query_frames": {k: v[0] for k, v in entry["source_frame_ids"].items()},
                "heldout_frames": heldout_ids, "train_frames": train_ids, "raw_surface_cache_exact": True,
                "target_range": [float(target_cpu["frame_target_joints"].min()), float(target_cpu["frame_target_joints"].max())],
                **roundtrip,
                "shard_bytes": meta["shard_bytes"], "token_dtype": str(entry["target"]["tokens"].dtype)})
            report["cache_progress"] = {"assets": sum(map(len, entries.values())),
                                        "bytes": sum(r["shard_bytes"] for v in entries.values() for r in v),
                                        "peak_cpu_gib": peak_cpu_gib()}
            check_memory_budget(config)
            write_json(report_path, report)
            print(f"CACHE {split} index={row['index']} J={row['joint_count']} "
                  f"assets={report['cache_progress']['assets']} bytes={meta['shard_bytes']}", flush=True)
            del cpu_batch, target_cpu, entry, restored, refs
    if versions != {name: p._version for name, p in original.conditioner.named_parameters()}:
        raise RuntimeError("cache construction changed the frozen encoder")
    return entries


def evaluation_views(config, ordinal: int, label: str) -> list[tuple[str, str, int]]:
    views = [("T8", "train_a", 8)]
    if label != "initialized" and ordinal < config["probe"]["detailed_eval_assets"]:
        views += [("T2", "train_a", 2), ("T24", "train_a", 24),
                  ("T8_other_query", "train_b", 8), ("T8_heldout_source", "heldout", 8)]
    return views


def generation_metrics(result, batch, heldout_positions, *, padding: float) -> dict[str, Any]:
    row = {"success": result["success"], "status": result["status"],
           "natural_eos": result["success"], "tokens": result["tokens"],
           "count_correct": False, "exact_tree": False,
           "heldout_unique_joint_coverage": 0.0, "heldout_located_edge_recall": 0.0,
           "heldout_stable_identity_coverage_all": 0.0, "heldout_identity_mpjpe": None,
           "heldout_identity_mpjpe_bbox_side": None}
    if not result["success"]:
        return row
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
    from report_inactive_joint_capability import unique_matches
    parents = result["parents"]
    actual = batch["target_parents"][0, batch["joint_mask"][0]].cpu().tolist()
    prediction = result["joints"][0].float().cpu()
    gt = batch["frame_target_joints"][0][:, batch["joint_mask"][0]].float().cpu()
    rows = [unique_matches(gt[i].numpy(), actual, prediction[i].numpy(), parents) for i in heldout_positions]
    edges = np.asarray(actual) >= 0
    coverage = float(np.mean([np.mean(item["covered"]) for item in rows]))
    edge_recall = float(np.mean([np.asarray(item["located_edge_found"])[edges].mean() for item in rows])) if edges.any() else 1.0
    same_count = len(parents) == len(actual)
    if same_count:
        errors = (prediction[heldout_positions] - gt[heldout_positions]).norm(dim=-1)
        row["heldout_identity_mpjpe"] = float(errors.mean())
        row["heldout_identity_mpjpe_bbox_side"] = float(errors.mean()) * padding / 2
        row["heldout_stable_identity_coverage_all"] = float((errors <= .05).all(dim=0).float().mean())
    row.update(parents=parents, joint_count=len(parents), gt_joint_count=len(actual),
               count_correct=same_count, exact_tree=parents == actual,
               absolute_count_error=abs(len(parents) - len(actual)),
               heldout_unique_joint_coverage=coverage, heldout_located_edge_recall=edge_recall)
    return row


def summarize_rows(rows) -> dict[str, Any]:
    result = {}
    for split, view in sorted({(row["split"], row["view"]) for row in rows}):
        group = [row for row in rows if (row["split"], row["view"]) == (split, view)]
        generated = [row["generation"] for row in group]
        identity = [row["heldout_identity_mpjpe"] for row in generated if row["heldout_identity_mpjpe"] is not None]
        result[f"{split}/{view}"] = {
            "assets": len(group), "natural_eos": sum(row["success"] for row in generated),
            "max_joints_failures": sum(row["status"] == "max_joints" for row in generated),
            "exact_generated_count": sum(row["count_correct"] for row in generated),
            "exact_generated_tree": sum(row["exact_tree"] for row in generated),
            "structure_ce": float(np.mean([row["structure_ce"] for row in group])),
            "teacher_forced_train_pose_mpjpe": float(np.mean([row["teacher_train_mpjpe"] for row in group])),
            "teacher_forced_heldout_pose_mpjpe": float(np.mean([row["teacher_heldout_mpjpe"] for row in group])),
            "generated_heldout_identity_mpjpe_count_matched": float(np.mean(identity)) if identity else None,
            "generated_heldout_identity_assets": len(identity),
            "generated_heldout_unique_joint_coverage_all": float(np.mean([row["heldout_unique_joint_coverage"] for row in generated])),
            "generated_heldout_located_edge_recall_all": float(np.mean([row["heldout_located_edge_recall"] for row in generated])),
            "generated_heldout_stable_identity_coverage_all": float(np.mean([row["heldout_stable_identity_coverage_all"] for row in generated])),
        }
    return result


@torch.no_grad()
def evaluate(model, entries, config, output_root, label, report, report_path, *, original=None):
    model.eval()
    output_dir = output_root / f"evaluation_{label}"
    output_dir.mkdir(exist_ok=False)
    rows, artifacts, comparisons = [], [], []
    for split, values in entries.items():
        if split == "train":
            values = values[:config["probe"]["evaluate_train_assets"]]
        for ordinal, meta in enumerate(values):
            entry = meta if original is not None else load_entry(meta)
            saved_views = {}
            for view, bank, frames in evaluation_views(config, ordinal, label):
                if original is not None:
                    source, target, batch, selection = select_online_input(original, entry, config, bank, frames, training=False)
                else:
                    source, target, batch, selection = select_cached_input(entry, bank, frames, training=False)
                heldout = selection["heldout_target_positions"]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_surface(source, target, batch)
                    teacher = model.skeleton.quantizer.decode(output["xyz_logits"].argmax(-1))
                    generated = model.generate_structure_surface(source, joint_limit=config["probe"]["generation_joint_limit"])
                    sequence_record = None
                    if generated["success"]:
                        sequence_record = generated["skeleton_sequence"].cpu_record()
                        restored = SkeletonSequence.from_record(sequence_record).to("cuda:0")
                        localized = model.localize_surface(restored, target)
                        generated = {**generated, **localized}
                        del restored, localized
                    zero_metrics = {}
                    if view == "T8" and ordinal < config["probe"]["detailed_eval_assets"]:
                        zero_structure = model.skeleton.structure.teacher_force(torch.zeros_like(output["shared_memory"]),
                            batch["target_parents"], batch["joint_mask"])
                        zero_sequence = SkeletonSequence(batch["target_parents"], zero_structure["joint_features"], batch["joint_mask"])
                        zero_xyz = model.localize_surface(zero_sequence, target)["joints"]
                        zero_metrics = {"structure_ce": float(zero_structure["loss_per_asset"].mean()),
                            "heldout_mpjpe": float((zero_xyz[:, heldout] - batch["frame_target_joints"][:, heldout]).norm(dim=-1).mean())}
                        del zero_structure, zero_sequence, zero_xyz
                errors = (teacher - batch["frame_target_joints"]).norm(dim=-1)
                metrics = generation_metrics(generated, batch, heldout, padding=config["data"]["target_mesh_padding"])
                row = {"split": split, "index": entry["index"], "asset_id": entry["asset_id"],
                       "integration_source": entry["integration_source"], "view": view, "selection": selection,
                       "structure_ce": float(output["structure_loss"]), "xyz_ce": float(output["xyz_loss"]),
                       "teacher_train_mpjpe": float(errors[:, :8].mean()),
                       "teacher_heldout_mpjpe": float(errors[:, heldout].mean()),
                       "zero_memory": zero_metrics, "generation": metrics}
                rows.append(row)
                saved_views[view] = {"skeleton_sequence": sequence_record,
                    "generation": metrics, "selection": selection,
                    "gt_parents": batch["target_parents"].cpu(),
                    "gt_joints_mesh_local": batch["frame_target_joints"].cpu(),
                    "pred_joints_mesh_local": generated["joints"].float().cpu() if generated["success"] else None,
                    "teacher_joints_mesh_local": teacher.float().cpu(),
                    "target_source_centers": batch["target_source_centers"].cpu(),
                    "target_source_scales": batch["target_source_scales"].cpu()}
                print(f"EVAL {label} {split} {entry['index']} {view} "
                      f"structure_ce={row['structure_ce']:.5f} heldout_mpjpe={row['teacher_heldout_mpjpe']:.5f} "
                      f"generation={generated['status']} exact_tree={metrics['exact_tree']}", flush=True)
                del source, target, batch, output, teacher, generated, errors
            for other in ("T8_other_query", "T8_heldout_source"):
                if other not in saved_views:
                    continue
                a, b = saved_views["T8"], saved_views[other]
                ga, gb = a["generation"], b["generation"]
                same = ga["success"] and gb["success"] and ga["parents"] == gb["parents"]
                difference = (float((a["pred_joints_mesh_local"][:, 8:] - b["pred_joints_mesh_local"][:, 8:]).norm(dim=-1).mean())
                              if same else None)
                comparisons.append({"split": split, "index": entry["index"], "other_view": other,
                    "same_parent_sequence": bool(same), "both_gt_tree_correct": ga["exact_tree"] and gb["exact_tree"],
                    "same_target_identity_prediction_difference": difference,
                    "source_frame_overlap": sorted(set(a["selection"]["source_raw_frames"]) & set(b["selection"]["source_raw_frames"]))})
            artifact = output_dir / f"{split}_{entry['index']}.pt"
            torch.save({"asset_id": entry["asset_id"], "partition": entry["partition"], "views": saved_views}, artifact)
            artifacts.append({"path": str(artifact), "sha256": file_hash(artifact), "views": list(saved_views)})
            report["evaluation_progress"] = {"label": label, "assets": len(artifacts), "rows": len(rows),
                                               "latest_asset": entry["asset_id"], "summary_so_far": summarize_rows(rows)}
            write_json(report_path, report)
            check_memory_budget(config)
            del saved_views, entry
    return {"rows": rows, "summary": summarize_rows(rows), "cross_pose_sequence_comparisons": comparisons,
            "artifacts": artifacts, "coordinate_units": "padded mesh half-extent; bbox-side fraction = value*padding/2",
            "summary_by_integration": {name: summarize_rows([r for r in rows if r["integration_source"] == name])
                                       for name in sorted({r["integration_source"] for r in rows})},
            "heldout_contract": "heldout raw frames were never structure inputs or XYZ supervision during THIS probe; same action only; initialization may have seen base-asset frames"}


def preflight(model, entries, config, report, report_path):
    feature_versions = {name: p._version for name, p in model.conditioner.named_parameters()}
    model.eval()
    checks = []
    for meta in entries["train"] + entries["valid"]:
        entry = load_entry(meta)
        cases = [("train_a", frames, True) for frames in config["preflight"]["frames"]]
        cases += [("train_b", 8, False), ("heldout", 8, False)]
        for bank, frames, training in cases:
            source, target, batch, selection = select_cached_input(entry, bank, frames, training=training)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward_surface(source, target, batch)
                if not bool(torch.isfinite(output["loss"])):
                    raise RuntimeError("nonfinite preflight loss")
                parity = _prefix_parity(model, output, batch, config)
                sequence = SkeletonSequence.from_record(output["skeleton_sequence"].cpu_record()).to("cuda:0")
                rebound = model.localize_surface(sequence, target)
                torch.testing.assert_close(output["xyz_logits"], rebound["xyz_logits"], atol=0, rtol=0)
                alone = model.localize_surface(sequence, target.select_frames(torch.tensor([target.tokens.shape[1] - 1])))
                torch.testing.assert_close(rebound["xyz_logits"][:, -1:], alone["xyz_logits"],
                                           atol=config["preflight"]["bf16_prefix_atol"], rtol=.01)
            checks.append({"split": entry["split"], "index": entry["index"], "bank": bank,
                "frames": frames, "target_frames": target.tokens.shape[1], "selection": selection,
                "prefix_maxabs": parity, "loss": float(output["loss"]), "sequence_roundtrip_rebind": True})
            del source, target, batch, output, sequence, rebound, alone
        del entry
    report["preflight_cases"] = checks
    write_json(report_path, report)
    meta = max(entries["train"], key=lambda row: row["joint_count"])
    entry = load_entry(meta)
    source, target, batch, selection = select_cached_input(entry, "train_a", config["data"]["frames_max"], training=True)
    optimizer = _optimizer(model, config)
    model.train()
    watched_names = {"memory.queries", "structure.embedding.weight", "structure.head.weight",
                     "positions.xyz_head.weight", "structure.backbone.decoder.layers.0.self_attn.q_proj.weight"}
    snapshots = {name: p.detach().clone() for name, p in model.skeleton.named_parameters() if name in watched_names}
    if len(snapshots) != len(watched_names):
        raise RuntimeError("preflight cannot identify all five update-check parameters")
    updates = []
    started = time.monotonic()
    for step in range(config["preflight"]["optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_surface(source, target, batch)
        if step == 0:
            watched = [model.skeleton.memory.queries, model.skeleton.structure.embedding.weight,
                       model.skeleton.structure.backbone.decoder.layers[0].self_attn.q_proj.weight]
            gradients = torch.autograd.grad(output["xyz_loss"], watched, retain_graph=True)
            if any(not bool(torch.isfinite(value).all()) or float(value.abs().sum()) <= 0 for value in gradients):
                raise RuntimeError("XYZ supervision does not train the shared skeleton representation")
            report["xyz_only_structure_gradients"] = [float(value.detach().abs().sum()) for value in gradients]
            del gradients
        output["loss"].backward()
        groups = _finite_gradients(model)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      config["probe"]["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        updates.append({"step": step + 1, "loss": float(output["loss"].detach()), "gradients": groups})
    torch.cuda.synchronize()
    changed = {name: bool((dict(model.skeleton.named_parameters())[name].detach() != value).any())
               for name, value in snapshots.items()}
    if not all(changed.values()):
        raise RuntimeError("preflight did not update every required trainable component")
    report["preflight_optimizer_seconds"] = time.monotonic() - started
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
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if peak >= config["runtime"]["memory_fraction"] * total:
        raise RuntimeError("GPU stress exceeded the recorded memory safety budget")
    if feature_versions != {name: p._version for name, p in model.conditioner.named_parameters()}:
        raise RuntimeError("frozen feature parameters changed during preflight")
    if any(p.grad is not None for p in model.conditioner.parameters()):
        raise RuntimeError("the frozen feature encoder received gradients")
    report.update(preflight_updates=updates, update_checks=changed, frozen_features_unchanged=True,
        stress={"source_frames": source.tokens.shape[1], "target_frames": target.tokens.shape[1], "joints": count,
                "synthetic_targets": True, "allocated_peak_gib": peak, "total_gpu_gib": total},
        peak_cpu_gib=peak_cpu_gib())
    del optimizer, output, stress, source, target, batch, entry
    model.zero_grad(set_to_none=True)


def save_training_checkpoint(model, optimizer, scheduler, config, source, output_root, step, sampler, training):
    path = output_root / f"checkpoint_step_{step:05d}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save({"schema_version": 1, "skeleton": model.skeleton.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step,
        "sampler": sampler, "configuration": config, "source": source,
        "source_checkpoint": _checkpoint_identity(config), "feature_encoder_frozen": True,
        "training_accounting": training, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate()}, path)
    return {"path": str(path), "sha256": file_hash(path), "step": step,
            "contains_optimizer_and_sampler": True}


def train_probe(model, entries, config, source_identity, output_root, report, report_path, *, original=None):
    plan = config["probe"]
    model.train()
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[plan["lr_new"], plan["lr_backbone"]],
                                                   total_steps=plan["steps"])
    rng = np.random.default_rng(config["data"]["seed"])
    versions = {name: p._version for name, p in model.conditioner.named_parameters()}
    order, cursor = list(map(int, rng.permutation(len(entries)))), 0
    exposure_counts = {str(row["index"]): 0 for row in entries}
    bank_counts = {name: 0 for name in TRAIN_BANKS}
    assets, source_frames, target_frames = 0, 0, 0
    trace, checkpoints = [], []
    started = time.monotonic()
    gradients = {}
    watched_names = {"memory.queries", "structure.embedding.weight", "structure.head.weight",
                     "positions.xyz_head.weight", "structure.backbone.decoder.layers.0.self_attn.q_proj.weight"}
    snapshots = {name: p.detach().clone() for name, p in model.skeleton.named_parameters()
                 if original is not None and name in watched_names}
    for step in range(1, plan["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        losses, structures, positions = [], [], []
        for _ in range(plan["accumulation"]):
            if cursor == len(order):
                order, cursor = list(map(int, rng.permutation(len(entries)))), 0
            meta = entries[order[cursor]]
            cursor += 1
            entry = meta if original is not None else load_entry(meta)
            bank = TRAIN_BANKS[int(rng.integers(0, len(TRAIN_BANKS)))]
            frames = int(rng.integers(config["data"]["frames_min"], config["data"]["frames_max"] + 1))
            view_seed = int(rng.integers(0, 2 ** 31))
            if original is not None:
                source, target, batch, selection = select_online_input(original, entry, config, bank, frames,
                                                                      training=True, seed=view_seed)
            else:
                source, target, batch, selection = select_cached_input(entry, bank, frames, training=True, seed=view_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward_surface(source, target, batch, xyz_weight=plan["xyz_weight"])
            if not bool(torch.isfinite(output["loss"])):
                raise RuntimeError(f"nonfinite training loss at update {step}")
            if original is not None and assets == 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    parity = _prefix_parity(model, output, batch, config)
                watched = [model.skeleton.memory.queries, model.skeleton.structure.embedding.weight,
                           model.skeleton.structure.backbone.decoder.layers[0].self_attn.q_proj.weight]
                xyz_gradients = torch.autograd.grad(output["xyz_loss"], watched, retain_graph=True)
                if any(not bool(torch.isfinite(g).all()) or float(g.abs().sum()) <= 0 for g in xyz_gradients):
                    raise RuntimeError("online XYZ supervision does not train the skeleton representation")
                report["online_startup"] = {"prefix_maxabs": parity, "selection": selection,
                    "xyz_only_structure_gradients": [float(g.abs().sum()) for g in xyz_gradients]}
                del xyz_gradients
            (output["loss"] / plan["accumulation"]).backward()
            losses.append(float(output["loss"].detach()))
            structures.append(float(output["structure_loss"].detach()))
            positions.append(float(output["xyz_loss"].detach()))
            assets += 1
            exposure_counts[str(meta["index"])] += 1
            bank_counts[bank] += 1
            source_frames += frames
            target_frames += len(selection["target_raw_frames"])
            del source, target, batch, output, entry
        if step == 1 or step % plan["gradient_audit_every"] == 0 or step == plan["steps"]:
            gradients = _finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              plan["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if original is not None and step == 1:
            changed = {name: bool((dict(model.skeleton.named_parameters())[name].detach() != value).any())
                       for name, value in snapshots.items()}
            if len(changed) != 5 or not all(changed.values()):
                raise RuntimeError("online startup did not update all five monitored components")
            if any(p.grad is not None for p in model.conditioner.parameters()):
                raise RuntimeError("online feature extraction produced encoder gradients")
            report["online_startup"].update(update_checks=changed, accepted=True,
                                           feature_mode="online_no_disk_cache")
            del snapshots
        item = {"step": step, "loss": float(np.mean(losses)), "structure_ce": float(np.mean(structures)),
                "xyz_ce": float(np.mean(positions)), "gradient_norm": float(norm), "assets": assets,
                "source_frame_exposures": source_frames, "target_frame_exposures": target_frames,
                "elapsed_seconds": time.monotonic() - started}
        trace.append(item)
        training = {"trace": trace, "assets": assets, "asset_exposure_counts": exposure_counts,
                    "bank_exposure_counts": bank_counts, "source_frame_exposures": source_frames,
                    "target_frame_exposures": target_frames, "heldout_frame_exposures": 0,
                    "latest_gradient_groups": gradients, "peak_cpu_gib": peak_cpu_gib(),
                    "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        if step in plan["checkpoint_steps"]:
            checkpoints.append(save_training_checkpoint(model, optimizer, scheduler, config, source_identity,
                output_root, step, {"order": order, "cursor": cursor, "numpy_rng": rng.bit_generator.state}, training))
            report["checkpoints"] = checkpoints
        if step == 1 or step % plan["log_every"] == 0 or step == plan["steps"]:
            check_memory_budget(config)
            report["training"] = training
            write_json(report_path, report)
            print(f"TRAIN step={step}/{plan['steps']} loss={item['loss']:.5f} "
                  f"structure={item['structure_ce']:.5f} xyz={item['xyz_ce']:.5f} "
                  f"assets={assets} seconds={item['elapsed_seconds']:.1f}", flush=True)
    if versions != {name: p._version for name, p in model.conditioner.named_parameters()}:
        raise RuntimeError("medium probe changed frozen feature parameters")
    if len(set(exposure_counts.values())) != 1:
        raise RuntimeError("completed budget did not expose every selected asset equally")
    training["frozen_features_unchanged"] = True
    report["training"] = training
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "cache", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_medium_config(config)
    stage_plan = build_stage_plan(config, args.stage, args.config)
    if args.output.resolve() != Path(stage_plan["evaluation_path"]).resolve() or args.output.exists():
        raise ValueError("child output is not the unused recorded evaluation path")
    source = source_metadata(config)
    for field in ("selection_audit", "coordinate_audit"):
        if file_hash(config["data"][field]) != config["data"][field + "_sha256"]:
            raise RuntimeError(f"pinned preparation evidence changed: {field}")
    for operation in stage_plan["guard_operations"]:
        guard(config, operation)
    online = config["runtime"].get("feature_mode", "cached") == "online"
    if online:
        preflight_reference = require_online_preflight(config)
    elif args.stage != "preflight":
        require_prior_stage(config, args.config, "preflight", source)
    if args.stage == "probe" and not online:
        require_prior_stage(config, args.config, "cache", source)
    output_root = Path(stage_plan["output_dir"])
    output_root.mkdir(parents=True, exist_ok=False)
    identity = _checkpoint_identity(config)
    checkpoint = Path(identity["path"])
    stat = checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns
    if file_hash(checkpoint) != identity["sha256"]:
        raise RuntimeError("source checkpoint checksum changed")
    report = {"schema_version": 1, "experiment": config["experiment"], "stage": args.stage,
        "source": source, "source_checkpoint": identity, "manifests": verify_manifests(config),
        "started_at": datetime.now(timezone.utc).isoformat(), "complete": False, "accepted": False,
        "configuration": config, "scope": "medium frozen-feature expansion, not formal full retraining or cross-action proof"}
    write_json(args.output, report)
    try:
        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("requires exactly the one recorded physical GPU")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["memory_fraction"])
        torch.backends.mha.set_fastpath_enabled(False)
        original, _, _ = _load_original(config)
        selected = selected_rows(config, preflight=args.stage == "preflight")
        report["selected"] = selected
        if online:
            entries = online_entries(selected, config)
            report.update(feature_mode="online_no_disk_cache", preflight_reference=preflight_reference,
                          initialized_evaluation="deferred; training starts before generation diagnostics")
        elif args.stage in {"preflight", "cache"}:
            entries = build_cache(original, config, selected, args.stage, report, args.output)
        else:
            prior = json.loads(Path(build_stage_plan(config, "cache", args.config)["evaluation_path"]).read_text())
            cache_meta = validate_cache_report(prior, config)
            payload = torch.load(cache_meta["path"], map_location="cpu", weights_only=False)
            if payload["source_checkpoint"] != identity or payload["selected"] != selected:
                raise RuntimeError("cache source or asset selection differs from this probe")
            entries = payload["entries"]
            report["cache_verification_before"] = verify_shards(entries)
            del payload
        if args.stage == "cache":
            cache_path = output_root / "surface_cache.pt"
            torch.save({"schema_version": 1, "storage": "per_asset_shards_on_verified_node_local_ssd",
                "source_checkpoint": identity, "selected": selected, "entries": entries}, cache_path)
            report["cache"] = {"path": str(cache_path), "sha256": file_hash(cache_path),
                               "train_assets": len(entries["train"]), "valid_assets": len(entries["valid"]),
                               "shards": verify_shards(entries)}
        else:
            initialization = {**config, "data": {**config["data"], "seed": config["data"]["model_seed"]}}
            model = _new_model(original, initialization)
            if not online:
                del original
            report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if args.stage == "preflight":
                preflight(model, entries, config, report, args.output)
            else:
                if not online:
                    report["initialized"] = evaluate(model, entries, config, output_root, "initialized", report, args.output)
                    write_json(args.output, report)
                train_probe(model, entries["train"], config, source, output_root, report, args.output,
                            original=original if online else None)
                report["trained"] = evaluate(model, entries, config, output_root, "trained", report, args.output,
                                             original=original if online else None)
                if online:
                    write_json(args.output, report)
                    del model, original
                    original, _, _ = _load_original(config)
                    model = _new_model(original, initialization)
                    report["initialized"] = evaluate(model, entries, config, output_root, "initialized", report,
                                                     args.output, original=original)
                    report["initialized_evaluation"] = "completed after training from the original checkpoint and model seed"
                if not online:
                    report["cache_verification_after"] = verify_shards(entries)
                report["checkpoint"] = report["checkpoints"][-1]
                report["method_quality_accepted"] = False
                report["quality_review"] = "Primary review required; complete/accepted means execution only."
        if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != stat or file_hash(checkpoint) != identity["sha256"]:
            raise RuntimeError("source checkpoint changed during this stage")
        if source_metadata(config) != source:
            raise RuntimeError("source checkout changed while the stage was running")
        check_memory_budget(config)
        report.update(complete=True, accepted=True, source_checkpoint_unchanged=True,
                      peak_cpu_gib=peak_cpu_gib(), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output, report)
        print(f"COMPLETE stage={args.stage} execution_accepted=true", flush=True)
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", complete=False, accepted=False,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    main()
