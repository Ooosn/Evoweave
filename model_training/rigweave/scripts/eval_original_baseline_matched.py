"""Evaluate the immutable July single-query checkpoint on the current valid panel.

Stage authorization belongs to original_baseline_eval_guard. Importing this module
does not load vendor code, checkpoints, datasets, or initialize CUDA.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


FAMILY = "original_single_query_matched_eval_v1"
CHECKPOINT_SHA = "4cdde3cac8b8b83d690128785065bcde6b6186044bcbdeac6f701acff3b7bb55"
CHECKPOINT_BYTES = 7995123574
MODEL_YAML_SHA = "5baa90902a4ae8ef64693df612646df149543d410290f5a48b1db0442111f511"
TOKENIZER_YAML_SHA = "20ad7eb9b9dfec8a3d2943e8c28223adbc16da3bdb65bd94aa38d92527f51109"
SLOTS = (0, 2, 4, 6)
REPO = Path(__file__).resolve().parents[3]
FRAME_FIELDS = (
    "frame_vertices", "vertex_normals", "face_normals", "frame_target_joints",
    "selected_frames", "frame_mask", "frame_numbers", "frame_mesh_bbox_side",
    "root_rt_rotations", "root_rt_translations",
)
REQUIRED_FIELDS = FRAME_FIELDS + (
    "target_joints", "target_parents", "joint_count", "joint_mask", "faces",
    "vertex_count", "face_count", "query_center", "query_scale",
    "root_rt_augmentation", "root_rt_root_index", "root_rt_relative_nonrigid",
    "asset_id", "path", "cls", "sampling_seed", "heldout_raw_ids", "frame_partition",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest}


def expand_local_artifacts(raw_config):
    if "local_artifacts" in raw_config and "local_artifacts_manifest" in raw_config:
        raise ValueError("inline and external local artifact mappings cannot both be specified")
    config = deepcopy(raw_config)
    if "local_artifacts_manifest" not in config:
        return config, None
    spec = config["local_artifacts_manifest"]
    if not isinstance(spec, dict) or not {"path", "sha256"} <= set(spec):
        raise ValueError("local_artifacts_manifest requires a path and pinned sha256")
    path = Path(spec["path"]).resolve(strict=True)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("local artifact manifest changed while reading")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError("local artifact manifest raw SHA256 differs from the pinned value")
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get("complete") is not True
            or not isinstance(payload.get("local_artifacts"), dict)):
        raise ValueError("local artifact manifest must be complete with a local_artifacts dictionary")
    config["local_artifacts"] = deepcopy(payload["local_artifacts"])
    return config, {"path": str(path), "bytes": len(raw), "mtime_ns": before.st_mtime_ns,
                    "sha256": digest}


def write_json(path, value):
    # Each report has one owner. Existing output is never replaced or resumed.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def context_budget(max_positions, max_new_tokens, condition_tokens=1024, prefix_tokens=2):
    values = (max_positions, max_new_tokens, condition_tokens, prefix_tokens)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("context dimensions must be positive integers")
    total = condition_tokens + prefix_tokens + max_new_tokens
    if total > max_positions:
        raise ValueError(f"generation context {total} exceeds decoder capacity {max_positions}")
    return {"condition_tokens": condition_tokens, "prefix_tokens": prefix_tokens,
            "max_new_tokens": max_new_tokens, "reserved_positions": total,
            "max_position_embeddings": max_positions, "fits": True}


def memory_fraction(runtime):
    fraction = runtime.get("gpu_memory_fraction_limit", runtime.get("cuda_memory_fraction", .85))
    if ("gpu_memory_fraction_limit" in runtime and "cuda_memory_fraction" in runtime
            and runtime["gpu_memory_fraction_limit"] != runtime["cuda_memory_fraction"]):
        raise ValueError("conflicting GPU memory fraction settings")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction <= .85:
        raise ValueError("CUDA memory fraction must be positive and at most 0.85")
    return fraction


def local_artifact(original, artifacts, verified):
    if original not in artifacts:
        raise ValueError(f"local_artifacts lacks original path: {original}")
    spec = artifacts[original]
    if (not isinstance(spec, dict) or not {"path", "sha256", "bytes"} <= set(spec)
            or not Path(spec["path"]).is_absolute() or type(spec["bytes"]) is not int
            or spec["bytes"] < 0 or not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64):
        raise ValueError(f"invalid local artifact specification: {original}")
    if original not in verified:
        identity = file_identity(spec["path"])
        if identity["bytes"] != spec["bytes"] or identity["sha256"] != spec["sha256"]:
            raise ValueError(f"local artifact bytes/hash differs: {original}")
        verified[original] = {"original_path": original, **identity}
    elif (Path(verified[original]["path"]) != Path(spec["path"]).resolve()
          or verified[original]["sha256"] != spec["sha256"] or verified[original]["bytes"] != spec["bytes"]):
        raise ValueError(f"artifact mapping changed during evaluation: {original}")
    stat = Path(verified[original]["path"]).stat()
    if (stat.st_size != verified[original]["bytes"]
            or stat.st_mtime_ns != verified[original]["mtime_ns"]):
        raise RuntimeError(f"local artifact size/mtime changed: {original}")
    return verified[original]["path"]


def relocate_panel_config(panel, artifacts, verified):
    relocated = deepcopy(panel)
    specifications = [relocated["data"]["manifests"][split] for split in ("train", "valid")]
    specifications.extend(relocated["reference_evaluation"]["files"].values())
    for spec in specifications:
        original = spec["path"]
        spec["path"] = local_artifact(original, artifacts, verified)
        if spec["sha256"] != verified[original]["sha256"]:
            raise ValueError(f"local artifact differs from the pinned panel's original hash: {original}")
    return relocated


def matched_dataset(valid_original, panel_config, config, verified):
    from virtual_query_xyz_runtime import fixed_dataset
    resolver = None
    if "local_artifacts" in config:
        resolver = lambda original: local_artifact(original, config["local_artifacts"], verified)
    return fixed_dataset(valid_original, panel_config, 8, mode="seen_eval", file_resolver=resolver)


def validate_config(config, panel):
    if config.get("family") != FAMILY:
        raise ValueError("unexpected evaluator family")
    expected = {"input_frames": 8, "frame_slots": list(SLOTS), "valid_assets": 16,
                "max_new_tokens": 2048, "seed": 20261022}
    if config.get("evaluation") != expected:
        raise ValueError("evaluation must be the exact 16-asset, four-slot panel")
    checkpoint = config["checkpoint"]
    for key, value in {"sha256": CHECKPOINT_SHA, "bytes": CHECKPOINT_BYTES,
                       "step": 1667, "sample_seen": 80016}.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"original checkpoint {key} differs from the approved identity")
    if not Path(checkpoint["path"]).is_absolute():
        raise ValueError("checkpoint path must be absolute")
    evaluation = panel["evaluation"]
    for key, value in {"input_frames": 8, "frame_indices": list(SLOTS),
                       "valid_assets": 16, "seed": 20261022, "max_new_tokens": 2048}.items():
        if evaluation.get(key) != value:
            raise ValueError(f"pinned panel {key} changed")
    data = panel["data"]
    if (data.get("normalization") != "input_sequence_mesh_bbox"
            or data.get("global_rt_removal") is not True
            or data.get("coordinate_clipping") is not False):
        raise ValueError("pinned panel normalization/RT contract changed")
    if config.get("reference_B65") != panel["reference_evaluation"]:
        raise ValueError("reference_B65 must equal the pinned panel reference_evaluation")
    runtime = config["runtime"]
    if Path(runtime["repo"]).resolve() != REPO.resolve():
        raise ValueError("runtime.repo is not this evaluator's checkout")
    if Path(runtime["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("runtime.python differs from the executing interpreter")
    memory_fraction(runtime)


def derive_cpu_yaml(original_path, output_path):
    import yaml
    original = yaml.safe_load(Path(original_path).read_text(encoding="utf-8"))
    if original["llm"].get("_attn_implementation") != "flash_attention_2":
        raise ValueError("CPU override requires the recorded FlashAttention2 source YAML")
    derived = deepcopy(original)
    derived["llm"]["_attn_implementation"] = "sdpa"
    with Path(output_path).open("x", encoding="utf-8") as handle:
        yaml.safe_dump(derived, handle, sort_keys=False)
    reloaded = yaml.safe_load(Path(output_path).read_text(encoding="utf-8"))
    restored = deepcopy(reloaded)
    restored["llm"]["_attn_implementation"] = original["llm"]["_attn_implementation"]
    if restored != original or reloaded != derived:
        raise ValueError("derived CPU YAML changed more than the attention backend")
    return {"original": file_identity(original_path), "derived": file_identity(output_path),
            "sole_override": {"llm._attn_implementation": {
                "from": "flash_attention_2", "to": "sdpa"}},
            "all_other_fields_recursively_equal": True, "scope": "CPU audit construction only"}


def construction_backend(runtime, mode, flash_available, platform=None):
    if mode == "audit":
        return {"backend": "sdpa", "cpu_only_override": True,
                "mechanism": "sole attention override in a separate CPU audit YAML"}
    requested = runtime.get("attention_backend", "flash_attention_2")
    platform = os.name if platform is None else platform
    if requested == "flash_attention_2" and flash_available:
        return {"backend": requested, "cpu_only_override": False, "mechanism": "original saved YAML"}
    if requested == "sdpa" and platform == "nt" and not flash_available:
        return {"backend": requested, "cpu_only_override": False,
                "mechanism": "explicit Windows SDPA fallback in load_unirig; original saved YAML retained",
                "authorization_field": "runtime.attention_backend=sdpa", "flash_attn_available": False}
    raise ValueError("requested generation backend unavailable or not explicitly authorized for this platform")


def checkpoint_settings(saved_args, defaults, runtime, checkpoint, artifacts=None, verified=None):
    settings = {**defaults, **saved_args, "checkpoint": Path(checkpoint), "frames": 8}
    paths = {}
    for key in ("model_config", "tokenizer_config"):
        saved = str(saved_args[key])
        effective = (local_artifact(saved, artifacts, verified) if artifacts is not None
                     else str(runtime.get(key, saved)))
        if not Path(effective).is_absolute():
            raise ValueError(f"provide an absolute local runtime.{key} when relocating checkpoint assets")
        settings[key] = Path(effective)
        paths[key] = {"saved": saved, "effective": effective}
    settings["unirig_checkpoint"] = None
    paths["unirig_checkpoint"] = {"saved": str(saved_args["unirig_checkpoint"]),
                                 "effective": None, "loaded": False}
    return settings, paths


def strict_restore(model, state):
    if any(value.device.type != "cpu" for value in model.state_dict().values()):
        raise ValueError("strict restoration must finish on CPU before a GPU move")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("strict restoration returned incompatible keys")
    loaded = model.state_dict()
    if set(loaded) != set(state):
        raise ValueError("restored checkpoint key set changed")
    for key, value in state.items():
        if (loaded[key].dtype != value.dtype or loaded[key].shape != value.shape
                or not torch.equal(loaded[key], value)):
            raise ValueError(f"restored tensor differs from checkpoint: {key}")
    return {"strict": True, "missing_keys": [], "unexpected_keys": [],
            "all_tensors_equal": True, "tensors": len(state),
            "elements_including_aliases": sum(value.numel() for value in state.values()),
            "model_parameters": sum(value.numel() for value in model.parameters())}


def geometry_digest(batch):
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(memoryview(array).cast("B"))
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def validate_geometry(batch):
    missing = set(REQUIRED_FIELDS) - set(batch)
    if missing:
        raise ValueError(f"geometry missing fields: {sorted(missing)}")
    vertices, targets = batch["frame_vertices"], batch["frame_target_joints"]
    if vertices.ndim != 4 or vertices.shape[:2] != (1, 8) or vertices.shape[-1] != 3:
        raise ValueError("geometry must contain one complete eight-frame window")
    if targets.ndim != 4 or targets.shape[:2] != (1, 8) or targets.shape[-1] != 3:
        raise ValueError("GT must contain one complete eight-frame window")
    count, vertex_count = targets.shape[2], vertices.shape[2]
    faces = batch["faces"]
    if faces.ndim != 3 or faces.shape[0] != 1 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [1,F,3]")
    face_count = faces.shape[1]
    shapes = {"vertex_normals": vertices.shape, "face_normals": (1, 8, face_count, 3),
              "target_joints": (1, count, 3), "target_parents": (1, count),
              "joint_mask": (1, count), "selected_frames": (1, 8), "frame_mask": (1, 8),
              "frame_numbers": (1, 8), "frame_mesh_bbox_side": (1, 8),
              "query_center": (1, 3), "query_scale": (1,),
              "root_rt_rotations": (1, 8, 3, 3), "root_rt_translations": (1, 8, 3),
              "root_rt_augmentation": (1, 3, 3), "root_rt_root_index": (1,),
              "root_rt_relative_nonrigid": (1,), "joint_count": (1,),
              "vertex_count": (1,), "face_count": (1,)}
    for key, shape in shapes.items():
        if not isinstance(batch[key], torch.Tensor) or tuple(batch[key].shape) != tuple(shape):
            raise ValueError(f"geometry field has wrong shape: {key}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite geometry field: {key}")
    if (count < 1 or vertex_count < 1 or face_count < 1
            or int(batch["joint_count"][0]) != count
            or int(batch["vertex_count"][0]) != vertex_count
            or int(batch["face_count"][0]) != face_count):
        raise ValueError("geometry counts disagree with unpadded arrays")
    for key in ("selected_frames", "frame_numbers", "target_parents", "faces"):
        if batch[key].dtype != torch.long:
            raise ValueError(f"geometry integer field must be int64: {key}")
    if (batch["frame_mask"].dtype != torch.bool or batch["joint_mask"].dtype != torch.bool
            or not bool(batch["frame_mask"].all()) or not bool(batch["joint_mask"].all())):
        raise ValueError("geometry must retain all frames and GT joints")
    raw_ids = batch["selected_frames"][0].tolist()
    if min(raw_ids) < 0 or len(set(raw_ids)) != 8 or set(raw_ids) & set(batch["heldout_raw_ids"]):
        raise ValueError("selected raw frames are duplicated, invalid, or held out")
    if batch["frame_partition"] != "seen_eval":
        raise ValueError("validation must use the current seen_eval frame partition")
    if not torch.equal(batch["target_joints"], targets[:, 0]):
        raise ValueError("target_joints must equal current query slot zero")
    if float(batch["query_scale"][0]) <= 0 or float(targets.abs().max()) > 1:
        raise ValueError("normalization scale or GT bounds changed")
    if int(faces.min()) < 0 or int(faces.max()) >= vertex_count:
        raise ValueError("faces index outside the original mesh")
    parents = batch["target_parents"][0].tolist()
    if parents[0] != -1 or any(parent < 0 or parent >= index for index, parent in enumerate(parents[1:], 1)):
        raise ValueError("GT parent list must preserve the complete source-order tree")
    if (not isinstance(batch["asset_id"], str) or not batch["asset_id"]
            or len(batch["path"]) != 1 or len(batch["cls"]) != 1
            or type(batch["sampling_seed"]) is not int or batch["sampling_seed"] < 0):
        raise ValueError("geometry identity or sampling seed is invalid")


def verify_permutation(original, query, slot):
    validate_geometry(original)
    validate_geometry(query)
    order = [slot] + [index for index in range(8) if index != slot]
    if set(query) != set(original):
        raise ValueError("query permutation changed the geometry field set")
    for key, value in original.items():
        expected = (value[:, order] if key in FRAME_FIELDS else
                    original["frame_target_joints"][:, slot] if key == "target_joints" else value)
        actual = query[key]
        if isinstance(expected, torch.Tensor):
            equal = (isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                     and actual.device == expected.device and torch.equal(actual, expected))
        else:
            equal = actual == expected
        if not equal:
            raise ValueError(f"query permutation drift: {key}")
    return order


def permute_query(batch, slot):
    if type(slot) is not int or slot not in SLOTS:
        raise ValueError("query slot must be one of 0,2,4,6")
    validate_geometry(batch)
    order = [slot] + [index for index in range(8) if index != slot]
    query = dict(batch)
    for key in FRAME_FIELDS:
        query[key] = batch[key][:, order].clone()
    query["target_joints"] = query["frame_target_joints"][:, 0].clone()
    verify_permutation(batch, query, slot)
    return query, order


def verify_reference(batch, target_tokens, reference):
    validate_geometry(batch)
    if len(target_tokens) != 8:
        raise ValueError("reference audit needs all eight GT token sequences")
    for frame, tokens in enumerate(target_tokens):
        key = ("valid", batch["asset_id"], frame)
        if key not in reference:
            raise ValueError(f"reference missing original window row: {key}")
        row = reference[key]
        if int(row["raw_frame_index"]) != int(batch["selected_frames"][0, frame]):
            raise ValueError(f"reference raw_frame_index differs: {key}")
        if row["target_tokens"] != list(tokens):
            raise ValueError(f"reference target_tokens differ: {key}")
        value = row["original_edge_f1"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid reference F1: {key}")


def validate_coverage(rows, asset_ids, slots=SLOTS):
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("duplicate expected assets")
    expected = {(identity, slot) for identity in asset_ids for slot in slots}
    actual = [(row["asset_id"], row["frame_index"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("missing, duplicated, or unexpected evaluation views")


def summarize(rows, asset_ids):
    validate_coverage(rows, asset_ids)
    assets = []
    for identity in asset_ids:
        views = [row for row in rows if row["asset_id"] == identity]
        counts = {row["gt_joint_count"] for row in views}
        if len(counts) != 1:
            raise ValueError("GT count changed across an asset's frames")
        for row in views:
            score = row["original_edge_f1"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid original-GT F1")
            if not row["success"] and (score != 0 or row["structure_signature"] is not None):
                raise ValueError("failed generations must score zero and have no valid signature")
        signatures = [row["structure_signature"] if row["success"] else None for row in views]
        pair_equal = [left is not None and right is not None and left == right
                      for left, right in itertools.combinations(signatures, 2)]
        assets.append({"asset_id": identity, "gt_joint_count": counts.pop(), "views": 4,
                       "original_edge_f1": float(np.mean([row["original_edge_f1"] for row in views])),
                       "reference_B65_f1": float(np.mean([row["reference_B65_f1"] for row in views])),
                       "all_four_same_parent_and_count": all(pair_equal),
                       "pairwise_same_parent_and_count": sum(pair_equal) / 6})
    def group(selected):
        identities = {row["asset_id"] for row in selected}
        views = [row for row in rows if row["asset_id"] in identities]
        if not selected:
            return {"assets": 0, "views": 0, "original_edge_f1": None}
        return {"assets": len(selected), "views": len(views),
                "original_edge_f1": float(np.mean([row["original_edge_f1"] for row in selected])),
                "reference_B65_f1": float(np.mean([row["reference_B65_f1"] for row in selected])),
                "success_fraction": sum(row["success"] for row in views) / len(views),
                "parse_failures": sum(not row["parse_ok"] for row in views),
                "eos_failures": sum(not row["has_eos"] for row in views),
                "max_token_failures": sum(row["status"] == "max_tokens" for row in views),
                "failed_generations": sum(not row["success"] for row in views),
                "exact_count_fraction": sum(row["exact_count"] for row in views) / len(views),
                "exact_parent_list_fraction": sum(row["exact_parent_list"] for row in views) / len(views),
                "all_four_same_parent_and_count": sum(row["all_four_same_parent_and_count"] for row in selected) / len(selected),
                "pairwise_same_parent_and_count": float(np.mean([row["pairwise_same_parent_and_count"] for row in selected]))}
    return {"all": group(assets), "gt52": group([row for row in assets if row["gt_joint_count"] == 52]),
            "non52": group([row for row in assets if row["gt_joint_count"] != 52]), "per_asset": assets,
            "aggregation": "mean of per-asset means over exactly four views; failures zero",
            "consistency_denominator": "all assets and all six frame pairs; failed views never match"}


def source_identities(runtime, saved_args):
    root = REPO / "model_training" / "rigweave"
    scripts = ("eval_original_baseline_matched.py", "original_baseline_eval_guard.py",
               "eval_dynamic_rig_ce.py", "eval_dynamic_rig_generation.py", "train_dynamic_rig.py",
               "virtual_query_xyz_runtime.py", "structure_xyz_fill_runtime.py", "fusion_evaluation.py",
               "probe_virtual_query_lambda.py", "virtual_fusion_normal_runtime.py", "virtual_frame_runtime.py",
               "structure_xyz_fill_initialization.py", "train_legacy_single_target.py")
    paths = [root / "scripts" / name for name in scripts]
    paths.extend((root / "src" / "rigweave" / "dynamic_rig").rglob("*.py"))
    vendor = Path(runtime["unirig_root"])
    paths.extend((vendor / "src").rglob("*.py"))
    paths.extend(Path(saved_args[key]) for key in ("model_config", "tokenizer_config"))
    paths.extend((vendor / "configs").rglob("*.yaml"))
    paths.extend(Path(runtime["opt_config_root"]).glob("*.json"))
    records = {str(path.resolve()): file_identity(path) for path in sorted(set(paths))}
    for name, digest in runtime.get("source_hashes", {}).items():
        path = Path(name) if Path(name).is_absolute() else REPO / name
        if sha256(path) != digest:
            raise ValueError(f"pinned source changed: {name}")
    return records


def assert_files_unchanged(records, checkpoint):
    for identity in records.values():
        stat = Path(identity["path"]).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise RuntimeError(f"source/config/reference size/mtime changed: {identity['path']}")
        if sha256(identity["path"]) != identity["sha256"]:
            raise RuntimeError(f"source/config/reference changed during evaluation: {identity['path']}")
    stat = Path(checkpoint["path"]).stat()
    if stat.st_size != checkpoint["bytes"] or stat.st_mtime_ns != checkpoint["mtime_ns"]:
        raise RuntimeError("source checkpoint changed during evaluation")


def load_all_valid_reference(panel_config, asset_ids):
    result = {}
    for rank in (0, 1):
        path = panel_config["reference_evaluation"]["files"][f"rank_{rank}"]["path"]
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in report["groups"]["valid"]["rows"]:
            if row["asset_id"] not in asset_ids:
                continue
            key = ("valid", row["asset_id"], row["frame_index"])
            if key in result:
                raise ValueError("duplicate original reference row")
            result[key] = row
    expected = {("valid", identity, frame) for identity in asset_ids for frame in range(8)}
    if set(result) != expected:
        raise ValueError("reference cannot establish every selected eight-frame window")
    return result


def prepare_cpu(config, panel_config, mode, rank, world, output, verified_artifacts=None):
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer
    from virtual_query_xyz_runtime import read_entries
    from structure_xyz_fill_runtime import load_small_reference_panel
    from rigweave.dynamic_rig.data import _make_token_input

    checkpoint = file_identity(config["checkpoint"]["path"])
    if checkpoint["sha256"] != CHECKPOINT_SHA or checkpoint["bytes"] != CHECKPOINT_BYTES:
        raise ValueError("original July checkpoint bytes/hash mismatch")
    payload = torch.load(checkpoint["path"], map_location="cpu", mmap=True, weights_only=False)
    if payload["step"] != 1667 or payload["sample_seen"] != 80016 or payload["args"]["frames"] != 24:
        raise ValueError("checkpoint training identity differs from July sample80000")
    state = payload["model"]
    if len(state) != 1117 or sum(value.numel() for value in state.values()) != 728319023:
        raise ValueError("checkpoint tensor inventory differs from the approved CPU inspection")
    settings, asset_paths = checkpoint_settings(payload["args"], CHECKPOINT_DEFAULTS,
        config["runtime"], checkpoint["path"], config.get("local_artifacts"), verified_artifacts)
    for key, expected in (("model_config", MODEL_YAML_SHA), ("tokenizer_config", TOKENIZER_YAML_SHA)):
        if sha256(settings[key]) != expected:
            raise ValueError(f"original checkpoint {key} YAML changed")
    sources = source_identities(config["runtime"], settings)
    construction = construction_backend(config["runtime"], mode, importlib.util.find_spec("flash_attn") is not None)
    construction["original_yaml"] = file_identity(settings["model_config"])
    if mode == "audit":
        derived = output / f"rank_{rank}_model_cpu_sdpa.yaml"
        construction.update(derive_cpu_yaml(settings["model_config"], derived))
        settings["model_config"] = derived
    tokenizer = build_tokenizer(Path(settings["tokenizer_config"]))
    if tokenizer.vocab_size != 267 or tokenizer.num_discrete != 256 or list(tokenizer.continuous_range) != [-1, 1]:
        raise ValueError("original 267-token coordinate vocabulary changed")
    model = _build_dynamic_model(SimpleNamespace(**settings), tokenizer, torch.device("cpu"),
        strict_checkpoint=True, initialize_from_official=False, checkpoint_payload=payload)
    restoration = strict_restore(model, state)
    restoration.update(initialize_from_official=False, verified_mmap_payload_reused=True)
    if restoration["model_parameters"] != 673925679:
        raise ValueError("restored model parameter count differs from the verified original")
    del payload, state
    actual_backend = model.transformer.config._attn_implementation
    if actual_backend != construction["backend"]:
        raise ValueError(f"unapproved decoder construction backend: {actual_backend}")
    construction["verified_decoder_backend"] = actual_backend
    if (model.condition_fusion != "dynamic" or model.branch_prior is not None
            or model.explicit_tree_decoder is not None or model.use_grammar_state_embedding
            or model.uses_action_group_bias
            or getattr(model.conditioner.motion_encoder, "motion_evidence_fusion", "off") != "off"
            or getattr(model.conditioner.motion_encoder, "frame_role_policy", "query_first") != "query_first"):
        raise ValueError("model is not the original real-query-only generation path")
    if (model.transformer.get_input_embeddings().weight.shape[0] != 267
            or model.transformer.get_output_embeddings().weight.shape[0] != 267):
        raise ValueError("decoder vocabulary differs from original 267 tokens")
    budget = context_budget(int(model.transformer.config.max_position_embeddings), 2048)
    # Import the actual vendor processor on CPU. Annotation/import defects must
    # stop the stage rather than select another decoder or grammar implementation.
    from src.model.unirig_ar import VocabSwitchingLogitsProcessor
    vendor_module = sys.modules[VocabSwitchingLogitsProcessor.__module__]
    vendor_path = Path(vendor_module.__file__).resolve()
    if not vendor_path.is_relative_to(Path(config["runtime"]["unirig_root"]).resolve()):
        raise ValueError("generation grammar was imported from a different vendor tree")
    entries = read_entries(panel_config)
    panel, _, manifest = load_small_reference_panel(panel_config, entries)
    valid = panel["valid"]
    if len(valid) != 16 or len({row["asset_id"] for row in valid}) != 16:
        raise ValueError("current panel must contain sixteen distinct valid assets")
    selected = valid[:2] if mode == "smoke" else valid
    asset_ids = [row["asset_id"] for row in selected]
    reference = load_all_valid_reference(panel_config, {row["asset_id"] for row in valid})
    dataset = matched_dataset(valid, panel_config, config, verified_artifacts)
    audits, work = [], []
    for index, entry in enumerate(selected):
        if index % world != rank:
            continue
        batch = dataset[index]
        validate_geometry(batch)
        if batch["asset_id"] != entry["asset_id"] or batch["path"] != [entry["path"]]:
            raise ValueError("dataset asset/path differs from the pinned manifest row")
        parents = batch["target_parents"][0].tolist()
        parent_input = [None if value < 0 else value for value in parents]
        tokens, roundtrips = [], []
        for frame in range(8):
            ids = _make_token_input(batch["frame_target_joints"][0, frame].numpy(), parent_input,
                                    batch["cls"][0], tokenizer, tails=None).astype(np.int64).tolist()
            if ids[:2] != [int(tokenizer.bos), int(tokenizer.cls_name_to_token(batch["cls"][0]))] or ids[-1] != tokenizer.eos:
                raise ValueError("GT tokenization lacks the original BOS/class/EOS contract")
            decoded = tokenizer.detokenize(np.asarray(ids, dtype=np.int64))
            decoded_parents = parsed_parents(decoded)
            if len(decoded_parents) != len(parents):
                raise ValueError("GT token roundtrip changed joint count")
            tokens.append(ids)
            roundtrips.append({"frame_index": frame, "tokens": len(ids), "parsed": True,
                               "exact_source_parents": decoded_parents == parents})
        verify_reference(batch, tokens, reference)
        for slot in SLOTS:
            query, _ = permute_query(batch, slot)
            del query
        audit = {"asset_id": batch["asset_id"], "path": batch["path"][0],
                 "resolved_path": batch.get("resolved_path"), "asset_index": index,
                 "selected_raw_frames": batch["selected_frames"][0].tolist(),
                 "frame_numbers": batch["frame_numbers"][0].tolist(),
                 "query_center": batch["query_center"][0].tolist(), "query_scale": float(batch["query_scale"][0]),
                 "sampling_seed": batch["sampling_seed"], "gt_roundtrips": roundtrips,
                 "geometry_sha256": geometry_digest(batch),
                 "all_eight_reference_raw_frames_and_gt_tokens_match": True,
                 "all_four_query_permutations_verified": True}
        audits.append(audit)
        work.append((index, tokens, audit))
        del batch
    return model, tokenizer, reference, dataset, work, {
        "checkpoint": checkpoint, "checkpoint_asset_paths": asset_paths,
        "strict_restoration": restoration, "construction": construction,
        "sources": sources, "context_budget": budget, "panel": manifest, "asset_ids": asset_ids,
        "audited_assets": audits, "inference_change": {"trained_input_frames": 24, "evaluation_input_frames": 8,
            "normalization": "current root-RT / selected-eight-input mesh-union bbox; July historical chart not replayed"},
        "grammar_source": str(vendor_path), "quality_approval": False}


def parsed_parents(output):
    parents = getattr(output, "parents", None)
    if parents is None:
        parents = output._get_parents()
    return [-1 if value is None else int(value) for value in parents]


def score_generation(generated, tokenizer, batch, slot, permutation, reference_f1, metric):
    generated = np.asarray(generated, dtype=np.int64)
    prefix = [int(tokenizer.bos), int(tokenizer.cls_name_to_token(batch["cls"][0]))]
    if generated.ndim != 1 or generated[:2].tolist() != prefix or len(generated) > 2050:
        raise RuntimeError("generation output violated the prefix/token budget contract")
    if bool(((generated < 0) | (generated >= 267)).any()):
        raise RuntimeError("generation returned IDs outside the original vocabulary")
    has_eos = bool(np.any(generated[2:] == int(tokenizer.eos)))
    parsed, error = None, None
    try:
        output = tokenizer.detokenize(generated)
        joints = np.asarray(output.joints, dtype=np.float32)
        parents = parsed_parents(output)
        if joints.shape != (len(parents), 3) or not len(parents) or not np.isfinite(joints).all():
            raise ValueError("detokenized skeleton has invalid/nonfinite geometry")
        if parents.count(-1) != 1 or parents[0] != -1 or any(parent < 0 or parent >= index for index, parent in enumerate(parents[1:], 1)):
            raise ValueError("detokenized parents do not form a source-order tree")
        parsed = joints, parents
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = has_eos and parsed is not None
    gt = batch["frame_target_joints"][0, slot].numpy()
    gt_parents = batch["target_parents"][0].tolist()
    score = float(metric(parsed[0], parsed[1], gt, gt_parents)["edge_f1"]) if success else 0.
    status = "eos" if success else "missing_eos" if not has_eos else "parse_failure"
    if not has_eos and len(generated) == 2050:
        status = "max_tokens"
    return {"asset_id": batch["asset_id"], "path": batch["path"][0],
            "resolved_path": batch.get("resolved_path"), "group": "valid",
            "frame_index": slot, "original_slot": slot, "query_slot": 0,
            "raw_frame_index": int(batch["selected_frames"][0, slot]),
            "selected_raw_frames": batch["selected_frames"][0].tolist(), "permutation": permutation,
            "permuted_raw_frames": batch["selected_frames"][0, permutation].tolist(),
            "sampling_seed": batch["sampling_seed"], "success": success, "has_eos": has_eos,
            "parse_ok": parsed is not None, "status": status, "parse_error": error,
            "generated_tokens": generated.tolist(), "new_tokens": len(generated) - 2,
            "predicted_joints": parsed[0].tolist() if parsed else None,
            "predicted_parents": parsed[1] if parsed else None,
            "original_target_joints": gt.tolist(), "original_target_parents": gt_parents,
            "gt_joint_count": len(gt_parents), "original_edge_f1": score,
            "exact_count": success and len(parsed[1]) == len(gt_parents),
            "exact_parent_list": success and parsed[1] == gt_parents,
            "structure_signature": parsed[1] if success else None, "reference_B65_f1": reference_f1}


def generate_rows(model, tokenizer, reference, dataset, work, config, local_rank):
    from eval_dynamic_rig_ce import _move_dynamic_model_to_device
    from eval_dynamic_rig_generation import _dynamic_generate
    from train_dynamic_rig import move_batch
    from rigweave.dynamic_rig.sampling import sample_trackable_surface
    from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(memory_fraction(config["runtime"]), device)
    _move_dynamic_model_to_device(model, device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, targets, audit in work:
            batch = dataset[index]
            validate_geometry(batch)
            if geometry_digest(batch) != audit["geometry_sha256"]:
                raise RuntimeError("geometry changed between the CPU audit and inference")
            for slot in SLOTS:
                query, order = permute_query(batch, slot)
                query = move_batch(query, device)
                generator = torch.Generator(device=device).manual_seed(batch["sampling_seed"])
                refs = sample_trackable_surface(query["frame_vertices"][:, 0], query["faces"],
                    num_samples=model.num_surface_samples, vertex_samples=model.vertex_samples,
                    query_tokens=model.query_tokens, vertex_counts=query["vertex_count"],
                    face_counts=query["face_count"], generator=generator)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    condition = model.build_condition(query, refs=refs)
                    context_budget(int(model.transformer.config.max_position_embeddings), 2048,
                                   condition_tokens=condition.shape[1])
                    tokens = _dynamic_generate(model, tokenizer, query, 2048, precomputed_cond=condition,
                        generation_kwargs={"do_sample": False, "num_beams": 1, "use_cache": True,
                                           "forced_eos_token_id": None, "forced_bos_token_id": None,
                                           "repetition_penalty": 1., "no_repeat_ngram_size": 0,
                                           "min_length": 0, "min_new_tokens": 0})
                row = score_generation(tokens, tokenizer, batch, slot, order,
                    reference[("valid", batch["asset_id"], slot)]["original_edge_f1"], _topology_metrics)
                row.update(target_tokens=targets[slot], query_center=audit["query_center"],
                           query_scale=audit["query_scale"], geometry_sha256=audit["geometry_sha256"])
                rows.append(row)
                del query, refs, condition
            print(json.dumps({"asset_complete": batch["asset_id"], "rank": os.environ.get("RANK", "0")}), flush=True)
            del batch
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "smoke", "evaluate"), required=True)
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    from original_baseline_eval_guard import verify_stage
    source_revision = verify_stage(args, raw_config)
    config, artifact_manifest_identity = expand_local_artifacts(raw_config)
    panel_spec = config["panel_config"]
    if sha256(panel_spec["path"]) != panel_spec["sha256"]:
        raise ValueError("pinned panel configuration changed")
    panel_config = json.loads(Path(panel_spec["path"]).read_text(encoding="utf-8"))
    validate_config(config, panel_config)
    verified_artifacts = {}
    if "local_artifacts" in config:
        panel_config = relocate_panel_config(panel_config, config["local_artifacts"], verified_artifacts)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world not in (1, 2) or not 0 <= rank < world or local_rank != rank:
        raise ValueError("only one or two local workers with one CUDA device per rank are supported")
    if "local_artifacts" in config and (world != 1 or rank != 0 or config["runtime"].get("device_index") != 0):
        raise ValueError("local artifact evaluation requires world size one and runtime.device_index=0")
    torch.set_num_threads(1)
    runtime = config["runtime"]
    os.environ["EVOWEAVE_UNIRIG_ROOT"] = runtime["unirig_root"]
    os.environ["EVOWEAVE_OPT_CONFIG_ROOT"] = runtime["opt_config_root"]
    os.environ["RIGWEAVE_DISABLE_LIGHTNING_IMPORT"] = "1"
    os.environ["RIGWEAVE_DISABLE_OPEN3D"] = "1"
    if args.mode == "audit":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in (REPO / "model_training" / "rigweave" / "src", Path(runtime["extra_pythonpath"]), Path(runtime["unirig_root"])):
        sys.path.insert(0, str(path))
    distributed = torch.distributed
    if world == 2:
        distributed.init_process_group("gloo", timeout=timedelta(minutes=30))
    try:
        creation = [None]
        if rank == 0:
            try:
                args.output.mkdir(parents=True, exist_ok=False)
            except Exception as error:
                creation[0] = f"{type(error).__name__}: {error}"
        if world == 2:
            distributed.broadcast_object_list(creation, src=0)
        if creation[0] is not None:
            raise RuntimeError(f"output must be fresh: {creation[0]}")
        files = {str(args.config.resolve()): file_identity(args.config),
                 str(Path(panel_spec["path"]).resolve()): file_identity(panel_spec["path"])}
        if artifact_manifest_identity is not None:
            files[artifact_manifest_identity["path"]] = artifact_manifest_identity
        for spec in (*panel_config["data"]["manifests"].values(),
                     *panel_config["reference_evaluation"]["files"].values()):
            identity = file_identity(spec["path"])
            if identity["sha256"] != spec["sha256"]:
                raise ValueError(f"pinned input identity changed: {spec['path']}")
            files[identity["path"]] = identity
        prepared, failure = None, None
        try:
            prepared = prepare_cpu(config, panel_config, args.mode, rank, world, args.output, verified_artifacts)
        except Exception as error:
            failure = {"rank": rank, "error": f"{type(error).__name__}: {error}"}
        failures = [failure]
        if world == 2:
            failures = [None] * world
            distributed.all_gather_object(failures, failure)
        if any(failures):
            write_json(args.output / f"failure_rank_{rank}.json", {"complete": False, "mode": args.mode,
                "phase": "CPU strict restore/panel audit", "failures": failures, "quality_approval": False})
            raise RuntimeError(f"CPU gate failed: {failures}")
        model, tokenizer, reference, dataset, work, metadata = prepared
        files.update(metadata["sources"])
        files.update({record["path"]: record for record in verified_artifacts.values()})
        metadata["local_artifacts_verified"] = verified_artifacts
        local_ids = [audit["asset_id"] for _, _, audit in work]
        write_json(args.output / f"cpu_audit_rank_{rank}.json", metadata)
        failure = None
        try:
            rows = [] if args.mode == "audit" else generate_rows(model, tokenizer, reference, dataset, work, config, local_rank)
            if args.mode != "audit":
                validate_coverage(rows, local_ids)
            files.update({record["path"]: record for record in verified_artifacts.values()})
            assert_files_unchanged(files, metadata["checkpoint"])
        except Exception as error:
            failure = {"rank": rank, "error": f"{type(error).__name__}: {error}"}
        failures = [failure]
        if world == 2:
            failures = [None] * world
            distributed.all_gather_object(failures, failure)
        if any(failures):
            write_json(args.output / f"failure_rank_{rank}.json", {"complete": False, "mode": args.mode,
                "phase": "generation/output identity", "failures": failures, "quality_approval": False})
            raise RuntimeError(f"evaluation failed: {failures}")
        report = {"complete": True, "family": FAMILY, "mode": args.mode, "rank": rank, "world_size": world,
                  "quality_approval": False, "metadata": metadata, "input_files": files, "rows": rows,
                  "source_revision": source_revision,
                  "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                              "threads": torch.get_num_threads(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "cuda_device_index": local_rank if args.mode != "audit" else None},
                  "checkpoint_stat_unchanged": True, "source_config_reference_hashes_unchanged": True}
        write_json(args.output / f"rank_{rank}.json", report)
        if world == 2:
            distributed.barrier()
        if rank == 0:
            reports = [json.loads((args.output / f"rank_{index}.json").read_text(encoding="utf-8")) for index in range(world)]
            if any(not value["complete"] or value["mode"] != args.mode for value in reports):
                raise RuntimeError("incomplete or mismatched worker reports")
            audited = [item["asset_id"] for value in reports for item in value["metadata"]["audited_assets"]]
            if len(audited) != len(metadata["asset_ids"]) or set(audited) != set(metadata["asset_ids"]):
                raise RuntimeError("CPU audit missed or duplicated panel assets")
            all_rows = [row for value in reports for row in value["rows"]]
            summary = {"complete": True, "family": FAMILY, "mode": args.mode, "quality_approval": False,
                       "panel": metadata["panel"], "evaluated_asset_ids": metadata["asset_ids"],
                       "strict_restoration": metadata["strict_restoration"], "checkpoint": metadata["checkpoint"],
                       "context_budget": metadata["context_budget"], "audited_assets": len(audited),
                       "generated_views": len(all_rows), "failed_generations_scored_zero": True,
                       "optimizer_created": False, "backward_performed": False,
                       "source_config_reference_hashes_unchanged": True}
            if args.mode != "audit":
                summary["valid"] = summarize(all_rows, metadata["asset_ids"])
            write_json(args.output / "summary.json", summary)
    finally:
        if world == 2 and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
