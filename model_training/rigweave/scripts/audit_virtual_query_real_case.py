"""Read-only source, CPU-only geometry replay of the fixed two-pose failure case.

Requires Python 3.10+, NumPy and CPU-capable PyTorch. No vendor/model imports.
See REAL_CASE_USAGE.txt for invocation, coordinate contracts and limitations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

# Set before importing torch or any repository module; suppress source pycache writes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np
import torch


ASSET_ID = "999df53fbf4b4de8a9a1d1824740303a"
FRAME_IDS = np.asarray([9, 7], dtype=np.int64)
NORMALIZED_ATOL = 2e-6
GEOMETRY_OVER_DIAG_ATOL = 1e-5
RIGID_ATOL = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def finite(name, value):
    require(np.isfinite(value).all(), f"{name}: NaN or infinity")


def max_abs(value):
    return float(np.max(np.abs(value))) if np.size(value) else 0.


def paths_inside(child, parent):
    return child == parent or parent in child.parents


@contextmanager
def consumer_modules(source):
    """Load only real geometry modules, bypassing unrelated package initializers."""
    package = source / "model_training/rigweave/src/rigweave"
    require(package.is_dir(), f"Missing rigweave geometry source: {package}")
    root = types.ModuleType("rigweave")
    root.__path__ = [str(package)]
    dynamic = types.ModuleType("rigweave.dynamic_rig")
    dynamic.__path__ = [str(package / "dynamic_rig")]
    with patch.dict(sys.modules):
        for key in list(sys.modules):
            if key == "rigweave" or key.startswith("rigweave."):
                del sys.modules[key]
        sys.modules.update({"rigweave": root, "rigweave.dynamic_rig": dynamic})
        medium = importlib.import_module("rigweave.dynamic_rig.shared_structure_medium_data")
        virtual = importlib.import_module("rigweave.dynamic_rig.virtual_frame_data")
        yield medium, virtual


def select_row(config, manifest, split, virtual):
    spec = config["data"]["manifests"][split]
    require(manifest == Path(spec["path"]).resolve(), "CLI manifest differs from config split path")
    rows = virtual.read_virtual_manifest(manifest, expected_count=spec["rows"],
                                         expected_sha256=spec["sha256"])
    matches = [row for row in rows if row["asset_id"] == ASSET_ID]
    require(len(matches) == 1, f"Expected exactly one {ASSET_ID}, found {len(matches)}")
    row = matches[0]
    rt = row.get("root_rt", {})
    require(rt.get("version") == "retained_root_rigid_v1", "Missing required retained-root RT descriptor")
    require(rt.get("normalization") == "input_sequence_mesh_bbox", "Unexpected RT normalization")
    require(rt.get("padding") == 1.1 and rt.get("reference_index") == 0, "Unexpected RT padding/reference")
    require(Path(row["path"]).is_absolute() and Path(rt["cache_path"]).is_absolute(), "Relative data/cache path")
    return row, len(rows)


def read_original(row):
    """Only the selected asset and its single RT shard are opened, never other assets."""
    # Numeric source arrays are readable without unpickling the legacy schema field.
    with np.load(row["path"], allow_pickle=False) as raw:
        mesh_all = raw["frame_vertices_rootspace"]
        joints_all = raw["target_joints_rootspace"]
        require(mesh_all.ndim == 3 and mesh_all.shape[-1] == 3, "Invalid raw mesh shape")
        require(joints_all.ndim == 3 and joints_all.shape[-1] == 3, "Invalid raw joint shape")
        count = len(mesh_all)
        require(count == len(joints_all) and count > int(FRAME_IDS.max()), "Mesh/GT frame count mismatch")
        finite("selected original mesh", mesh_all[FRAME_IDS])
        finite("selected original joints", joints_all[FRAME_IDS])
        require(mesh_all.dtype.kind == joints_all.dtype.kind == "f", "Expected floating-point geometry")
        original_mesh = mesh_all[FRAME_IDS].copy()
        original_joints = joints_all[FRAME_IDS].copy()
        numbers = raw["frame_numbers"].copy()
        faces, parents = raw["faces"].copy(), raw["target_parents"].copy()
        raw_indices = raw["target_raw_indices"].copy()
    require(numbers.shape == (count,) and numbers.dtype.kind in "iu", "Invalid original frame_numbers")
    require(raw_indices.shape == (original_joints.shape[1],) and raw_indices.dtype.kind in "iu",
            "Missing/invalid retained-joint source indices")
    retained_root = int(raw_indices[0])
    require(parents.shape == (original_joints.shape[1],) and parents.dtype.kind == "i", "Invalid parents")
    require(parents[0] == -1 and all(0 <= int(p) < child for child, p in enumerate(parents[1:], 1)),
            "Expected one root and parent-before-child target order")
    require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in "iu" and len(faces) > 0,
            "Invalid triangle faces")
    require(faces.min() >= 0 and faces.max() < original_mesh.shape[1], "Face index out of bounds")
    descriptor, cache = row["root_rt"], Path(row["root_rt"]["cache_path"])
    with np.load(cache, allow_pickle=False) as shard:
        index = descriptor["index"]
        require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, "Invalid cache row")
        counts = shard["frame_counts"]
        require(index < len(counts), "RT cache index out of bounds")
        require(shard["schema_version"].item() == "retained_root_rigid_v1", "RT cache schema mismatch")
        require(int(counts[index]) == descriptor["frame_count"] == count, "RT cache/asset frame count mismatch")
        require(shard["path_hashes"][index] == hashlib.sha256(row["path"].encode("utf-8")).hexdigest(),
                "RT cache asset-path hash mismatch")
        require(np.array_equal(shard["frame_numbers"][index, :count], numbers), "RT cache frame mapping mismatch")
        require(int(shard["root_indices"][index]) == retained_root, "RT cache retained-root mismatch")
        rotations = shard["rotations"][index, FRAME_IDS].copy()
        translations = shard["translations"][index, FRAME_IDS].copy()
        reference_r = shard["rotations"][index, 0].copy()
        reference_t = shard["translations"][index, 0].copy()
        nonrigid = bool(shard["relative_nonrigid"][index])
    for name, value in (("RT rotation", rotations), ("RT translation", translations),
                        ("reference rotation", reference_r), ("reference translation", reference_t)):
        finite(name, value)
    require(rotations.shape == (2, 3, 3) and translations.shape == (2, 3), "Invalid selected RT shape")
    require(max_abs(reference_r - np.eye(3)) <= RIGID_ATOL and max_abs(reference_t) <= RIGID_ATOL,
            "RT reference frame is not identity")
    return {"mesh": original_mesh, "joints": original_joints, "faces": faces, "parents": parents,
            "frame_numbers": numbers[FRAME_IDS], "rotations": rotations, "translations": translations,
            "retained_root_index": retained_root, "relative_nonrigid": nonrigid, "frame_count": count}


def independent_geometry(original):
    """Solve the raw-cache column-vector equation R*x+t=p; no model RT helpers."""
    rotations, translations = original["rotations"], original["translations"]
    orthogonal = max_abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3))
    determinants = np.linalg.det(rotations)
    require(orthogonal <= RIGID_ATOL and max_abs(determinants - 1) <= RIGID_ATOL, "RT is not a proper rotation")
    aligned64, aligned32, cast_errors = {}, {}, {}
    for name in ("mesh", "joints"):
        value = original[name]
        # The actual loader accepts original geometry as float32; measure this cast separately.
        as32 = value.astype(np.float32)
        finite(f"float32 {name}", as32)
        cast_errors[name] = max_abs(value.astype(np.float64) - as32.astype(np.float64))
        aligned64[name] = np.stack([np.linalg.solve(r, (p.astype(np.float64) - t).T).T
                                    for p, r, t in zip(as32, rotations, translations)])
        aligned32[name] = aligned64[name].astype(np.float32)
        finite(f"independent aligned {name}", aligned32[name])
    # Reproduce the published float32 bbox contract, using only independent mesh coordinates.
    mesh_points = aligned32["mesh"].reshape(-1, 3)
    low, high = np.min(mesh_points, axis=0), np.max(mesh_points, axis=0)
    center = np.asarray((low + high) / np.float32(2), dtype=np.float32)
    scale = np.float32(1.1) * np.max((high - low) / np.float32(2))
    require(np.isfinite(scale) and scale > 0 and np.isfinite(center).all(), "Degenerate independent bbox")
    normalized = {key: np.asarray((value - center) / scale, dtype=np.float32)
                  for key, value in aligned32.items()}
    for name, value in normalized.items():
        finite(f"independent normalized {name}", value)
        require(max_abs(value) <= 1., f"Independent normalized {name} outside [-1,1]; clipping prohibited")
    return aligned64, normalized, center, float(scale), {
        "rotation_orthogonality_max_abs": orthogonal, "rotation_determinants": determinants.tolist(),
        "source_to_float32_max_abs": cast_errors}


def measure_geometry(original, aligned64, expected, center, scale, batch):
    arrays = {"mesh": batch["frame_vertices"][0].numpy(), "joints": batch["frame_target_joints"][0].numpy()}
    for tensor in batch.values():
        if isinstance(tensor, torch.Tensor):
            require(tensor.device.type == "cpu", "Unexpected non-CPU loader output")
            if tensor.is_floating_point():
                require(torch.isfinite(tensor).all(), "Nonfinite loader tensor")
    require(np.array_equal(batch["selected_frames"][0].numpy(), FRAME_IDS), "Loader changed frame order")
    require(np.array_equal(batch["frame_numbers"][0].numpy(), original["frame_numbers"]), "Loader frame numbers changed")
    require(np.array_equal(batch["faces"][0].numpy(), original["faces"]), "Loader faces changed")
    require(np.array_equal(batch["target_parents"][0].numpy(), original["parents"]), "Loader parents changed")
    for key in ("rotations", "translations"):
        require(np.array_equal(batch[f"root_rt_{key}"][0].numpy(), original[key]), f"Loader RT {key} differ")
    require(np.array_equal(batch["root_rt_augmentation"][0].numpy(), np.eye(3)), "Unexpected extra augmentation")
    require(int(batch["root_rt_root_index"][0]) == original["retained_root_index"], "Loader retained-root mismatch")
    require(bool(batch["root_rt_relative_nonrigid"][0]) == original["relative_nonrigid"], "Loader nonrigid flag mismatch")
    actual_center, actual_scale = batch["query_center"][0].numpy(), float(batch["query_scale"][0])
    center_error = max_abs(actual_center - center) / scale
    scale_error = abs(actual_scale - scale) / scale
    require(center_error <= NORMALIZED_ATOL and scale_error <= NORMALIZED_ATOL, "Common mesh bbox differs")
    raw_mesh64 = original["mesh"].astype(np.float64)
    diagonals = np.linalg.norm(raw_mesh64.max(1) - raw_mesh64.min(1), axis=1)
    require(np.all(diagonals > 0), "Degenerate original frame bbox")
    report = {"bbox_center": actual_center.tolist(), "bbox_scale": actual_scale,
              "bbox_center_error_over_scale": center_error, "bbox_scale_relative_error": scale_error,
              "per_frame_original_mesh_bbox_diagonal": diagonals.tolist(), "geometry": {}, "bone_lengths": {}}
    for name, actual in arrays.items():
        finite(f"loader normalized {name}", actual)
        require(actual.shape == expected[name].shape, f"Loader {name} shape differs")
        require(max_abs(actual) <= 1., f"Loader {name} out of [-1,1]")
        difference = max_abs(actual - expected[name])
        require(difference <= NORMALIZED_ATOL, f"Independent {name} transform differs: {difference}")
        unnormalized = actual.astype(np.float64) * actual_scale + actual_center
        restored = np.stack([(r @ p.T).T + t for p, r, t in zip(unnormalized,
                             original["rotations"], original["translations"])])
        direct_inverse = np.stack([(r @ p.T).T + t for p, r, t in zip(aligned64[name],
                                  original["rotations"], original["translations"])])
        roundtrip = np.max(np.abs(restored - original[name].astype(np.float64)), axis=(1, 2))
        inverse = np.max(np.abs(direct_inverse - original[name].astype(np.float64)), axis=(1, 2))
        require(np.max(roundtrip / diagonals) <= GEOMETRY_OVER_DIAG_ATOL, f"{name} roundtrip error exceeds tolerance")
        require(np.max(inverse / diagonals) <= GEOMETRY_OVER_DIAG_ATOL, f"{name} rigid inverse error exceeds tolerance")
        report["geometry"][name] = {"independent_vs_loader_max_abs": difference,
            "max_normalized_abs": max_abs(actual), "out_of_bounds_coordinates": int(np.count_nonzero(np.abs(actual) > 1)),
            "raw_to_aligned_to_raw_max_abs_per_frame": inverse.tolist(),
            "raw_to_normalized_to_raw_max_abs_per_frame": roundtrip.tolist(),
            "raw_to_normalized_to_raw_over_mesh_diag": (roundtrip / diagonals).tolist(),
            "between_pose_normalized_max_abs": max_abs(actual[0] - actual[1])}
    parents = original["parents"]
    children = np.flatnonzero(parents >= 0)
    require(len(children) > 0, "No bones to verify")

    def lengths(points):
        points = points.astype(np.float64)
        return np.linalg.norm(points[:, children] - points[:, parents[children]], axis=-1)

    before = lengths(original["joints"])
    stages = {"independent_after_rt": lengths(aligned64["joints"]),
              "loader_after_rt_unscaled": lengths(arrays["joints"]) * actual_scale}
    report["bone_lengths"]["bone_count"] = len(children)
    report["bone_lengths"]["before_range_per_frame"] = [[float(x.min()), float(x.max())] for x in before]
    for name, after in stages.items():
        errors = np.abs(after - before)
        absolute = errors.max(1)
        over_diag = absolute / diagonals
        require(np.max(over_diag) <= GEOMETRY_OVER_DIAG_ATOL, f"{name}: bone lengths changed")
        nonzero = before > diagonals[:, None] * 1e-12
        relative = np.divide(errors, before, out=np.zeros_like(errors), where=nonzero)
        report["bone_lengths"][name] = {"max_abs_error_per_frame": absolute.tolist(),
            "max_error_over_mesh_diag": over_diag.tolist(), "max_relative_error_nonzero_bones": relative.max(1).tolist(),
            "near_zero_bones_excluded_from_relative_error": (~nonzero).sum(1).tolist()}
    report["no_silent_clipping"] = "Independent un-clipped geometry is in bounds and agrees with actual loader geometry"
    return arrays, report


def execute(args, report):
    source, config_path, manifest = (path.resolve() for path in (args.source_root, args.config, args.manifest))
    require(not paths_inside(args.output_dir.resolve(), source), "Output must be outside read-only source checkout")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    data = config["data"]
    require(data.get("global_rt_removal") is True and data.get("coordinate_clipping") is False,
            "Config must enable global RT removal and prohibit coordinate clipping")
    require(data.get("normalization") == "input_sequence_mesh_bbox", "Unexpected config normalization")
    dynamic = source / "model_training/rigweave/src/rigweave/dynamic_rig"
    source_files = [dynamic / name for name in ("data.py", "root_rt.py", "shared_structure_data.py",
        "shared_structure_medium_data.py", "virtual_frame_data.py", "frame_batch_sampler.py", "sampling.py")]
    tracked = {str(path): sha256(path) for path in [manifest, *source_files]}
    tracked[str(config_path)] = hashlib.sha256(config_bytes).hexdigest()
    with consumer_modules(source) as (medium, virtual):
        row, manifest_count = select_row(config, manifest, args.split, virtual)
        for path in (Path(row["path"]), Path(row["root_rt"]["cache_path"])):
            tracked[str(path)] = sha256(path)
        report["inputs"] = {"source_root": str(source), "config": str(config_path), "manifest": str(manifest),
            "split": args.split, "manifest_rows_checked": manifest_count, "asset_npz": row["path"],
            "rt_cache": row["root_rt"]["cache_path"], "rt_cache_index": row["root_rt"]["index"],
            "sha256": tracked}
        original = read_original(row)
        aligned, expected, center, scale, independent = independent_geometry(original)
        # Exact consumer code, explicit original NPZ row IDs: no sampler or learned forward.
        batch = medium.load_medium_batch(row, FRAME_IDS.tolist())
        arrays, measurements = measure_geometry(original, aligned, expected, center, scale, batch)
        report["measurements"] = {**independent, **measurements}
        require(all(sha256(path) == digest for path, digest in tracked.items()), "An input/source changed during audit")
        report["inputs_unchanged"] = True
        report["frame_numbers"] = original["frame_numbers"].tolist()
        report["retained_root_index"] = original["retained_root_index"]
        report["relative_nonrigid"] = original["relative_nonrigid"]
        payload = {"schema_version": np.asarray("fixed_normalized_geometry_v1"), "asset_id": np.asarray(ASSET_ID),
            "raw_frame_ids": FRAME_IDS, "frame_numbers": original["frame_numbers"],
            "vertices": arrays["mesh"], "faces": original["faces"].astype(np.int64),
            "gt_joints": arrays["joints"], "parents": original["parents"].astype(np.int64),
            "center": batch["query_center"][0].numpy(), "scale": batch["query_scale"][0].numpy(),
            "root_rt_rotations": original["rotations"], "root_rt_translations": original["translations"],
            "augmentation": batch["root_rt_augmentation"][0].numpy(),
            "retained_root_index": np.asarray(original["retained_root_index"], dtype=np.int64)}
        destination = args.output_dir / "fixed_poses.npz"
        temporary = args.output_dir / "fixed_poses.npz.tmp"
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        with np.load(temporary, allow_pickle=False) as verify:
            require(set(verify.files) == set(payload), "Export keys differ on readback")
            require(all(np.array_equal(verify[key], value) for key, value in payload.items()), "Export readback mismatch")
        temporary.replace(destination)
        report["export"] = {"path": str(destination.resolve()), "sha256": sha256(destination),
            "bytes": destination.stat().st_size, "shapes": {key: list(value.shape) for key, value in payload.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    require(not paths_inside(args.output_dir, args.source_root.resolve()), "Output must not write inside source checkout")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "audit_real_case.json"
    require(not any((args.output_dir / name).exists() for name in
        ("audit_real_case.json", "fixed_poses.npz", "fixed_poses.npz.tmp")), "Use a fresh output directory; refusing overwrite")
    torch.set_num_threads(1)
    report = {"schema_version": 1, "status": "running", "asset_id": ASSET_ID,
        "raw_frame_ids": FRAME_IDS.tolist(), "selection_mode": "explicit_original_NPZ_rows_no_resampling",
        "execution": {"device": "cpu", "numpy": np.__version__, "torch": torch.__version__,
            "script_sha256": sha256(__file__), "learned_model_loaded": False, "surface_FPS_run": False},
        "tolerances": {"normalized_coordinate_max_abs": NORMALIZED_ATOL,
            "geometry_or_bone_error_over_original_frame_mesh_diagonal": GEOMETRY_OVER_DIAG_ATOL,
            "rotation_orthogonality_and_determinant": RIGID_ATOL, "coordinate_abs_limit": 1.},
        "limitations": ["Explicit fixed frames, not a rerun of frame selection or surface sampling",
            "Verifies NPZ-to-RT-cache binding and application, not derivation of RT from source animation",
            "Bone preservation is within each pose; cross-pose articulation/scale need not preserve bone lengths",
            "No tokenizer, learned model, checkpoint, gradients or GPU inference"]}
    try:
        execute(args, report)
        report["status"] = "passed"
        code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 1
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
