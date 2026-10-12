#!/usr/bin/env python3
"""Publish immutable root-RT caches and manifests without rewriting source NPZs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

SCHEMA = "retained_root_rigid_v1"
PADDING = 1.1


class QualityError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_hash(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def root_rigid_transforms(transforms: np.ndarray, rest_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    d = np.asarray(transforms, dtype=np.float64)
    rest = np.asarray(rest_joint, dtype=np.float64)
    if d.ndim != 3 or d.shape[1:] != (4, 4) or rest.shape != (3,) or len(d) < 1:
        raise QualityError("invalid_root_transform_shape")
    if not np.isfinite(d).all() or not np.isfinite(rest).all():
        raise QualityError("nonfinite_root_transform")
    if np.max(np.abs(d[:, 3] - np.array([0., 0., 0., 1.]))) > 1e-6:
        raise QualityError("nonaffine_root_transform")
    if np.linalg.cond(d[0, :3, :3]) > 1e8:
        raise QualityError("ill_conditioned_reference_root")
    relative = d @ np.linalg.inv(d[0])
    a = relative[:, :3, :3]
    u, singular, vh = np.linalg.svd(a)
    if np.any(singular[:, -1] <= 1e-8) or np.any(singular[:, 0] / singular[:, -1] > 1e8):
        raise QualityError("singular_or_ill_conditioned_root_motion")
    if np.any(np.linalg.det(a) <= 0):
        raise QualityError("root_motion_changes_orientation")
    sign = np.ones((len(d), 3), dtype=np.float64)
    sign[:, -1] = np.linalg.det(u @ vh)
    rotations = (u * sign[:, None, :]) @ vh
    positions = d[:, :3, :3] @ rest + d[:, :3, 3]
    translations = positions - rotations @ positions[0]
    rotations[0] = np.eye(3)
    translations[0] = 0
    orth = float(np.max(np.abs(np.swapaxes(a, -1, -2) @ a - np.eye(3))))
    det_error = float(np.max(np.abs(np.linalg.det(a) - 1)))
    projected_error = float(np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3))))
    if projected_error > 1e-10 or np.max(np.abs(np.linalg.det(rotations) - 1)) > 1e-10:
        raise QualityError("invalid_projected_rotation")
    return rotations, translations, {
        "relative_nonrigid": bool(orth > 1e-3 or det_error > 1e-3),
        "relative_orthogonality_error": orth,
        "relative_determinant_error": det_error,
        "relative_singular_min": float(singular.min()),
        "relative_singular_max": float(singular.max()),
        "root_position_displacement": float(np.linalg.norm(positions - positions[0], axis=1).max()),
    }


def align_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return ((np.asarray(points, dtype=np.float64) - translation) @ rotation).astype(np.float32)


def pair_window_bounds(mesh_lo: np.ndarray, mesh_hi: np.ndarray, joint_lo: np.ndarray,
                       joint_hi: np.ndarray, padding: float = PADDING) -> dict:
    """Single/pair windows bound every subset containing the supervised frames.

    For a positive-axis violation in a larger set, retain the joint's frame and
    the frame attaining that axis' minimum mesh bound. The pair's center cannot
    increase and its scale cannot increase. Negative violations use the maximum.
    """
    lo = np.minimum(mesh_lo[:, None], mesh_lo[None, :])
    hi = np.maximum(mesh_hi[:, None], mesh_hi[None, :])
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = (padding * np.max((hi - lo) * 0.5, axis=-1)).astype(np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise QualityError("invalid_aligned_mesh_extent")
    jlo = np.minimum(joint_lo[:, None], joint_lo[None, :])
    jhi = np.maximum(joint_hi[:, None], joint_hi[None, :])
    normalized = np.maximum(np.abs((jlo - center) / scale[..., None]),
                            np.abs((jhi - center) / scale[..., None]))
    index = np.unravel_index(np.argmax(normalized), normalized.shape)
    return {"pair_window_normalized_joint_maxabs": float(normalized[index]),
            "worst_pair_frame_indices": [int(index[0]), int(index[1])],
            "worst_pair_axis": int(index[2]),
            "violating_frame_pairs": int(np.count_nonzero(np.triu(np.max(normalized, axis=-1) > 1.0)))}


def analyze_asset(item: tuple[int, str, dict, dict | None]) -> dict:
    ordinal, split, row, exclusion = item
    source = str(row["path"])
    result: dict[str, Any] = {"ordinal": ordinal, "split": split, "row": row,
                              "path": source, "status": "rejected", "reasons": []}
    if exclusion is not None:
        result["reasons"] = [str(exclusion["reason"])]
        result["exclusion_evidence"] = exclusion
        result["relative_nonrigid"] = bool(exclusion.get("relative_nonrigid", False))
        return result
    try:
        with np.load(source, allow_pickle=False) as z:
            vertices = z["frame_vertices_rootspace"]
            joints = z["target_joints_rootspace"]
            target_parents = z["target_parents"].astype(np.int64)
            raw_indices = z["target_raw_indices"].astype(np.int64)
            entries = z["target_entry_raw_indices"].astype(np.int64).reshape(-1)
            roots = np.flatnonzero(target_parents < 0)
            if len(roots) != 1 or len(entries) != 1 or raw_indices[roots[0]] != entries[0]:
                raise QualityError("retained_root_mapping_invalid")
            root = int(entries[0])
            all_transforms = z["bone_transforms"]
            rest_joints = z["rest_joints"]
            if root < 0 or root >= len(rest_joints) or all_transforms.shape[1] != len(rest_joints):
                raise QualityError("raw_root_index_invalid")
            frame_numbers = z["frame_numbers"].astype(np.int64)
            if (vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[1] < 1 or
                    joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[1] != len(target_parents) or
                    len(vertices) != len(joints) or len(vertices) != len(all_transforms) or
                    len(vertices) != len(frame_numbers)):
                raise QualityError("invalid_frame_or_joint_shapes")
            rotations, translations, metrics = root_rigid_transforms(all_transforms[:, root], rest_joints[root])
            if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
                raise QualityError("nonfinite_geometry")
            mesh_lo, mesh_hi, joint_lo, joint_hi = [], [], [], []
            roundtrip = 0.0
            root_drift = 0.0
            reference_root = joints[0, roots[0]].astype(np.float64)
            for frame in range(len(vertices)):
                r, t = rotations[frame], translations[frame]
                v = align_points(vertices[frame], r, t)
                j = align_points(joints[frame], r, t)
                lo, hi = v.min(axis=0), v.max(axis=0)
                diag = float(np.linalg.norm(hi.astype(float) - lo.astype(float)))
                if not np.isfinite(diag) or diag <= 0:
                    raise QualityError("zero_aligned_mesh_extent")
                mesh_lo.append(lo)
                mesh_hi.append(hi)
                joint_lo.append(j.min(axis=0))
                joint_hi.append(j.max(axis=0))
                ids = np.linspace(0, len(v) - 1, min(128, len(v)), dtype=np.int64)
                back = v[ids].astype(float) @ r.T + t
                roundtrip = max(roundtrip, float(np.max(np.abs(back - vertices[frame, ids]))) / diag)
                root_drift = max(root_drift, float(np.linalg.norm(j[roots[0]] - reference_root)) / diag)
            mesh_lo = np.asarray(mesh_lo)
            mesh_hi = np.asarray(mesh_hi)
            joint_lo = np.asarray(joint_lo)
            joint_hi = np.asarray(joint_hi)
            metrics.update(pair_window_bounds(mesh_lo, mesh_hi, joint_lo, joint_hi))
            lo, hi = mesh_lo.min(axis=0), mesh_hi.max(axis=0)
            center = ((lo + hi) * .5).astype(np.float32)
            scale = np.float32(PADDING * np.max((hi - lo) * .5))
            metrics["sequence_normalized_joint_maxabs"] = float(max(
                np.max(np.abs((joint_lo - center) / scale)), np.max(np.abs((joint_hi - center) / scale))))
            metrics["float32_roundtrip_error_over_mesh_diag"] = roundtrip
            metrics["retained_root_drift_over_mesh_diag"] = root_drift
            metrics["frames"] = len(vertices)
            metrics["vertices"] = vertices.shape[1]
            metrics["root_index"] = root
            metrics["source_motion_sha256"] = hashlib.sha256(
                all_transforms[:, root].tobytes() + rest_joints[root].tobytes() + frame_numbers.tobytes()).hexdigest()
            result.update(metrics)
            if metrics["pair_window_normalized_joint_maxabs"] > 1.0:
                result["reasons"].append("aligned_input_window_joint_coordinate_out_of_bounds")
            if roundtrip > 1e-5:
                result["reasons"].append("float32_rt_roundtrip_error")
            if root_drift > 1e-5:
                result["reasons"].append("retained_root_position_inconsistent")
            if not result["reasons"]:
                result.update(status="accepted", rotations=rotations, translations=translations,
                              frame_numbers=frame_numbers)
    except (QualityError, np.linalg.LinAlgError) as exc:
        result["reasons"] = [str(exc)]
    except Exception as exc:
        result["status"] = "error"
        result["reasons"] = ["unexpected_processing_error"]
        result["error"] = repr(exc)
    return result


def read_manifests(paths: dict[str, Path], expected: dict[str, str]) -> list[tuple[int, str, dict]]:
    items = []
    seen: set[str] = set()
    for split, path in paths.items():
        actual = sha256_file(path)
        if actual != expected[split]:
            raise ValueError(f"{split} manifest hash mismatch: {actual}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("invalid manifest row")
            if row["path"] in seen:
                raise ValueError("duplicate source path across input manifests")
            if "root_rt" in row:
                raise ValueError("input already contains a root_rt overlay")
            seen.add(row["path"])
            items.append((len(items), split, row))
    return items


def publish(args: argparse.Namespace) -> dict:
    started = time.time()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"immutable output already exists: {root}")
    paths = {"train": args.train_manifest, "valid": args.valid_manifest}
    expected = {"train": args.train_sha256, "valid": args.valid_sha256}
    original = read_manifests(paths, expected)
    items = original
    if args.selection_json:
        selection = json.loads(args.selection_json.read_text(encoding="utf-8-sig"))
        selected_paths = {v["path"] if isinstance(v, dict) else v for v in selection}
        if not selected_paths <= {r["path"] for _, _, r in original}:
            raise ValueError("selection contains paths outside accepted manifests")
        items = [i for i in original if i[2]["path"] in selected_paths]
    exclusions = {}
    if args.exclusions:
        for line in args.exclusions.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                value = json.loads(line)
                exclusions[value["path"]] = value
        if not set(exclusions) <= {r["path"] for _, _, r in original}:
            raise ValueError("exclusion outside current accepted manifest")
    manifest_dir = root / "quality_distributions/rootless_bbox_consistency/final_manifests"
    cache_dir = root / "root_rt_cache"
    quarantine_dir = root / "quarantine/root_rt_quality"
    provenance_dir = root / "provenance"
    for directory in (manifest_dir, cache_dir, quarantine_dir, provenance_dir):
        directory.mkdir(parents=True)
    accepted = {s: [] for s in paths}
    rejected = {s: [] for s in paths}
    batch = []
    shard_hashes = {}
    stats = {"accepted": Counter(), "rejected": Counter(), "reasons": Counter(), "nonrigid_accepted": Counter(),
             "nonrigid_rejected": Counter(), "errors": [], "max_pair_normalized_joint": 0., "max_roundtrip": 0.}

    def flush_batch() -> None:
        if not batch:
            return
        shard_path = cache_dir / f"shard_{len(shard_hashes):04d}.npz"
        n, max_t = len(batch), max(len(r["rotations"]) for r in batch)
        rs = np.zeros((n, max_t, 3, 3), dtype=np.float64)
        ts = np.zeros((n, max_t, 3), dtype=np.float64)
        fns = np.full((n, max_t), -1, dtype=np.int64)
        for index, r in enumerate(batch):
            count = len(r["rotations"])
            rs[index, :count] = r["rotations"]
            ts[index, :count] = r["translations"]
            fns[index, :count] = r["frame_numbers"]
        payload = dict(schema_version=np.asarray(SCHEMA), rotations=rs, translations=ts,
                       frame_counts=np.asarray([len(r["rotations"]) for r in batch], dtype=np.int32),
                       path_hashes=np.asarray([path_hash(r["path"]) for r in batch], dtype="U64"),
                       frame_numbers=fns, root_indices=np.asarray([r["root_index"] for r in batch], dtype=np.int32),
                       relative_nonrigid=np.asarray([r["relative_nonrigid"] for r in batch], dtype=bool))
        np.savez_compressed(shard_path, **payload)
        with np.load(shard_path, allow_pickle=False) as z:
            if set(z.files) != set(payload) or any(not np.array_equal(z[k], v) for k, v in payload.items()):
                raise RuntimeError("cache serialization roundtrip mismatch")
        digest = sha256_file(shard_path)
        shard_hashes[str(shard_path.relative_to(root))] = digest
        for index, r in enumerate(batch):
            row = dict(r["row"])
            row["root_rt"] = dict(version=SCHEMA, cache_path=str(shard_path), index=index,
                                  frame_count=len(r["rotations"]), reference_index=0,
                                  normalization="input_sequence_mesh_bbox", padding=PADDING,
                                  cache_sha256=digest)
            accepted[r["split"]].append(row)
        batch.clear()

    task_items = [(i, s, r, exclusions.get(r["path"])) for i, s, r in items]
    with (provenance_dir / "asset_metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for processed, result in enumerate(pool.map(analyze_asset, task_items, chunksize=4), 1):
                split = result["split"]
                diagnostic = {k: v for k, v in result.items() if k not in {"row", "rotations", "translations", "frame_numbers"}}
                metrics_file.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                if result["status"] == "accepted":
                    stats["accepted"][split] += 1
                    stats["nonrigid_accepted"][split] += int(result["relative_nonrigid"])
                    stats["max_pair_normalized_joint"] = max(stats["max_pair_normalized_joint"], result["pair_window_normalized_joint_maxabs"])
                    stats["max_roundtrip"] = max(stats["max_roundtrip"], result["float32_roundtrip_error_over_mesh_diag"])
                    batch.append(result)
                    if len(batch) >= args.shard_size:
                        flush_batch()
                else:
                    stats["rejected"][split] += 1
                    stats["nonrigid_rejected"][split] += int(result.get("relative_nonrigid", False))
                    stats["reasons"].update(result["reasons"])
                    rejected[split].append({**result["row"], "root_rt_rejection": diagnostic})
                    if result["status"] == "error":
                        stats["errors"].append(diagnostic)
                if processed % 100 == 0 or processed == len(items):
                    metrics_file.flush()
                    print(json.dumps({"processed": processed, "total": len(items), "accepted": sum(stats["accepted"].values()),
                                      "rejected": sum(stats["rejected"].values()), "seconds": round(time.time() - started, 1)}), flush=True)
        flush_batch()
    manifest_hashes = {}
    for split in paths:
        for directory, rows in ((manifest_dir, accepted[split]), (quarantine_dir, rejected[split])):
            path = directory / f"{split}_manifest.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
            manifest_hashes[str(path.relative_to(root))] = sha256_file(path)
    final_paths = [r["path"] for rows in accepted.values() for r in rows]
    rejected_paths = [r["path"] for rows in rejected.values() for r in rows]
    if len(set(final_paths + rejected_paths)) != len(items) or set(final_paths) & set(rejected_paths):
        raise RuntimeError("accepted/quarantined partition mismatch")
    if set(final_paths + rejected_paths) != {r["path"] for _, _, r in items}:
        raise RuntimeError("missing input rows in partition")
    if any(sha256_file(paths[s]) != expected[s] for s in paths):
        raise RuntimeError("input manifest changed during processing")
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    receipt = {"schema_version": SCHEMA, "status": "failed" if stats["errors"] else "validated",
               "mode": "selection_smoke" if args.selection_json else "full",
               "source_git_commit": git_sha, "source_script_sha256": sha256_file(Path(__file__)),
               "input_manifest_paths": {s: str(p) for s, p in paths.items()}, "input_manifest_sha256": expected,
               "input_counts": dict(Counter(s for _, s, _ in items)), "counts": {k: dict(v) for k, v in stats.items() if isinstance(v, Counter)},
               "errors": stats["errors"], "max_pair_normalized_joint": stats["max_pair_normalized_joint"],
               "max_float32_roundtrip_over_mesh_diag": stats["max_roundtrip"], "padding": PADDING,
               "coordinate_limit": 1.0, "window_coverage": "all subsets of stored frames, without extra rotation augmentation",
               "root_drift_and_roundtrip_limit": 1e-5, "nonrigid_policy": "proper polar rotation with retained-root pivot; preserve scale/shear",
               "manifest_sha256": manifest_hashes, "cache_sha256": shard_hashes,
               "source_npz_files_changed": 0, "seconds": time.time() - started}
    receipt_path = provenance_dir / "root_rt_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not stats["errors"]:
        (root / ".READY").write_text(sha256_file(receipt_path) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"cache_sha256", "manifest_sha256", "errors"}}), flush=True)
    if stats["errors"]:
        raise RuntimeError(f"{len(stats['errors'])} unexpected errors; output not published READY")
    return receipt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--valid-manifest", type=Path, required=True)
    p.add_argument("--train-sha256", required=True)
    p.add_argument("--valid-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclusions", type=Path)
    p.add_argument("--selection-json", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=256)
    args = p.parse_args()
    if args.workers < 1 or args.shard_size < 1:
        p.error("workers and shard-size must be positive")
    return args


if __name__ == "__main__":
    publish(parse_args())
