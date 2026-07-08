#!/usr/bin/env python3
"""Precheck Pass1 NPZ files before the rootless target rewrite.

This stage validates that raw Blender exports are readable and internally
consistent.  It does not make training-target quality decisions and it does not
apply root/rootless policy.  Root removal, forest rejection, mesh component
filtering, and target quality gates happen after this stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = (
    "rest_vertices",
    "faces",
    "frame_vertices",
    "skin_weights",
    "parents",
    "rest_joints",
    "rest_tails",
    "rest_tails_raw",
    "bone_transforms",
    "frame_numbers",
)


def asset_id_from_path(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_seq0") else name


def sequence_id_from_path(path: Path) -> str:
    stem = path.stem
    if "_seq" in stem:
        return stem.rsplit("_seq", 1)[-1]
    return "0"


def safe_float(value: float | np.floating, default: float = 0.0) -> float:
    out = float(value)
    return out if math.isfinite(out) else float(default)


def read_manifest(path: Path) -> list[Path]:
    rows: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_path = Path(str(row.get("path", "")))
            rows.append(raw_path if raw_path.is_absolute() else (path.parent / raw_path).resolve())
    return rows


def collect_npz_files(npz_dirs: list[Path], manifests: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for manifest in manifests:
        paths.extend(read_manifest(manifest))
    for npz_dir in npz_dirs:
        paths.extend(sorted(npz_dir.rglob("*.npz")))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def bbox_diag(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return 0.0
    pts = pts.reshape(-1, 3)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def face_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "face_index_valid_ratio": 0.0,
        "degenerate_face_ratio": 1.0,
    }
    if faces.ndim != 2 or faces.shape[1] != 3 or vertices.ndim != 2:
        return metrics
    if faces.shape[0] == 0:
        metrics["degenerate_face_ratio"] = 0.0
        return metrics
    valid = (faces >= 0) & (faces < vertices.shape[0])
    metrics["face_index_valid_ratio"] = safe_float(valid.all(axis=1).mean())
    tri = faces.astype(np.int64, copy=False)
    degenerate = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    metrics["degenerate_face_ratio"] = safe_float(degenerate.mean())
    return metrics


def parent_graph_issues(parents: np.ndarray) -> list[str]:
    issues: list[str] = []
    arr = np.asarray(parents, dtype=np.int64).reshape(-1)
    n = int(arr.shape[0])
    roots = [int(i) for i, p in enumerate(arr.tolist()) if int(p) < 0]
    if not roots:
        issues.append("no_parent_negative_root")
    children: list[list[int]] = [[] for _ in range(n)]
    for child, parent in enumerate(arr.tolist()):
        p = int(parent)
        if p < 0:
            continue
        if p >= n:
            issues.append("parent_index_out_of_range")
            continue
        if p == child:
            issues.append("self_parent")
            continue
        if p >= child:
            issues.append("parent_order_violation")
        children[p].append(int(child))
    if issues:
        return sorted(set(issues))

    seen: set[int] = set()
    stack = roots[:]
    while stack:
        node = int(stack.pop())
        if node in seen:
            issues.append("tree_cycle_or_duplicate_visit")
            break
        seen.add(node)
        stack.extend(children[node])
    if len(seen) != n:
        issues.append("tree_not_fully_reachable")
    return sorted(set(issues))


def deterministic_ids(count: int, sample_count: int) -> np.ndarray:
    n = min(max(int(sample_count), 0), int(count))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.linspace(0, int(count) - 1, n).round().astype(np.int64)


def lbs_metrics(
    rest_vertices: np.ndarray,
    frame_vertices: np.ndarray,
    skin: np.ndarray,
    bone_transforms: np.ndarray,
    rest_diag: float,
    *,
    max_vertices: int,
    max_frames: int,
) -> dict[str, Any]:
    vertex_ids = deterministic_ids(rest_vertices.shape[0], max_vertices)
    frame_ids = deterministic_ids(frame_vertices.shape[0], max_frames)
    if vertex_ids.size == 0 or frame_ids.size == 0:
        return {
            "lbs_sample_vertex_count": int(vertex_ids.size),
            "lbs_sample_frame_count": int(frame_ids.size),
            "lbs_recon_p50_bbox": 999.0,
            "lbs_recon_p95_bbox": 999.0,
            "lbs_recon_p99_bbox": 999.0,
            "lbs_recon_max_bbox": 999.0,
        }
    rest = rest_vertices[vertex_ids].astype(np.float32, copy=False)
    weights = skin[vertex_ids].astype(np.float32, copy=False)
    rest_h = np.concatenate([rest, np.ones((rest.shape[0], 1), dtype=np.float32)], axis=1)
    transforms = bone_transforms[frame_ids].astype(np.float32, copy=False)
    target = frame_vertices[frame_ids][:, vertex_ids].astype(np.float32, copy=False)
    posed = np.einsum("fjab,vb->fjva", transforms, rest_h, optimize=True)[..., :3]
    pred = np.einsum("vj,fjva->fva", weights, posed, optimize=True)
    err = np.linalg.norm(pred - target, axis=-1) / max(float(rest_diag), 1.0e-8)
    return {
        "lbs_sample_vertex_count": int(vertex_ids.size),
        "lbs_sample_frame_count": int(frame_ids.size),
        "lbs_recon_p50_bbox": safe_float(np.percentile(err, 50)),
        "lbs_recon_p95_bbox": safe_float(np.percentile(err, 95)),
        "lbs_recon_p99_bbox": safe_float(np.percentile(err, 99)),
        "lbs_recon_max_bbox": safe_float(err.max()),
    }


def check_one(
    path: Path,
    *,
    min_frames: int,
    min_vertices: int,
    min_faces: int,
    min_joints: int,
    max_joints: int,
    max_lbs_recon_p95_bbox: float,
    max_lbs_recon_p99_bbox: float,
    max_lbs_recon_max_bbox: float,
    lbs_sample_vertices: int,
    lbs_sample_frames: int,
) -> tuple[bool, dict[str, Any], list[str]]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "path": str(path),
        "asset_id": asset_id_from_path(path),
        "sequence_id": sequence_id_from_path(path),
    }
    try:
        raw = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return False, metrics, [f"npz_load_error:{repr(exc)}"]

    with raw:
        missing = [key for key in REQUIRED_KEYS if key not in raw.files]
        if missing:
            metrics["missing_keys"] = missing
            return False, metrics, [f"missing_keys:{','.join(missing)}"]
        try:
            rest_vertices = np.asarray(raw["rest_vertices"], dtype=np.float32)
            faces = np.asarray(raw["faces"], dtype=np.int64)
            frame_vertices = np.asarray(raw["frame_vertices"], dtype=np.float32)
            skin = np.asarray(raw["skin_weights"], dtype=np.float32)
            parents = np.asarray(raw["parents"], dtype=np.int64).reshape(-1)
            rest_joints = np.asarray(raw["rest_joints"], dtype=np.float32)
            rest_tails = np.asarray(raw["rest_tails"], dtype=np.float32)
            rest_tails_raw = np.asarray(raw["rest_tails_raw"], dtype=np.float32)
            bone_transforms = np.asarray(raw["bone_transforms"], dtype=np.float32)
            frame_numbers = np.asarray(raw["frame_numbers"], dtype=np.int64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return False, metrics, [f"array_cast_error:{repr(exc)}"]

    metrics.update(
        {
            "vertex_count": int(rest_vertices.shape[0]) if rest_vertices.ndim >= 1 else 0,
            "face_count": int(faces.shape[0]) if faces.ndim >= 1 else 0,
            "frame_count": int(frame_vertices.shape[0]) if frame_vertices.ndim >= 1 else 0,
            "joint_count": int(rest_joints.shape[0]) if rest_joints.ndim >= 1 else 0,
            "root_count": int((parents < 0).sum()) if parents.ndim == 1 else 0,
            "pass1_precheck_policy": "raw_readability_only_rootless_later",
        }
    )

    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        reasons.append("bad_rest_vertices_shape")
    if faces.ndim != 2 or faces.shape[1] != 3:
        reasons.append("bad_faces_shape")
    if frame_vertices.ndim != 3 or frame_vertices.shape[1:] != rest_vertices.shape:
        reasons.append("bad_frame_vertices_shape")
    if rest_joints.ndim != 2 or rest_joints.shape[1] != 3:
        reasons.append("bad_rest_joints_shape")
    if rest_tails.shape != rest_joints.shape:
        reasons.append("bad_rest_tails_shape")
    if rest_tails_raw.shape != rest_joints.shape:
        reasons.append("bad_rest_tails_raw_shape")
    if parents.shape != (rest_joints.shape[0],):
        reasons.append("bad_parents_shape")
    if skin.ndim != 2 or skin.shape != (rest_vertices.shape[0], rest_joints.shape[0]):
        reasons.append("bad_skin_shape")
    if bone_transforms.ndim != 4 or bone_transforms.shape[:2] != (frame_vertices.shape[0], rest_joints.shape[0]) or bone_transforms.shape[2:] != (4, 4):
        reasons.append("bad_bone_transforms_shape")
    if frame_numbers.shape[0] != frame_vertices.shape[0]:
        reasons.append("frame_number_count_mismatch")
    if reasons:
        return False, metrics, reasons

    for name, arr in {
        "rest_vertices": rest_vertices,
        "faces": faces,
        "frame_vertices": frame_vertices,
        "skin_weights": skin,
        "parents": parents,
        "rest_joints": rest_joints,
        "rest_tails": rest_tails,
        "rest_tails_raw": rest_tails_raw,
        "bone_transforms": bone_transforms,
    }.items():
        if not np.isfinite(arr).all():
            reasons.append(f"nonfinite_{name}")

    if rest_vertices.shape[0] < int(min_vertices):
        reasons.append("too_few_vertices")
    if faces.shape[0] < int(min_faces):
        reasons.append("too_few_faces")
    if rest_joints.shape[0] < int(min_joints):
        reasons.append("too_few_joints")
    if int(max_joints) > 0 and rest_joints.shape[0] > int(max_joints):
        reasons.append("too_many_joints")
    if frame_vertices.shape[0] < int(min_frames):
        reasons.append("too_few_frames")

    reasons.extend(parent_graph_issues(parents))

    rest_diag = bbox_diag(rest_vertices)
    metrics["rest_bbox_diag"] = safe_float(rest_diag)
    if rest_diag <= 1.0e-8:
        reasons.append("degenerate_rest_bbox")

    face = face_metrics(rest_vertices, faces)
    metrics.update(face)
    if face["face_index_valid_ratio"] < 1.0:
        reasons.append("invalid_face_indices")
    if face["degenerate_face_ratio"] > 0.05:
        reasons.append("too_many_degenerate_faces")

    row_sum = skin.sum(axis=1)
    metrics["skin_row_sum_max_abs_err"] = safe_float(np.max(np.abs(row_sum - 1.0))) if row_sum.size else 999.0
    metrics["skin_negative_count"] = int((skin < -1.0e-6).sum())
    if metrics["skin_row_sum_max_abs_err"] > 1.0e-3:
        reasons.append("skin_row_sum_bad")
    if metrics["skin_negative_count"] > 0:
        reasons.append("skin_negative")
    active = skin.sum(axis=0) > 1.0e-5
    metrics["active_skin_joints"] = int(active.sum())
    if int(active.sum()) < 1:
        reasons.append("no_active_skin_joints")

    affine_last = bone_transforms[..., 3, :]
    affine_target = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    metrics["bone_transform_last_row_max_err"] = safe_float(np.abs(affine_last - affine_target).max())
    if metrics["bone_transform_last_row_max_err"] > 1.0e-3:
        reasons.append("bone_transform_bad_last_row")

    try:
        lbs = lbs_metrics(
            rest_vertices,
            frame_vertices,
            skin,
            bone_transforms,
            rest_diag,
            max_vertices=lbs_sample_vertices,
            max_frames=lbs_sample_frames,
        )
        metrics.update(lbs)
        if lbs["lbs_recon_p95_bbox"] > float(max_lbs_recon_p95_bbox):
            reasons.append("lbs_recon_p95_bbox_bad")
        if lbs["lbs_recon_p99_bbox"] > float(max_lbs_recon_p99_bbox):
            reasons.append("lbs_recon_p99_bbox_bad")
        if lbs["lbs_recon_max_bbox"] > float(max_lbs_recon_max_bbox):
            reasons.append("lbs_recon_max_bbox_bad")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"lbs_reconstruction_error:{repr(exc)}")

    return not reasons, metrics, reasons


def check_job(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job["path"]))
    ok, metrics, reasons = check_one(
        path,
        min_frames=int(job["min_frames"]),
        min_vertices=int(job["min_vertices"]),
        min_faces=int(job["min_faces"]),
        min_joints=int(job["min_joints"]),
        max_joints=int(job["max_joints"]),
        max_lbs_recon_p95_bbox=float(job["max_lbs_recon_p95_bbox"]),
        max_lbs_recon_p99_bbox=float(job["max_lbs_recon_p99_bbox"]),
        max_lbs_recon_max_bbox=float(job["max_lbs_recon_max_bbox"]),
        lbs_sample_vertices=int(job["lbs_sample_vertices"]),
        lbs_sample_frames=int(job["lbs_sample_frames"]),
    )
    return {
        "asset_id": str(metrics.get("asset_id") or asset_id_from_path(path)),
        "sequence_id": str(metrics.get("sequence_id") or sequence_id_from_path(path)),
        "path": str(path),
        "status": "pass1_precheck_accept" if ok else "pass1_precheck_reject",
        "metrics": metrics,
        "reasons": reasons,
    }


def load_split_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            asset_id = str(row.get("asset_id") or "").strip()
            split = str(row.get("split") or "").strip()
            if asset_id and split in {"train", "val", "test"}:
                out[asset_id] = split
    return out


def split_assets(keys: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = sorted(keys)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(round(n * float(val_ratio)))
    n_test = int(round(n * float(test_ratio)))
    val = set(shuffled[:n_val])
    test = set(shuffled[n_val : n_val + n_test])
    return {key: ("val" if key in val else "test" if key in test else "train") for key in keys}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.get("metrics", {}).keys():
            if key not in seen:
                seen.add(key)
                metric_keys.append(key)
    fields = ["status", "asset_id", "sequence_id", "path", "reasons", *metric_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "status": row.get("status", ""),
                "asset_id": row.get("asset_id", ""),
                "sequence_id": row.get("sequence_id", ""),
                "path": row.get("path", ""),
                "reasons": ";".join(row.get("reasons", [])),
            }
            out.update({key: row.get("metrics", {}).get(key, "") for key in metric_keys})
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, action="append", default=[])
    parser.add_argument("--manifest-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-assets", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--min-faces", type=int, default=20)
    parser.add_argument("--min-joints", type=int, default=4)
    parser.add_argument("--max-joints", type=int, default=256)
    parser.add_argument("--max-lbs-recon-p95-bbox", type=float, default=0.05)
    parser.add_argument("--max-lbs-recon-p99-bbox", type=float, default=0.15)
    parser.add_argument("--max-lbs-recon-max-bbox", type=float, default=0.5)
    parser.add_argument("--lbs-sample-vertices", type=int, default=2048)
    parser.add_argument("--lbs-sample-frames", type=int, default=8)
    parser.add_argument("--split-map-csv", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    paths = collect_npz_files(args.npz_dir, args.manifest_jsonl)
    jobs = [
        {
            "path": str(path),
            "min_frames": int(args.min_frames),
            "min_vertices": int(args.min_vertices),
            "min_faces": int(args.min_faces),
            "min_joints": int(args.min_joints),
            "max_joints": int(args.max_joints),
            "max_lbs_recon_p95_bbox": float(args.max_lbs_recon_p95_bbox),
            "max_lbs_recon_p99_bbox": float(args.max_lbs_recon_p99_bbox),
            "max_lbs_recon_max_bbox": float(args.max_lbs_recon_max_bbox),
            "lbs_sample_vertices": int(args.lbs_sample_vertices),
            "lbs_sample_frames": int(args.lbs_sample_frames),
        }
        for path in paths
    ]

    rows: list[dict[str, Any]]
    if int(args.workers) <= 1:
        rows = []
        for idx, job in enumerate(jobs, 1):
            rows.append(check_job(job))
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(json.dumps({"event": "pass1_precheck_progress", "done": idx}, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            rows = list(executor.map(check_job, jobs, chunksize=8))

    accepted = [row for row in rows if row["status"] == "pass1_precheck_accept"]
    rejected = [row for row in rows if row["status"] != "pass1_precheck_accept"]

    seen_keys: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicate_rejects: list[dict[str, Any]] = []
    for row in sorted(accepted, key=lambda item: (item["asset_id"], item["sequence_id"], item["path"])):
        key = f"{row['asset_id']}::{row['sequence_id']}"
        if key in seen_keys:
            duplicate_rejects.append({**row, "status": "pass1_precheck_reject", "reasons": ["duplicate_asset_sequence"]})
            continue
        seen_keys.add(key)
        deduped.append(row)
    accepted = deduped
    rejected.extend(duplicate_rejects)

    if int(args.target_assets) > 0 and len(accepted) > int(args.target_assets):
        rng = random.Random(int(args.seed))
        shuffled = accepted[:]
        rng.shuffle(shuffled)
        keep = {f"{row['asset_id']}::{row['sequence_id']}" for row in shuffled[: int(args.target_assets)]}
        overflow = [row for row in accepted if f"{row['asset_id']}::{row['sequence_id']}" not in keep]
        rejected.extend({**row, "status": "pass1_precheck_reject", "reasons": ["over_target_assets"]} for row in overflow)
        accepted = [row for row in accepted if f"{row['asset_id']}::{row['sequence_id']}" in keep]

    split_map = load_split_map(args.split_map_csv)
    if split_map:
        missing = [row["asset_id"] for row in accepted if row["asset_id"] not in split_map]
        if missing:
            preview = ", ".join(missing[:20])
            raise SystemExit(f"split_map_csv missing {len(missing)} accepted assets; first ids: {preview}")
        splits = {f"{row['asset_id']}::{row['sequence_id']}": split_map[row["asset_id"]] for row in accepted}
    else:
        splits = split_assets(
            [f"{row['asset_id']}::{row['sequence_id']}" for row in accepted],
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )

    manifest_rows: list[dict[str, Any]] = []
    for row in accepted:
        key = f"{row['asset_id']}::{row['sequence_id']}"
        split = splits[key]
        path = Path(str(row["path"]))
        manifest_rows.append(
            {
                "asset_id": row["asset_id"],
                "sequence_id": row["sequence_id"],
                "path": str(path),
                "raw_path": str(path),
                "rel": f"articulationxl/{path.name}",
                "split": split,
                "pass1_precheck_metrics": row["metrics"],
            }
        )

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    train = [row for row in manifest_rows if row["split"] == "train"]
    val = [row for row in manifest_rows if row["split"] == "val"]
    test = [row for row in manifest_rows if row["split"] == "test"]
    write_jsonl(out / "accepted.jsonl", manifest_rows)
    write_jsonl(out / "rejected.jsonl", rejected)
    write_jsonl(out / "train_manifest.jsonl", train)
    write_jsonl(out / "val_manifest.jsonl", val)
    write_jsonl(out / "test_manifest.jsonl", test)
    write_metrics_csv(out / "pass1_precheck_metrics.csv", accepted + rejected)

    reason_counts = Counter(reason for row in rejected for reason in row.get("reasons", []))
    summary = {
        "stage": "pass1_precheck",
        "input_files": len(paths),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "top_reject_reasons": reason_counts.most_common(50),
        "policy": "raw readability only; root/rootless target quality happens after rootless rewrite",
        "split_map_csv": str(args.split_map_csv) if args.split_map_csv is not None else None,
        "seed": int(args.seed),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
