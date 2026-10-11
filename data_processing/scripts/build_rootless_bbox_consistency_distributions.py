#!/usr/bin/env python3
"""Build final skeleton/mesh bbox-consistency histogram screening.

This is the compact final geometry screening stage after rootless rewriting. It
reads rootless NPZ target fields and compares rootless skeleton extent to the
retained mesh extent using symmetric bbox diagonal consistency scores.

Scores are in ``[0, 1]``: 1 means skeleton and mesh bbox diagonals match, while
small scores indicate either a collapsed/under-covering skeleton or an
over-expanded skeleton/coordinate mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = [
    "target_joint_mesh_bbox_diag_consistency",
    "active_joint_mesh_bbox_diag_consistency",
]
DEFAULT_TARGET_MESH_PADDING = 1.1
COORDINATE_ABS_LIMIT = 1.0
COORDINATE_METRIC_COLUMNS = [
    "target_joint_mesh_normalized_coordinate_maxabs",
    "target_joint_mesh_normalized_coordinate_violating_joint_frame_count",
    "target_joint_mesh_normalized_coordinate_violating_frame_count",
    "target_joint_mesh_normalized_coordinate_worst_frame",
    "target_joint_mesh_normalized_coordinate_worst_joint",
    "target_joint_mesh_normalized_coordinate_worst_axis",
    "target_joint_mesh_normalized_coordinate_worst_value",
    "target_mesh_padding",
    "coordinate_abs_limit",
]
SCREENING_EVIDENCE_COLUMNS = [*METRIC_COLUMNS, *COORDINATE_METRIC_COLUMNS]
PROGRESS_INTERVAL = 200


def threshold_args(args: argparse.Namespace) -> dict[str, float]:
    target_mesh_padding = validate_target_mesh_padding(
        getattr(args, "target_mesh_padding", DEFAULT_TARGET_MESH_PADDING)
    )
    return {
        "min_target_joint_mesh_bbox_consistency": float(args.min_target_joint_mesh_bbox_consistency),
        "min_active_joint_mesh_bbox_consistency": float(args.min_active_joint_mesh_bbox_consistency),
        "target_mesh_padding": target_mesh_padding,
        "coordinate_abs_limit": COORDINATE_ABS_LIMIT,
    }


def safe_float(value: float | np.floating) -> float:
    out = float(value)
    return out if math.isfinite(out) else 0.0


def validate_target_mesh_padding(value: float) -> float:
    padding = float(value)
    if not math.isfinite(padding) or padding < 1.0:
        raise ValueError("target_mesh_padding must be finite and >= 1.0")
    with np.errstate(over="ignore", invalid="ignore"):
        padding32 = np.float32(padding)
    if not np.isfinite(padding32):
        raise ValueError("target_mesh_padding must be representable as finite float32")
    return padding


def choose_frame(count: int, frame_index: int) -> int:
    if count <= 0:
        raise ValueError("empty frame array")
    if frame_index < 0:
        return max(0, count + frame_index)
    return min(int(frame_index), count - 1)


def bbox_diag(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        return 0.0
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))


def bbox_diag_consistency(a: float, b: float) -> float:
    x = safe_float(a)
    y = safe_float(b)
    denom = max(x, y, 1.0e-8)
    return safe_float(min(x, y) / denom)


def compute_target_joint_mesh_normalized_coordinate_metrics(
    frame_vertices: np.ndarray,
    joints_all: np.ndarray,
    *,
    target_mesh_padding: float = DEFAULT_TARGET_MESH_PADDING,
) -> dict[str, Any]:
    vertices = np.asarray(frame_vertices, dtype=np.float32)
    joints = np.asarray(joints_all, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] == 0 or vertices.shape[1] == 0:
        raise ValueError(f"frame_vertices_rootspace must be nonempty [T,V,3], got {vertices.shape}")
    if joints.ndim != 3 or joints.shape[0] != vertices.shape[0] or joints.shape[-1] != 3 or joints.shape[1] == 0:
        raise ValueError(f"target_joints_rootspace must be nonempty [T,J,3] compatible with frames {vertices.shape}, got {joints.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("frame_vertices_rootspace contains nonfinite values")
    if not np.isfinite(joints).all():
        raise ValueError("target_joints_rootspace contains nonfinite values")

    padding = validate_target_mesh_padding(target_mesh_padding)
    padding32 = np.float32(padding)
    limit32 = np.float32(COORDINATE_ABS_LIMIT)
    maxabs32 = np.float32(0.0)
    violating_joint_frame_count = 0
    violating_frame_count = 0
    worst_frame = -1
    worst_joint = -1
    worst_axis = -1
    worst_value = np.float32(0.0)

    for frame_idx in range(int(vertices.shape[0])):
        frame = vertices[frame_idx]
        joints_frame = joints[frame_idx]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lo = np.min(frame, axis=0)
            hi = np.max(frame, axis=0)
            center = (lo + hi) * np.float32(0.5)
            scale = np.max((hi - lo) * np.float32(0.5)) * padding32
            normalized = (joints_frame - center) / scale
        if not np.isfinite(center).all() or not np.isfinite(scale) or float(scale) <= 0.0:
            raise ValueError(f"invalid target mesh coordinate scale at frame {frame_idx}: {scale!r}")
        if not np.isfinite(normalized).all():
            raise ValueError(f"nonfinite normalized target joints at frame {frame_idx}")

        absolute = np.abs(normalized)
        frame_maxabs32 = np.max(absolute)
        if worst_frame < 0 or frame_maxabs32 > maxabs32:
            maxabs32 = frame_maxabs32
            frame_worst = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
            worst_frame = int(frame_idx)
            worst_joint = int(frame_worst[0])
            worst_axis = int(frame_worst[1])
            worst_value = normalized[frame_worst]

        violating_joint_mask = np.any(absolute > limit32, axis=1)
        violating_joint_frame_count += int(np.count_nonzero(violating_joint_mask))
        violating_frame_count += int(np.any(violating_joint_mask))

    return {
        "target_joint_mesh_normalized_coordinate_maxabs": float(maxabs32),
        "target_joint_mesh_normalized_coordinate_violating_joint_frame_count": violating_joint_frame_count,
        "target_joint_mesh_normalized_coordinate_violating_frame_count": violating_frame_count,
        "target_joint_mesh_normalized_coordinate_worst_frame": worst_frame,
        "target_joint_mesh_normalized_coordinate_worst_joint": worst_joint,
        "target_joint_mesh_normalized_coordinate_worst_axis": worst_axis,
        "target_joint_mesh_normalized_coordinate_worst_value": float(worst_value),
        "target_mesh_padding": padding,
        "coordinate_abs_limit": COORDINATE_ABS_LIMIT,
    }


def compute_bbox_consistency_metrics(
    path: Path,
    *,
    active_skin_threshold: float,
    frame_index: int,
    target_mesh_padding: float = DEFAULT_TARGET_MESH_PADDING,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        required = [
            "frame_vertices_rootspace",
            "target_joints_rootspace",
            "target_skin_weights",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"missing rootless fields: {','.join(missing)}")
        frame_vertices = np.asarray(data["frame_vertices_rootspace"], dtype=np.float32)
        joints_all = np.asarray(data["target_joints_rootspace"], dtype=np.float32)
        skin = np.asarray(data["target_skin_weights"], dtype=np.float32)

    if frame_vertices.ndim != 3 or frame_vertices.shape[-1] != 3:
        raise ValueError(f"frame_vertices_rootspace must be [T,V,3], got {frame_vertices.shape}")
    if joints_all.ndim != 3 or joints_all.shape[0] != frame_vertices.shape[0] or joints_all.shape[-1] != 3:
        raise ValueError(f"target_joints_rootspace shape {joints_all.shape} incompatible with frames {frame_vertices.shape}")
    if skin.ndim != 2 or skin.shape != (frame_vertices.shape[1], joints_all.shape[1]):
        raise ValueError(f"target_skin_weights shape {skin.shape} incompatible with vertices/joints")
    if frame_vertices.shape[0] == 0 or frame_vertices.shape[1] == 0 or joints_all.shape[1] == 0:
        raise ValueError(
            "frame_vertices_rootspace and target_joints_rootspace must contain at least one frame, vertex, and joint"
        )
    if not np.isfinite(skin).all():
        raise ValueError("target_skin_weights contains nonfinite values")

    coordinate_metrics = compute_target_joint_mesh_normalized_coordinate_metrics(
        frame_vertices,
        joints_all,
        target_mesh_padding=target_mesh_padding,
    )

    idx = choose_frame(int(frame_vertices.shape[0]), int(frame_index))
    vertices = frame_vertices[idx]
    joints = joints_all[idx]
    mesh_diag = max(bbox_diag(vertices), 1.0e-8)
    joint_diag = bbox_diag(joints)

    active = skin.sum(axis=0) > float(active_skin_threshold)
    active_joints = joints[active] if np.any(active) else np.zeros((0, 3), dtype=np.float32)
    active_joint_diag = bbox_diag(active_joints)

    return {
        **coordinate_metrics,
        "rootless_bbox_frame_index": int(idx),
        "rootless_mesh_bbox_diag": safe_float(mesh_diag),
        "rootless_target_joint_bbox_diag": safe_float(joint_diag),
        "rootless_active_joint_bbox_diag": safe_float(active_joint_diag),
        "rootless_joint_count": int(joints.shape[0]),
        "rootless_active_joint_count": int(active_joints.shape[0]),
        "target_joint_mesh_bbox_diag_consistency": bbox_diag_consistency(joint_diag, mesh_diag),
        "active_joint_mesh_bbox_diag_consistency": bbox_diag_consistency(active_joint_diag, mesh_diag),
    }


def split_from_manifest(path: Path) -> str:
    for split in ("train", "val", "test"):
        if path.name.startswith(split):
            return split
    return ""


def load_manifest_jobs(manifests: list[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for manifest in manifests:
        split = split_from_manifest(manifest)
        with manifest.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_path = Path(str(row.get("path", "")))
                path = raw_path if raw_path.is_absolute() else (manifest.parent / raw_path).resolve()
                jobs.append(
                    {
                        "split": str(row.get("split") or split),
                        "row_index": int(row_index),
                        "asset_id": str(row.get("asset_id") or path.stem),
                        "path": str(path),
                        "manifest_row": row,
                    }
                )
    return jobs


def analyze_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        metrics = compute_bbox_consistency_metrics(
            Path(str(job["path"])),
            active_skin_threshold=float(job["active_skin_threshold"]),
            frame_index=int(job["frame_index"]),
            target_mesh_padding=float(job.get("target_mesh_padding", DEFAULT_TARGET_MESH_PADDING)),
        )
        return {**job, "status": "ok", "reason": "", **metrics}
    except Exception as exc:  # noqa: BLE001
        return {**job, "status": "error", "reason": repr(exc)}


def analyze_jobs(jobs: list[dict[str, Any]], *, workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(jobs)

    def append_with_progress(row: dict[str, Any]) -> None:
        rows.append(row)
        done = len(rows)
        if done % PROGRESS_INTERVAL == 0 or done == total:
            errors = sum(result.get("status") == "error" for result in rows)
            print(f"bbox consistency progress: {done}/{total} done, errors={errors}", flush=True)

    if int(workers) <= 1:
        for job in jobs:
            append_with_progress(analyze_job(job))
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            for row in executor.map(analyze_job, jobs, chunksize=8):
                append_with_progress(row)
    return rows


def screening_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if row.get("status") != "ok":
        return ["bbox_consistency_metric_error"]

    reasons: list[str] = []
    try:
        coordinate_values = {
            key: float(row[key])
            for key in COORDINATE_METRIC_COLUMNS
        }
    except (TypeError, ValueError):
        return ["bbox_consistency_metric_error"]
    if not all(math.isfinite(value) for value in coordinate_values.values()):
        return ["bbox_consistency_metric_error"]
    if coordinate_values["coordinate_abs_limit"] != COORDINATE_ABS_LIMIT:
        return ["bbox_consistency_metric_error"]
    if coordinate_values["target_mesh_padding"] < 1.0:
        return ["bbox_consistency_metric_error"]
    coordinate_maxabs = coordinate_values["target_joint_mesh_normalized_coordinate_maxabs"]
    if coordinate_maxabs < 0.0:
        return ["bbox_consistency_metric_error"]
    if coordinate_maxabs > COORDINATE_ABS_LIMIT:
        reasons.append("target_joint_mesh_normalized_coordinate_out_of_bounds")

    min_joint = float(args.min_target_joint_mesh_bbox_consistency)
    min_active = float(args.min_active_joint_mesh_bbox_consistency)

    target_joint = float(row.get("target_joint_mesh_bbox_diag_consistency", 0.0))
    active_joint = float(row.get("active_joint_mesh_bbox_diag_consistency", 0.0))

    if min_joint > 0.0 and target_joint < min_joint:
        reasons.append("target_joint_mesh_bbox_diag_consistency_below_threshold")
    if min_active > 0.0 and active_joint < min_active:
        reasons.append("active_joint_mesh_bbox_diag_consistency_below_threshold")

    return reasons


def apply_screening(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    thresholds = threshold_args(args)
    for row in rows:
        reasons = screening_reasons(row, args)
        row["screening_status"] = "reject" if reasons else "accept"
        row["screening_reasons"] = reasons
        row["screening_thresholds"] = thresholds


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "status",
        "screening_status",
        "split",
        "row_index",
        "asset_id",
        "path",
        "reason",
        "screening_reasons",
        "rootless_bbox_frame_index",
        "rootless_mesh_bbox_diag",
        "rootless_target_joint_bbox_diag",
        "rootless_active_joint_bbox_diag",
        "rootless_joint_count",
        "rootless_active_joint_count",
        *METRIC_COLUMNS,
        *COORDINATE_METRIC_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["screening_reasons"] = json.dumps(row.get("screening_reasons", []), ensure_ascii=False)
            writer.writerow(out)


def final_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row.get("manifest_row") or {})
    if row.get("source_split") is not None:
        out.setdefault("source_split", row.get("source_split"))
    out["final_bbox_consistency_screening"] = {
        "status": row.get("screening_status"),
        "reasons": list(row.get("screening_reasons", [])),
        "thresholds": dict(row.get("screening_thresholds", {})),
        "metrics": {
            key: row.get(key)
            for key in [
                "rootless_bbox_frame_index",
                "rootless_mesh_bbox_diag",
                "rootless_target_joint_bbox_diag",
                "rootless_active_joint_bbox_diag",
                "rootless_joint_count",
                "rootless_active_joint_count",
                *SCREENING_EVIDENCE_COLUMNS,
            ]
        },
    }
    return out


def final_split(split: str) -> str:
    value = str(split).strip().lower()
    if value in {"train", "test"}:
        return "train"
    if value in {"val", "valid", "validation"}:
        return "valid"
    raise ValueError(f"unknown split for final manifest: {split!r}")


def with_final_split(row: dict[str, Any]) -> dict[str, Any]:
    split = final_split(str(row.get("split", "")))
    out = dict(row)
    manifest = dict(out.get("manifest_row") or {})
    source_split = str(
        manifest.get("source_split")
        or manifest.get("split")
        or row.get("source_split")
        or row.get("split")
        or ""
    )
    manifest["split"] = split
    out["split"] = split
    out["source_split"] = source_split
    out["manifest_row"] = manifest
    return out


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(final_manifest_row(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_screened_manifests(rows: list[dict[str, Any]], out_dir: Path) -> None:
    manifest_dir = out_dir / "final_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    accepted = [with_final_split(row) for row in rows if row.get("screening_status") == "accept"]
    rejected = [with_final_split(row) for row in rows if row.get("screening_status") == "reject"]

    write_jsonl(accepted, manifest_dir / "accepted_manifest.jsonl")
    write_jsonl(rejected, manifest_dir / "rejected_manifest.jsonl")
    for stale_name in ("val_manifest.jsonl", "val_rejected.jsonl", "test_manifest.jsonl", "test_rejected.jsonl"):
        stale_path = manifest_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for split in ("train", "valid"):
        split_accepted = [row for row in accepted if str(row.get("split")) == split]
        split_rejected = [row for row in rejected if str(row.get("split")) == split]
        write_jsonl(split_accepted, manifest_dir / f"{split}_manifest.jsonl")
        write_jsonl(split_rejected, manifest_dir / f"{split}_rejected.jsonl")

    summary = {
        "total": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "status_counts": {
            "accept": len(accepted),
            "reject": len(rejected),
        },
        "split_policy": "final manifests use train/valid only; original test rows are merged into train",
        "accepted_split_counts": {
            "train": sum(1 for row in accepted if row.get("split") == "train"),
            "valid": sum(1 for row in accepted if row.get("split") == "valid"),
        },
        "rejected_split_counts": {
            "train": sum(1 for row in rejected if row.get("split") == "train"),
            "valid": sum(1 for row in rejected if row.get("split") == "valid"),
        },
        "reject_reasons": {},
    }
    for row in rejected:
        for reason in row.get("screening_reasons", []):
            summary["reject_reasons"][reason] = int(summary["reject_reasons"].get(reason, 0)) + 1
    (manifest_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            value = float(row.get(key, ""))
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def plot_single(values: np.ndarray, key: str, out_path: Path, bins: int) -> None:
    if values.size == 0:
        return
    lo = float(np.quantile(values, 0.005))
    hi = float(np.quantile(values, 0.995))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    clipped = values[(values >= lo) & (values <= hi)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.hist(clipped, bins=bins, color="#2563eb", alpha=0.78)
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    ax.set_title(f"{key} distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_grid(rows: list[dict[str, Any]], out_path: Path, bins: int) -> None:
    fig, axes = plt.subplots(1, len(METRIC_COLUMNS), figsize=(5.2 * len(METRIC_COLUMNS), 4.0), dpi=160)
    axes = np.asarray(axes).ravel()
    for ax, key in zip(axes, METRIC_COLUMNS):
        values = metric_array(rows, key)
        if values.size:
            lo = float(np.quantile(values, 0.005))
            hi = float(np.quantile(values, 0.995))
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                lo = float(values.min())
                hi = float(values.max())
            if hi <= lo:
                hi = lo + 1.0
            clipped = values[(values >= lo) & (values <= hi)]
            ax.hist(clipped, bins=bins, color="#2563eb", alpha=0.78)
        ax.set_title(key, fontsize=8)
        ax.grid(alpha=0.18)
    fig.suptitle("Final skeleton/mesh bbox-consistency histogram screening", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    status_counts: dict[str, int] = {}
    screening_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        screening_status = str(row.get("screening_status", ""))
        screening_counts[screening_status] = screening_counts.get(screening_status, 0) + 1
    lines = [
        "# Final Skeleton/Mesh BBox-Consistency Histogram Screening",
        "",
        "These metrics are computed from rootless NPZ target fields.",
        "They compare rootless skeleton extent to retained mesh extent by symmetric bbox diagonal consistency.",
        "Score formula: `min(skeleton_bbox_diag, mesh_bbox_diag) / max(skeleton_bbox_diag, mesh_bbox_diag)`.",
        "This is the final histogram-based data-screening layer.",
        "",
        f"Frame index: {args.frame_index}",
        f"Active skin threshold: {args.active_skin_threshold}",
        f"Screening thresholds: `{json.dumps(threshold_args(args), sort_keys=True)}`",
        "Coordinate gate: all target joints are normalized against each frame's retained mesh bounds in float32; inclusive [-1, 1] is accepted.",
        f"Total rows: {len(rows)}",
        f"Status counts: {status_counts}",
        f"Screening counts: {screening_counts}",
        "",
        "| metric | count | min | p05 | p25 | p50 | p75 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METRIC_COLUMNS:
        values = metric_array(rows, key)
        q = quantiles(values)
        lines.append(
            f"| {key} | {values.size} | {q['min']:.6g} | {q['p05']:.6g} | {q['p25']:.6g} | "
            f"{q['p50']:.6g} | {q['p75']:.6g} | {q['p95']:.6g} | {q['max']:.6g} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--active-skin-threshold", type=float, default=1.0e-4)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--target-mesh-padding", type=float, default=DEFAULT_TARGET_MESH_PADDING)
    parser.add_argument("--min-target-joint-mesh-bbox-consistency", type=float, default=0.4)
    parser.add_argument("--min-active-joint-mesh-bbox-consistency", type=float, default=0.0)
    args = parser.parse_args()
    try:
        args.target_mesh_padding = validate_target_mesh_padding(args.target_mesh_padding)
    except ValueError as exc:
        parser.error(str(exc))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics").mkdir(exist_ok=True)
    jobs = load_manifest_jobs(args.manifest)
    for job in jobs:
        job["active_skin_threshold"] = float(args.active_skin_threshold)
        job["frame_index"] = int(args.frame_index)
        job["target_mesh_padding"] = float(args.target_mesh_padding)

    rows = analyze_jobs(jobs, workers=int(args.workers))

    apply_screening(rows, args)
    write_csv(rows, args.out_dir / "rootless_bbox_consistency_metrics.csv")
    plot_grid(rows, args.out_dir / "rootless_bbox_consistency_distribution_grid.png", args.bins)
    for key in METRIC_COLUMNS:
        plot_single(metric_array(rows, key), key, args.out_dir / "metrics" / f"{key}.png", args.bins)
    write_screened_manifests(rows, args.out_dir)
    write_report(rows, args.out_dir / "rootless_bbox_consistency_report.md", args)


if __name__ == "__main__":
    main()
