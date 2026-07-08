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


def threshold_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_target_joint_mesh_bbox_consistency": float(args.min_target_joint_mesh_bbox_consistency),
        "min_active_joint_mesh_bbox_consistency": float(args.min_active_joint_mesh_bbox_consistency),
    }


def safe_float(value: float | np.floating) -> float:
    out = float(value)
    return out if math.isfinite(out) else 0.0


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


def compute_bbox_consistency_metrics(path: Path, *, active_skin_threshold: float, frame_index: int) -> dict[str, Any]:
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

    idx = choose_frame(int(frame_vertices.shape[0]), int(frame_index))
    vertices = frame_vertices[idx]
    joints = joints_all[idx]
    mesh_diag = max(bbox_diag(vertices), 1.0e-8)
    joint_diag = bbox_diag(joints)

    active = skin.sum(axis=0) > float(active_skin_threshold)
    active_joints = joints[active] if np.any(active) else np.zeros((0, 3), dtype=np.float32)
    active_joint_diag = bbox_diag(active_joints)

    return {
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
        )
        return {**job, "status": "ok", "reason": "", **metrics}
    except Exception as exc:  # noqa: BLE001
        return {**job, "status": "error", "reason": repr(exc)}


def screening_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if row.get("status") != "ok":
        return ["bbox_consistency_metric_error"]

    reasons: list[str] = []
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
                *METRIC_COLUMNS,
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
    source_split = str(manifest.get("split") or row.get("split") or "")
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
    parser.add_argument("--min-target-joint-mesh-bbox-consistency", type=float, default=0.4)
    parser.add_argument("--min-active-joint-mesh-bbox-consistency", type=float, default=0.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics").mkdir(exist_ok=True)
    jobs = load_manifest_jobs(args.manifest)
    for job in jobs:
        job["active_skin_threshold"] = float(args.active_skin_threshold)
        job["frame_index"] = int(args.frame_index)

    if int(args.workers) <= 1:
        rows = [analyze_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            rows = list(executor.map(analyze_job, jobs, chunksize=8))

    apply_screening(rows, args)
    write_csv(rows, args.out_dir / "rootless_bbox_consistency_metrics.csv")
    plot_grid(rows, args.out_dir / "rootless_bbox_consistency_distribution_grid.png", args.bins)
    for key in METRIC_COLUMNS:
        plot_single(metric_array(rows, key), key, args.out_dir / "metrics" / f"{key}.png", args.bins)
    write_screened_manifests(rows, args.out_dir)
    write_report(rows, args.out_dir / "rootless_bbox_consistency_report.md", args)


if __name__ == "__main__":
    main()
