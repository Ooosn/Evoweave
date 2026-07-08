#!/usr/bin/env python3
"""Build final dynamic quality histogram screening from rootless NPZs.

This is the final data-screening layer after the rootless NPZ rewrite. It reads
rootless training manifests and computes query-frame-normalized dynamic metrics
directly from ``frame_vertices_rootspace`` and rootless target fields.
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

from rigweave.data.query_space import normalize_query_root_space


SCORE_COLUMNS = [
    "motion_coverage_score",
    "motion_amount_score",
    "motion_validity_score",
    "bbox_stability_score",
    "edge_stretch_stability_score",
    "edge_collapse_stability_score",
    "spike_cleanliness_score",
    "geometry_stability_score",
    "overall_quality_score",
]

METRIC_COLUMNS = [
    "qr_query_count",
    "qr_motion_rate_max",
    "qr_motion_rate_p50",
    "qr_motion_p50_max",
    "qr_motion_p95_max",
    "qr_motion_p99_max",
    "qr_motion_max",
    "qr_spike_score_max",
    "qr_bbox_ratio_min",
    "qr_bbox_ratio_max",
    "qr_edge_stretch_ratio_p99_max",
    "qr_edge_stretch_ratio_max",
    "qr_edge_collapse_ratio_min",
]


def safe_float(value: float | np.floating, default: float = 0.0) -> float:
    out = float(value)
    return out if math.isfinite(out) else float(default)


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def increasing_ramp(value: float, bad: float, good: float) -> float:
    if good <= bad:
        raise ValueError("good must be greater than bad")
    return clip01((value - bad) / (good - bad))


def decreasing_log_ramp(value: float, good: float, bad: float) -> float:
    if good <= 0 or bad <= good:
        raise ValueError("need 0 < good < bad")
    value = max(float(value), 1.0e-12)
    t = (math.log(value) - math.log(good)) / (math.log(bad) - math.log(good))
    return clip01(1.0 - t)


def increasing_log_ramp(value: float, bad: float, good: float) -> float:
    if bad <= 0 or good <= bad:
        raise ValueError("need 0 < bad < good")
    value = max(float(value), 1.0e-12)
    t = (math.log(value) - math.log(bad)) / (math.log(good) - math.log(bad))
    return clip01(t)


def bbox_stability(min_ratio: float, max_ratio: float, good_factor: float, bad_factor: float) -> float:
    min_ratio = max(float(min_ratio), 1.0e-12)
    max_ratio = max(float(max_ratio), 1.0e-12)
    factor = max(max_ratio, 1.0 / min_ratio)
    return decreasing_log_ramp(factor, good=good_factor, bad=bad_factor)


def compute_scores(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    motion_coverage = clip01(float(metrics.get("qr_motion_rate_max", 0.0)))
    motion_amount = increasing_ramp(
        float(metrics.get("qr_motion_p95_max", 0.0)),
        bad=float(args.motion_amount_bad),
        good=float(args.motion_amount_good),
    )
    motion_validity = min(motion_coverage, motion_amount)
    bbox_score = bbox_stability(
        float(metrics.get("qr_bbox_ratio_min", 1.0)),
        float(metrics.get("qr_bbox_ratio_max", 1.0)),
        good_factor=float(args.bbox_good_factor),
        bad_factor=float(args.bbox_bad_factor),
    )
    edge_stretch_score = decreasing_log_ramp(
        float(metrics.get("qr_edge_stretch_ratio_p99_max", 1.0)),
        good=float(args.edge_stretch_good),
        bad=float(args.edge_stretch_bad),
    )
    edge_collapse_score = increasing_log_ramp(
        float(metrics.get("qr_edge_collapse_ratio_min", 1.0)),
        bad=float(args.edge_collapse_bad),
        good=float(args.edge_collapse_good),
    )
    spike_score = decreasing_log_ramp(
        max(float(metrics.get("qr_spike_score_max", 1.0)), 1.0e-8),
        good=float(args.spike_good),
        bad=float(args.spike_bad),
    )
    geometry_stability = min(bbox_score, edge_stretch_score, edge_collapse_score, spike_score)
    overall = min(motion_validity, geometry_stability)
    return {
        "motion_coverage_score": motion_coverage,
        "motion_amount_score": motion_amount,
        "motion_validity_score": motion_validity,
        "bbox_stability_score": bbox_score,
        "edge_stretch_stability_score": edge_stretch_score,
        "edge_collapse_stability_score": edge_collapse_score,
        "spike_cleanliness_score": spike_score,
        "geometry_stability_score": geometry_stability,
        "overall_quality_score": overall,
    }


def sample_edges(faces: np.ndarray, max_edges: int = 20000) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    max_faces = max(1, int(max_edges) // 3)
    if faces.shape[0] > max_faces:
        ids = np.linspace(0, faces.shape[0] - 1, max_faces).round().astype(np.int64)
        faces = faces[ids]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if edges.shape[0] > max_edges:
        ids = np.linspace(0, edges.shape[0] - 1, max_edges).round().astype(np.int64)
        edges = edges[ids]
    return edges.astype(np.int64)


def rootless_dynamic_metrics(
    frame_vertices: np.ndarray,
    faces: np.ndarray,
    target_joints: np.ndarray,
    *,
    motion_rate_eps: float,
    max_queries: int,
) -> dict[str, Any]:
    frames = np.asarray(frame_vertices, dtype=np.float32)
    joints = np.asarray(target_joints, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError(f"frame_vertices_rootspace must be [T,V,3], got {frames.shape}")
    if joints.ndim != 3 or joints.shape[0] != frames.shape[0] or joints.shape[-1] != 3:
        raise ValueError(f"target_joints_rootspace shape {joints.shape} incompatible with frames {frames.shape}")

    query_ids = np.linspace(0, frames.shape[0] - 1, min(int(max_queries), frames.shape[0])).round().astype(np.int64)
    edges = sample_edges(faces)
    motion_rates: list[float] = []
    motion_p50: list[float] = []
    motion_p95: list[float] = []
    motion_p99: list[float] = []
    motion_max: list[float] = []
    spike_scores: list[float] = []
    bbox_ratio_mins: list[float] = []
    bbox_ratio_maxs: list[float] = []
    edge_stretch_p99: list[float] = []
    edge_stretch_max: list[float] = []
    edge_collapse_min: list[float] = []

    for query_idx in query_ids.tolist():
        frames_n, _joints_n, _tails_n, _transform = normalize_query_root_space(
            frames,
            joints,
            None,
            None,
            int(query_idx),
        )
        disp = np.linalg.norm(frames_n - frames_n[int(query_idx) : int(query_idx) + 1], axis=-1)
        per_vertex_motion = disp.max(axis=0)
        motion_rates.append(float(np.mean(per_vertex_motion > float(motion_rate_eps))))
        motion_p50.append(float(np.percentile(disp, 50)))
        motion_p95.append(float(np.percentile(disp, 95)))
        motion_p99.append(float(np.percentile(disp, 99)))
        max_disp = float(disp.max())
        motion_max.append(max_disp)
        p99 = max(float(np.percentile(disp, 99)), 1.0e-8)
        spike_scores.append(max_disp / p99)

        bbox_diags = np.linalg.norm(frames_n.max(axis=1) - frames_n.min(axis=1), axis=1)
        query_diag = max(float(bbox_diags[int(query_idx)]), 1.0e-8)
        bbox_ratios = bbox_diags / query_diag
        bbox_ratio_mins.append(float(bbox_ratios.min()))
        bbox_ratio_maxs.append(float(bbox_ratios.max()))

        if edges.shape[0] > 0:
            edge_len = np.linalg.norm(frames_n[:, edges[:, 0]] - frames_n[:, edges[:, 1]], axis=-1)
            ref = np.maximum(edge_len[int(query_idx) : int(query_idx) + 1], 1.0e-8)
            ratio = edge_len / ref
            edge_stretch_p99.append(float(np.percentile(ratio, 99)))
            edge_stretch_max.append(float(ratio.max()))
            edge_collapse_min.append(float(ratio.min()))

    return {
        "qr_query_count": int(query_ids.shape[0]),
        "qr_motion_rate_max": safe_float(max(motion_rates) if motion_rates else 0.0),
        "qr_motion_rate_p50": safe_float(np.percentile(motion_rates, 50) if motion_rates else 0.0),
        "qr_motion_p50_max": safe_float(max(motion_p50) if motion_p50 else 0.0),
        "qr_motion_p95_max": safe_float(max(motion_p95) if motion_p95 else 0.0),
        "qr_motion_p99_max": safe_float(max(motion_p99) if motion_p99 else 0.0),
        "qr_motion_max": safe_float(max(motion_max) if motion_max else 0.0),
        "qr_spike_score_max": safe_float(max(spike_scores) if spike_scores else 0.0),
        "qr_bbox_ratio_min": safe_float(min(bbox_ratio_mins) if bbox_ratio_mins else 0.0),
        "qr_bbox_ratio_max": safe_float(max(bbox_ratio_maxs) if bbox_ratio_maxs else 0.0),
        "qr_edge_stretch_ratio_p99_max": safe_float(max(edge_stretch_p99) if edge_stretch_p99 else 0.0),
        "qr_edge_stretch_ratio_max": safe_float(max(edge_stretch_max) if edge_stretch_max else 0.0),
        "qr_edge_collapse_ratio_min": safe_float(min(edge_collapse_min) if edge_collapse_min else 1.0),
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
                    }
                )
    return jobs


def analyze_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        with np.load(Path(str(job["path"])), allow_pickle=True) as data:
            required = ["frame_vertices_rootspace", "target_joints_rootspace", "faces"]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"missing rootless fields: {','.join(missing)}")
            metrics = rootless_dynamic_metrics(
                np.asarray(data["frame_vertices_rootspace"], dtype=np.float32),
                np.asarray(data["faces"], dtype=np.int64),
                np.asarray(data["target_joints_rootspace"], dtype=np.float32),
                motion_rate_eps=float(job["motion_rate_eps"]),
                max_queries=int(job["max_queries"]),
            )
        scores = compute_scores(metrics, argparse.Namespace(**job["score_args"]))
        return {**job, "status": "ok", "reason": "", **metrics, **scores}
    except Exception as exc:  # noqa: BLE001
        return {**job, "status": "error", "reason": repr(exc)}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["status", "split", "row_index", "asset_id", "path", "reason", *METRIC_COLUMNS, *SCORE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def score_array(rows: list[dict[str, Any]], score: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            value = float(row.get(score, ""))
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def plot_grid(rows: list[dict[str, Any]], out_path: Path, bins: int) -> None:
    cols = 3
    grid_rows = int(math.ceil(len(SCORE_COLUMNS) / cols))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(13.5, 3.5 * grid_rows), dpi=160)
    axes = np.asarray(axes).ravel()
    for ax, score in zip(axes, SCORE_COLUMNS):
        values = score_array(rows, score)
        counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
        frac = counts.astype(np.float64) / max(1, values.size)
        ax.bar(edges[:-1], frac, width=np.diff(edges), align="edge", color="#2563eb", alpha=0.72)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(score, fontsize=9)
        ax.grid(alpha=0.18)
    for ax in axes[len(SCORE_COLUMNS) :]:
        ax.axis("off")
    fig.suptitle("Final query-normalized dynamic quality histogram screening", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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


def write_report(rows: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Final Dynamic Quality Histogram Screening",
        "",
        "These histograms are computed from rootless NPZ files after the rootless rewrite.",
        "This is the final histogram-based data-screening layer.",
        "",
        f"Total rows: {len(rows)}",
        f"Status counts: {status_counts}",
        "",
        "## Quantiles",
        "",
        "| score | min | p05 | p25 | p50 | p75 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for score in SCORE_COLUMNS:
        q = quantiles(score_array(rows, score))
        lines.append(
            f"| {score} | {q['min']:.4f} | {q['p05']:.4f} | {q['p25']:.4f} | "
            f"{q['p50']:.4f} | {q['p75']:.4f} | {q['p95']:.4f} | {q['max']:.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--motion-rate-eps", type=float, default=0.01)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--motion-amount-bad", type=float, default=0.02)
    parser.add_argument("--motion-amount-good", type=float, default=0.50)
    parser.add_argument("--bbox-good-factor", type=float, default=1.25)
    parser.add_argument("--bbox-bad-factor", type=float, default=4.0)
    parser.add_argument("--edge-stretch-good", type=float, default=1.5)
    parser.add_argument("--edge-stretch-bad", type=float, default=8.0)
    parser.add_argument("--edge-collapse-bad", type=float, default=0.01)
    parser.add_argument("--edge-collapse-good", type=float, default=0.05)
    parser.add_argument("--spike-good", type=float, default=2.0)
    parser.add_argument("--spike-bad", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_manifest_jobs(args.manifest)
    score_args = {
        "motion_amount_bad": args.motion_amount_bad,
        "motion_amount_good": args.motion_amount_good,
        "bbox_good_factor": args.bbox_good_factor,
        "bbox_bad_factor": args.bbox_bad_factor,
        "edge_stretch_good": args.edge_stretch_good,
        "edge_stretch_bad": args.edge_stretch_bad,
        "edge_collapse_bad": args.edge_collapse_bad,
        "edge_collapse_good": args.edge_collapse_good,
        "spike_good": args.spike_good,
        "spike_bad": args.spike_bad,
    }
    for job in jobs:
        job["motion_rate_eps"] = float(args.motion_rate_eps)
        job["max_queries"] = int(args.max_queries)
        job["score_args"] = score_args

    if int(args.workers) <= 1:
        rows = [analyze_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            rows = list(executor.map(analyze_job, jobs, chunksize=8))

    write_csv(rows, args.out_dir / "rootless_query_quality_scores.csv")
    plot_grid(rows, args.out_dir / "rootless_query_quality_score_distribution_grid.png", args.bins)
    write_report(rows, args.out_dir / "rootless_query_quality_score_report.md", args)


if __name__ == "__main__":
    main()
