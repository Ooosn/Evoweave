#!/usr/bin/env python3
"""Build the first formal hard-rejected dataset from rootless target NPZs.

Input manifests must already point to rootless v3 NPZ files.  This stage checks
the actual training target fields and materializes an accepted-only dataset root
for validation, histograms, and training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_rootless_quality_score_distributions import compute_scores, rootless_dynamic_metrics
from validate_rootless_dynamic_npz import validate_one


SPLITS = ("train", "val", "test")


def split_from_manifest(path: Path) -> str:
    for split in SPLITS:
        if path.name.startswith(split):
            return split
    return ""


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split = split_from_manifest(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_path = Path(str(row.get("path", "")))
            row["path"] = str(raw_path if raw_path.is_absolute() else (path.parent / raw_path).resolve())
            row.setdefault("split", split)
            row["_manifest_index"] = int(index)
            rows.append(row)
    return rows


def bbox_diag(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return 0.0
    pts = pts.reshape(-1, 3)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def face_issues(faces: np.ndarray, vertex_count: int) -> list[str]:
    issues: list[str] = []
    if faces.ndim != 2 or faces.shape[1] != 3:
        return ["bad_faces_shape"]
    if faces.size:
        if int(faces.min()) < 0 or int(faces.max()) >= int(vertex_count):
            issues.append("invalid_face_indices")
        degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
        if float(np.mean(degenerate)) > 0.05:
            issues.append("too_many_degenerate_faces")
    return issues


def analyze_rootless_row(job: dict[str, Any]) -> dict[str, Any]:
    row = dict(job["row"])
    path = Path(str(row["path"]))
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "asset_id": str(row.get("asset_id") or path.stem),
        "sequence_id": str(row.get("sequence_id", "0")),
        "split": str(row.get("split", "")),
        "path": str(path),
        "row_index": int(job["row_index"]),
    }

    try:
        validation_metrics, validation_issues = validate_one(
            path,
            sample_vertices=int(job["sample_vertices"]),
            alignment_threshold=float(job["alignment_threshold"]),
        )
        metrics.update(validation_metrics)
        reasons.extend(str(issue) for issue in validation_issues)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "reject",
            "row": row,
            "metrics": metrics,
            "reasons": [f"rootless_validation_error:{repr(exc)}"],
        }

    try:
        with np.load(path, allow_pickle=True) as data:
            frame_vertices_rootspace = np.asarray(data["frame_vertices_rootspace"], dtype=np.float32)
            target_joints_rootspace = np.asarray(data["target_joints_rootspace"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.int64)
            target_skin = np.asarray(data["target_skin_weights"], dtype=np.float32)
            target_has_skin = np.asarray(data["target_has_skin"], dtype=bool).reshape(-1)
        metrics.update(
            {
                "frame_count": int(frame_vertices_rootspace.shape[0]) if frame_vertices_rootspace.ndim >= 1 else 0,
                "vertex_count": int(frame_vertices_rootspace.shape[1]) if frame_vertices_rootspace.ndim >= 2 else 0,
                "face_count": int(faces.shape[0]) if faces.ndim >= 1 else 0,
                "target_joint_count": int(target_joints_rootspace.shape[1]) if target_joints_rootspace.ndim >= 2 else 0,
                "target_active_joint_count": int(target_has_skin.sum()),
                "mesh_bbox_diag_frame0": bbox_diag(frame_vertices_rootspace[0]) if frame_vertices_rootspace.ndim == 3 and frame_vertices_rootspace.shape[0] else 0.0,
            }
        )
        reasons.extend(face_issues(faces, int(metrics["vertex_count"])))
        if int(metrics["frame_count"]) < int(job["min_frames"]):
            reasons.append("too_few_frames")
        if int(metrics["vertex_count"]) < int(job["min_vertices"]):
            reasons.append("too_few_vertices")
        if int(metrics["face_count"]) < int(job["min_faces"]):
            reasons.append("too_few_faces")
        if int(metrics["target_joint_count"]) < int(job["min_target_joints"]):
            reasons.append("too_few_target_joints")
        if int(metrics["target_active_joint_count"]) < int(job["min_active_target_joints"]):
            reasons.append("too_few_active_target_joints")
        if target_skin.ndim != 2 or target_skin.shape != (int(metrics["vertex_count"]), int(metrics["target_joint_count"])):
            reasons.append("bad_target_skin_shape")

        dyn = rootless_dynamic_metrics(
            frame_vertices_rootspace,
            faces,
            target_joints_rootspace,
            motion_rate_eps=float(job["motion_rate_eps"]),
            max_queries=int(job["max_queries"]),
        )
        scores = compute_scores(dyn, argparse.Namespace(**job["score_args"]))
        metrics.update(dyn)
        metrics.update(scores)

        quality_reasons: list[str] = []
        for score_name, threshold_name in (
            ("motion_coverage_score", "min_motion_coverage_score"),
            ("motion_amount_score", "min_motion_amount_score"),
            ("bbox_stability_score", "min_bbox_stability_score"),
            ("edge_stretch_stability_score", "min_edge_stretch_stability_score"),
            ("edge_collapse_stability_score", "min_edge_collapse_stability_score"),
            ("spike_cleanliness_score", "min_spike_cleanliness_score"),
        ):
            if float(metrics.get(score_name, 0.0)) < float(job[threshold_name]):
                quality_reasons.append(f"{score_name}_below_rootless_target_threshold")
        if float(metrics.get("lbs_recon_p95_bbox", 0.0)) > float(job["max_lbs_recon_p95_bbox"]):
            quality_reasons.append("lbs_recon_p95_bbox_above_rootless_target_threshold")
        metrics["rootless_target_quality_gate_mode"] = str(job["quality_gate_mode"])
        metrics["rootless_target_quality_gate_reasons"] = quality_reasons
        if str(job["quality_gate_mode"]) == "hard":
            reasons.extend(quality_reasons)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"rootless_target_metric_error:{repr(exc)}")

    return {
        "status": "reject" if reasons else "accept",
        "row": row,
        "metrics": metrics,
        "reasons": reasons,
    }


def output_rel(row: dict[str, Any], source_path: Path) -> Path:
    rel = row.get("rel")
    if rel:
        path = Path(str(rel))
        return path if path.suffix == ".npz" else path.with_suffix(".npz")
    return Path("articulationxl") / source_path.name


def materialize_npz(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown materialize mode {mode!r}")


def final_manifest_row(result: dict[str, Any], output_root: Path, materialize_mode: str) -> dict[str, Any]:
    row = dict(result["row"])
    src = Path(str(row["path"]))
    rel = output_rel(row, src)
    dst = output_root / "npz" / rel
    materialize_npz(src, dst, materialize_mode)
    row["rootless_path"] = str(src)
    row["path"] = str(dst)
    row["rel"] = str(rel).replace("\\", "/")
    row["rootless_target_strict_metrics"] = result["metrics"]
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_metrics_csv(path: Path, results: list[dict[str, Any]]) -> None:
    metric_keys: list[str] = []
    seen: set[str] = set()
    for result in results:
        for key in result.get("metrics", {}).keys():
            if key not in seen:
                seen.add(key)
                metric_keys.append(key)
    fields = ["screening_status", "asset_id", "sequence_id", "split", "path", "reasons", *metric_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result.get("metrics", {})
            out = {
                "screening_status": result.get("status", ""),
                "asset_id": metrics.get("asset_id", result.get("row", {}).get("asset_id", "")),
                "sequence_id": metrics.get("sequence_id", result.get("row", {}).get("sequence_id", "")),
                "split": metrics.get("split", result.get("row", {}).get("split", "")),
                "path": metrics.get("path", result.get("row", {}).get("path", "")),
                "reasons": ";".join(result.get("reasons", [])),
            }
            out.update({key: metrics.get(key, "") for key in metric_keys})
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--materialize-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-vertices", type=int, default=256)
    parser.add_argument("--alignment-threshold", type=float, default=0.15)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--min-faces", type=int, default=20)
    parser.add_argument("--min-target-joints", type=int, default=4)
    parser.add_argument("--min-active-target-joints", type=int, default=1)
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
    parser.add_argument("--min-motion-coverage-score", type=float, default=0.15)
    parser.add_argument("--min-motion-amount-score", type=float, default=0.15)
    parser.add_argument("--min-bbox-stability-score", type=float, default=0.50)
    parser.add_argument("--min-edge-stretch-stability-score", type=float, default=0.50)
    parser.add_argument("--min-edge-collapse-stability-score", type=float, default=0.20)
    parser.add_argument("--min-spike-cleanliness-score", type=float, default=0.50)
    parser.add_argument("--max-lbs-recon-p95-bbox", type=float, default=0.02)
    parser.add_argument("--quality-gate-mode", choices=("hard", "record"), default="hard")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for manifest in args.manifest:
        rows.extend(read_manifest(manifest))

    score_args = {
        "motion_amount_bad": float(args.motion_amount_bad),
        "motion_amount_good": float(args.motion_amount_good),
        "bbox_good_factor": float(args.bbox_good_factor),
        "bbox_bad_factor": float(args.bbox_bad_factor),
        "edge_stretch_good": float(args.edge_stretch_good),
        "edge_stretch_bad": float(args.edge_stretch_bad),
        "edge_collapse_bad": float(args.edge_collapse_bad),
        "edge_collapse_good": float(args.edge_collapse_good),
        "spike_good": float(args.spike_good),
        "spike_bad": float(args.spike_bad),
    }
    jobs = []
    for idx, row in enumerate(rows):
        jobs.append(
            {
                "row": row,
                "row_index": int(idx),
                "sample_vertices": int(args.sample_vertices),
                "alignment_threshold": float(args.alignment_threshold),
                "min_frames": int(args.min_frames),
                "min_vertices": int(args.min_vertices),
                "min_faces": int(args.min_faces),
                "min_target_joints": int(args.min_target_joints),
                "min_active_target_joints": int(args.min_active_target_joints),
                "motion_rate_eps": float(args.motion_rate_eps),
                "max_queries": int(args.max_queries),
                "score_args": score_args,
                "min_motion_coverage_score": float(args.min_motion_coverage_score),
                "min_motion_amount_score": float(args.min_motion_amount_score),
                "min_bbox_stability_score": float(args.min_bbox_stability_score),
                "min_edge_stretch_stability_score": float(args.min_edge_stretch_stability_score),
                "min_edge_collapse_stability_score": float(args.min_edge_collapse_stability_score),
                "min_spike_cleanliness_score": float(args.min_spike_cleanliness_score),
                "max_lbs_recon_p95_bbox": float(args.max_lbs_recon_p95_bbox),
                "quality_gate_mode": str(args.quality_gate_mode),
            }
        )

    if int(args.workers) <= 1:
        results = [analyze_rootless_row(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            results = list(executor.map(analyze_rootless_row, jobs, chunksize=8))

    args.output_root.mkdir(parents=True, exist_ok=True)
    accepted_results = [result for result in results if result["status"] == "accept"]
    rejected_results = [result for result in results if result["status"] != "accept"]

    accepted_rows = [final_manifest_row(result, args.output_root, args.materialize_mode) for result in accepted_results]
    rejected_rows = [
        {
            **dict(result["row"]),
            "rootless_target_strict_status": "reject",
            "rootless_target_strict_reasons": list(result["reasons"]),
            "rootless_target_strict_metrics": result["metrics"],
        }
        for result in rejected_results
    ]

    by_split = {split: [row for row in accepted_rows if str(row.get("split")) == split] for split in SPLITS}
    write_jsonl(args.output_root / "accepted.jsonl", accepted_rows)
    write_jsonl(args.output_root / "rejected.jsonl", rejected_rows)
    for split in SPLITS:
        write_jsonl(args.output_root / f"{split}_manifest.jsonl", by_split[split])
        write_jsonl(args.output_root / f"{split}_manifest.westlake.jsonl", by_split[split])
        write_jsonl(args.output_root / f"{split}_rejected.jsonl", [row for row in rejected_rows if str(row.get("split")) == split])
    write_metrics_csv(args.output_root / "rootless_target_strict_metrics.csv", results)

    reason_counts = Counter(reason for result in rejected_results for reason in result.get("reasons", []))
    summary = {
        "stage": "rootless_target_strict",
        "input_rows": len(rows),
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "split_counts": {split: len(by_split[split]) for split in SPLITS},
        "top_reject_reasons": reason_counts.most_common(80),
        "quality_gate_mode": str(args.quality_gate_mode),
        "materialize_mode": str(args.materialize_mode),
        "policy": "first formal hard reject after rootless target rewrite",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
