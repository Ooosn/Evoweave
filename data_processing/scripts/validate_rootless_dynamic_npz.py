#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def bbox_diag(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return 0.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def deterministic_vertex_ids(vertex_count: int, sample_vertices: int) -> np.ndarray:
    count = min(max(int(sample_vertices), 0), int(vertex_count))
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.linspace(0, int(vertex_count) - 1, count).round().astype(np.int64)


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, direction) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * direction)))


def target_alignment_median(vertices: np.ndarray, skin: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> float:
    diag = max(bbox_diag(vertices), 1.0e-8)
    active = skin.sum(axis=0) > 1.0e-5
    ids = np.flatnonzero(active).astype(np.int64)
    if ids.size == 0:
        return 999.0
    active_skin = skin[:, ids].astype(np.float32, copy=False)
    denom = active_skin.sum(axis=0).clip(min=1.0e-8)
    centroids = (active_skin.T @ vertices.astype(np.float32, copy=False)) / denom[:, None]
    children = [[] for _ in range(parents.shape[0])]
    for child, parent in enumerate(parents.tolist()):
        p = int(parent)
        if p >= 0:
            children[p].append(int(child))
    distances = []
    for centroid, joint_id in zip(centroids, ids):
        incident = []
        parent = int(parents[int(joint_id)])
        if parent >= 0:
            incident.append((joints[parent], joints[joint_id]))
        for child in children[int(joint_id)]:
            incident.append((joints[joint_id], joints[child]))
        if incident:
            distances.append(min(point_segment_distance(centroid, a, b) for a, b in incident))
        else:
            distances.append(float(np.linalg.norm(centroid - joints[int(joint_id)])))
    arr = np.asarray(distances, dtype=np.float64) / diag
    return float(np.median(arr))


def world_from_rootspace(points: np.ndarray, root_positions: np.ndarray, world_to_root_rotations: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    roots = np.asarray(root_positions, dtype=np.float32)
    rots = np.asarray(world_to_root_rotations, dtype=np.float32)
    world = np.einsum("t...c,tdc->t...d", pts, rots, optimize=True)
    world = world + roots.reshape((roots.shape[0],) + (1,) * (pts.ndim - 2) + (3,))
    return world.astype(np.float32)


def lbs_reconstruct_sample(rest_vertices: np.ndarray, skin: np.ndarray, bone_transforms: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    rv = np.asarray(rest_vertices[vertex_ids], dtype=np.float32)
    weights = np.asarray(skin[vertex_ids], dtype=np.float32)
    ones = np.ones((rv.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([rv, ones], axis=1)
    posed_by_bone = np.einsum("tjbc,vc->tvjb", bone_transforms.astype(np.float32), homog, optimize=True)[..., :3]
    return np.einsum("vj,tvjc->tvc", weights, posed_by_bone, optimize=True).astype(np.float32)


def lbs_error_metrics(rest_vertices: np.ndarray, skin: np.ndarray, bone_transforms: np.ndarray, frame_vertices: np.ndarray, vertex_ids: np.ndarray) -> dict[str, float]:
    pred = lbs_reconstruct_sample(rest_vertices, skin, bone_transforms, vertex_ids)
    target = frame_vertices[:, vertex_ids].astype(np.float32, copy=False)
    abs_linf = float(np.max(np.abs(pred - target))) if vertex_ids.size else 0.0
    diag = max(bbox_diag(frame_vertices.reshape(-1, 3)), 1.0e-8)
    err = np.linalg.norm(pred - target, axis=-1) / diag if vertex_ids.size else np.zeros((0,), dtype=np.float32)
    return {
        "lbs_recon_linf": abs_linf,
        "lbs_recon_p50_bbox": float(np.percentile(err, 50)) if err.size else 0.0,
        "lbs_recon_p95_bbox": float(np.percentile(err, 95)) if err.size else 0.0,
        "lbs_recon_p99_bbox": float(np.percentile(err, 99)) if err.size else 0.0,
        "lbs_recon_max_bbox": float(err.max()) if err.size else 0.0,
    }


def validate_one(path: Path, *, sample_vertices: int, alignment_threshold: float) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    with np.load(path, allow_pickle=True) as z:
        required = [
            "canonical_schema_version",
            "frame_vertices",
            "rest_vertices",
            "skin_weights",
            "parents",
            "bone_transforms",
            "root_positions",
            "root_rotations_world_to_root",
            "root_rotations_root_to_world",
            "frame_vertices_rootspace",
            "target_parents",
            "target_raw_indices",
            "target_joint_names",
            "target_joints_rootspace",
            "target_skin_weights",
            "target_is_synthetic_root",
        ]
        missing = [key for key in required if key not in z.files]
        if missing:
            return {"path": str(path), "status": "invalid", "missing": missing}, [f"missing fields {missing}"]

        target_names = np.asarray(z["target_joint_names"], dtype=object).reshape(-1).tolist()
        target_parents = np.asarray(z["target_parents"], dtype=np.int64).reshape(-1)
        target_raw_indices = np.asarray(z["target_raw_indices"], dtype=np.int64).reshape(-1)
        target_skin = np.asarray(z["target_skin_weights"], dtype=np.float32)
        target_joints = np.asarray(z["target_joints_rootspace"], dtype=np.float32)
        target_is_synthetic = np.asarray(z["target_is_synthetic_root"], dtype=bool).reshape(-1)
        frame_vertices = np.asarray(z["frame_vertices"], dtype=np.float32)
        frame_vertices_rootspace = np.asarray(z["frame_vertices_rootspace"], dtype=np.float32)
        root_positions = np.asarray(z["root_positions"], dtype=np.float32)
        root_rot_world_to_root = np.asarray(z["root_rotations_world_to_root"], dtype=np.float32)
        raw_skin = np.asarray(z["skin_weights"], dtype=np.float32)
        rest_vertices = np.asarray(z["rest_vertices"], dtype=np.float32)
        bone_transforms = np.asarray(z["bone_transforms"], dtype=np.float32)

        if str(np.asarray(z["canonical_schema_version"]).item()) != "rootless_dynamic_npz_v3":
            issues.append("schema version is not rootless_dynamic_npz_v3")
        if any(str(name) == "__object_root__" for name in target_names):
            issues.append("target contains __object_root__")
        if bool(target_is_synthetic.any()):
            issues.append("target_is_synthetic_root has true values")
        if target_parents.shape[0] != len(target_names) or target_raw_indices.shape[0] != len(target_names):
            issues.append("target name/parent/raw-index lengths differ")
        roots = [int(i) for i, p in enumerate(target_parents.tolist()) if int(p) < 0]
        if roots != [0]:
            issues.append(f"target roots are {roots}, expected [0]")
        for child, parent in enumerate(target_parents.tolist()):
            p = int(parent)
            if child == 0:
                if p != -1:
                    issues.append(f"target root parent is {p}")
            elif p < 0 or p >= child:
                issues.append(f"invalid parent edge child={child} parent={p}")
                break
        if target_skin.shape[1] != len(target_names):
            issues.append("target_skin_weights column count differs from target count")
        raw_sum = raw_skin.sum(axis=1)
        target_sum = target_skin.sum(axis=1)
        skin_loss = float(np.max(np.abs(raw_sum - target_sum))) if raw_sum.size else 0.0
        if not math.isfinite(skin_loss) or skin_loss > 1.0e-4:
            issues.append(f"skin weight loss {skin_loss}")

        vertex_count = int(frame_vertices.shape[1])
        vertex_ids = deterministic_vertex_ids(vertex_count, sample_vertices)
        count = int(vertex_ids.size)
        recon_world = world_from_rootspace(frame_vertices_rootspace[:, vertex_ids], root_positions, root_rot_world_to_root)
        rootspace_recon = float(np.max(np.abs(recon_world - frame_vertices[:, vertex_ids]))) if count > 0 else 0.0
        bbox_extent = float(np.max(frame_vertices.max(axis=(0, 1)) - frame_vertices.min(axis=(0, 1))))
        rootspace_tol = max(1.0e-4, 1.0e-5 * max(bbox_extent, 1.0e-8))
        if not math.isfinite(rootspace_recon) or rootspace_recon > rootspace_tol:
            issues.append(f"rootspace recon {rootspace_recon} > {rootspace_tol}")

        lbs_metrics = lbs_error_metrics(rest_vertices, raw_skin, bone_transforms, frame_vertices, vertex_ids)
        if (
            not math.isfinite(lbs_metrics["lbs_recon_p95_bbox"])
            or lbs_metrics["lbs_recon_p95_bbox"] > 0.05
            or lbs_metrics["lbs_recon_p99_bbox"] > 0.15
            or lbs_metrics["lbs_recon_max_bbox"] > 0.5
        ):
            issues.append(
                "lbs normalized recon "
                f"p95={lbs_metrics['lbs_recon_p95_bbox']} "
                f"p99={lbs_metrics['lbs_recon_p99_bbox']} "
                f"max={lbs_metrics['lbs_recon_max_bbox']}"
            )

        align = target_alignment_median(
            frame_vertices_rootspace[0],
            target_skin,
            target_joints[0],
            target_parents,
        )

        metrics = {
            "path": str(path),
            "status": "ok" if not issues else "invalid",
            "target_joint_count": int(len(target_names)),
            "skin_weight_lost_linf": skin_loss,
            "rootspace_recon_linf": rootspace_recon,
            **lbs_metrics,
            "target_alignment_median": align,
        }
    return metrics, issues


def validate_job(job: dict[str, Any]) -> dict[str, Any]:
    idx = int(job["index"])
    row = dict(job["row"])
    metrics, issues = validate_one(
        Path(row["path"]),
        sample_vertices=int(job["sample_vertices"]),
        alignment_threshold=float(job["alignment_threshold"]),
    )
    return {
        "index": idx,
        "asset_id": row.get("asset_id"),
        "path": row.get("path"),
        "metrics": metrics,
        "issues": issues,
    }


def update_summary(summary: dict[str, Any], result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    issues = result["issues"]
    summary["max_skin_weight_lost_linf"] = max(summary["max_skin_weight_lost_linf"], float(metrics.get("skin_weight_lost_linf", 0.0)))
    summary["max_rootspace_recon_linf"] = max(summary["max_rootspace_recon_linf"], float(metrics.get("rootspace_recon_linf", 0.0)))
    summary["max_lbs_recon_linf"] = max(summary["max_lbs_recon_linf"], float(metrics.get("lbs_recon_linf", 0.0)))
    summary["max_lbs_recon_p95_bbox"] = max(summary["max_lbs_recon_p95_bbox"], float(metrics.get("lbs_recon_p95_bbox", 0.0)))
    summary["max_lbs_recon_p99_bbox"] = max(summary["max_lbs_recon_p99_bbox"], float(metrics.get("lbs_recon_p99_bbox", 0.0)))
    summary["max_lbs_recon_max_bbox"] = max(summary["max_lbs_recon_max_bbox"], float(metrics.get("lbs_recon_max_bbox", 0.0)))
    summary["max_target_alignment_median"] = max(summary["max_target_alignment_median"], float(metrics.get("target_alignment_median", 0.0)))
    if issues:
        summary["invalid"] += 1
        summary["issues"].append(
            {
                "index": int(result["index"]),
                "asset_id": result.get("asset_id"),
                "path": result.get("path"),
                "issues": issues,
                "metrics": metrics,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-vertices", type=int, default=128)
    parser.add_argument("--alignment-threshold", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for manifest in args.manifest:
        all_rows.extend(read_jsonl(Path(manifest)))
    if args.limit > 0:
        all_rows = all_rows[: args.limit]

    summary: dict[str, Any] = {
        "rows": len(all_rows),
        "invalid": 0,
        "issues": [],
        "max_skin_weight_lost_linf": 0.0,
        "max_rootspace_recon_linf": 0.0,
        "max_lbs_recon_linf": 0.0,
        "max_lbs_recon_p95_bbox": 0.0,
        "max_lbs_recon_p99_bbox": 0.0,
        "max_lbs_recon_max_bbox": 0.0,
        "max_target_alignment_median": 0.0,
    }
    jobs = [
        {
            "index": idx,
            "row": row,
            "sample_vertices": int(args.sample_vertices),
            "alignment_threshold": float(args.alignment_threshold),
        }
        for idx, row in enumerate(all_rows)
    ]
    done = 0
    if int(args.workers) <= 1:
        for job in jobs:
            update_summary(summary, validate_job(job))
            done += 1
            if done % 250 == 0:
                print(json.dumps({"event": "validate_progress", "done": done, "invalid": summary["invalid"]}, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(validate_job, job) for job in jobs]
            for future in as_completed(futures):
                update_summary(summary, future.result())
                done += 1
                if done % 250 == 0:
                    print(json.dumps({"event": "validate_progress", "done": done, "invalid": summary["invalid"]}, sort_keys=True), flush=True)
    summary["issues"].sort(key=lambda item: int(item.get("index", 0)))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
