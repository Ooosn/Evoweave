#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_MANIFESTS = (
    "train_manifest.westlake.jsonl",
    "val_manifest.westlake.jsonl",
    "test_manifest.westlake.jsonl",
)

REQUIRED_FIELDS = (
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
    "target_has_skin",
    "target_is_connector",
    "target_is_synthetic_root",
    "target_is_tail_or_end",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def bbox_diag(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return 0.0
    pts = pts.reshape(-1, 3)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def deterministic_vertex_ids(vertex_count: int, sample_vertices: int) -> np.ndarray:
    count = min(max(int(sample_vertices), 0), int(vertex_count))
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.linspace(0, int(vertex_count) - 1, count).round().astype(np.int64)


def bbox_diag_consistency(a: float, b: float) -> float:
    x = float(a)
    y = float(b)
    denom = max(x, y, 1.0e-8)
    return float(min(x, y) / denom)


def bbox_center(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((3,), dtype=np.float32)
    return ((pts.max(axis=0) + pts.min(axis=0)) * 0.5).astype(np.float32)


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, direction) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * direction)))


def children_from_parents(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(int(parents.shape[0]))]
    for child, parent in enumerate(parents.tolist()):
        p = int(parent)
        if p >= 0 and p < len(children):
            children[p].append(int(child))
    return children


def target_alignment_stats(
    vertices: np.ndarray,
    skin: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
) -> dict[str, float]:
    diag = max(bbox_diag(vertices), 1.0e-8)
    active = skin.sum(axis=0) > 1.0e-5
    ids = np.flatnonzero(active).astype(np.int64)
    if ids.size == 0:
        return {
            "target_alignment_median": 999.0,
            "target_alignment_p90": 999.0,
            "target_alignment_max": 999.0,
            "skin_centroid_count": 0.0,
            "skin_centroid_spread_to_mesh_bbox": 0.0,
        }

    active_skin = skin[:, ids].astype(np.float32, copy=False)
    denom = active_skin.sum(axis=0).clip(min=1.0e-8)
    centroids = (active_skin.T @ vertices.astype(np.float32, copy=False)) / denom[:, None]
    children = children_from_parents(parents)
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
    return {
        "target_alignment_median": float(np.median(arr)),
        "target_alignment_p90": float(np.percentile(arr, 90)),
        "target_alignment_max": float(arr.max()),
        "skin_centroid_count": float(ids.size),
        "skin_centroid_spread_to_mesh_bbox": float(bbox_diag(centroids) / diag),
    }


def world_from_rootspace(points: np.ndarray, root_positions: np.ndarray, world_to_root_rotations: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    roots = np.asarray(root_positions, dtype=np.float32)
    rots = np.asarray(world_to_root_rotations, dtype=np.float32)
    world = np.einsum("t...c,tdc->t...d", pts, rots, optimize=True)
    world = world + roots.reshape((roots.shape[0],) + (1,) * (pts.ndim - 2) + (3,))
    return world.astype(np.float32)


def lbs_reconstruct_sample(
    rest_vertices: np.ndarray,
    skin: np.ndarray,
    bone_transforms: np.ndarray,
    vertex_ids: np.ndarray,
) -> np.ndarray:
    rv = np.asarray(rest_vertices[vertex_ids], dtype=np.float32)
    weights = np.asarray(skin[vertex_ids], dtype=np.float32)
    ones = np.ones((rv.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([rv, ones], axis=1)
    posed_by_bone = np.einsum("tjbc,vc->tvjb", bone_transforms.astype(np.float32), homog, optimize=True)[..., :3]
    return np.einsum("vj,tvjc->tvc", weights, posed_by_bone, optimize=True).astype(np.float32)


def lbs_error_metrics(
    rest_vertices: np.ndarray,
    skin: np.ndarray,
    bone_transforms: np.ndarray,
    frame_vertices: np.ndarray,
    vertex_ids: np.ndarray,
) -> dict[str, float]:
    if vertex_ids.size == 0:
        return {
            "lbs_recon_linf": 0.0,
            "lbs_recon_p50_bbox": 0.0,
            "lbs_recon_p95_bbox": 0.0,
            "lbs_recon_p99_bbox": 0.0,
            "lbs_recon_max_bbox": 0.0,
        }
    pred = lbs_reconstruct_sample(rest_vertices, skin, bone_transforms, vertex_ids)
    target = frame_vertices[:, vertex_ids].astype(np.float32, copy=False)
    abs_linf = float(np.max(np.abs(pred - target)))
    diag = max(bbox_diag(frame_vertices.reshape(-1, 3)), 1.0e-8)
    err = np.linalg.norm(pred - target, axis=-1) / diag
    return {
        "lbs_recon_linf": abs_linf,
        "lbs_recon_p50_bbox": float(np.percentile(err, 50)),
        "lbs_recon_p95_bbox": float(np.percentile(err, 95)),
        "lbs_recon_p99_bbox": float(np.percentile(err, 99)),
        "lbs_recon_max_bbox": float(err.max()),
    }


def tree_issues(parents: np.ndarray) -> list[str]:
    issues: list[str] = []
    roots = [int(i) for i, p in enumerate(parents.tolist()) if int(p) < 0]
    if roots != [0]:
        issues.append(f"target roots are {roots}, expected [0]")
    for child, parent in enumerate(parents.tolist()):
        p = int(parent)
        if child == 0:
            if p != -1:
                issues.append(f"target root parent is {p}")
        elif p < 0 or p >= child:
            issues.append(f"invalid parent edge child={child} parent={p}")
            break
    return issues


def rotation_stats(rotations: np.ndarray) -> dict[str, float]:
    r = np.asarray(rotations, dtype=np.float32)
    if r.size == 0:
        return {"root_rot_orthonormal_max": 0.0, "root_rot_det_absdiff_max": 0.0}
    eye = np.eye(3, dtype=np.float32)
    gram = np.einsum("tac,tbc->tab", r, r, optimize=True)
    orth = float(np.max(np.abs(gram - eye)))
    det = np.linalg.det(r.astype(np.float64))
    return {
        "root_rot_orthonormal_max": orth,
        "root_rot_det_absdiff_max": float(np.max(np.abs(det - 1.0))),
    }


def edge_stats(
    joints: np.ndarray,
    parents: np.ndarray,
    mesh_diag: float,
) -> dict[str, float]:
    diag = max(float(mesh_diag), 1.0e-8)
    children = children_from_parents(parents)
    edge_lengths = []
    zero_edges = 0
    for child, parent in enumerate(parents.tolist()):
        p = int(parent)
        if p >= 0:
            length = float(np.linalg.norm(joints[child] - joints[p]))
            edge_lengths.append(length / diag)
            if length <= 1.0e-7 * diag:
                zero_edges += 1

    branch_count = 0
    for parent, kids in enumerate(children):
        if not kids:
            continue
        branch_count += 1

    if edge_lengths:
        edge_arr = np.asarray(edge_lengths, dtype=np.float64)
        min_edge = float(edge_arr.min())
        p05_edge = float(np.percentile(edge_arr, 5))
    else:
        min_edge = 0.0
        p05_edge = 0.0
    return {
        "zero_edge_count": float(zero_edges),
        "min_edge_len_bbox": min_edge,
        "p05_edge_len_bbox": p05_edge,
        "branch_node_count": float(branch_count),
    }


def duplicate_joint_pairs(joints: np.ndarray, mesh_diag: float) -> int:
    n = int(joints.shape[0])
    if n <= 1:
        return 0
    tol = max(1.0e-6, 1.0e-5 * max(float(mesh_diag), 1.0e-8))
    count = 0
    for i in range(n):
        d = np.linalg.norm(joints[i + 1 :] - joints[i], axis=1)
        count += int(np.count_nonzero(d <= tol))
    return count


def face_stats(faces: np.ndarray | None, vertex_count: int) -> tuple[dict[str, float], list[str]]:
    if faces is None:
        return {
            "face_count": 0.0,
            "invalid_face_count": 0.0,
            "degenerate_face_frac": 0.0,
        }, []
    f = np.asarray(faces)
    issues: list[str] = []
    if f.ndim != 2 or f.shape[1] < 3:
        return {
            "face_count": float(f.shape[0]) if f.ndim >= 1 else 0.0,
            "invalid_face_count": 1.0,
            "degenerate_face_frac": 1.0,
        }, ["faces shape is not Nx3+"]
    tri = f[:, :3].astype(np.int64, copy=False)
    invalid = int(np.count_nonzero((tri < 0) | (tri >= int(vertex_count))))
    if invalid:
        issues.append(f"invalid face indices count={invalid}")
    deg = int(np.count_nonzero((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])))
    frac = float(deg / max(int(tri.shape[0]), 1))
    return {
        "face_count": float(tri.shape[0]),
        "invalid_face_count": float(invalid),
        "degenerate_face_frac": frac,
    }, issues


def finite_check(name: str, arr: np.ndarray) -> list[str]:
    if not np.all(np.isfinite(arr)):
        return [f"{name} contains non-finite values"]
    return []


def analyse_one(path: Path, row: dict[str, Any], split: str, row_index: int, sample_vertices: int) -> tuple[dict[str, Any], list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    base = {
        "split": split,
        "row_index": row_index,
        "asset_id": row.get("asset_id", ""),
        "path": str(path),
        "raw_path": row.get("raw_path", ""),
    }
    if not path.exists():
        return dict(base, status="missing"), ["manifest path missing"], []

    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:
        return dict(base, status="load_error"), [f"np.load failed: {exc}"], []

    with z:
        missing = [key for key in REQUIRED_FIELDS if key not in z.files]
        if missing:
            return dict(base, status="missing_fields", missing=";".join(missing)), [f"missing fields {missing}"], []

        try:
            schema = str(np.asarray(z["canonical_schema_version"]).item())
            frame_vertices = np.asarray(z["frame_vertices"], dtype=np.float32)
            rest_vertices = np.asarray(z["rest_vertices"], dtype=np.float32)
            raw_skin = np.asarray(z["skin_weights"], dtype=np.float32)
            raw_parents = np.asarray(z["parents"], dtype=np.int64).reshape(-1)
            bone_transforms = np.asarray(z["bone_transforms"], dtype=np.float32)
            root_positions = np.asarray(z["root_positions"], dtype=np.float32)
            root_rot_world_to_root = np.asarray(z["root_rotations_world_to_root"], dtype=np.float32)
            root_rot_root_to_world = np.asarray(z["root_rotations_root_to_world"], dtype=np.float32)
            frame_vertices_rootspace = np.asarray(z["frame_vertices_rootspace"], dtype=np.float32)
            target_parents = np.asarray(z["target_parents"], dtype=np.int64).reshape(-1)
            target_raw_indices = np.asarray(z["target_raw_indices"], dtype=np.int64).reshape(-1)
            target_names = np.asarray(z["target_joint_names"], dtype=object).reshape(-1).tolist()
            target_joints = np.asarray(z["target_joints_rootspace"], dtype=np.float32)
            target_skin = np.asarray(z["target_skin_weights"], dtype=np.float32)
            target_has_skin = np.asarray(z["target_has_skin"], dtype=bool).reshape(-1)
            target_is_connector = np.asarray(z["target_is_connector"], dtype=bool).reshape(-1)
            target_is_synthetic = np.asarray(z["target_is_synthetic_root"], dtype=bool).reshape(-1)
            target_is_tail_or_end = np.asarray(z["target_is_tail_or_end"], dtype=bool).reshape(-1)
            faces = np.asarray(z["faces"]) if "faces" in z.files else None
        except Exception as exc:
            return dict(base, status="field_read_error"), [f"field read failed: {exc}"], []

        hard.extend(finite_check("frame_vertices", frame_vertices))
        hard.extend(finite_check("rest_vertices", rest_vertices))
        hard.extend(finite_check("skin_weights", raw_skin))
        hard.extend(finite_check("bone_transforms", bone_transforms))
        hard.extend(finite_check("root_positions", root_positions))
        hard.extend(finite_check("root_rotations_world_to_root", root_rot_world_to_root))
        hard.extend(finite_check("frame_vertices_rootspace", frame_vertices_rootspace))
        hard.extend(finite_check("target_joints_rootspace", target_joints))
        hard.extend(finite_check("target_skin_weights", target_skin))

        if schema != "rootless_dynamic_npz_v3":
            hard.append(f"schema version is {schema}")
        if any(str(name) == "__object_root__" for name in target_names):
            hard.append("target contains __object_root__")
        if bool(target_is_synthetic.any()):
            hard.append("target_is_synthetic_root has true values")

        target_count = int(len(target_names))
        frame_count = int(frame_vertices.shape[0]) if frame_vertices.ndim >= 3 else 0
        vertex_count = int(frame_vertices.shape[1]) if frame_vertices.ndim >= 3 else 0
        raw_joint_count = int(raw_parents.shape[0])

        expected_target_shape = (frame_count, target_count, 3)
        if target_joints.shape != expected_target_shape:
            hard.append(f"target_joints shape {target_joints.shape} != {expected_target_shape}")
        if target_skin.shape != (vertex_count, target_count):
            hard.append(f"target_skin shape {target_skin.shape} != {(vertex_count, target_count)}")
        if raw_skin.shape != (vertex_count, raw_joint_count):
            hard.append(f"skin_weights shape {raw_skin.shape} != {(vertex_count, raw_joint_count)}")
        if rest_vertices.shape != (vertex_count, 3):
            hard.append(f"rest_vertices shape {rest_vertices.shape} != {(vertex_count, 3)}")
        if target_raw_indices.shape[0] != target_count:
            hard.append("target_raw_indices length differs from target count")
        if (
            target_has_skin.shape[0] != target_count
            or target_is_connector.shape[0] != target_count
            or target_is_tail_or_end.shape[0] != target_count
            or target_is_synthetic.shape[0] != target_count
        ):
            hard.append("target boolean metadata length differs from target count")

        hard.extend(tree_issues(target_parents))

        if target_raw_indices.size and (target_raw_indices.min() < 0 or target_raw_indices.max() >= raw_joint_count):
            hard.append("target_raw_indices out of raw joint range")

        raw_row_sum = raw_skin.sum(axis=1) if raw_skin.ndim == 2 else np.zeros((0,), dtype=np.float32)
        target_row_sum = target_skin.sum(axis=1) if target_skin.ndim == 2 else np.zeros((0,), dtype=np.float32)
        skin_loss = float(np.max(np.abs(raw_row_sum - target_row_sum))) if raw_row_sum.size and target_row_sum.size else 999.0
        raw_weight_sum_dev = float(np.max(np.abs(raw_row_sum - 1.0))) if raw_row_sum.size else 999.0
        target_weight_sum_dev = float(np.max(np.abs(target_row_sum - 1.0))) if target_row_sum.size else 999.0
        raw_weight_min = float(raw_skin.min()) if raw_skin.size else 0.0
        target_weight_min = float(target_skin.min()) if target_skin.size else 0.0
        if not math.isfinite(skin_loss) or skin_loss > 1.0e-4:
            hard.append(f"skin weight loss {skin_loss}")
        if not math.isfinite(raw_weight_sum_dev) or raw_weight_sum_dev > 1.0e-3:
            hard.append(f"raw skin row-sum deviation {raw_weight_sum_dev}")
        if not math.isfinite(target_weight_sum_dev) or target_weight_sum_dev > 1.0e-3:
            hard.append(f"target skin row-sum deviation {target_weight_sum_dev}")
        if raw_weight_min < -1.0e-6 or target_weight_min < -1.0e-6:
            hard.append(f"negative skin weight min raw={raw_weight_min} target={target_weight_min}")

        vertex_ids = deterministic_vertex_ids(vertex_count, sample_vertices)
        count = int(vertex_ids.size)
        if count > 0:
            recon_world = world_from_rootspace(frame_vertices_rootspace[:, vertex_ids], root_positions, root_rot_world_to_root)
            rootspace_recon = float(np.max(np.abs(recon_world - frame_vertices[:, vertex_ids])))
        else:
            rootspace_recon = 0.0
        bbox_extent = float(np.max(frame_vertices.max(axis=(0, 1)) - frame_vertices.min(axis=(0, 1)))) if frame_vertices.size else 0.0
        rootspace_tol = max(1.0e-4, 1.0e-5 * max(bbox_extent, 1.0e-8))
        if not math.isfinite(rootspace_recon) or rootspace_recon > rootspace_tol:
            hard.append(f"rootspace recon {rootspace_recon} > {rootspace_tol}")

        lbs = lbs_error_metrics(rest_vertices, raw_skin, bone_transforms, frame_vertices, vertex_ids)
        if (
            not math.isfinite(lbs["lbs_recon_p95_bbox"])
            or lbs["lbs_recon_p95_bbox"] > 0.05
            or lbs["lbs_recon_p99_bbox"] > 0.15
            or lbs["lbs_recon_max_bbox"] > 0.5
        ):
            hard.append(
                "lbs normalized recon "
                f"p95={lbs['lbs_recon_p95_bbox']} p99={lbs['lbs_recon_p99_bbox']} max={lbs['lbs_recon_max_bbox']}"
            )

        mesh_diag_frame0 = bbox_diag(frame_vertices_rootspace[0]) if frame_vertices_rootspace.ndim == 3 else 0.0
        mesh_diag_all = bbox_diag(frame_vertices_rootspace) if frame_vertices_rootspace.ndim == 3 else 0.0
        joint_frame0 = target_joints[0] if target_joints.ndim == 3 and target_joints.shape[0] else np.zeros((0, 3), dtype=np.float32)
        active_ids = np.flatnonzero(target_has_skin).astype(np.int64) if target_has_skin.shape[0] == target_count else np.zeros((0,), dtype=np.int64)
        joint_bbox = bbox_diag(joint_frame0)
        active_bbox = bbox_diag(joint_frame0[active_ids]) if active_ids.size else 0.0
        mesh_diag = max(mesh_diag_frame0, 1.0e-8)
        joint_bbox_consistency = bbox_diag_consistency(joint_bbox, mesh_diag)
        active_bbox_consistency = bbox_diag_consistency(active_bbox, mesh_diag)
        center_offset_ratio = float(np.linalg.norm(bbox_center(joint_frame0) - bbox_center(frame_vertices_rootspace[0])) / mesh_diag)
        joint_max_abs = float(np.max(np.abs(joint_frame0))) if joint_frame0.size else 0.0
        mesh_max_abs = float(np.max(np.abs(frame_vertices_rootspace[0]))) if frame_vertices_rootspace.size else 0.0
        duplicate_pairs = duplicate_joint_pairs(joint_frame0, mesh_diag)

        if target_count >= 3 and joint_bbox_consistency < 0.2:
            soft.append(f"low joint/mesh bbox consistency={joint_bbox_consistency}")
        if active_ids.size >= 3 and active_bbox_consistency < 0.2:
            soft.append(f"low active joint/mesh bbox consistency={active_bbox_consistency}")
        if center_offset_ratio > 0.75:
            soft.append(f"skeleton bbox center offset ratio={center_offset_ratio}")

        alignment = target_alignment_stats(
            frame_vertices_rootspace[0],
            target_skin,
            joint_frame0,
            target_parents,
        )
        if not math.isfinite(alignment["target_alignment_median"]) or alignment["target_alignment_median"] > 0.1:
            soft.append(f"target alignment median {alignment['target_alignment_median']} > 0.1")
        if alignment["target_alignment_p90"] > 0.35:
            soft.append(f"target alignment p90 {alignment['target_alignment_p90']} > 0.35")

        rot = rotation_stats(root_rot_world_to_root)
        if rot["root_rot_orthonormal_max"] > 1.0e-2 or rot["root_rot_det_absdiff_max"] > 1.0e-2:
            hard.append(
                "root rotation invalid "
                f"orth={rot['root_rot_orthonormal_max']} det_absdiff={rot['root_rot_det_absdiff_max']}"
            )
        if root_rot_root_to_world.shape == root_rot_world_to_root.shape:
            rt_inverse_err = float(np.max(np.abs(root_rot_root_to_world - np.swapaxes(root_rot_world_to_root, 1, 2))))
            if rt_inverse_err > 1.0e-5:
                hard.append(f"root_rotations_root_to_world is not transpose, err={rt_inverse_err}")
        else:
            rt_inverse_err = 999.0
            hard.append("root_rotations_root_to_world shape differs from world_to_root")

        edge_metric = edge_stats(joint_frame0, target_parents, mesh_diag)
        if edge_metric["zero_edge_count"] > 0:
            soft.append(f"zero-length parent-child edges={int(edge_metric['zero_edge_count'])}")
        if duplicate_pairs > 0:
            soft.append(f"duplicate joint coordinate pairs={duplicate_pairs}")

        face_metric, face_issues = face_stats(faces, vertex_count)
        hard.extend(face_issues)
        if face_metric["degenerate_face_frac"] > 0.2:
            soft.append(f"high degenerate face fraction={face_metric['degenerate_face_frac']}")

        metrics = {
            **base,
            "status": "hard_fail" if hard else ("soft_flag" if soft else "ok"),
            "frame_count": float(frame_count),
            "vertex_count": float(vertex_count),
            "raw_joint_count": float(raw_joint_count),
            "target_joint_count": float(target_count),
            "target_active_joint_count": float(active_ids.size),
            "target_connector_count": float(np.count_nonzero(target_is_connector)),
            "target_tail_or_end_count": float(np.count_nonzero(target_is_tail_or_end)),
            "skin_weight_lost_linf": skin_loss,
            "raw_weight_sum_dev_max": raw_weight_sum_dev,
            "target_weight_sum_dev_max": target_weight_sum_dev,
            "raw_weight_min": raw_weight_min,
            "target_weight_min": target_weight_min,
            "rootspace_recon_linf": rootspace_recon,
            "rootspace_recon_tol": rootspace_tol,
            **lbs,
            "mesh_bbox_diag_frame0": mesh_diag_frame0,
            "mesh_bbox_diag_all_frames": mesh_diag_all,
            "rootless_joint_bbox_diag": joint_bbox,
            "active_rootless_joint_bbox_diag": active_bbox,
            "joint_mesh_bbox_diag_consistency": joint_bbox_consistency,
            "active_joint_mesh_bbox_diag_consistency": active_bbox_consistency,
            "skeleton_mesh_center_offset_bbox": center_offset_ratio,
            "mesh_rootspace_max_abs_frame0": mesh_max_abs,
            "joint_rootspace_max_abs_frame0": joint_max_abs,
            "duplicate_joint_pairs": float(duplicate_pairs),
            **alignment,
            **rot,
            "root_rot_transpose_err": rt_inverse_err,
            **edge_metric,
            **face_metric,
            "hard_issue_count": float(len(hard)),
            "soft_issue_count": float(len(soft)),
        }
        return metrics, hard, soft


def quantile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([x for x in values if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0.0,
            "nan_or_inf": float(len(values)),
            "min": float("nan"),
            "p1": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "count": float(arr.size),
        "nan_or_inf": float(len(values) - arr.size),
        "min": float(arr.min()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def top_cases(metrics: list[dict[str, Any]], key: str, n: int = 20, reverse: bool = True) -> list[dict[str, Any]]:
    rows = []
    for item in metrics:
        value = item.get(key)
        try:
            numeric = float(value)
        except Exception:
            continue
        if not math.isfinite(numeric):
            continue
        rows.append((numeric, item))
    rows.sort(key=lambda x: x[0], reverse=reverse)
    out = []
    for value, item in rows[:n]:
        out.append(
            {
                "value": value,
                "split": item.get("split"),
                "asset_id": item.get("asset_id"),
                "path": item.get("path"),
                "target_joint_count": item.get("target_joint_count"),
                "target_active_joint_count": item.get("target_active_joint_count"),
            }
        )
    return out


def write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    preferred = [
        "split",
        "row_index",
        "asset_id",
        "status",
        "path",
        "frame_count",
        "vertex_count",
        "target_joint_count",
        "target_active_joint_count",
        "target_connector_count",
        "target_tail_or_end_count",
        "target_alignment_median",
        "target_alignment_p90",
        "lbs_recon_p95_bbox",
        "lbs_recon_p99_bbox",
        "lbs_recon_max_bbox",
        "rootspace_recon_linf",
        "joint_mesh_bbox_diag_consistency",
        "active_joint_mesh_bbox_diag_consistency",
        "skeleton_mesh_center_offset_bbox",
        "duplicate_joint_pairs",
        "zero_edge_count",
    ]
    for key in preferred:
        if any(key in row for row in metrics):
            keys.append(key)
            seen.add(key)
    for row in metrics:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def load_all_rows(dataset: Path) -> list[tuple[str, int, dict[str, Any]]]:
    all_rows: list[tuple[str, int, dict[str, Any]]] = []
    for manifest_name in SPLIT_MANIFESTS:
        split = manifest_name.split("_", 1)[0]
        rows = read_jsonl(dataset / manifest_name)
        for idx, row in enumerate(rows):
            all_rows.append((split, idx, row))
    return all_rows


def dataset_level_checks(dataset: Path, all_rows: list[tuple[str, int, dict[str, Any]]]) -> dict[str, Any]:
    manifest_paths = [Path(str(row["path"])) for _split, _idx, row in all_rows]
    path_counts = Counter(str(path) for path in manifest_paths)
    duplicate_paths = [path for path, count in path_counts.items() if count > 1]
    npz_root = dataset / "npz"
    npz_files = set(str(path) for path in npz_root.rglob("*.npz")) if npz_root.exists() else set()
    manifest_set = set(str(path) for path in manifest_paths)
    orphan_npz = sorted(npz_files - manifest_set)
    missing_npz = sorted(manifest_set - npz_files)
    return {
        "manifest_rows": len(all_rows),
        "manifest_unique_paths": len(path_counts),
        "duplicate_manifest_paths": duplicate_paths[:100],
        "duplicate_manifest_path_count": len(duplicate_paths),
        "npz_files_under_npz": len(npz_files),
        "orphan_npz_count": len(orphan_npz),
        "orphan_npz_examples": orphan_npz[:100],
        "missing_npz_count": len(missing_npz),
        "missing_npz_examples": missing_npz[:100],
    }


def try_import_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(json.dumps({"event": "matplotlib_unavailable", "error": str(exc)}, sort_keys=True), flush=True)
        return None


def plot_distributions(plt: Any, out_path: Path, metrics: list[dict[str, Any]]) -> None:
    numeric_keys = [
        ("target_alignment_median", "alignment median"),
        ("target_alignment_p90", "alignment p90"),
        ("joint_mesh_bbox_diag_consistency", "joint/mesh bbox consistency"),
        ("active_joint_mesh_bbox_diag_consistency", "active joint/mesh bbox consistency"),
        ("skeleton_mesh_center_offset_bbox", "center offset / mesh bbox"),
        ("lbs_recon_p95_bbox", "LBS p95 / bbox"),
        ("lbs_recon_max_bbox", "LBS max / bbox"),
        ("target_joint_count", "target joint count"),
        ("target_connector_count", "connector count"),
        ("duplicate_joint_pairs", "duplicate joint pairs"),
        ("mesh_bbox_diag_frame0", "mesh bbox diag"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, (key, title) in zip(axes.reshape(-1), numeric_keys):
        values = []
        for row in metrics:
            try:
                value = float(row.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
        if values:
            arr = np.asarray(values, dtype=np.float64)
            if key in {
                "target_alignment_median",
                "target_alignment_p90",
                "joint_mesh_bbox_diag_consistency",
                "active_joint_mesh_bbox_diag_consistency",
                "skeleton_mesh_center_offset_bbox",
                "lbs_recon_p95_bbox",
                "lbs_recon_max_bbox",
                "mesh_bbox_diag_frame0",
            }:
                hi = float(np.percentile(arr, 99.5))
                lo = float(np.percentile(arr, 0.5))
                arr = arr[(arr >= lo) & (arr <= hi)]
            ax.hist(arr, bins=60, color="#4c78a8", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


def best_projection(points: np.ndarray) -> tuple[int, int, str]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    span = pts.max(axis=0) - pts.min(axis=0) if pts.size else np.zeros((3,), dtype=np.float32)
    candidates = [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]
    best = max(candidates, key=lambda item: float(span[item[0][0]] * span[item[0][1]]))
    return int(best[0][0]), int(best[0][1]), str(best[1])


def draw_case(ax: Any, path: str, title: str, rng: np.random.Generator, viz_vertices: int) -> None:
    try:
        with np.load(path, allow_pickle=True) as z:
            vertices = np.asarray(z["frame_vertices_rootspace"], dtype=np.float32)[0]
            joints = np.asarray(z["target_joints_rootspace"], dtype=np.float32)[0]
            parents = np.asarray(z["target_parents"], dtype=np.int64).reshape(-1)
            has_skin = np.asarray(z["target_has_skin"], dtype=bool).reshape(-1)
            connector = np.asarray(z["target_is_connector"], dtype=bool).reshape(-1)
            skin = np.asarray(z["target_skin_weights"], dtype=np.float32)
        count = min(int(viz_vertices), int(vertices.shape[0]))
        if count < int(vertices.shape[0]):
            ids = rng.choice(int(vertices.shape[0]), size=count, replace=False)
            v = vertices[ids]
        else:
            v = vertices
        a, b, plane = best_projection(np.concatenate([vertices, joints], axis=0))
        ax.scatter(v[:, a], v[:, b], s=1, color="#a8adb3", alpha=0.24, linewidths=0)
        for child, parent in enumerate(parents.tolist()):
            p = int(parent)
            if p >= 0:
                ax.plot([joints[p, a], joints[child, a]], [joints[p, b], joints[child, b]], color="#1f77b4", linewidth=1.2, alpha=0.9)
        ax.scatter(joints[:, a], joints[:, b], s=14, color="#1f77b4", zorder=3)
        if connector.shape[0] == joints.shape[0] and np.any(connector):
            cids = np.flatnonzero(connector)
            ax.scatter(joints[cids, a], joints[cids, b], s=16, color="#ff7f0e", zorder=4)
        if has_skin.shape[0] == joints.shape[0] and np.any(has_skin):
            aids = np.flatnonzero(has_skin)
            ax.scatter(joints[aids, a], joints[aids, b], s=18, facecolors="none", edgecolors="#d62728", linewidths=0.8, zorder=5)
        active = skin.sum(axis=0) > 1.0e-5
        ids = np.flatnonzero(active).astype(np.int64)
        if ids.size:
            active_skin = skin[:, ids].astype(np.float32, copy=False)
            denom = active_skin.sum(axis=0).clip(min=1.0e-8)
            centroids = (active_skin.T @ vertices.astype(np.float32, copy=False)) / denom[:, None]
            ax.scatter(centroids[:, a], centroids[:, b], s=10, color="#2ca02c", alpha=0.75, zorder=4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\n{plane}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    except Exception as exc:
        ax.text(0.5, 0.5, f"plot failed\n{Path(path).name}\n{exc}", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_gallery(plt: Any, out_path: Path, rows: list[dict[str, Any]], metric_key: str, title: str, viz_vertices: int) -> None:
    if not rows:
        return
    rows = rows[:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    rng = np.random.default_rng(20260701)
    for ax, row in zip(axes.reshape(-1), rows):
        try:
            value = float(row.get(metric_key, row.get("value", float("nan"))))
        except Exception:
            value = float("nan")
        label = f"{row.get('asset_id', '')}\n{metric_key}={value:.4g}"
        draw_case(ax, str(row["path"]), label, rng, viz_vertices)
    for ax in axes.reshape(-1)[len(rows) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


def analyse_job(job: dict[str, Any]) -> dict[str, Any]:
    global_idx = int(job["global_index"])
    split_idx = int(job["split_index"])
    split = str(job["split"])
    row = dict(job["row"])
    path = Path(str(row["path"]))
    item, hard, soft = analyse_one(path, row, split, global_idx, int(job["sample_vertices"]))
    return {
        "global_index": global_idx,
        "split_index": split_idx,
        "split": split,
        "row": row,
        "metrics": item,
        "hard": hard,
        "soft": soft,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-vertices", type=int, default=128)
    parser.add_argument("--viz-vertices", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_rows(dataset)
    dataset_checks = dataset_level_checks(dataset, all_rows)
    metrics: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    hard_counter: Counter[str] = Counter()
    soft_counter: Counter[str] = Counter()

    jobs = [
        {
            "global_index": global_idx,
            "split_index": split_idx,
            "split": split,
            "row": row,
            "sample_vertices": int(args.sample_vertices),
        }
        for global_idx, (split, split_idx, row) in enumerate(all_rows)
    ]
    if int(args.workers) <= 1:
        result_iter = map(analyse_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=int(args.workers))
        result_iter = executor.map(analyse_job, jobs, chunksize=4)

    try:
        for done, result in enumerate(result_iter, start=1):
            split = str(result["split"])
            split_idx = int(result["split_index"])
            global_idx = int(result["global_index"])
            row = dict(result["row"])
            item = result["metrics"]
            hard = [str(x) for x in result["hard"]]
            soft = [str(x) for x in result["soft"]]
            metrics.append(item)
            for issue in hard:
                hard_counter[issue.split(" ", 3)[0] if issue else "unknown"] += 1
            for issue in soft:
                soft_counter[issue.split(" ", 3)[0] if issue else "unknown"] += 1
            if hard or soft:
                issue_rows.append(
                    {
                        "split": split,
                        "row_index": split_idx,
                        "global_index": global_idx,
                        "asset_id": row.get("asset_id"),
                        "path": row.get("path"),
                        "hard_issues": hard,
                        "soft_issues": soft,
                        "metrics": item,
                    }
                )
            if done % 250 == 0:
                print(
                    json.dumps(
                        {
                            "event": "audit_progress",
                            "done": done,
                            "total": len(all_rows),
                            "hard": sum(1 for row2 in metrics if row2.get("status") == "hard_fail"),
                            "soft": sum(1 for row2 in metrics if row2.get("status") == "soft_flag"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if "executor" in locals():
            executor.shutdown(wait=True)

    numeric_keys = sorted(
        {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    distributions = {key: quantile_summary([float(row.get(key, float("nan"))) for row in metrics]) for key in numeric_keys}
    status_counts = Counter(str(row.get("status", "")) for row in metrics)
    split_counts = Counter(str(row.get("split", "")) for row in metrics)

    important_top_cases = {
        "target_alignment_median_high": top_cases(metrics, "target_alignment_median", 20, True),
        "target_alignment_p90_high": top_cases(metrics, "target_alignment_p90", 20, True),
        "lbs_recon_p95_bbox_high": top_cases(metrics, "lbs_recon_p95_bbox", 20, True),
        "lbs_recon_max_bbox_high": top_cases(metrics, "lbs_recon_max_bbox", 20, True),
        "joint_bbox_consistency_low": top_cases(metrics, "joint_mesh_bbox_diag_consistency", 20, False),
        "active_joint_bbox_consistency_low": top_cases(metrics, "active_joint_mesh_bbox_diag_consistency", 20, False),
        "center_offset_high": top_cases(metrics, "skeleton_mesh_center_offset_bbox", 20, True),
        "duplicate_joint_pairs_high": top_cases(metrics, "duplicate_joint_pairs", 20, True),
        "connector_count_high": top_cases(metrics, "target_connector_count", 20, True),
        "mesh_bbox_diag_high": top_cases(metrics, "mesh_bbox_diag_frame0", 20, True),
        "mesh_bbox_diag_low": top_cases(metrics, "mesh_bbox_diag_frame0", 20, False),
    }

    hard_fail_count = int(status_counts.get("hard_fail", 0))
    soft_flag_count = int(status_counts.get("soft_flag", 0))
    summary = {
        "dataset": str(dataset),
        "out_dir": str(out_dir),
        "rows": len(metrics),
        "status_counts": dict(status_counts),
        "split_counts": dict(split_counts),
        "hard_fail_count": hard_fail_count,
        "soft_flag_count": soft_flag_count,
        "hard_issue_counter": dict(hard_counter),
        "soft_issue_counter": dict(soft_counter),
        "dataset_level_checks": dataset_checks,
        "distributions": distributions,
        "top_cases": important_top_cases,
    }

    write_metrics_csv(out_dir / "rootless_final_metrics.csv", metrics)
    write_jsonl(out_dir / "rootless_final_audit_issues.jsonl", issue_rows)
    (out_dir / "rootless_final_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.no_figures:
        plt = try_import_matplotlib()
        if plt is not None:
            figures = out_dir / "figures"
            plot_distributions(plt, figures / "distributions.png", metrics)
            rng = np.random.default_rng(20260701)
            random_rows = [metrics[i] for i in rng.choice(len(metrics), size=min(12, len(metrics)), replace=False)] if metrics else []
            plot_gallery(plt, figures / "random_gallery.png", random_rows, "target_alignment_median", "Random final dataset samples", int(args.viz_vertices))
            plot_gallery(plt, figures / "worst_alignment_gallery.png", [x for x in sorted(metrics, key=lambda r: float(r.get("target_alignment_median", -1.0)), reverse=True)], "target_alignment_median", "Worst target alignment median", int(args.viz_vertices))
            plot_gallery(plt, figures / "worst_lbs_gallery.png", [x for x in sorted(metrics, key=lambda r: float(r.get("lbs_recon_max_bbox", -1.0)), reverse=True)], "lbs_recon_max_bbox", "Worst normalized LBS max", int(args.viz_vertices))
            plot_gallery(plt, figures / "smallest_joint_bbox_consistency_gallery.png", [x for x in sorted(metrics, key=lambda r: float(r.get("joint_mesh_bbox_diag_consistency", 999.0)))], "joint_mesh_bbox_diag_consistency", "Lowest joint/mesh bbox consistency", int(args.viz_vertices))
            plot_gallery(plt, figures / "largest_center_offset_gallery.png", [x for x in sorted(metrics, key=lambda r: float(r.get("skeleton_mesh_center_offset_bbox", -1.0)), reverse=True)], "skeleton_mesh_center_offset_bbox", "Largest skeleton/mesh center offset", int(args.viz_vertices))
            plot_gallery(plt, figures / "highest_connector_gallery.png", [x for x in sorted(metrics, key=lambda r: float(r.get("target_connector_count", -1.0)), reverse=True)], "target_connector_count", "Highest connector count", int(args.viz_vertices))

    print(json.dumps({"event": "audit_done", "summary": summary}, ensure_ascii=False, sort_keys=True), flush=True)
    if dataset_checks["duplicate_manifest_path_count"] or dataset_checks["missing_npz_count"] or dataset_checks["orphan_npz_count"]:
        return 2
    return 1 if hard_fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
