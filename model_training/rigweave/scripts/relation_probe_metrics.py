"""Offline motion strata and missing-structure diagnostics, never model inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def select_rows(manifest, count, seed):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    buckets = {}
    for index, row in enumerate(rows):
        joints = int(row["canonical_metrics"]["target_joint_count"])
        band = int(np.searchsorted([16, 40, 64], joints, side="left"))
        buckets.setdefault((row["dataset_source"], band), []).append(index)
    rng = np.random.default_rng(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        before = len(selected)
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                index = buckets[key].pop()
                row = rows[index]
                selected.append({"index": index, "path": row["path"], "asset_id": row["asset_id"],
                                 "source": key[0], "joint_band": key[1],
                                 "joint_count": int(row["canonical_metrics"]["target_joint_count"])})
        if len(selected) == before:
            raise ValueError("insufficient manifest rows")
    return selected


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_motion(transforms, raw_indices, parents, rest_joints, query, scale):
    """Cancel parent motion before measuring rotation and joint-head translation."""
    transforms = np.asarray(transforms, dtype=np.float64)
    count = len(parents)
    angles = np.zeros((len(transforms), count), dtype=np.float64)
    shifts = np.zeros_like(angles)
    valid = np.zeros(count, dtype=bool)
    for joint, parent in enumerate(parents):
        if parent < 0 or parent >= count:
            continue
        raw_j, raw_p = int(raw_indices[joint]), int(raw_indices[parent])
        pair = transforms[:, [raw_j, raw_p]]
        if not np.isfinite(pair).all() or np.max(np.abs(pair[..., 3, :] - [0, 0, 0, 1])) > 1e-5:
            continue
        singular = np.linalg.svd(pair[..., :3, :3], compute_uv=False)
        if np.max(np.abs(singular - 1)) > 0.01 or np.any(np.linalg.det(pair[..., :3, :3]) <= 0):
            continue
        local = np.linalg.solve(pair[:, 1], pair[:, 0])
        rotation = local[:, :3, :3] @ np.linalg.inv(local[query, :3, :3])
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2
        angles[:, joint] = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        head = local @ np.r_[rest_joints[raw_j], 1.0]
        shifts[:, joint] = np.linalg.norm(head[:, :3] - head[query, :3], axis=1) / scale
        valid[joint] = True
    angles[query] = 0
    shifts[query] = 0
    return angles, shifts, valid


def observability(path, frames, scale):
    with np.load(path, allow_pickle=True) as raw:
        angle, shift, valid = local_motion(raw["bone_transforms"], raw["target_raw_indices"],
            raw["target_parents"], raw["rest_joints"], int(frames[0]), scale)
    selected = np.asarray(frames, dtype=int)
    unseen = np.setdiff1d(np.arange(len(angle)), selected)
    input_angle, input_shift = angle[selected].max(axis=0), shift[selected].max(axis=0)
    other_angle = angle[unseen].max(axis=0) if len(unseen) else np.zeros_like(input_angle)
    other_shift = shift[unseen].max(axis=0) if len(unseen) else np.zeros_like(input_shift)
    moving = valid & ((input_angle >= 5) | (input_shift >= 0.02))
    out = {"valid": valid.tolist(), "input_angle_degrees": input_angle.tolist(),
           "input_head_shift_query_units": input_shift.tolist(),
           "noninput_angle_degrees": other_angle.tolist(), "noninput_head_shift_query_units": other_shift.tolist()}
    strata = {"observed_active": moving}
    for label, angle_limit, shift_limit in (("strict", 1.0, 0.005), ("loose", 3.0, 0.01)):
        quiet = valid & (input_angle <= angle_limit) & (input_shift <= shift_limit)
        elsewhere = (other_angle >= 5) | (other_shift >= 0.02)
        strata["quiet_" + label] = quiet
        strata["hidden_demonstrated_" + label] = quiet & elsewhere & moving.any()
    out["strata"] = {name: mask.tolist() for name, mask in strata.items()}
    return out


def structure_by_stratum(pred_joints, pred_parents, target_joints, target_parents, observation):
    count = len(target_joints)
    distance = np.full(count, np.inf)
    edge_found = np.zeros(count, dtype=bool)
    if pred_joints is not None and len(pred_joints):
        d = np.linalg.norm(np.asarray(target_joints)[:, None] - np.asarray(pred_joints)[None], axis=-1)
        nearest = d.argmin(axis=1)
        distance = d.min(axis=1)
        edges = {(int(p), i) for i, p in enumerate(pred_parents) if p is not None and p >= 0 and p != i}
        for j, p in enumerate(target_parents):
            if p >= 0:
                edge_found[j] = (int(nearest[p]), int(nearest[j])) in edges
    out = {}
    for name, values in observation["strata"].items():
        mask = np.asarray(values, dtype=bool)
        edges = mask & (np.asarray(target_parents) >= 0)
        out[name] = {"joints": int(mask.sum()), "edges": int(edges.sum()),
                     "mean_gt_to_pred_distance": float(distance[mask].mean()) if mask.any() and np.isfinite(distance[mask]).all() else None,
                     "joint_coverage_005": float((distance[mask] <= 0.05).mean()) if mask.any() else None,
                     "edge_recall": float(edge_found[edges].mean()) if edges.any() else None}
    return out
