from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def manifest_rows(
    manifest_path: str | Path | None,
    statuses: Iterable[str] = ("keep_strict", "keep_repaired_multi_root"),
    split: str | None = None,
) -> dict[str, dict] | None:
    """Load strict-cleaning manifest rows keyed by resolved npz path."""

    if manifest_path is None:
        return None
    allowed = set(statuses)
    rows: dict[str, dict] = {}
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed and row.get("status") not in allowed:
                continue
            if split is not None and row.get("split") != split:
                continue
            rows[Path(row["path"]).resolve().as_posix()] = row
    return rows


def apply_repair(parents: np.ndarray, manifest_row: dict | None) -> np.ndarray:
    """Apply the stored strict-cleaning parent repair, if any."""

    repaired = np.asarray(parents, dtype=np.int64).copy()
    repair = None if manifest_row is None else manifest_row.get("repair")
    if not repair:
        return repaired
    if repair.get("rule") != "attach_secondary_roots_to_primary_by_skin_mass":
        raise ValueError(f"unsupported repair rule: {repair}")
    primary = int(repair["primary_root"])
    for root in repair.get("attached_roots", []):
        repaired[int(root)] = primary
    return repaired


def sample_vertices(vertex_count: int, sample_count: int, seed: int) -> np.ndarray:
    """Deterministically sample vertex ids with replacement only when needed."""

    rng = np.random.default_rng(seed)
    return rng.choice(vertex_count, size=sample_count, replace=vertex_count < sample_count).astype(np.int64)


def apply_bone_transforms(bone_transforms: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Apply per-frame bone transforms to rest joints.

    Args:
        bone_transforms: `(T, J, 4, 4)` transforms.
        joints: `(J, 3)` rest-pose joint positions.

    Returns:
        `(T, J, 3)` posed joint positions.
    """

    transforms = np.asarray(bone_transforms, dtype=np.float32)
    joints_f = np.asarray(joints, dtype=np.float32)
    ones = np.ones((joints_f.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([joints_f, ones], axis=1)
    posed = np.einsum("tjbc,jc->tjb", transforms, homogeneous)
    return posed[..., :3].astype(np.float32)


# Backwards-compatible aliases for scripts that still use the old private names.
_manifest_rows = manifest_rows
_apply_repair = apply_repair
_sample_vertices = sample_vertices
_apply_bone_transforms = apply_bone_transforms
