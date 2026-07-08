from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuerySpaceTransform:
    center: np.ndarray
    scale: float
    query_idx: int


def _bbox_center_scale(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(vertices, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"query vertices must be [V,3], got {pts.shape}")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = float(np.max((hi - lo) * 0.5))
    if not np.isfinite(scale) or scale < 1.0e-8:
        raise ValueError("degenerate query mesh scale")
    return center, scale


def _normalize(points: np.ndarray | None, center: np.ndarray, scale: float) -> np.ndarray | None:
    if points is None:
        return None
    return ((np.asarray(points, dtype=np.float32) - center) / float(scale)).astype(np.float32)


def normalize_query_root_space(
    frame_vertices: np.ndarray,
    posed_joints: np.ndarray,
    posed_tails: np.ndarray | None,
    bone_transforms: np.ndarray | None,
    query_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, QuerySpaceTransform]:
    """Normalize a dynamic sequence into the query frame mesh-bbox space.

    The strict data filters and tokenizer audit use one query frame as the
    coordinate reference.  All frames, joints, and tails are transformed by the
    same query-frame center and scale so temporal motion remains comparable.
    ``bone_transforms`` is accepted for API compatibility with older callers.
    """

    frames = np.asarray(frame_vertices, dtype=np.float32)
    joints = np.asarray(posed_joints, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError(f"frame_vertices must be [T,V,3], got {frames.shape}")
    if joints.ndim != 3 or joints.shape[0] != frames.shape[0] or joints.shape[-1] != 3:
        raise ValueError(f"posed_joints shape {joints.shape} does not match frames {frames.shape}")
    q = int(query_idx)
    if q < 0 or q >= frames.shape[0]:
        raise IndexError(f"query_idx {q} out of range for {frames.shape[0]} frames")
    center, scale = _bbox_center_scale(frames[q])
    tails_n = None if posed_tails is None else _normalize(np.asarray(posed_tails, dtype=np.float32), center, scale)
    return (
        _normalize(frames, center, scale),
        _normalize(joints, center, scale),
        tails_n,
        QuerySpaceTransform(center=center, scale=float(scale), query_idx=q),
    )
