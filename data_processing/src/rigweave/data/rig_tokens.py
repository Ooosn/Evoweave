from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RestSpaceTransform:
    center: np.ndarray
    scale: float


def _bbox_center_scale(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(vertices, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"rest vertices must be [V,3], got {pts.shape}")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = float(np.max((hi - lo) * 0.5))
    if not np.isfinite(scale) or scale < 1.0e-8:
        raise ValueError("degenerate rest mesh scale")
    return center, scale


def normalize_rest_space(
    rest_vertices: np.ndarray,
    frame_vertices: np.ndarray,
    rest_joints: np.ndarray,
    posed_joints: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], RestSpaceTransform, None]:
    """Normalize rest/posed geometry by the rest mesh bounding box.

    Returns the legacy tuple shape used by strict data scripts:
    ``(frame_vertices_norm, (rest_vertices_norm, rest_joints_norm,
    posed_joints_norm), transform, None)``.
    """

    center, scale = _bbox_center_scale(rest_vertices)
    rest_vertices_n = ((np.asarray(rest_vertices, dtype=np.float32) - center) / scale).astype(np.float32)
    frame_vertices_n = ((np.asarray(frame_vertices, dtype=np.float32) - center) / scale).astype(np.float32)
    rest_joints_n = ((np.asarray(rest_joints, dtype=np.float32) - center) / scale).astype(np.float32)
    posed_joints_n = ((np.asarray(posed_joints, dtype=np.float32) - center) / scale).astype(np.float32)
    return (
        frame_vertices_n,
        (rest_vertices_n, rest_joints_n, posed_joints_n),
        RestSpaceTransform(center=center, scale=float(scale)),
        None,
    )
