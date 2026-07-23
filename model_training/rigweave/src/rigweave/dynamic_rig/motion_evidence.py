"""Local relative-motion evidence derived from tracked mesh edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "mean_signed",
    "std_signed",
    "mean_abs",
    "rms",
    "max_abs",
    "abs_q50",
    "abs_q75",
    "abs_q90",
    "max_stretch",
    "max_collapse",
    "active_frame_ratio",
)


@dataclass(frozen=True)
class LocalMotionEvidence:
    edges: np.ndarray
    features: np.ndarray
    skin_overlap: np.ndarray
    boundary: np.ndarray
    observability: np.ndarray
    query_edge_lengths: np.ndarray
    dropped_degenerate_edges: int


def unique_mesh_edges(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """Return sorted unique undirected edges from triangular faces."""

    tri = np.asarray(faces, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {tri.shape}")
    if tri.size == 0:
        raise ValueError("faces are empty")
    if int(tri.min()) < 0 or int(tri.max()) >= int(vertex_count):
        raise ValueError("faces contain a vertex index outside the mesh")

    edges = np.concatenate(
        (tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]),
        axis=0,
    )
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size == 0:
        raise ValueError("faces contain no non-degenerate edge")
    return np.unique(edges, axis=0)


def relative_edge_length_trajectories(
    frame_vertices: np.ndarray,
    edges: np.ndarray,
    *,
    query_index: int = 0,
    min_length_ratio: float = 1.0e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Measure edge-length change relative to one query frame.

    Returns `(relative_change, valid_edges, query_lengths, dropped_count)`.
    The relative-change array includes the query frame, whose value is exactly
    zero apart from floating-point roundoff.
    """

    frames = np.asarray(frame_vertices, dtype=np.float32)
    edge_array = np.asarray(edges, dtype=np.int64)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError(f"frame_vertices must have shape (T,N,3), got {frames.shape}")
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError(f"edges must have shape (E,2), got {edge_array.shape}")
    if not 0 <= int(query_index) < int(frames.shape[0]):
        raise ValueError(f"query_index={query_index} outside {frames.shape[0]} frames")
    if edge_array.size == 0:
        raise ValueError("edges are empty")
    if int(edge_array.min()) < 0 or int(edge_array.max()) >= int(frames.shape[1]):
        raise ValueError("edges contain a vertex index outside frame_vertices")

    query = frames[int(query_index)]
    bbox_diag = float(np.linalg.norm(np.ptp(query, axis=0)))
    if not np.isfinite(bbox_diag) or bbox_diag <= 0.0:
        raise ValueError("query mesh has a degenerate bounding box")

    edge_vectors = frames[:, edge_array[:, 0]] - frames[:, edge_array[:, 1]]
    lengths = np.linalg.norm(edge_vectors, axis=-1)
    query_lengths = lengths[int(query_index)]
    threshold = max(float(min_length_ratio) * bbox_diag, np.finfo(np.float32).eps)
    valid = np.isfinite(query_lengths) & (query_lengths > threshold)
    valid_edges = edge_array[valid]
    valid_lengths = query_lengths[valid].astype(np.float32, copy=False)
    if valid_edges.shape[0] == 0:
        raise ValueError("all mesh edges are degenerate in the query frame")

    relative = lengths[:, valid] / valid_lengths[None, :] - 1.0
    relative[int(query_index)] = 0.0
    relative = relative.T.astype(np.float32, copy=False)
    if not np.isfinite(relative).all():
        raise ValueError("relative edge-length trajectories contain NaN or Inf")
    return relative, valid_edges, valid_lengths, int((~valid).sum())


def summarize_relative_trajectories(
    relative_change: np.ndarray,
    *,
    query_index: int = 0,
    active_threshold: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create order-independent per-edge motion features."""

    values = np.asarray(relative_change, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"relative_change must have shape (E,T), got {values.shape}")
    if values.shape[1] < 2:
        raise ValueError("at least one query and one evidence frame are required")
    if not 0 <= int(query_index) < int(values.shape[1]):
        raise ValueError(f"query_index={query_index} outside {values.shape[1]} frames")

    evidence = np.delete(values, int(query_index), axis=1)
    absolute = np.abs(evidence)
    quantiles = np.quantile(absolute, (0.50, 0.75, 0.90), axis=1).T.astype(np.float32)
    features = np.column_stack(
        (
            evidence.mean(axis=1),
            evidence.std(axis=1),
            absolute.mean(axis=1),
            np.sqrt(np.mean(np.square(evidence), axis=1)),
            absolute.max(axis=1),
            quantiles,
            np.maximum(evidence, 0.0).max(axis=1),
            np.maximum(-evidence, 0.0).max(axis=1),
            (absolute > float(active_threshold)).mean(axis=1),
        )
    ).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"feature width {features.shape[1]} does not match names {len(FEATURE_NAMES)}"
        )
    observability = features[:, FEATURE_NAMES.index("rms")].copy()
    return features, observability


def soft_skin_overlap(skin_weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram-intersection overlap for the skin distributions at edge ends."""

    weights = np.asarray(skin_weights, dtype=np.float32)
    edge_array = np.asarray(edges, dtype=np.int64)
    if weights.ndim != 2:
        raise ValueError(f"skin_weights must have shape (N,J), got {weights.shape}")
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError(f"edges must have shape (E,2), got {edge_array.shape}")
    if edge_array.size == 0:
        raise ValueError("edges are empty")
    if int(edge_array.min()) < 0 or int(edge_array.max()) >= int(weights.shape[0]):
        raise ValueError("edges contain a vertex index outside skin_weights")
    if not np.isfinite(weights).all() or float(weights.min()) < -1.0e-6:
        raise ValueError("skin_weights contain invalid values")

    row_sum = weights.sum(axis=1, keepdims=True)
    normalized = np.divide(
        weights,
        np.maximum(row_sum, 1.0e-12),
        out=np.zeros_like(weights),
        where=row_sum > 1.0e-12,
    )
    overlap = np.minimum(normalized[edge_array[:, 0]], normalized[edge_array[:, 1]]).sum(axis=1)
    return np.clip(overlap, 0.0, 1.0).astype(np.float32)


def extract_local_motion_evidence(
    frame_vertices: np.ndarray,
    faces: np.ndarray,
    skin_weights: np.ndarray,
    *,
    query_index: int = 0,
    min_length_ratio: float = 1.0e-7,
    active_threshold: float = 1.0e-3,
) -> LocalMotionEvidence:
    """Extract model-independent local articulation evidence for one asset pose."""

    frames = np.asarray(frame_vertices, dtype=np.float32)
    edges = unique_mesh_edges(faces, vertex_count=int(frames.shape[1]))
    relative, valid_edges, query_lengths, dropped = relative_edge_length_trajectories(
        frames,
        edges,
        query_index=query_index,
        min_length_ratio=min_length_ratio,
    )
    features, observability = summarize_relative_trajectories(
        relative,
        query_index=query_index,
        active_threshold=active_threshold,
    )
    overlap = soft_skin_overlap(skin_weights, valid_edges)
    boundary = (1.0 - overlap).astype(np.float32)
    return LocalMotionEvidence(
        edges=valid_edges,
        features=features,
        skin_overlap=overlap,
        boundary=boundary,
        observability=observability,
        query_edge_lengths=query_lengths,
        dropped_degenerate_edges=dropped,
    )
