"""Skin-relation targets aligned with the motion-evidence query anchors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences

from .encoder import (
    _incident_vertex_features,
    _materialize_query_features,
    _unique_edges,
)


@dataclass(frozen=True)
class QuerySkinBoundaryTargets:
    """Incident skin-boundary mean/max at each sampled query anchor."""

    values: torch.Tensor
    valid_mask: torch.BoolTensor
    valid_edge_counts: torch.LongTensor


def query_aligned_skin_boundary_targets(
    skin_weights: torch.Tensor,
    faces: torch.LongTensor,
    refs: TrackableSurfaceReferences,
    *,
    vertex_counts: torch.LongTensor | None = None,
    face_counts: torch.LongTensor | None = None,
    min_skin_sum: float = 1.0e-6,
) -> QuerySkinBoundaryTargets:
    """Build soft local boundary targets without exposing skin at inference."""

    if skin_weights.ndim != 3:
        raise ValueError(
            f"skin_weights must have shape (B,N,J), got {tuple(skin_weights.shape)}"
        )
    if min_skin_sum <= 0.0:
        raise ValueError("min_skin_sum must be positive")
    batch_size, padded_vertices, _ = skin_weights.shape
    if faces.ndim == 2:
        faces = faces.unsqueeze(0).expand(batch_size, -1, -1)
    if faces.ndim != 3 or faces.shape[0] != batch_size or faces.shape[-1] != 3:
        raise ValueError(f"faces must have shape (B,F,3), got {tuple(faces.shape)}")

    device = skin_weights.device
    faces = faces.to(device=device, dtype=torch.long)
    if vertex_counts is None:
        vertex_counts = torch.full(
            (batch_size,), padded_vertices, device=device, dtype=torch.long
        )
    else:
        vertex_counts = vertex_counts.to(device=device, dtype=torch.long)
    if face_counts is None:
        face_counts = torch.full(
            (batch_size,), faces.shape[1], device=device, dtype=torch.long
        )
    else:
        face_counts = face_counts.to(device=device, dtype=torch.long)

    padded_targets = torch.zeros(
        (batch_size, padded_vertices, 2),
        device=device,
        dtype=torch.float32,
    )
    padded_valid = torch.zeros(
        (batch_size, padded_vertices, 1),
        device=device,
        dtype=torch.float32,
    )
    valid_edge_counts: list[int] = []

    for batch_index in range(batch_size):
        vertex_count = int(vertex_counts[batch_index])
        face_count = int(face_counts[batch_index])
        weights = skin_weights[batch_index, :vertex_count].float().clamp_min(0.0)
        weight_sum = weights.sum(dim=-1)
        skinned = weight_sum > min_skin_sum
        normalized = weights / weight_sum[:, None].clamp_min(min_skin_sum)
        edges = _unique_edges(faces[batch_index, :face_count], vertex_count)
        valid_edges = skinned[edges[:, 0]] & skinned[edges[:, 1]]
        edges = edges[valid_edges]
        valid_edge_counts.append(int(edges.shape[0]))
        if edges.shape[0] == 0:
            continue

        overlap = torch.minimum(
            normalized[edges[:, 0]],
            normalized[edges[:, 1]],
        ).sum(dim=-1)
        boundary = (1.0 - overlap).clamp(0.0, 1.0)[:, None]
        padded_targets[batch_index, :vertex_count] = _incident_vertex_features(
            boundary,
            edges,
            vertex_count,
        )
        endpoints = torch.cat((edges[:, 0], edges[:, 1]), dim=0)
        incident = torch.zeros((vertex_count, 1), device=device, dtype=torch.float32)
        incident.scatter_add_(
            0,
            endpoints[:, None],
            torch.ones((endpoints.shape[0], 1), device=device),
        )
        padded_valid[batch_index, :vertex_count] = (incident > 0).float()

    query_targets = _materialize_query_features(padded_targets, faces, refs)
    query_validity = _materialize_query_features(padded_valid, faces, refs)[..., 0]
    return QuerySkinBoundaryTargets(
        values=query_targets,
        valid_mask=query_validity > 0.999,
        valid_edge_counts=torch.tensor(valid_edge_counts, device=device, dtype=torch.long),
    )
