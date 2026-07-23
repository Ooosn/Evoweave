"""GPU-compatible topology-local relative-motion evidence extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences


@dataclass(frozen=True)
class MotionEvidenceValues:
    """Motion-only values aligned with existing FPS query references."""

    query_features: torch.Tensor
    confidence: torch.Tensor
    example_motion_q90_rms: torch.Tensor
    source_edge_counts: torch.LongTensor


def _unique_edges(faces: torch.LongTensor, vertex_count: int) -> torch.LongTensor:
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {tuple(faces.shape)}")
    if faces.numel() == 0:
        raise ValueError("faces are empty")
    if int(faces.min()) < 0 or int(faces.max()) >= int(vertex_count):
        raise ValueError("faces contain a vertex index outside the valid mesh")
    edges = torch.cat(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.numel() == 0:
        raise ValueError("faces contain no non-degenerate edge")
    return torch.unique(edges, dim=0, sorted=True)


def _incident_vertex_features(
    edge_features: torch.Tensor,
    edges: torch.LongTensor,
    vertex_count: int,
) -> torch.Tensor:
    """Aggregate mean and max edge evidence at each endpoint."""

    feature_dim = int(edge_features.shape[1])
    endpoint_ids = torch.cat((edges[:, 0], edges[:, 1]), dim=0)
    endpoint_features = torch.cat((edge_features, edge_features), dim=0)

    sums = edge_features.new_zeros((vertex_count, feature_dim))
    sums.scatter_add_(0, endpoint_ids[:, None].expand(-1, feature_dim), endpoint_features)
    degree = edge_features.new_zeros((vertex_count, 1))
    degree.scatter_add_(
        0,
        endpoint_ids[:, None],
        torch.ones((endpoint_ids.shape[0], 1), device=edge_features.device, dtype=edge_features.dtype),
    )
    means = sums / degree.clamp_min(1.0)

    maxima = edge_features.new_zeros((vertex_count, feature_dim))
    maxima.scatter_reduce_(
        0,
        endpoint_ids[:, None].expand(-1, feature_dim),
        endpoint_features,
        reduce="amax",
        include_self=True,
    )
    return torch.cat((means, maxima), dim=-1)


def _materialize_query_features(
    vertex_features: torch.Tensor,
    faces: torch.LongTensor,
    refs: TrackableSurfaceReferences,
) -> torch.Tensor:
    """Apply the existing vertex/face references to per-vertex evidence."""

    batch_size, _, feature_dim = vertex_features.shape
    refs = refs.to(vertex_features.device)
    vertex_gather = refs.vertex_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
    vertex_samples = torch.gather(vertex_features, dim=1, index=vertex_gather)

    face_samples: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        triangle_ids = faces[batch_index, refs.face_indices[batch_index]]
        triangle_values = vertex_features[batch_index, triangle_ids]
        face_samples.append(
            (triangle_values * refs.barycentric[batch_index, :, :, None]).sum(dim=1)
        )
    interpolated = torch.stack(face_samples, dim=0)
    dense = torch.cat((vertex_samples, interpolated), dim=1)
    query_gather = refs.query_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
    return torch.gather(dense, dim=1, index=query_gather)


class TopologyLocalMotionEvidence(nn.Module):
    """Extract motion-only values from real mesh edges and align them to Q."""

    edge_feature_dim = 4
    vertex_feature_dim = edge_feature_dim * 2

    def __init__(
        self,
        *,
        active_threshold: float = 1.0e-3,
        confidence_scale: float = 1.0e-2,
        min_edge_length_ratio: float = 1.0e-7,
    ) -> None:
        super().__init__()
        if active_threshold <= 0.0:
            raise ValueError("active_threshold must be positive")
        if confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be positive")
        if min_edge_length_ratio <= 0.0:
            raise ValueError("min_edge_length_ratio must be positive")
        self.active_threshold = float(active_threshold)
        self.confidence_scale = float(confidence_scale)
        self.min_edge_length_ratio = float(min_edge_length_ratio)

    def _one(
        self,
        frames: torch.Tensor,
        faces: torch.LongTensor,
        vertex_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        edges = _unique_edges(faces, vertex_count)
        vectors = frames[:, edges[:, 0]] - frames[:, edges[:, 1]]
        lengths = torch.linalg.vector_norm(vectors.float(), dim=-1)
        query_lengths = lengths[0]
        query = frames[0, :vertex_count].float()
        bbox_diag = torch.linalg.vector_norm(query.amax(dim=0) - query.amin(dim=0))
        threshold = torch.clamp(
            bbox_diag * self.min_edge_length_ratio,
            min=torch.finfo(torch.float32).eps,
        )
        valid = torch.isfinite(query_lengths) & (query_lengths > threshold)
        edges = edges[valid]
        lengths = lengths[:, valid]
        query_lengths = query_lengths[valid]
        if edges.shape[0] == 0:
            raise ValueError("all mesh edges are degenerate in the query frame")

        relative = lengths / query_lengths[None] - 1.0
        relative = relative.clone()
        relative[0] = 0.0
        evidence = relative[1:]
        absolute = evidence.abs()
        rms = torch.sqrt(torch.mean(evidence.square(), dim=0))
        std = torch.std(evidence, dim=0, unbiased=False)
        max_abs = absolute.amax(dim=0)
        active_ratio = (absolute > self.active_threshold).float().mean(dim=0)
        edge_features = torch.stack((rms, std, max_abs, active_ratio), dim=-1)
        vertex_features = _incident_vertex_features(edge_features, edges, vertex_count)
        example_motion = torch.quantile(rms, 0.90)
        return vertex_features, example_motion, int(edges.shape[0])

    def forward(
        self,
        frame_vertices: torch.Tensor,
        faces: torch.LongTensor,
        refs: TrackableSurfaceReferences,
        *,
        vertex_counts: torch.LongTensor | None = None,
        face_counts: torch.LongTensor | None = None,
    ) -> MotionEvidenceValues:
        if frame_vertices.ndim != 4 or frame_vertices.shape[-1] != 3:
            raise ValueError(
                f"frame_vertices must have shape (B,T,N,3), got {tuple(frame_vertices.shape)}"
            )
        if frame_vertices.shape[1] < 2:
            raise ValueError("at least one query and one evidence frame are required")
        if faces.ndim == 2:
            faces = faces.unsqueeze(0).expand(frame_vertices.shape[0], -1, -1)
        if faces.ndim != 3 or faces.shape[0] != frame_vertices.shape[0] or faces.shape[-1] != 3:
            raise ValueError(f"faces must have shape (B,F,3), got {tuple(faces.shape)}")

        batch_size, _, padded_vertices, _ = frame_vertices.shape
        if vertex_counts is None:
            vertex_counts = torch.full(
                (batch_size,), padded_vertices, device=frame_vertices.device, dtype=torch.long
            )
        else:
            vertex_counts = vertex_counts.to(device=frame_vertices.device, dtype=torch.long)
        if face_counts is None:
            face_counts = torch.full(
                (batch_size,), faces.shape[1], device=faces.device, dtype=torch.long
            )
        else:
            face_counts = face_counts.to(device=faces.device, dtype=torch.long)

        padded_features = frame_vertices.new_zeros(
            (batch_size, padded_vertices, self.vertex_feature_dim),
            dtype=torch.float32,
        )
        motion_amounts: list[torch.Tensor] = []
        edge_counts: list[int] = []
        for batch_index in range(batch_size):
            vertex_count = int(vertex_counts[batch_index])
            face_count = int(face_counts[batch_index])
            if vertex_count <= 0 or vertex_count > padded_vertices:
                raise ValueError(f"invalid vertex_count={vertex_count} for batch item {batch_index}")
            if face_count <= 0 or face_count > faces.shape[1]:
                raise ValueError(f"invalid face_count={face_count} for batch item {batch_index}")
            values, amount, edge_count = self._one(
                frame_vertices[batch_index, :, :vertex_count],
                faces[batch_index, :face_count],
                vertex_count,
            )
            padded_features[batch_index, :vertex_count] = values
            motion_amounts.append(amount)
            edge_counts.append(edge_count)

        example_motion = torch.stack(motion_amounts, dim=0)
        confidence = example_motion / (example_motion + self.confidence_scale)
        query_features = _materialize_query_features(padded_features, faces, refs)
        return MotionEvidenceValues(
            query_features=query_features,
            confidence=confidence,
            example_motion_q90_rms=example_motion,
            source_edge_counts=torch.tensor(
                edge_counts, device=frame_vertices.device, dtype=torch.long
            ),
        )


class TopologyMotionValueEncoder(nn.Module):
    """Encode deterministic local deformation statistics into motion values."""

    def __init__(
        self,
        hidden_size: int,
        *,
        feature_scale: float = 1.0e-2,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if feature_scale <= 0.0:
            raise ValueError("feature_scale must be positive")
        width = int(intermediate_size or hidden_size)
        self.feature_scale = float(feature_scale)
        self.network = nn.Sequential(
            nn.Linear(TopologyLocalMotionEvidence.vertex_feature_dim + 1, width),
            nn.GELU(),
            nn.Linear(width, hidden_size),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if query_features.ndim != 3:
            raise ValueError(
                f"query_features must have shape (B,Q,C), got {tuple(query_features.shape)}"
            )
        if query_features.shape[-1] != TopologyLocalMotionEvidence.vertex_feature_dim:
            raise ValueError(
                f"expected {TopologyLocalMotionEvidence.vertex_feature_dim} query features, "
                f"got {query_features.shape[-1]}"
            )
        if confidence.shape != (query_features.shape[0],):
            raise ValueError(
                f"confidence must have shape {(query_features.shape[0],)}, got {tuple(confidence.shape)}"
            )
        temporal = query_features[..., :3]
        active = query_features[..., 3:4]
        temporal_max = query_features[..., 4:7]
        active_max = query_features[..., 7:8]
        normalized = torch.cat(
            (
                torch.log1p(temporal / self.feature_scale),
                active,
                torch.log1p(temporal_max / self.feature_scale),
                active_max,
                confidence[:, None, None].expand(-1, query_features.shape[1], 1),
            ),
            dim=-1,
        )
        compute_dtype = self.network[0].weight.dtype
        values = self.network(normalized.to(dtype=compute_dtype))
        return values * confidence.to(dtype=values.dtype)[:, None, None]
