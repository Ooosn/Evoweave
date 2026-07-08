from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrackableSurfaceReferences:
    """Clip-level surface correspondence sampled on frame 0.

    `vertex_indices` produce the first dense samples directly from mesh
    vertices. `face_indices` and `barycentric` produce the remaining dense
    samples from triangle interiors. `query_indices` select the FPS query
    anchors from the concatenated dense sample list.
    """

    vertex_indices: torch.LongTensor
    face_indices: torch.LongTensor
    barycentric: torch.Tensor
    query_indices: torch.LongTensor

    @property
    def dense_count(self) -> int:
        return int(self.vertex_indices.shape[1] + self.face_indices.shape[1])

    @property
    def query_count(self) -> int:
        return int(self.query_indices.shape[1])

    def to(self, device: torch.device | str) -> "TrackableSurfaceReferences":
        return TrackableSurfaceReferences(
            vertex_indices=self.vertex_indices.to(device),
            face_indices=self.face_indices.to(device),
            barycentric=self.barycentric.to(device),
            query_indices=self.query_indices.to(device),
        )


@dataclass(frozen=True)
class TrackableSurfaceSamples:
    """Materialized surface samples for one frame or one frame batch."""

    dense_points: torch.Tensor
    dense_normals: torch.Tensor
    query_points: torch.Tensor
    query_normals: torch.Tensor
    query_indices: torch.LongTensor


def _as_batched_faces(faces: torch.LongTensor, batch_size: int) -> torch.LongTensor:
    if faces.dim() == 2:
        return faces.unsqueeze(0).expand(batch_size, -1, -1)
    if faces.dim() == 3 and faces.shape[0] == batch_size:
        return faces
    raise ValueError(f"faces must be (F,3) or (B,F,3), got {tuple(faces.shape)}")


def _gather_vertices(values: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    # values: (B, N, C), indices: (B, M) -> (B, M, C)
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, dim=1, index=gather_idx)


def _barycentric_points(
    vertices: torch.Tensor,
    faces: torch.LongTensor,
    face_indices: torch.LongTensor,
    barycentric: torch.Tensor,
) -> torch.Tensor:
    batch_size = vertices.shape[0]
    faces_b = _as_batched_faces(faces, batch_size)
    out = []
    for b in range(batch_size):
        tri_ids = faces_b[b, face_indices[b]]
        tri = vertices[b, tri_ids]
        out.append((tri * barycentric[b].unsqueeze(-1)).sum(dim=1))
    return torch.stack(out, dim=0)


def materialize_trackable_surface(
    vertices: torch.Tensor,
    faces: torch.LongTensor,
    refs: TrackableSurfaceReferences,
    vertex_normals: torch.Tensor | None = None,
    face_normals: torch.Tensor | None = None,
) -> TrackableSurfaceSamples:
    """Apply reset-frame correspondence to a posed frame.

    Args:
        vertices: `(B, N, 3)` posed vertices.
        faces: `(F, 3)` or `(B, F, 3)` triangle indices.
        refs: correspondence sampled once on frame 0.
        vertex_normals: optional `(B, N, 3)` normals for vertex samples.
        face_normals: optional `(B, F, 3)` normals for face samples.
    """

    if vertices.dim() != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must be (B,N,3), got {tuple(vertices.shape)}")

    batch_size = vertices.shape[0]
    refs = refs.to(vertices.device)

    vertex_points = _gather_vertices(vertices, refs.vertex_indices)
    surface_points = _barycentric_points(vertices, faces.to(vertices.device), refs.face_indices, refs.barycentric)
    dense_points = torch.cat([vertex_points, surface_points], dim=1)

    if vertex_normals is None:
        vertex_normal_samples = torch.zeros_like(vertex_points)
    else:
        vertex_normal_samples = _gather_vertices(vertex_normals.to(vertices.device), refs.vertex_indices)

    if face_normals is None:
        surface_normal_samples = torch.zeros_like(surface_points)
    else:
        face_normals = face_normals.to(vertices.device)
        face_normals_b = face_normals.unsqueeze(0).expand(batch_size, -1, -1) if face_normals.dim() == 2 else face_normals
        surface_normal_samples = _gather_vertices(face_normals_b, refs.face_indices)

    dense_normals = torch.cat([vertex_normal_samples, surface_normal_samples], dim=1)

    query_points = _gather_vertices(dense_points, refs.query_indices)
    query_normals = _gather_vertices(dense_normals, refs.query_indices)

    return TrackableSurfaceSamples(
        dense_points=dense_points,
        dense_normals=dense_normals,
        query_points=query_points,
        query_normals=query_normals,
        query_indices=refs.query_indices,
    )


def _triangle_areas(vertices: torch.Tensor, faces: torch.LongTensor) -> torch.Tensor:
    tri = vertices[faces]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return cross.norm(dim=-1).clamp_min(1.0e-12)


def _random_barycentric(count: int, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    uv = torch.rand((count, 2), device=device, generator=generator)
    mask = uv.sum(dim=-1) > 1.0
    uv[mask] = 1.0 - uv[mask]
    u = uv[:, 0]
    v = uv[:, 1]
    return torch.stack([1.0 - u - v, u, v], dim=-1)


def farthest_point_sample(points: torch.Tensor, count: int, generator: torch.Generator | None = None) -> torch.LongTensor:
    """Simple FPS over `(B, P, 3)` points.

    This is intentionally dependency-free. For full-scale training, it can be
    replaced by a CUDA FPS kernel without changing the sampler contract.
    """

    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (B,P,3), got {tuple(points.shape)}")
    batch_size, point_count, _ = points.shape
    if count > point_count:
        raise ValueError(f"FPS count {count} exceeds point count {point_count}")

    selected = torch.empty((batch_size, count), device=points.device, dtype=torch.long)
    min_dist = torch.full((batch_size, point_count), float("inf"), device=points.device)
    farthest = torch.randint(point_count, (batch_size,), device=points.device, generator=generator)
    batch = torch.arange(batch_size, device=points.device)

    for i in range(count):
        selected[:, i] = farthest
        centroid = points[batch, farthest].unsqueeze(1)
        dist = ((points - centroid) ** 2).sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = min_dist.argmax(dim=-1)
    return selected


def sample_trackable_surface(
    rest_vertices: torch.Tensor,
    faces: torch.LongTensor,
    *,
    num_samples: int = 65536,
    vertex_samples: int = 8192,
    query_tokens: int = 1024,
    vertex_counts: torch.LongTensor | None = None,
    face_counts: torch.LongTensor | None = None,
    generator: torch.Generator | None = None,
) -> TrackableSurfaceReferences:
    """Sample dense surface points and FPS query anchors on frame 0.

    Different training samples may sample different references. All frames
    inside the same sample must reuse the returned references.

    In the current query-frame baseline, `rest_vertices` is the normalized
    query-pose mesh, not necessarily the asset's bind/rest mesh.
    """

    if rest_vertices.dim() != 3 or rest_vertices.shape[-1] != 3:
        raise ValueError(f"rest_vertices must be (B,N,3), got {tuple(rest_vertices.shape)}")
    if faces.dim() not in (2, 3) or faces.shape[-1] != 3:
        raise ValueError(f"faces must be (F,3) or (B,F,3), got {tuple(faces.shape)}")
    if num_samples < query_tokens:
        raise ValueError("num_samples must be >= query_tokens")
    if vertex_samples > num_samples:
        raise ValueError("vertex_samples must be <= num_samples")

    device = rest_vertices.device
    batch_size, vertex_count, _ = rest_vertices.shape
    surface_count = num_samples - vertex_samples
    faces_b = _as_batched_faces(faces.to(device), batch_size)
    if vertex_counts is None:
        vertex_counts = torch.full((batch_size,), vertex_count, dtype=torch.long, device=device)
    else:
        vertex_counts = vertex_counts.to(device=device, dtype=torch.long)
    if face_counts is None:
        face_counts = torch.full((batch_size,), faces_b.shape[1], dtype=torch.long, device=device)
    else:
        face_counts = face_counts.to(device=device, dtype=torch.long)

    vertex_refs = []
    face_refs = []
    bary_refs = []
    for b in range(batch_size):
        valid_vertices = int(vertex_counts[b].item())
        valid_faces = int(face_counts[b].item())
        if valid_vertices <= 0:
            raise ValueError(f"batch item {b} has no valid vertices")
        if valid_faces <= 0:
            raise ValueError(f"batch item {b} has no valid faces")

        if vertex_samples <= valid_vertices:
            perm = torch.randperm(valid_vertices, device=device, generator=generator)[:vertex_samples]
        else:
            perm = torch.randint(valid_vertices, (vertex_samples,), device=device, generator=generator)
        vertex_refs.append(perm)

        valid_face_tensor = faces_b[b, :valid_faces]
        if int(valid_face_tensor.max().item()) >= valid_vertices or int(valid_face_tensor.min().item()) < 0:
            raise ValueError(f"batch item {b} has face indices outside valid vertex range")
        areas = _triangle_areas(rest_vertices[b, :valid_vertices], valid_face_tensor)
        face_idx = torch.multinomial(areas, surface_count, replacement=True, generator=generator)
        face_refs.append(face_idx)
        bary_refs.append(_random_barycentric(surface_count, device=device, generator=generator))

    vertex_indices = torch.stack(vertex_refs, dim=0)
    face_indices = torch.stack(face_refs, dim=0)
    barycentric = torch.stack(bary_refs, dim=0)

    refs_without_query = TrackableSurfaceReferences(
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        barycentric=barycentric,
        query_indices=torch.empty((batch_size, 0), dtype=torch.long, device=device),
    )
    rest_samples = materialize_trackable_surface(rest_vertices, faces.to(device), refs_without_query)
    query_indices = farthest_point_sample(rest_samples.dense_points, query_tokens, generator=generator)

    return TrackableSurfaceReferences(
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        barycentric=barycentric,
        query_indices=query_indices,
    )
