from __future__ import annotations

import torch
from torch import nn

from .motion_encoder import AnchorWiseAlternatingMotionEncoder
from .motion_evidence import grouped_motion_evidence
from .sampling import TrackableSurfaceReferences, materialize_trackable_surface
from .surface_tokenizer import FixedQuerySurfaceTokenizer


class DynamicRigConditioner(nn.Module):
    """Dynamic mesh sequence encoder producing UniRig-compatible condition tokens."""

    def __init__(
        self,
        surface_tokenizer: FixedQuerySurfaceTokenizer,
        motion_encoder: AnchorWiseAlternatingMotionEncoder,
    ) -> None:
        super().__init__()
        self.surface_tokenizer = surface_tokenizer
        self.motion_encoder = motion_encoder

    def forward(
        self,
        frame_vertices: torch.Tensor,
        faces: torch.LongTensor,
        refs: TrackableSurfaceReferences,
        vertex_normals: torch.Tensor | None = None,
        face_normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode `(B,T,N,3)` dynamic vertices into `(B,Q,D)` condition tokens."""

        if frame_vertices.dim() != 4 or frame_vertices.shape[-1] != 3:
            raise ValueError(f"frame_vertices must be (B,T,N,3), got {tuple(frame_vertices.shape)}")

        frame_tokens = []
        frame_query_points = []
        dense_frames = []
        needs_evidence = getattr(self.motion_encoder, "motion_evidence_fusion", "off") != "off"
        for t in range(frame_vertices.shape[1]):
            v_normals_t = None if vertex_normals is None else vertex_normals[:, t]
            f_normals_t = None if face_normals is None else face_normals[:, t]
            samples = materialize_trackable_surface(
                frame_vertices[:, t],
                faces,
                refs,
                vertex_normals=v_normals_t,
                face_normals=f_normals_t,
            )
            tokens_t = self.surface_tokenizer(
                samples.dense_points,
                samples.dense_normals,
                samples.query_points,
                samples.query_normals,
            )
            frame_tokens.append(tokens_t)
            frame_query_points.append(samples.query_points)
            if needs_evidence:
                dense_frames.append(samples.dense_points.detach())

        z_seq = torch.stack(frame_tokens, dim=1)
        query_points = torch.stack(frame_query_points, dim=1)
        if needs_evidence:
            evidence = grouped_motion_evidence(torch.stack(dense_frames, dim=1), refs.query_indices)
            return self.motion_encoder(z_seq, query_points=query_points, motion_evidence=evidence)
        return self.motion_encoder(z_seq, query_points=query_points)
