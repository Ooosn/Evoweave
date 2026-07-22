from __future__ import annotations

import torch
from torch import nn

from .motion_encoder import AnchorWiseAlternatingMotionEncoder
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

    def tokenize_frames(
        self,
        frame_vertices: torch.Tensor,
        faces: torch.LongTensor,
        refs: TrackableSurfaceReferences,
        vertex_normals: torch.Tensor | None = None,
        face_normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize every frame while preserving the shared anchor ordering."""

        if frame_vertices.dim() != 4 or frame_vertices.shape[-1] != 3:
            raise ValueError(f"frame_vertices must be (B,T,N,3), got {tuple(frame_vertices.shape)}")

        frame_tokens = []
        frame_query_points = []
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

        return torch.stack(frame_tokens, dim=1), torch.stack(frame_query_points, dim=1)

    def forward(
        self,
        frame_vertices: torch.Tensor,
        faces: torch.LongTensor,
        refs: TrackableSurfaceReferences,
        vertex_normals: torch.Tensor | None = None,
        face_normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode `(B,T,N,3)` dynamic vertices into `(B,Q,D)` condition tokens."""

        z_seq, query_points = self.tokenize_frames(
            frame_vertices,
            faces,
            refs,
            vertex_normals=vertex_normals,
            face_normals=face_normals,
        )
        return self.motion_encoder(z_seq, query_points=query_points)
