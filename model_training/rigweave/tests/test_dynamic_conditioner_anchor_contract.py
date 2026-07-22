from __future__ import annotations

import torch
from torch import nn

from rigweave.dynamic_rig.model import DynamicRigConditioner
from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences


class _SurfaceTokenizer(nn.Module):
    def forward(
        self,
        dense_points: torch.Tensor,
        dense_normals: torch.Tensor,
        query_points: torch.Tensor,
        query_normals: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([query_points, query_points[..., :1]], dim=-1)


class _MotionEncoder(nn.Module):
    dim = 4

    def forward(self, tokens: torch.Tensor, *, query_points: torch.Tensor) -> torch.Tensor:
        assert tokens.shape[:3] == query_points.shape[:3]
        return tokens.mean(dim=1)


def test_tokenize_frames_exposes_the_same_anchor_order_used_by_motion_encoder() -> None:
    conditioner = DynamicRigConditioner(_SurfaceTokenizer(), _MotionEncoder())
    frame_vertices = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
          [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]]
    )
    faces = torch.tensor([[[0, 1, 2]]])
    refs = TrackableSurfaceReferences(
        vertex_indices=torch.tensor([[0, 1]]),
        face_indices=torch.empty((1, 0), dtype=torch.long),
        barycentric=torch.empty((1, 0, 3)),
        query_indices=torch.tensor([[0, 1]]),
    )
    vertex_normals = torch.zeros_like(frame_vertices)
    face_normals = torch.zeros((1, 2, 1, 3))

    frame_tokens, query_points = conditioner.tokenize_frames(
        frame_vertices,
        faces,
        refs,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
    )
    output = conditioner(
        frame_vertices,
        faces,
        refs,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
    )

    assert frame_tokens.shape == (1, 2, 2, 4)
    assert torch.equal(frame_tokens[:, 0, :, :3], query_points[:, 0])
    assert torch.equal(frame_tokens[:, 1, :, :3], query_points[:, 1])
    assert torch.equal(output, frame_tokens.mean(dim=1))
