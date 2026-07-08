from __future__ import annotations

import torch
from torch import nn


class FixedQuerySurfaceTokenizer(nn.Module):
    """Michelangelo-style tokenizer with externally supplied query anchors.

    UniRig's Michelangelo encoder samples/FPS query points inside `forward`.
    RigWeave supplies those query anchors from frame-0 trackable surface
    correspondence, then reuses the same encoder layers and weights.
    """

    def __init__(self, michelangelo_encoder: nn.Module, output_proj: nn.Module | None = None):
        super().__init__()
        if not hasattr(michelangelo_encoder, "encoder"):
            raise TypeError("michelangelo_encoder must expose an `.encoder` module")
        self.michelangelo_encoder = michelangelo_encoder
        self.output_proj = output_proj

    @property
    def width(self) -> int:
        return int(getattr(self.michelangelo_encoder, "width"))

    def _encode_data(self, points: torch.Tensor, feats: torch.Tensor | None) -> torch.Tensor:
        enc = self.michelangelo_encoder.encoder
        data = enc.fourier_embedder(points)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        return enc.input_proj(data)

    def forward(
        self,
        dense_points: torch.Tensor,
        dense_feats: torch.Tensor | None,
        query_points: torch.Tensor,
        query_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        """Tokenize a posed frame.

        Args:
            dense_points: `(B, P, 3)` dense vertex/surface samples.
            dense_feats: optional `(B, P, C)` point features, usually normals.
            query_points: `(B, Q, 3)` externally fixed query anchors.
            query_feats: optional `(B, Q, C)` query features.

        Returns:
            `(B, Q, D)` surface tokens, optionally projected to UniRig hidden dim.
        """

        enc = self.michelangelo_encoder.encoder
        data = self._encode_data(dense_points, dense_feats)
        query = self._encode_data(query_points, query_feats)

        latents = enc.cross_attn(query, data)
        latents = enc.self_attn(latents)
        if getattr(enc, "ln_post", None) is not None:
            latents = enc.ln_post(latents)
        if self.output_proj is not None:
            latents = self.output_proj(latents)
        return latents

