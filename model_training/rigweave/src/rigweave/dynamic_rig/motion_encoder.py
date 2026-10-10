from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .motion_evidence import MotionEvidence, MotionEvidenceInjection
from .motion_relation_residual import MotionRelationResidual


class AnchorWiseAlternatingBlock(nn.Module):
    """One pose-inner / anchor-temporal block over `(B, T, S, D)` tokens."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        motion_evidence_fusion: str = "off",
        motion_evidence_heads: int = 0,
    ) -> None:
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        self.pose_inner = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.anchor_temporal = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.motion_injection = MotionEvidenceInjection(
            dim, heads, fusion=motion_evidence_fusion, biased_heads=motion_evidence_heads,
        )
        self.relation_residual: MotionRelationResidual | None = None

    def forward(self, tokens: torch.Tensor, motion_evidence: MotionEvidence | None = None,
                relation_states: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError(f"tokens must be (B,T,S,D), got {tuple(tokens.shape)}")
        batch_size, frame_count, slot_count, dim = tokens.shape

        # Pose-inner attention: each posed mesh updates its anchors using the
        # current-pose geometric context.
        if self.motion_injection.fusion == "off":
            z = tokens.reshape(batch_size * frame_count, slot_count, dim)
            z = self.pose_inner(z)
            z = z.reshape(batch_size, frame_count, slot_count, dim)
        else:
            if motion_evidence is None:
                raise ValueError("enabled motion fusion requires motion evidence")
            z = self.motion_injection(self.pose_inner, tokens, motion_evidence)

        if self.relation_residual is not None:
            if relation_states is None:
                raise ValueError("enabled relation residual requires pair states")
            prefix_count = slot_count - relation_states.shape[1]
            if prefix_count < 1:
                raise ValueError("relation residual must preserve the role/register prefix")
            z = torch.cat((z[:, :, :prefix_count],
                           self.relation_residual(z[:, :, prefix_count:], relation_states)), dim=2)

        # Anchor-wise temporal attention: slot q attends only to slot q across
        # poses. This uses our fixed vertex/surface correspondence.
        z = z.permute(0, 2, 1, 3).reshape(batch_size * slot_count, frame_count, dim)
        z = self.anchor_temporal(z)
        z = z.reshape(batch_size, slot_count, frame_count, dim).permute(0, 2, 1, 3)
        return z


class AnchorWiseAlternatingMotionEncoder(nn.Module):
    """Contextualize dynamic surface anchors into canonical UniRig tokens.

    `frame_tokens` are continuous Michelangelo/UniRig condition tokens with
    shape `(B, T, Q, D)`. Slot `q` must be the same vertex/surface point in all
    frames. Each layer first lets anchors inside one pose exchange geometry,
    then lets the same anchor exchange motion evidence across poses.

    Each frame prepends a learned role token: slot 0 is a canonical-rig target
    token for frame 0 and a motion-evidence token for all other frames. Register
    tokens follow the role token and are used only inside the encoder. The
    UniRig decoder still receives only the updated canonical anchor tokens, so
    the condition length remains compatible with UniRig.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        register_tokens: int = 32,
        max_frames: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_motion_features: bool = False,
        use_time_embedding: bool = False,
        gradient_checkpointing: bool = False,
        motion_evidence_fusion: str = "off",
        motion_evidence_heads: int = 0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_tokens = int(register_tokens)
        self.max_frames = int(max_frames)
        self.use_motion_features = bool(use_motion_features)
        self.use_time_embedding = bool(use_time_embedding)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.motion_evidence_fusion = str(motion_evidence_fusion)
        self.motion_evidence_heads = int(motion_evidence_heads)

        self.role_token = nn.Parameter(torch.randn(1, 2, 1, dim) * 0.02)
        if self.register_tokens > 0:
            self.register = nn.Parameter(torch.randn(register_tokens, dim) * 0.02)
        else:
            self.register = None
        self.time_embed = nn.Parameter(torch.randn(max_frames, dim) * 0.02)
        self.motion_feature_mlp = nn.Sequential(
            nn.Linear(13, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        if not self.use_time_embedding:
            self.time_embed.requires_grad_(False)
        if not self.use_motion_features:
            for parameter in self.motion_feature_mlp.parameters():
                parameter.requires_grad_(False)
        self.blocks = nn.ModuleList(
            [
                AnchorWiseAlternatingBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    motion_evidence_fusion=self.motion_evidence_fusion,
                    motion_evidence_heads=self.motion_evidence_heads,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def enable_relation_residual(self, bottleneck_dim: int = 64, *, reference_subtraction: bool = False) -> None:
        """Attach an explicit adapter after strict loading of the base checkpoint."""
        if not self.blocks:
            raise ValueError("relation residual requires at least one motion block")
        if self.motion_evidence_fusion == "off":
            raise ValueError("relation residual requires computed motion evidence")
        if any(block.relation_residual is not None for block in self.blocks):
            raise RuntimeError("relation residual is already enabled")
        for block in self.blocks:
            block.relation_residual = MotionRelationResidual(
                self.dim, bottleneck_dim, reference_subtraction=reference_subtraction).to(
                device=self.role_token.device, dtype=self.role_token.dtype)

    @staticmethod
    def _motion_features(query_points: torch.Tensor) -> torch.Tensor:
        if query_points.dim() != 4 or query_points.shape[-1] != 3:
            raise ValueError(f"query_points must be (B,T,Q,3), got {tuple(query_points.shape)}")
        rest = query_points[:, :1].expand_as(query_points)
        delta = query_points - rest
        velocity = torch.zeros_like(query_points)
        if query_points.shape[1] > 1:
            velocity[:, 1:] = query_points[:, 1:] - query_points[:, :-1]
        delta_norm = delta.norm(dim=-1, keepdim=True)
        return torch.cat([query_points, rest, delta, velocity, delta_norm], dim=-1)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        *,
        query_points: torch.Tensor | None = None,
        return_all: bool = False,
        motion_evidence: MotionEvidence | None = None,
        relation_states: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode `(B,T,Q,D)` frame tokens into `(B,Q,D)` canonical tokens."""

        if frame_tokens.dim() != 4:
            raise ValueError(f"frame_tokens must be (B,T,Q,D), got {tuple(frame_tokens.shape)}")
        batch_size, frame_count, _, dim = frame_tokens.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if frame_count > self.max_frames:
            raise ValueError(f"frame_count {frame_count} exceeds max_frames {self.max_frames}")
        if (self.motion_evidence_fusion != "off") != (motion_evidence is not None):
            raise ValueError("motion evidence presence must match the encoder fusion configuration")
        relation_enabled = bool(self.blocks) and self.blocks[0].relation_residual is not None
        if relation_enabled:
            if relation_states is None:
                relation_states = motion_evidence.states
            query_count = frame_tokens.shape[2]
            if relation_states.shape != (batch_size, query_count, query_count, 3):
                raise ValueError("relation states must align with all input anchors")
        elif relation_states is not None:
            raise ValueError("relation states supplied without an enabled residual")

        z = frame_tokens
        if self.use_motion_features:
            if query_points is None:
                raise ValueError("query_points are required when use_motion_features=True")
            if query_points.shape[:3] != frame_tokens.shape[:3]:
                raise ValueError(
                    "query_points must align with frame_tokens in (B,T,Q), "
                    f"got {tuple(query_points.shape)} vs {tuple(frame_tokens.shape)}"
                )
            motion = self._motion_features(query_points).to(dtype=z.dtype)
            z = z + self.motion_feature_mlp(motion)

        if self.register_tokens > 0:
            assert self.register is not None
            regs = self.register.view(1, 1, self.register_tokens, dim).expand(batch_size, frame_count, -1, -1)
        else:
            regs = z.new_empty((batch_size, frame_count, 0, dim))

        canonical_role = self.role_token[:, 0:1].expand(batch_size, 1, -1, -1)
        motion_roles = self.role_token[:, 1:2].expand(batch_size, max(0, frame_count - 1), -1, -1)
        role = torch.cat([canonical_role, motion_roles], dim=1)
        z = torch.cat([role, regs, z], dim=2)
        if self.use_time_embedding:
            z = z + self.time_embed[:frame_count].view(1, frame_count, 1, dim)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(block, z, motion_evidence, relation_states, use_reentrant=False)
            else:
                z = block(z, motion_evidence, relation_states)
        z = self.norm(z)

        canonical_tokens = z[:, 0, 1 + self.register_tokens :]
        if return_all:
            return canonical_tokens, z
        return canonical_tokens


# Backward-compatible name for older scripts. The implementation is now the
# explicit anchor-wise alternating encoder.
TemporalMotionEncoder = AnchorWiseAlternatingMotionEncoder


class FrameTypeAnchorWiseAlternatingMotionEncoder(nn.Module):
    """Load-compatible encoder for checkpoints before learned role tokens."""

    def __init__(
        self,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        register_tokens: int = 32,
        max_frames: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_motion_features: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_tokens = int(register_tokens)
        self.max_frames = int(max_frames)
        self.use_motion_features = bool(use_motion_features)
        if self.register_tokens > 0:
            self.register = nn.Parameter(torch.randn(register_tokens, dim) * 0.02)
        else:
            self.register = None
        self.frame_type_embed = nn.Embedding(2, dim)
        self.time_embed = nn.Parameter(torch.randn(max_frames, dim) * 0.02)
        self.motion_feature_mlp = nn.Sequential(
            nn.Linear(13, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.blocks = nn.ModuleList(
            [
                AnchorWiseAlternatingBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        *,
        query_points: torch.Tensor | None = None,
        return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if frame_tokens.dim() != 4:
            raise ValueError(f"frame_tokens must be (B,T,Q,D), got {tuple(frame_tokens.shape)}")
        batch_size, frame_count, _, dim = frame_tokens.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if frame_count > self.max_frames:
            raise ValueError(f"frame_count {frame_count} exceeds max_frames {self.max_frames}")

        z = frame_tokens
        if self.use_motion_features:
            if query_points is None:
                raise ValueError("query_points are required when use_motion_features=True")
            if query_points.shape[:3] != frame_tokens.shape[:3]:
                raise ValueError(
                    "query_points must align with frame_tokens in (B,T,Q), "
                    f"got {tuple(query_points.shape)} vs {tuple(frame_tokens.shape)}"
                )
            motion = AnchorWiseAlternatingMotionEncoder._motion_features(query_points).to(dtype=z.dtype)
            z = z + self.motion_feature_mlp(motion)

        if self.register_tokens > 0:
            assert self.register is not None
            regs = self.register.view(1, 1, self.register_tokens, dim).expand(batch_size, frame_count, -1, -1)
            z = torch.cat([regs, z], dim=2)

        frame_type = torch.ones(frame_count, device=frame_tokens.device, dtype=torch.long)
        frame_type[0] = 0
        z = z + self.frame_type_embed(frame_type).view(1, frame_count, 1, dim)
        z = z + self.time_embed[:frame_count].view(1, frame_count, 1, dim)

        for block in self.blocks:
            z = block(z)
        z = self.norm(z)

        canonical_tokens = z[:, 0, self.register_tokens :]
        if return_all:
            return canonical_tokens, z
        return canonical_tokens


class LegacyTemporalMotionBlock(nn.Module):
    """Legacy block used by older DynamicRig checkpoints.

    The computation is the same factorized pose/temporal pattern, but the
    module names were `spatial` and `temporal` and there were no explicit
    query-point motion features. Keeping this class lets evaluation load old
    checkpoints without silently dropping their motion encoder weights.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        self.spatial = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError(f"tokens must be (B,T,S,D), got {tuple(tokens.shape)}")
        batch_size, frame_count, slot_count, dim = tokens.shape
        z = tokens.reshape(batch_size * frame_count, slot_count, dim)
        z = self.spatial(z)
        z = z.reshape(batch_size, frame_count, slot_count, dim)
        z = z.permute(0, 2, 1, 3).reshape(batch_size * slot_count, frame_count, dim)
        z = self.temporal(z)
        return z.reshape(batch_size, slot_count, frame_count, dim).permute(0, 2, 1, 3)


class LegacyTemporalMotionEncoder(nn.Module):
    """Load-compatible encoder for pre-anchor-alt checkpoints."""

    def __init__(
        self,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        register_tokens: int = 32,
        max_frames: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_tokens = int(register_tokens)
        self.max_frames = int(max_frames)
        if self.register_tokens > 0:
            self.register = nn.Parameter(torch.randn(register_tokens, dim) * 0.02)
        else:
            self.register = None
        self.frame_type_embed = nn.Embedding(2, dim)
        self.time_embed = nn.Parameter(torch.randn(max_frames, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                LegacyTemporalMotionBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        *,
        query_points: torch.Tensor | None = None,
        return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del query_points
        if frame_tokens.dim() != 4:
            raise ValueError(f"frame_tokens must be (B,T,Q,D), got {tuple(frame_tokens.shape)}")
        batch_size, frame_count, _, dim = frame_tokens.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if frame_count > self.max_frames:
            raise ValueError(f"frame_count {frame_count} exceeds max_frames {self.max_frames}")
        z = frame_tokens
        if self.register_tokens > 0:
            assert self.register is not None
            regs = self.register.view(1, 1, self.register_tokens, dim).expand(batch_size, frame_count, -1, -1)
            z = torch.cat([regs, z], dim=2)
        frame_type = torch.ones(frame_count, device=frame_tokens.device, dtype=torch.long)
        frame_type[0] = 0
        z = z + self.frame_type_embed(frame_type).view(1, frame_count, 1, dim)
        z = z + self.time_embed[:frame_count].view(1, frame_count, 1, dim)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        canonical_tokens = z[:, 0, self.register_tokens :]
        if return_all:
            return canonical_tokens, z
        return canonical_tokens
