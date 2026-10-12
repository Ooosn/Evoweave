"""Shared topology from a mesh-free frame and localization from final anchors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .motion_encoder import AnchorWiseAlternatingMotionEncoder
from .motion_evidence import MotionEvidence, grouped_motion_evidence
from .shared_structure import SharedParentDecoder, SharedStructureConfig, SkeletonSequence


@dataclass(frozen=True)
class VirtualFrameRigConfig:
    dim: int = 1024
    heads: int = 8
    query_tokens: int = 1024
    register_tokens: int = 96
    motion_depth: int = 12
    position_depth: int = 2
    position_self_attention: bool = True
    max_joints: int = 512
    max_frames: int = 48
    motion_evidence_fusion: str = "bias"
    motion_evidence_heads: int = 8
    dropout: float = 0.0
    motion_checkpointing: bool = True
    surface_checkpointing: bool = True
    position_checkpointing: bool = True
    generation_cache: bool = True
    temporal_chunk_size: int = 4096
    position_coordinate_readout: bool = False
    position_coordinate_dim: int = 128

    def __post_init__(self) -> None:
        counts = (self.dim, self.heads, self.query_tokens, self.register_tokens,
                  self.motion_depth, self.position_depth, self.max_joints, self.max_frames,
                  self.temporal_chunk_size, self.position_coordinate_dim)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in counts):
            raise ValueError("dimensions, counts and depths must be positive integers")
        if self.dim % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("invalid attention width or dropout")
        if self.motion_evidence_fusion not in {"off", "bias"}:
            raise ValueError("virtual-frame baseline supports off or bias motion evidence")
        if not isinstance(self.position_self_attention, bool):
            raise ValueError("position_self_attention must be boolean")
        if not isinstance(self.position_coordinate_readout, bool):
            raise ValueError("position_coordinate_readout must be boolean")


class VirtualFrameMotionEncoder(AnchorWiseAlternatingMotionEncoder):
    """Real frames have role/register/anchor slots; the virtual frame has no anchors."""

    def __init__(self, config: VirtualFrameRigConfig) -> None:
        super().__init__(
            dim=config.dim, depth=config.motion_depth, heads=config.heads,
            register_tokens=config.register_tokens, max_frames=config.max_frames,
            dropout=config.dropout, gradient_checkpointing=config.motion_checkpointing,
            motion_evidence_fusion=config.motion_evidence_fusion,
            motion_evidence_heads=config.motion_evidence_heads,
            register_temporal_mode="aligned",
        )
        self.query_tokens = config.query_tokens
        self.temporal_chunk_size = config.temporal_chunk_size

    @staticmethod
    def _temporal(layer: nn.Module, sequences: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # The leading axis contains independent aligned-slot sequences. Keep
        # every frame together while bounding FlashAttention's backward workspace.
        if sequences.shape[0] <= chunk_size:
            return layer(sequences)
        return torch.cat([layer(part) for part in sequences.split(chunk_size, dim=0)], dim=0)

    @staticmethod
    def _block(
        block: nn.Module, real: torch.Tensor, virtual: torch.Tensor,
        evidence: MotionEvidence | None, chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, slots, dim = real.shape
        prefix = virtual.shape[1]
        if block.motion_injection.fusion == "off":
            real = block.pose_inner(real.reshape(batch * frames, slots, dim)).reshape(
                batch, frames, slots, dim)
        else:
            real = block.motion_injection(block.pose_inner, real, evidence)
        virtual = block.pose_inner(virtual)

        # Only role/register slots extend to the virtual frame. Anchors retain
        # real-frame correspondence, with no padding or invented virtual mesh.
        shared = torch.cat((real[:, :, :prefix], virtual[:, None]), dim=1)
        shared = shared.permute(0, 2, 1, 3).reshape(batch * prefix, frames + 1, dim)
        shared = VirtualFrameMotionEncoder._temporal(block.anchor_temporal, shared, chunk_size)
        shared = shared.reshape(batch, prefix, frames + 1, dim).permute(0, 2, 1, 3)
        anchors = real[:, :, prefix:].permute(0, 2, 1, 3)
        anchors = VirtualFrameMotionEncoder._temporal(
            block.anchor_temporal, anchors.reshape(batch * (slots - prefix), frames, dim), chunk_size)
        anchors = anchors.reshape(batch, slots - prefix, frames, dim).permute(0, 2, 1, 3)
        return torch.cat((shared[:, :frames], anchors), dim=2), shared[:, -1]

    def forward(
        self, frame_tokens: torch.Tensor, *, query_points: torch.Tensor | None = None,
        motion_evidence: MotionEvidence | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_tokens.ndim != 4 or frame_tokens.shape[2:] != (self.query_tokens, self.dim):
            raise ValueError("expected aligned (B,T,query_tokens,dim) surface features")
        batch, frames, _, dim = frame_tokens.shape
        if not 1 <= frames <= self.max_frames:
            raise ValueError("input frame count exceeds the encoder capacity")
        if (self.motion_evidence_fusion != "off") != (motion_evidence is not None):
            raise ValueError("motion evidence must match the configured fusion")
        if query_points is not None and query_points.shape != (*frame_tokens.shape[:3], 3):
            raise ValueError("anchor coordinates must align with all real frame tokens")
        if self.register_temporal_mode != "aligned":
            raise ValueError("virtual-frame registers require aligned-index temporal attention")
        if any(block.relation_residual is not None for block in self.blocks):
            raise ValueError("the virtual-frame baseline does not use a relation adapter")
        registers = self.register[None, None].expand(batch, frames, -1, -1)
        roles = self.role_token[:, 1:2].expand(batch, frames, -1, -1)
        real = torch.cat((roles, registers, frame_tokens), dim=2)
        virtual = torch.cat((self.role_token[:, 0], self.register[None]), dim=1).expand(batch, -1, -1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                real, virtual = checkpoint(self._block, block, real, virtual, motion_evidence,
                                           self.temporal_chunk_size, use_reentrant=False)
            else:
                real, virtual = self._block(block, real, virtual, motion_evidence, self.temporal_chunk_size)
        real, virtual = self.norm(real), self.norm(virtual)
        return real[:, :, 1 + self.register_tokens:], virtual[:, 1:]


class JointLocalizationLayer(nn.Module):
    def __init__(self, config: VirtualFrameRigConfig) -> None:
        super().__init__()
        self.self_attention = (
            nn.MultiheadAttention(config.dim, config.heads, dropout=config.dropout, batch_first=True)
            if config.position_self_attention else None
        )
        self.self_norm = nn.LayerNorm(config.dim) if self.self_attention is not None else None
        self.cross_attention = nn.MultiheadAttention(
            config.dim, config.heads, dropout=config.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(config.dim)
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * config.dim, config.dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, joints: torch.Tensor, memory: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        if self.self_attention is not None:
            query = self.self_norm(joints)
            joints = joints + self.dropout(self.self_attention(
                query, query, query, key_padding_mask=padding, need_weights=False)[0])
        joints = joints + self.dropout(self.cross_attention(
            self.cross_norm(joints), memory, memory, need_weights=False)[0])
        return joints + self.dropout(self.ffn(self.ffn_norm(joints)))


class AnchorCoordinateReadout(nn.Module):
    """Read a spatial correction directly from learned anchor probabilities."""

    def __init__(self, dim: int, attention_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, attention_dim, bias=False)
        self.key = nn.Linear(dim, attention_dim, bias=False)
        self.scale = attention_dim ** -0.5
        nn.init.zeros_(self.key.weight)

    def forward(self, joints: torch.Tensor, memory: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        query, key = self.query(joints), self.key(memory)
        logits = torch.bmm(query, key.transpose(1, 2)) * self.scale
        # A zero key projection makes the correction zero at attachment. The
        # original unbounded XYZ head still predicts the absolute component.
        with torch.autocast(device_type=joints.device.type, enabled=False):
            weights = logits.float().softmax(dim=-1)
            centered = points.float() - points.float().mean(dim=1, keepdim=True)
            return torch.bmm(weights, centered)


class ContinuousJointLocalizer(nn.Module):
    def __init__(self, config: VirtualFrameRigConfig) -> None:
        super().__init__()
        self.config = config
        self.point_embedding = nn.Sequential(
            nn.Linear(3, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.layers = nn.ModuleList([JointLocalizationLayer(config) for _ in range(config.position_depth)])
        self.norm = nn.LayerNorm(config.dim)
        self.xyz_head = nn.Linear(config.dim, 3)
        self.coordinate_readout = (
            AnchorCoordinateReadout(config.dim, config.position_coordinate_dim)
            if config.position_coordinate_readout else None
        )

    def forward(
        self, joints: torch.Tensor, joint_mask: torch.Tensor,
        anchors: torch.Tensor, anchor_points: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.ndim != 4 or anchor_points.shape != (*anchors.shape[:3], 3):
            raise ValueError("final anchors and their coordinates must have aligned (B,T,Q) shapes")
        batch, frames, queries, dim = anchors.shape
        if joints.ndim != 3 or joints.shape[0] != batch or joints.shape[-1] != dim or dim != self.config.dim:
            raise ValueError("shared joints and final anchors must have matching batch and width")
        if joint_mask.dtype != torch.bool or joint_mask.shape != joints.shape[:2] or not bool(joint_mask.any(1).all()):
            raise ValueError("every asset needs at least one valid joint")
        count = joints.shape[1]
        memory = (anchors + self.point_embedding(anchor_points.to(anchors.dtype))).reshape(
            batch * frames, queries, dim)
        posed = joints[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, count, dim)
        padding = (~joint_mask[:, None].expand(-1, frames, -1)).reshape(batch * frames, count)
        for layer in self.layers:
            if self.config.position_checkpointing and self.training and torch.is_grad_enabled():
                posed = checkpoint(layer, posed, memory, padding, use_reentrant=False)
            else:
                posed = layer(posed, memory, padding)
        posed = self.norm(posed)
        xyz = self.xyz_head(posed)
        if self.coordinate_readout is not None:
            points = anchor_points.reshape(batch * frames, queries, 3)
            if self.config.position_checkpointing and self.training and torch.is_grad_enabled():
                correction = checkpoint(self.coordinate_readout, posed, memory, points, use_reentrant=False)
            else:
                correction = self.coordinate_readout(posed, memory, points)
            xyz = xyz.float() + correction
        xyz = xyz.reshape(batch, frames, count, 3)
        return xyz.masked_fill(~joint_mask[:, None, :, None], 0)


def coordinate_regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, joint_mask: torch.Tensor, frame_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim != 4 or prediction.shape[-1] != 3 or prediction.shape != target.shape:
        raise ValueError("coordinates must have identical (B,T,J,3) shapes")
    if joint_mask.shape != (prediction.shape[0], prediction.shape[2]) or frame_mask.shape != prediction.shape[:2]:
        raise ValueError("coordinate masks do not match the predicted frames and joints")
    if joint_mask.dtype != torch.bool or frame_mask.dtype != torch.bool:
        raise ValueError("coordinate masks must be boolean")
    valid = (frame_mask[:, :, None, None] & joint_mask[:, None, :, None]).expand_as(target)
    counts = valid.sum(dim=(1, 2, 3))
    if not bool((counts > 0).all()):
        raise ValueError("each asset needs valid coordinate supervision")
    if not bool(torch.isfinite(target[valid]).all()):
        raise ValueError("coordinate supervision must be finite")
    error = torch.where(valid, prediction.float() - target.float(), 0).abs()
    return error.sum(dim=(1, 2, 3)) / counts


class VirtualFrameRig(nn.Module):
    def __init__(self, surface_tokenizer: nn.Module, backbone: nn.Module, config: VirtualFrameRigConfig) -> None:
        super().__init__()
        self.config = config
        self.surface_tokenizer = surface_tokenizer
        self.motion_encoder = VirtualFrameMotionEncoder(config)
        parent_config = SharedStructureConfig(
            dim=config.dim, heads=config.heads, max_joints=config.max_joints, dropout=config.dropout)
        self.structure = SharedParentDecoder(backbone, parent_config)
        self.positions = ContinuousJointLocalizer(config)

    def encode(self, surface: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        points = surface["query_points"]
        if points.ndim != 4 or points.shape[2:] != (self.config.query_tokens, 3):
            raise ValueError("surface queries must have shape (B,T,query_tokens,3)")
        tokens = []
        for frame in range(points.shape[1]):
            inputs = tuple(surface[key][:, frame] for key in (
                "dense_points", "dense_normals", "query_points", "query_normals"))
            if self.config.surface_checkpointing and self.training and torch.is_grad_enabled():
                tokens.append(checkpoint(self.surface_tokenizer, *inputs, use_reentrant=False))
            else:
                tokens.append(self.surface_tokenizer(*inputs))
        evidence = None
        if self.config.motion_evidence_fusion != "off":
            evidence = grouped_motion_evidence(surface["dense_points"].detach(), surface["query_indices"])
        return self.motion_encoder(torch.stack(tokens, dim=1), query_points=points, motion_evidence=evidence)

    def forward(
        self, surface: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], *, xyz_weight: float,
    ) -> dict[str, Any]:
        anchors, memory = self.encode(surface)
        return self.forward_encoded(anchors, memory, surface["query_points"], targets, xyz_weight=xyz_weight)

    def forward_encoded(
        self, anchors: torch.Tensor, memory: torch.Tensor, anchor_points: torch.Tensor,
        targets: dict[str, torch.Tensor], *, xyz_weight: float,
    ) -> dict[str, Any]:
        if not 0 < xyz_weight < float("inf"):
            raise ValueError("XYZ loss weight must be finite and positive")
        structure = self.structure.teacher_force(memory, targets["target_parents"], targets["joint_mask"])
        xyz = self.positions(structure["joint_features"], targets["joint_mask"], anchors, anchor_points)
        per_asset_xyz = coordinate_regression_loss(
            xyz, targets["frame_target_joints"], targets["joint_mask"], targets["frame_mask"])
        per_asset = structure["loss_per_asset"] + xyz_weight * per_asset_xyz
        return {
            "loss": per_asset.mean(), "loss_per_asset": per_asset,
            "structure_loss": structure["loss_per_asset"].mean(), "xyz_loss": per_asset_xyz.mean(),
            "structure_logits": structure["logits"], "structure_labels": structure["labels"],
            "joints": xyz, "joint_features": structure["joint_features"],
        }

    @torch.no_grad()
    def generate(self, surface: dict[str, torch.Tensor], *, joint_limit: int | None = None) -> dict[str, Any]:
        anchors, memory = self.encode(surface)
        return self.generate_encoded(anchors, memory, surface["query_points"], joint_limit=joint_limit)

    @torch.no_grad()
    def generate_encoded(
        self, anchors: torch.Tensor, memory: torch.Tensor, anchor_points: torch.Tensor,
        *, joint_limit: int | None = None,
    ) -> dict[str, Any]:
        generate = self.structure.generate_cached if self.config.generation_cache else self.structure.generate
        result = generate(memory, joint_limit=joint_limit)
        if not result["success"]:
            return result
        mask = torch.ones(result["joint_features"].shape[:2], dtype=torch.bool, device=anchors.device)
        parents = torch.tensor([result["parents"]], dtype=torch.long, device=anchors.device)
        sequence = SkeletonSequence(parents, result["joint_features"], mask)
        return {**result, "skeleton_sequence": sequence,
                "joints": self.positions(sequence.joint_features, mask, anchors, anchor_points)}

    @torch.no_grad()
    def localize(self, sequence: SkeletonSequence, surface: dict[str, torch.Tensor]) -> torch.Tensor:
        anchors, _ = self.encode(surface)
        return self.positions(sequence.joint_features, sequence.joint_mask, anchors, surface["query_points"])

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
