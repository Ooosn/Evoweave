from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .model import DynamicRigConditioner
from .motion_evidence import grouped_motion_evidence
from .sampling import TrackableSurfaceReferences, materialize_trackable_surface
from .shared_structure import SharedStructureRig, SkeletonSequence


@dataclass(frozen=True)
class SharedSurfaceSequence:
    tokens: torch.Tensor
    query_points: torch.Tensor
    query_indices: torch.Tensor
    dense_points: torch.Tensor | None

    def to(self, device: torch.device | str) -> "SharedSurfaceSequence":
        return SharedSurfaceSequence(
            self.tokens.to(device), self.query_points.to(device), self.query_indices.to(device),
            None if self.dense_points is None else self.dense_points.to(device),
        )

    def select_frames(self, indices: torch.Tensor) -> "SharedSurfaceSequence":
        if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() < 1:
            raise ValueError("frame selection must be a non-empty long vector")
        if bool(((indices < 0) | (indices >= self.tokens.shape[1])).any()):
            raise ValueError("selected frame is outside cached surface sequence")
        return SharedSurfaceSequence(
            self.tokens.index_select(1, indices.to(self.tokens.device)),
            self.query_points.index_select(1, indices.to(self.query_points.device)),
            self.query_indices,
            None if self.dense_points is None else self.dense_points.index_select(1, indices.to(self.dense_points.device)),
        )

    def cpu_record(self) -> dict[str, torch.Tensor | None]:
        return {
            "tokens": self.tokens.detach().cpu(), "query_points": self.query_points.detach().cpu(),
            "query_indices": self.query_indices.detach().cpu(),
            "dense_points": None if self.dense_points is None else self.dense_points.detach().cpu(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SharedSurfaceSequence":
        return cls(record["tokens"], record["query_points"], record["query_indices"], record["dense_points"])


def capture_surface_sequence(
    conditioner: DynamicRigConditioner, batch: dict[str, Any], refs: TrackableSurfaceReferences,
    *, include_motion_geometry: bool = True,
) -> SharedSurfaceSequence:
    """Cache only independent per-frame geometry, never full-clip motion features."""
    frames = batch["frame_vertices"]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frame_vertices must be (B,T,V,3)")
    tokens, points, dense = [], [], []
    needs_evidence = include_motion_geometry and conditioner.motion_encoder.motion_evidence_fusion != "off"
    for frame in range(frames.shape[1]):
        sampled = materialize_trackable_surface(
            frames[:, frame], batch["faces"], refs,
            vertex_normals=None if batch.get("vertex_normals") is None else batch["vertex_normals"][:, frame],
            face_normals=None if batch.get("face_normals") is None else batch["face_normals"][:, frame],
        )
        tokens.append(conditioner.surface_tokenizer(
            sampled.dense_points, sampled.dense_normals, sampled.query_points, sampled.query_normals,
        ))
        points.append(sampled.query_points)
        if needs_evidence:
            dense.append(sampled.dense_points.detach())
    return SharedSurfaceSequence(
        torch.stack(tokens, dim=1), torch.stack(points, dim=1), refs.query_indices,
        torch.stack(dense, dim=1) if needs_evidence else None,
    )


class SharedStructureDynamicRig(nn.Module):
    def __init__(
        self, conditioner: DynamicRigConditioner, skeleton: SharedStructureRig,
        *, register_temporal_mode: str = "global", freeze_features: bool = False,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.skeleton = skeleton
        self.conditioner.motion_encoder.set_register_temporal_mode(register_temporal_mode)
        self.freeze_features = bool(freeze_features)
        if self.freeze_features:
            self.conditioner.requires_grad_(False)

    def train(self, mode: bool = True) -> "SharedStructureDynamicRig":
        super().train(mode)
        if self.freeze_features:
            self.conditioner.eval()
        return self

    def encode_surface(self, sequence: SharedSurfaceSequence) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.conditioner.motion_encoder
        needs_evidence = encoder.motion_evidence_fusion != "off"
        if needs_evidence != (sequence.dense_points is not None):
            raise ValueError("cached dense geometry must match the motion-evidence setting")
        # Recompute evidence and temporal features AFTER selecting the actual input frames.
        evidence = grouped_motion_evidence(sequence.dense_points, sequence.query_indices) if needs_evidence else None
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_features):
            _, encoded = encoder(sequence.tokens, query_points=sequence.query_points,
                                 motion_evidence=evidence, return_all=True)
        registers = encoded[:, :, 1:1 + encoder.register_tokens]
        anchors = encoded[:, :, 1 + encoder.register_tokens:]
        if anchors.shape != sequence.tokens.shape:
            raise RuntimeError("encoder changed the anchor layout")
        return anchors, registers

    def forward_surface(
        self, source: SharedSurfaceSequence, target: SharedSurfaceSequence, batch: dict[str, Any],
        *, xyz_weight: float = 1.0,
    ) -> dict[str, Any]:
        if batch["frame_mask"].shape != target.tokens.shape[:2] or not bool(batch["frame_mask"].all()):
            raise ValueError("target frames and supervision must have the same unpadded frame count")
        anchors, registers = self.encode_surface(source)
        source_mask = torch.ones(source.tokens.shape[:2], dtype=torch.bool, device=source.tokens.device)
        # The target side reads only its own per-frame geometry, never the source's temporal state.
        return self.skeleton(
            anchors, registers, source_mask, target.tokens, target.query_points,
            batch["target_parents"], batch["joint_mask"],
            batch["frame_mask"], batch["frame_target_joints"], xyz_weight=xyz_weight,
        )

    def forward(
        self, source_batch: dict[str, Any], target_batch: dict[str, Any], refs: TrackableSurfaceReferences,
        *, xyz_weight: float = 1.0,
    ) -> dict[str, Any]:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_features):
            source = capture_surface_sequence(self.conditioner, source_batch, refs)
            target = capture_surface_sequence(self.conditioner, target_batch, refs, include_motion_geometry=False)
        return self.forward_surface(source, target, target_batch, xyz_weight=xyz_weight)

    @torch.no_grad()
    def generate_structure_surface(
        self, sequence: SharedSurfaceSequence, *, joint_limit: int | None = None,
    ) -> dict[str, Any]:
        anchors, registers = self.encode_surface(sequence)
        frame_mask = torch.ones(sequence.tokens.shape[:2], dtype=torch.bool, device=sequence.tokens.device)
        return self.skeleton.generate_structure(anchors, registers, frame_mask, joint_limit=joint_limit)

    @torch.no_grad()
    def localize_surface(self, skeleton: SkeletonSequence, target: SharedSurfaceSequence) -> dict[str, torch.Tensor]:
        return self.skeleton.localize(skeleton, target.tokens, target.query_points)

    @torch.no_grad()
    def generate_surface(
        self, source: SharedSurfaceSequence, target: SharedSurfaceSequence, *, joint_limit: int | None = None,
    ) -> dict[str, Any]:
        result = self.generate_structure_surface(source, joint_limit=joint_limit)
        if not result["success"]:
            return result
        return {**result, **self.localize_surface(result["skeleton_sequence"], target)}


def subset_frame_targets(batch: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    """Use precisely the same selected frames for inputs and full-joint labels."""
    if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() < 1:
        raise ValueError("frame selection must be a non-empty long vector")
    count = batch["frame_target_joints"].shape[1]
    if bool(((indices < 0) | (indices >= count)).any()):
        raise ValueError("selected frame is outside target sequence")
    output = dict(batch)
    for key in ("frame_target_joints", "selected_frames", "frame_mask", "target_mesh_centers",
                "target_mesh_scales", "target_source_centers", "target_source_scales"):
        if key in batch:
            output[key] = batch[key].index_select(1, indices.to(batch[key].device))
    output["target_joints"] = output["frame_target_joints"][:, 0]
    return output
