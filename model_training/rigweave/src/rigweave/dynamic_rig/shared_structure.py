from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SharedStructureConfig:
    dim: int = 1024
    heads: int = 8
    shared_tokens: int = 128
    shared_depth: int = 2
    position_depth: int = 2
    max_joints: int = 512
    xyz_bins: int = 256
    xyz_min: float = -1.0
    xyz_max: float = 1.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.heads <= 0 or self.dim % self.heads:
            raise ValueError("dim must be a positive multiple of heads")
        if min(self.shared_tokens, self.shared_depth, self.position_depth, self.max_joints) <= 0:
            raise ValueError("token counts and layer depths must be positive")
        if self.xyz_bins < 2 or not self.xyz_min < self.xyz_max:
            raise ValueError("coordinate range and bin count are invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


def _validate_prefix_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"{name} must be a boolean (B,L) tensor")
    count = mask.sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < count[:, None]
    if bool((count == 0).any()) or not torch.equal(mask, expected):
        raise ValueError(f"{name} must contain a non-empty contiguous valid prefix")
    return count


class ParentSequenceTokens:
    """One token per joint, followed by EOS; positions never enter this stream."""

    def __init__(self, max_joints: int) -> None:
        if max_joints < 1:
            raise ValueError("max_joints must be positive")
        self.max_joints = int(max_joints)
        self.root = self.max_joints
        self.eos = self.max_joints + 1
        self.bos = self.max_joints + 2
        self.pad = self.max_joints + 3
        self.vocab_size = self.max_joints + 4

    def encode(self, parents: torch.Tensor, joint_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        count = _validate_prefix_mask(joint_mask, "joint_mask")
        if parents.dtype != torch.long or parents.shape != joint_mask.shape:
            raise ValueError("parents must be a long tensor matching joint_mask")
        batch, joints = parents.shape
        if joints > self.max_joints:
            raise ValueError("target exceeds max_joints; targets must not be truncated")
        steps = torch.arange(joints, device=parents.device)[None]
        invalid = joint_mask & (steps > 0) & ((parents < 0) | (parents >= steps))
        if bool((parents[:, 0] != -1).any()) or bool(invalid.any()):
            raise ValueError("expected root 0 and parent < child for every other joint")
        if bool((parents[~joint_mask] != -2).any()):
            raise ValueError("padded parents must use -2")

        joint_tokens = parents.clone()
        joint_tokens[:, 0] = self.root
        inputs = parents.new_full((batch, joints + 1), self.pad)
        inputs[:, 0] = self.bos
        inputs[:, 1:] = torch.where(joint_mask, joint_tokens, self.pad)
        labels = parents.new_full((batch, joints + 1), -100)
        labels[:, :joints] = torch.where(joint_mask, joint_tokens, -100)
        labels[torch.arange(batch, device=parents.device), count] = self.eos
        attention = torch.arange(joints + 1, device=parents.device)[None] <= count[:, None]
        return {"input_ids": inputs, "labels": labels, "attention_mask": attention, "joint_count": count}

    def allowed(self, steps: torch.Tensor) -> torch.Tensor:
        if steps.dtype != torch.long or steps.ndim != 1 or bool((steps < 0).any()):
            raise ValueError("steps must be nonnegative long indices")
        mask = torch.zeros((steps.numel(), self.vocab_size), dtype=torch.bool, device=steps.device)
        mask[:, :self.max_joints] = torch.arange(self.max_joints, device=steps.device)[None] < steps[:, None]
        mask[:, self.root] = steps == 0
        mask[:, self.eos] = steps > 0
        return mask

    def decode(self, tokens: list[int]) -> list[int]:
        if not tokens or tokens[0] != self.root:
            raise ValueError("a complete skeleton must start with ROOT")
        if tokens[-1] != self.eos:
            raise ValueError("a complete skeleton must terminate naturally with EOS")
        parents = [-1]
        for child, token in enumerate(tokens[1:-1], start=1):
            if not 0 <= token < child or child >= self.max_joints:
                raise ValueError("invalid parent token")
            parents.append(token)
        return parents


@dataclass(frozen=True)
class SkeletonSequence:
    """Reusable joint identity/features and topology, with no target-frame coordinates."""

    parents: torch.Tensor
    joint_features: torch.Tensor
    joint_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.joint_features.ndim != 3 or self.joint_features.shape[:2] != self.parents.shape:
            raise ValueError("skeleton features must have shape (B,J,D), not a frame dimension")
        if not torch.is_floating_point(self.joint_features) or not bool(torch.isfinite(self.joint_features).all()):
            raise ValueError("skeleton features must be finite floating-point values")
        if self.parents.device != self.joint_features.device or self.joint_mask.device != self.parents.device:
            raise ValueError("skeleton sequence tensors must share a device")
        ParentSequenceTokens(self.parents.shape[1]).encode(self.parents, self.joint_mask)

    def to(self, device: torch.device | str) -> "SkeletonSequence":
        return SkeletonSequence(self.parents.to(device), self.joint_features.to(device), self.joint_mask.to(device))

    def cpu_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "parents": self.parents.detach().cpu().clone(),
            "joint_features": self.joint_features.detach().cpu().clone(),
            "joint_mask": self.joint_mask.detach().cpu().clone(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SkeletonSequence":
        if record.get("schema_version") != 1:
            raise ValueError("unsupported skeleton-sequence record")
        return cls(record["parents"], record["joint_features"], record["joint_mask"])


class StrictXYZQuantizer:
    """Explicit range, preserving UniRig's round-to-grid convention in range."""

    def __init__(self, bins: int, minimum: float, maximum: float) -> None:
        if bins < 2 or not minimum < maximum:
            raise ValueError("invalid coordinate quantization settings")
        self.bins = int(bins)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = (self.maximum - self.minimum) / self.bins

    def encode(self, xyz: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if xyz.shape != (*valid.shape, 3) or valid.dtype != torch.bool:
            raise ValueError("XYZ and valid masks must have matching leading dimensions")
        selected = xyz[valid]
        if not selected.numel() or not bool(torch.isfinite(selected).all()):
            raise ValueError("valid XYZ targets must be non-empty and finite")
        tolerance = 1.0e-6 * max(1.0, abs(self.minimum), abs(self.maximum))
        if bool((selected < self.minimum - tolerance).any()) or bool((selected > self.maximum + tolerance).any()):
            raise ValueError(
                f"XYZ targets outside explicit [{self.minimum},{self.maximum}] range: "
                f"min={float(selected.min()):.8g}, max={float(selected.max()):.8g}; no clipping allowed"
            )
        labels = torch.full(xyz.shape, -100, dtype=torch.long, device=xyz.device)
        # Only the inclusive upper endpoint is assigned to the last grid bin.
        labels[valid] = ((selected.float() - self.minimum) / self.step).round().long().clamp(0, self.bins - 1)
        return labels

    def decode(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dtype != torch.long or bool(((labels < 0) | (labels >= self.bins)).any()):
            raise ValueError("coordinate tokens are outside the vocabulary")
        return labels.float() * self.step + self.minimum


class SharedStructureMemory(nn.Module):
    def __init__(self, config: SharedStructureConfig) -> None:
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(torch.randn(config.shared_tokens, config.dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                config.dim, config.heads, dim_feedforward=4 * config.dim,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.shared_depth)
        ])
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, anchors: torch.Tensor, registers: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if anchors.ndim != 4 or registers.ndim != 4 or anchors.shape[:2] != registers.shape[:2]:
            raise ValueError("anchors and registers must have aligned (B,T,slots,D) shapes")
        if anchors.shape[-1] != self.config.dim or registers.shape[-1] != self.config.dim:
            raise ValueError("feature dimensions do not match shared structure memory")
        if frame_mask.shape != anchors.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be boolean (B,T)")
        if bool((frame_mask.sum(dim=1) == 0).any()):
            raise ValueError("each asset needs an input frame")
        features = torch.cat((registers, anchors), dim=2)
        memory = features.flatten(1, 2)
        padding = (~frame_mask[:, :, None].expand(*features.shape[:3])).flatten(1, 2)
        slots = self.queries[None].expand(anchors.shape[0], -1, -1)
        for layer in self.layers:
            slots = layer(slots, memory, memory_key_padding_mask=padding)
        return self.norm(slots)


class SharedParentDecoder(nn.Module):
    """The backbone accepts continuous embeddings and returns last_hidden_state."""

    def __init__(self, backbone: nn.Module, config: SharedStructureConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.tokens = ParentSequenceTokens(config.max_joints)
        self.embedding = nn.Embedding(self.tokens.vocab_size, config.dim, padding_idx=self.tokens.pad)
        self.head = nn.Linear(config.dim, self.tokens.vocab_size)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.tokens.pad].zero_()

    def forward_prefix(
        self, memory: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if memory.ndim != 3 or memory.shape[0] != input_ids.shape[0] or memory.shape[-1] != self.config.dim:
            raise ValueError("structure memory and prefix batch/dimension differ")
        if input_ids.dtype != torch.long or input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("invalid structure prefix tensors")
        if input_ids.shape[1] > self.config.max_joints + 1:
            raise ValueError("structure prefix exceeds the configured capacity")
        prefix = self.embedding(input_ids)
        full_input = torch.cat((memory, prefix), dim=1)
        full_mask = torch.cat((attention_mask.new_ones(memory.shape[:2]), attention_mask), dim=1)
        result = self.backbone(
            inputs_embeds=full_input, attention_mask=full_mask, use_cache=False, return_dict=True,
        )
        hidden = result.last_hidden_state[:, memory.shape[1]:]
        if hidden.shape != prefix.shape:
            raise RuntimeError("backbone hidden state does not match joint-prefix embeddings")
        logits = self.head(hidden)
        allowed = self.tokens.allowed(torch.arange(input_ids.shape[1], device=input_ids.device))
        logits = logits.masked_fill(~allowed[None], -torch.finfo(logits.dtype).max)
        return logits, hidden

    def teacher_force(
        self, memory: torch.Tensor, parents: torch.Tensor, joint_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokens.encode(parents, joint_mask)
        logits, hidden = self.forward_prefix(memory, encoded["input_ids"], encoded["attention_mask"])
        per_token = F.cross_entropy(logits.float().transpose(1, 2), encoded["labels"], reduction="none")
        valid = encoded["labels"] != -100
        loss_per_asset = (per_token * valid).sum(dim=1) / valid.sum(dim=1)
        # A joint's feature is read AFTER its own parent token, not after a future token.
        joints = hidden[:, 1:parents.shape[1] + 1] * joint_mask[:, :, None]
        return {"logits": logits, "joint_features": joints, "loss_per_asset": loss_per_asset, **encoded}

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, *, joint_limit: int | None = None) -> dict[str, Any]:
        if memory.shape[0] != 1:
            raise ValueError("natural structure generation operates on one asset at a time")
        limit = self.config.max_joints if joint_limit is None else int(joint_limit)
        if not 1 <= limit <= self.config.max_joints:
            raise ValueError("generation joint limit is outside the model capacity")
        ids = torch.tensor([[self.tokens.bos]], dtype=torch.long, device=memory.device)
        generated: list[int] = []
        for _ in range(limit + 1):
            logits, hidden = self.forward_prefix(memory, ids, torch.ones_like(ids, dtype=torch.bool))
            token = int(logits[0, -1].argmax())
            if token == self.tokens.eos:
                generated.append(token)
                parents = self.tokens.decode(generated)
                return {
                    "success": True, "status": "eos", "tokens": generated,
                    "parents": parents, "joint_features": hidden[:, 1:],
                }
            if len(generated) == limit:
                return {"success": False, "status": "max_joints", "tokens": generated,
                        "parents": None, "joint_limit": limit}
            generated.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
        raise AssertionError("unreachable structure-generation termination")

    @torch.no_grad()
    def generate_cached(self, memory: torch.Tensor, *, joint_limit: int | None = None) -> dict[str, Any]:
        """Use the backbone's KV cache, retaining each joint's post-token feature."""
        if self.training or memory.shape[0] != 1:
            raise ValueError("cached structure generation requires eval mode and one asset")
        limit = self.config.max_joints if joint_limit is None else int(joint_limit)
        if not 1 <= limit <= self.config.max_joints:
            raise ValueError("generation joint limit is outside the model capacity")
        bos = torch.tensor([[self.tokens.bos]], dtype=torch.long, device=memory.device)
        attention = torch.ones((1, memory.shape[1] + 1), dtype=torch.bool, device=memory.device)
        result = self.backbone(inputs_embeds=torch.cat((memory, self.embedding(bos)), dim=1),
                               attention_mask=attention, use_cache=True, return_dict=True)
        generated, features = [], []
        for step in range(limit + 1):
            hidden = result.last_hidden_state[:, -1:]
            if step:
                features.append(hidden)
            logits = self.head(hidden)[:, -1]
            allowed = self.tokens.allowed(torch.tensor([step], dtype=torch.long, device=memory.device))
            token = int(logits.masked_fill(~allowed, -torch.finfo(logits.dtype).max)[0].argmax())
            if token == self.tokens.eos:
                generated.append(token)
                return {"success": True, "status": "eos", "tokens": generated,
                        "parents": self.tokens.decode(generated), "joint_features": torch.cat(features, dim=1)}
            if len(generated) == limit:
                return {"success": False, "status": "max_joints", "tokens": generated,
                        "parents": None, "joint_limit": limit}
            generated.append(token)
            past = result.past_key_values
            if past is None:
                raise RuntimeError("backbone did not return the requested generation cache")
            attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
            ids = torch.tensor([[token]], dtype=torch.long, device=memory.device)
            result = self.backbone(inputs_embeds=self.embedding(ids), attention_mask=attention,
                                   past_key_values=past, use_cache=True, return_dict=True)
        raise AssertionError("unreachable cached structure-generation termination")


class FrameJointPositionDecoder(nn.Module):
    def __init__(self, config: SharedStructureConfig) -> None:
        super().__init__()
        self.config = config
        self.point_embedding = nn.Sequential(nn.Linear(3, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                config.dim, config.heads, dim_feedforward=4 * config.dim,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.position_depth)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.xyz_head = nn.Linear(config.dim, 3 * config.xyz_bins)

    def forward(
        self, joints: torch.Tensor, joint_mask: torch.Tensor,
        anchors: torch.Tensor, anchor_points: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.ndim != 4 or anchor_points.shape != (*anchors.shape[:3], 3):
            raise ValueError("anchor features and positions must align in (B,T,Q)")
        batch, frames, queries, dim = anchors.shape
        if joints.ndim != 3 or joints.shape[0] != batch or joints.shape[-1] != dim or dim != self.config.dim:
            raise ValueError("joint features and frame anchors have incompatible dimensions")
        if joint_mask.shape != joints.shape[:2]:
            raise ValueError("joint mask and features differ")
        _validate_prefix_mask(joint_mask, "joint_mask")
        count = joints.shape[1]
        memory = anchors + self.point_embedding(anchor_points.to(dtype=anchors.dtype))
        memory = memory.reshape(batch * frames, queries, dim)
        pose_joints = joints[:, None].expand(batch, frames, count, dim).reshape(batch * frames, count, dim)
        padding = (~joint_mask[:, None].expand(batch, frames, count)).reshape(batch * frames, count)
        for layer in self.layers:
            pose_joints = layer(pose_joints, memory, tgt_key_padding_mask=padding)
        logits = self.xyz_head(self.norm(pose_joints))
        return logits.reshape(batch, frames, count, 3, self.config.xyz_bins)


def frame_coordinate_loss(
    logits: torch.Tensor, targets: torch.Tensor, joint_mask: torch.Tensor, frame_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 5 or targets.shape != logits.shape[:-1] or logits.shape[-2] != 3:
        raise ValueError("expected XYZ logits (B,T,J,3,V) and matching labels")
    if joint_mask.shape != (logits.shape[0], logits.shape[2]) or frame_mask.shape != logits.shape[:2]:
        raise ValueError("coordinate masks do not match logits")
    valid = frame_mask[:, :, None, None] & joint_mask[:, None, :, None]
    valid = valid.expand_as(targets)
    if bool((valid.sum(dim=(1, 2, 3)) == 0).any()):
        raise ValueError("each asset needs valid coordinate supervision")
    if bool((targets[valid] < 0).any()) or bool((targets[~valid] != -100).any()):
        raise ValueError("coordinate labels and padding masks disagree")
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    losses = losses.reshape_as(targets)
    return (losses * valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))


class SharedStructureRig(nn.Module):
    def __init__(self, backbone: nn.Module, config: SharedStructureConfig) -> None:
        super().__init__()
        self.config = config
        self.memory = SharedStructureMemory(config)
        self.structure = SharedParentDecoder(backbone, config)
        self.positions = FrameJointPositionDecoder(config)
        self.quantizer = StrictXYZQuantizer(config.xyz_bins, config.xyz_min, config.xyz_max)

    def forward(
        self, anchors: torch.Tensor, registers: torch.Tensor, source_frame_mask: torch.Tensor,
        target_anchors: torch.Tensor, target_points: torch.Tensor,
        parents: torch.Tensor, joint_mask: torch.Tensor, target_frame_mask: torch.Tensor,
        target_xyz: torch.Tensor, *, xyz_weight: float = 1.0,
    ) -> dict[str, Any]:
        if not xyz_weight > 0.0:
            raise ValueError("xyz_weight must be positive")
        memory = self.memory(anchors, registers, source_frame_mask)
        structure = self.structure.teacher_force(memory, parents, joint_mask)
        sequence = SkeletonSequence(parents, structure["joint_features"], joint_mask)
        xyz_logits = self.localize(sequence, target_anchors, target_points)["xyz_logits"]
        valid = target_frame_mask[:, :, None] & joint_mask[:, None]
        labels = self.quantizer.encode(target_xyz, valid)
        xyz_per_asset = frame_coordinate_loss(xyz_logits, labels, joint_mask, target_frame_mask)
        total = structure["loss_per_asset"] + xyz_weight * xyz_per_asset
        return {
            "loss": total.mean(), "structure_loss": structure["loss_per_asset"].mean(),
            "xyz_loss": xyz_per_asset.mean(), "loss_per_asset": total,
            "structure_logits": structure["logits"], "structure_labels": structure["labels"],
            "xyz_logits": xyz_logits, "xyz_labels": labels,
            "joint_features": structure["joint_features"], "shared_memory": memory,
            "skeleton_sequence": sequence,
        }

    @torch.no_grad()
    def generate_structure(
        self, anchors: torch.Tensor, registers: torch.Tensor,
        frame_mask: torch.Tensor, *, joint_limit: int | None = None,
    ) -> dict[str, Any]:
        memory = self.memory(anchors, registers, frame_mask)
        result = self.structure.generate(memory, joint_limit=joint_limit)
        if not result["success"]:
            return result
        joint_features = result["joint_features"]
        mask = torch.ones(joint_features.shape[:2], dtype=torch.bool, device=anchors.device)
        parents = torch.tensor([result["parents"]], dtype=torch.long, device=anchors.device)
        return {**result, "skeleton_sequence": SkeletonSequence(parents, joint_features, mask)}

    def localize(
        self, sequence: SkeletonSequence, target_anchors: torch.Tensor, target_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Bind an existing sequence to new meshes without rerunning structure generation."""
        logits = self.positions(sequence.joint_features, sequence.joint_mask, target_anchors, target_points)
        xyz_tokens = logits.argmax(dim=-1)
        return {"xyz_logits": logits, "xyz_tokens": xyz_tokens, "joints": self.quantizer.decode(xyz_tokens)}

    @torch.no_grad()
    def generate(
        self, anchors: torch.Tensor, registers: torch.Tensor, source_frame_mask: torch.Tensor,
        target_anchors: torch.Tensor, target_points: torch.Tensor, *, joint_limit: int | None = None,
    ) -> dict[str, Any]:
        result = self.generate_structure(anchors, registers, source_frame_mask, joint_limit=joint_limit)
        if not result["success"]:
            return result
        return {**result, **self.localize(result["skeleton_sequence"], target_anchors, target_points)}

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
