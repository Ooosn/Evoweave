from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .data import (
    _face_normals,
    _pad_branch_targets,
    _pad_faces,
    _pad_frame_faces,
    _pad_frame_vertices,
    _parents_to_list,
    _resolve_manifest_path,
    _select_query_sequence,
    _vertex_normals,
)
from .model import DynamicRigConditioner
from .sampling import sample_trackable_surface


@dataclass(frozen=True)
class PuppeteerTokenBatch:
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    labels: torch.LongTensor
    token_role: torch.LongTensor


class PuppeteerJointTokenizer:
    """Puppeteer joint-token representation for rootless Evoweave skeletons.

    Each joint is serialized as `(x, y, z, parent_index)`.  The raw coordinate
    and parent bins live in `[0, n_discrete_size)`, then receive Puppeteer's
    special-token offset: BOS=0, EOS=1, PAD=2, regular token=raw+3.
    """

    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    offset = 3
    bone_per_token = 4

    def __init__(
        self,
        *,
        n_discrete_size: int = 128,
        continuous_range: tuple[float, float] = (-0.5, 0.5),
        target_coord_scale: float = 0.25,
        strict_range: bool = True,
    ) -> None:
        self.n_discrete_size = int(n_discrete_size)
        if self.n_discrete_size <= 4:
            raise ValueError("n_discrete_size must be > 4")
        self.continuous_range = (float(continuous_range[0]), float(continuous_range[1]))
        if not self.continuous_range[1] > self.continuous_range[0]:
            raise ValueError(f"invalid continuous_range={self.continuous_range}")
        self.target_coord_scale = float(target_coord_scale)
        self.strict_range = bool(strict_range)
        self.vocab_size = self.n_discrete_size + self.offset

    def _discretize_xyz(self, joints: torch.Tensor, *, path: str = "") -> torch.LongTensor:
        xyz = joints.to(dtype=torch.float32) * self.target_coord_scale
        low, high = self.continuous_range
        if self.strict_range:
            too_low = xyz < (low - 1.0e-6)
            too_high = xyz > (high + 1.0e-6)
            if bool((too_low | too_high).any().item()):
                mn = float(xyz.min().detach().cpu())
                mx = float(xyz.max().detach().cpu())
                raise ValueError(
                    f"{path} target xyz after scale={self.target_coord_scale} is outside "
                    f"{self.continuous_range}: min={mn:.6g} max={mx:.6g}"
                )
        xyz = xyz.clamp(min=low, max=np.nextafter(high, low))
        raw = torch.floor((xyz - low) / (high - low) * self.n_discrete_size).to(dtype=torch.long)
        return raw.clamp_(0, self.n_discrete_size - 1)

    def _parent_bins(self, parents: torch.LongTensor, joint_count: int, *, path: str = "") -> torch.LongTensor:
        raw = torch.empty((joint_count,), dtype=torch.long, device=parents.device)
        for child in range(joint_count):
            parent = int(parents[child].item())
            if parent < 0:
                raw[child] = 0
            else:
                if parent >= child:
                    raise ValueError(f"{path} parent order is not autoregressive: child={child} parent={parent}")
                raw[child] = parent + 1
        if int(raw.max().item()) >= self.n_discrete_size:
            raise ValueError(
                f"{path} parent_index token max={int(raw.max().item())} exceeds "
                f"n_discrete_size={self.n_discrete_size}; increase vocab or filter joint_count"
            )
        return raw

    def decode_ids(self, ids: torch.Tensor | np.ndarray | list[int], *, require_eos: bool = False) -> dict[str, Any]:
        """Decode a strict rootless joint-token sequence.

        The accepted sequence form is `[BOS], (x,y,z,parent)*, [EOS]`.
        Returned coordinates are in the unscaled target coordinate space used by
        `target_joints`, i.e. the inverse of `_discretize_xyz`.
        """

        if isinstance(ids, torch.Tensor):
            raw_ids = ids.detach().cpu().to(dtype=torch.long).view(-1).tolist()
        else:
            raw_ids = [int(x) for x in np.asarray(ids, dtype=np.int64).reshape(-1).tolist()]
        if raw_ids and raw_ids[0] == self.bos_token_id:
            raw_ids = raw_ids[1:]
        has_eos = False
        if self.eos_token_id in raw_ids:
            eos_index = raw_ids.index(self.eos_token_id)
            raw_ids = raw_ids[:eos_index]
            has_eos = True
        if require_eos and not has_eos:
            raise ValueError("Puppeteer decode requires EOS but sequence did not contain one")
        if any(token in {self.bos_token_id, self.eos_token_id, self.pad_token_id} for token in raw_ids):
            raise ValueError("Puppeteer generated special token inside joint-token payload")
        if len(raw_ids) == 0 or len(raw_ids) % self.bone_per_token != 0:
            raise ValueError(f"Puppeteer joint-token payload length {len(raw_ids)} is not a positive multiple of 4")

        tokens = torch.as_tensor(raw_ids, dtype=torch.long)
        raw = tokens - self.offset
        if bool(((raw < 0) | (raw >= self.n_discrete_size)).any().item()):
            raise ValueError("Puppeteer joint-token payload contains id outside regular token range")
        joint_data = raw.view(-1, self.bone_per_token)
        coords_raw = joint_data[:, :3].to(dtype=torch.float32)
        low, high = self.continuous_range
        coords = coords_raw / float(self.n_discrete_size)
        coords = coords * float(high - low) + float(low)
        coords = coords / max(float(self.target_coord_scale), 1.0e-12)

        parent_raw = joint_data[:, 3].to(dtype=torch.long)
        parents: list[int] = []
        for child, value in enumerate(parent_raw.tolist()):
            parent_bin = int(value)
            if child == 0:
                if parent_bin != 0:
                    raise ValueError(f"Puppeteer root joint parent bin must be 0, got {parent_bin}")
                parents.append(-1)
                continue
            if parent_bin <= 0 or parent_bin > child:
                raise ValueError(f"Puppeteer invalid parent bin {parent_bin} for child {child}")
            parents.append(parent_bin - 1)
        return {
            "joints": coords.cpu().numpy().astype(np.float32),
            "parents": np.asarray(parents, dtype=np.int64),
            "has_eos": has_eos,
        }

    def encode_one(
        self,
        joints: torch.Tensor,
        parents: torch.LongTensor,
        joint_count: int,
        *,
        path: str = "",
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        if joint_count <= 0:
            raise ValueError(f"{path} joint_count must be positive")
        joints = joints[:joint_count]
        parents = parents[:joint_count]
        if joints.shape != (joint_count, 3):
            raise ValueError(f"{path} joints shape {tuple(joints.shape)} does not match joint_count={joint_count}")
        if parents.shape != (joint_count,):
            raise ValueError(f"{path} parents shape {tuple(parents.shape)} does not match joint_count={joint_count}")
        roots = [i for i, p in enumerate(parents.tolist()) if int(p) < 0]
        if roots != [0]:
            raise ValueError(f"{path} rootless Puppeteer tokens require root at joint 0, got roots={roots}")

        xyz = self._discretize_xyz(joints, path=path)
        parent = self._parent_bins(parents, joint_count, path=path)
        raw = torch.cat([xyz, parent.view(-1, 1)], dim=1).reshape(-1)
        token_ids = raw + self.offset
        roles = torch.arange(self.bone_per_token, device=token_ids.device, dtype=torch.long).repeat(joint_count)
        return token_ids.to(dtype=torch.long), roles

    def make_batch(
        self,
        joints: torch.Tensor,
        parents: torch.LongTensor,
        joint_counts: torch.LongTensor,
        paths: list[str],
    ) -> PuppeteerTokenBatch:
        encoded: list[torch.LongTensor] = []
        roles: list[torch.LongTensor] = []
        for b in range(int(joints.shape[0])):
            ids, role = self.encode_one(
                joints[b],
                parents[b],
                int(joint_counts[b].item()),
                path=paths[b] if b < len(paths) else "",
            )
            encoded.append(ids)
            roles.append(role)

        max_len = max(int(x.numel()) for x in encoded) + 2
        input_ids = joints.new_full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        labels = joints.new_full((len(encoded), max_len), -100, dtype=torch.long)
        attention_mask = joints.new_zeros((len(encoded), max_len), dtype=torch.float32)
        token_role = joints.new_full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        for b, ids in enumerate(encoded):
            n = int(ids.numel())
            input_ids[b, 0] = self.bos_token_id
            input_ids[b, 1 : 1 + n] = ids.to(input_ids.device)
            input_ids[b, 1 + n] = self.eos_token_id

            labels[b, 1 : 1 + n] = ids.to(labels.device)
            labels[b, 1 + n] = self.eos_token_id
            attention_mask[b, : n + 2] = 1.0

            token_role[b, 0] = self.bos_token_id
            token_role[b, 1 : 1 + n] = roles[b].to(token_role.device) + self.offset
            token_role[b, 1 + n] = self.eos_token_id
        return PuppeteerTokenBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_role=token_role,
        )


class PuppeteerDynamicRigDataset(Dataset):
    """Rootless dynamic-rig manifest dataset without UniRig tokenization."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        frame_count: int = 24,
        limit: int = 0,
        random_query: bool = True,
        seed: int = 20260529,
        motion_fps_ratio: float = 0.7,
        motion_vertex_samples: int = 512,
        max_joints: int = 128,
    ) -> None:
        self.paths, self.raw_rows, self.filtered_over_max_joints = _load_puppeteer_manifest(
            Path(manifest),
            limit=limit,
            max_joints=max_joints,
        )
        self.frame_count = int(frame_count)
        self.random_query = bool(random_query)
        self.seed = int(seed)
        self.motion_fps_ratio = float(motion_fps_ratio)
        self.motion_vertex_samples = int(motion_vertex_samples)
        self.max_joints = int(max_joints)
        if not self.paths:
            raise ValueError(f"{manifest} has no rows after Puppeteer max_joints={self.max_joints} filtering")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[int(index)]
        raw = np.load(path, allow_pickle=True)
        required = [
            "canonical_schema_version",
            "frame_vertices_rootspace",
            "faces",
            "rest_joints",
            "raw_root_index",
            "target_joints_rootspace",
            "target_parents",
            "target_has_skin",
            "target_is_synthetic_root",
        ]
        missing = [name for name in required if name not in raw.files]
        if missing:
            raise KeyError(f"{path} is not a strict rootless training NPZ; missing {missing}")
        schema_version = str(np.asarray(raw["canonical_schema_version"]).reshape(-1)[0])
        if schema_version not in {"rootless_dynamic_npz_v1", "rootless_dynamic_npz_v2", "rootless_dynamic_npz_v3"}:
            raise ValueError(f"{path} canonical_schema_version={schema_version!r}, expected rootless_dynamic_npz_v1/v2/v3")

        raw_frame_vertices = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)
        faces = np.asarray(raw["faces"], dtype=np.int64)
        target_parent_array = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
        posed_joints = np.asarray(raw["target_joints_rootspace"], dtype=np.float32)
        if posed_joints.ndim != 3 or posed_joints.shape[0] != raw_frame_vertices.shape[0] or posed_joints.shape[-1] != 3:
            raise ValueError(f"{path} target_joints_rootspace shape {posed_joints.shape} does not match frames")
        if target_parent_array.shape != (posed_joints.shape[1],):
            raise ValueError(f"{path} target_parents shape {target_parent_array.shape} != target joints {posed_joints.shape[1]}")
        if self.max_joints > 0 and posed_joints.shape[1] > self.max_joints:
            raise ValueError(
                f"{path} has {posed_joints.shape[1]} target joints, exceeding Puppeteer max_joints={self.max_joints}"
            )
        roots = [int(i) for i, p in enumerate(target_parent_array.tolist()) if int(p) < 0]
        if roots != [0]:
            raise ValueError(f"{path} rootless Puppeteer route requires exactly one root at joint 0, got roots={roots}")
        target_is_synthetic_root = np.asarray(raw["target_is_synthetic_root"], dtype=bool).reshape(-1)
        if bool(target_is_synthetic_root.any()):
            raise ValueError(f"{path} target still contains synthetic root entries")

        frame_vertices, target_joints, _target_tails, selected = _select_query_sequence(
            raw_frame_vertices,
            posed_joints,
            None,
            frame_count=self.frame_count,
            path=path,
            index=int(index),
            random_query=self.random_query,
            seed=self.seed,
            motion_fps_ratio=self.motion_fps_ratio,
            motion_vertex_samples=self.motion_vertex_samples,
            input_space_policy="mesh_query_bbox",
        )
        query_vertices = frame_vertices[0]
        lo = query_vertices.min(axis=0)
        hi = query_vertices.max(axis=0)
        query_center = ((lo + hi) * 0.5).astype(np.float32)
        query_scale = float(np.max((hi - lo) * 0.5))
        if not np.isfinite(query_scale) or query_scale < 1.0e-8:
            raise ValueError(f"{path} query mesh has degenerate bbox after query-root alignment")
        frames_n = ((frame_vertices - query_center) / query_scale).astype(np.float32)
        joints_n = ((target_joints - query_center) / query_scale).astype(np.float32)
        parents = _parents_to_list(target_parent_array)
        parent_tensor = np.asarray([-1 if p is None else int(p) for p in parents], dtype=np.int64)

        face_normals_seq = np.stack([_face_normals(frame, faces) for frame in frames_n], axis=0)
        vertex_normals_seq = np.stack(
            [_vertex_normals(frame, faces, face_normals_seq[i]) for i, frame in enumerate(frames_n)],
            axis=0,
        )
        return {
            "path": str(path),
            "frame_vertices": torch.from_numpy(frames_n),
            "faces": torch.from_numpy(faces),
            "vertex_normals": torch.from_numpy(vertex_normals_seq),
            "face_normals": torch.from_numpy(face_normals_seq),
            "target_joints": torch.from_numpy(joints_n.astype(np.float32)),
            "target_parents": torch.from_numpy(parent_tensor),
            "selected_frames": torch.from_numpy(selected.astype(np.int64)),
            "query_center": torch.from_numpy(query_center),
            "query_scale": torch.tensor(float(query_scale), dtype=torch.float32),
            "joint_count": int(joints_n.shape[0]),
            "source_joint_count": int(np.asarray(raw["rest_joints"]).shape[0]),
            "active_skin_joint_count": int(np.asarray(raw["target_has_skin"], dtype=bool).sum()),
            "target_start_index": 0,
            "alignment_root_index": int(np.asarray(raw["raw_root_index"], dtype=np.int64)),
            "vertex_count": int(raw_frame_vertices.shape[1]),
            "face_count": int(faces.shape[0]),
        }


def _manifest_joint_count(row: dict[str, Any]) -> int | None:
    candidates = [
        row.get("canonical_metrics", {}).get("target_joint_count"),
        row.get("final_bbox_consistency_screening", {}).get("metrics", {}).get("rootless_joint_count"),
        row.get("final_bbox_consistency_screening", {}).get("metrics", {}).get("rootless_target_joint_count"),
        row.get("metrics", {}).get("target_joint_count"),
        row.get("metrics", {}).get("joint_count"),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def _load_puppeteer_manifest(path: Path, *, limit: int = 0, max_joints: int = 128) -> tuple[list[Path], int, int]:
    paths: list[Path] = []
    raw_rows = 0
    filtered_over_max_joints = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw_rows += 1
        if line.startswith("{"):
            row = json.loads(line)
            value = row.get("path") or row.get("npz_path") or row.get("file")
            if value is None:
                raise ValueError(f"manifest JSON row has no path-like field: {line[:160]}")
            joint_count = _manifest_joint_count(row)
            if max_joints > 0 and joint_count is not None and joint_count > max_joints:
                filtered_over_max_joints += 1
                continue
            paths.append(_resolve_manifest_path(value))
        else:
            paths.append(_resolve_manifest_path(line))
        if limit > 0 and len(paths) >= limit:
            break
    return paths, raw_rows, filtered_over_max_joints


def puppeteer_dynamic_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": [str(item["path"]) for item in batch],
        "frame_vertices": _pad_frame_vertices([item["frame_vertices"] for item in batch]),
        "faces": _pad_faces([item["faces"] for item in batch]),
        "vertex_normals": _pad_frame_vertices([item["vertex_normals"] for item in batch]),
        "face_normals": _pad_frame_faces([item["face_normals"] for item in batch]),
        "target_joints": _pad_branch_targets([item["target_joints"] for item in batch]),
        "target_parents": _pad_1d([item["target_parents"] for item in batch], pad_value=-2),
        "selected_frames": torch.stack([item["selected_frames"] for item in batch], dim=0),
        "query_center": torch.stack([item["query_center"] for item in batch], dim=0),
        "query_scale": torch.stack([item["query_scale"] for item in batch], dim=0),
        "joint_count": torch.tensor([int(item["joint_count"]) for item in batch], dtype=torch.long),
        "source_joint_count": torch.tensor([int(item["source_joint_count"]) for item in batch], dtype=torch.long),
        "active_skin_joint_count": torch.tensor([int(item["active_skin_joint_count"]) for item in batch], dtype=torch.long),
        "target_start_index": torch.tensor([int(item["target_start_index"]) for item in batch], dtype=torch.long),
        "alignment_root_index": torch.tensor([int(item["alignment_root_index"]) for item in batch], dtype=torch.long),
        "vertex_count": torch.tensor([int(item["vertex_count"]) for item in batch], dtype=torch.long),
        "face_count": torch.tensor([int(item["face_count"]) for item in batch], dtype=torch.long),
    }


def _pad_1d(values: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(int(value.shape[0]) for value in values)
    out = values[0].new_full((len(values), max_len), int(pad_value))
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


class ConditionPrefixProjector(nn.Module):
    """Pool Evoweave condition tokens to Puppeteer's fixed condition slots."""

    def __init__(self, input_dim: int, output_dim: int, cond_length: int = 257, heads: int = 8) -> None:
        super().__init__()
        self.cond_length = int(cond_length)
        self.query = nn.Parameter(torch.randn(self.cond_length, int(output_dim)) * 0.02)
        self.input_proj = nn.Linear(int(input_dim), int(output_dim))
        self.query_norm = nn.LayerNorm(int(output_dim))
        self.key_norm = nn.LayerNorm(int(output_dim))
        self.cross_attn = nn.MultiheadAttention(int(output_dim), int(heads), batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), int(output_dim) * 2),
            nn.GELU(),
            nn.Linear(int(output_dim) * 2, int(output_dim)),
        )
        self.out_norm = nn.LayerNorm(int(output_dim))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        cond = self.input_proj(cond)
        q = self.query.view(1, self.cond_length, -1).expand(int(cond.shape[0]), -1, -1)
        attn, _ = self.cross_attn(self.query_norm(q.float()), self.key_norm(cond.float()), self.key_norm(cond.float()), need_weights=False)
        out = q.float() + attn
        out = out + self.ff(out)
        return self.out_norm(out).to(dtype=cond.dtype)


class IdentityConditionProjector(nn.Module):
    """Pass condition tokens through without changing their order or length."""

    def __init__(self, input_dim: int, output_dim: int, cond_length: int) -> None:
        super().__init__()
        if int(input_dim) != int(output_dim):
            raise ValueError(
                "identity condition projection requires matching dimensions: "
                f"input_dim={input_dim} output_dim={output_dim}"
            )
        self.cond_length = int(cond_length)
        self.output_dim = int(output_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 3:
            raise ValueError(f"identity condition projection expects [B,N,C], got {tuple(cond.shape)}")
        if int(cond.shape[1]) != self.cond_length or int(cond.shape[2]) != self.output_dim:
            raise ValueError(
                "identity condition projection contract mismatch: "
                f"expected [B,{self.cond_length},{self.output_dim}], got {tuple(cond.shape)}"
            )
        return cond


class PuppeteerDynamicRigModel(nn.Module):
    """Evoweave dynamic condition prefix plus a joint-token AR decoder.

    Puppeteer weights may initialize the decoder, but this wrapper owns the
    sequence contract: condition tokens, `[BOS]`, `(x, y, z, parent_index)*`,
    then `[EOS]`.
    """

    def __init__(
        self,
        *,
        conditioner: DynamicRigConditioner,
        decoder: nn.Module,
        tokenizer: PuppeteerJointTokenizer,
        num_surface_samples: int,
        vertex_samples: int,
        query_tokens: int,
        cond_length: int = 1024,
        projector_heads: int = 8,
        condition_projection: str = "identity",
        max_joints: int = 128,
        use_joint_slot_embedding: bool = True,
        target_aware_pos_embed: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.decoder = decoder
        self.tokenizer = tokenizer
        hidden_size = int(getattr(self.decoder.config, "word_embed_proj_dim", self.decoder.config.hidden_size))
        surface_dim = int(getattr(self.conditioner.motion_encoder, "dim"))
        if condition_projection == "cross_attention":
            self.prefix_projector = ConditionPrefixProjector(
                surface_dim,
                hidden_size,
                cond_length=cond_length,
                heads=projector_heads,
            )
        elif condition_projection == "identity":
            if int(cond_length) != int(query_tokens):
                raise ValueError(
                    "identity condition projection requires cond_length == query_tokens: "
                    f"cond_length={cond_length} query_tokens={query_tokens}"
                )
            self.prefix_projector = IdentityConditionProjector(surface_dim, hidden_size, cond_length)
        else:
            raise ValueError(f"unsupported condition_projection={condition_projection!r}")
        self.condition_projection = str(condition_projection)
        self.max_joints = int(max_joints)
        if self.max_joints <= 0:
            raise ValueError(f"max_joints must be positive, got {self.max_joints}")
        if use_joint_slot_embedding:
            slot_count = self.max_joints + 1
            target_aware = torch.empty((1, slot_count, hidden_size), dtype=torch.float32)
            nn.init.trunc_normal_(target_aware, mean=0.0, std=0.02)
            if target_aware_pos_embed is not None:
                source = target_aware_pos_embed.detach().to(dtype=torch.float32)
                if source.ndim != 3 or int(source.shape[0]) != 1 or int(source.shape[2]) != hidden_size:
                    raise ValueError(
                        "Puppeteer target_aware_pos_embed must have shape "
                        f"(1, N, {hidden_size}), got {tuple(source.shape)}"
                    )
                rows = min(int(source.shape[1]), slot_count)
                target_aware[:, :rows].copy_(source[:, :rows])
            self.target_aware_pos_embed = nn.Parameter(target_aware)
        elif target_aware_pos_embed is not None:
            raise ValueError("target_aware_pos_embed requires use_joint_slot_embedding=True")
        else:
            self.target_aware_pos_embed = None
        if self.target_aware_pos_embed is not None and int(self.target_aware_pos_embed.shape[1]) < self.max_joints + 1:
            raise ValueError(
                "joint slot embedding must contain condition slot plus every joint slot: "
                f"shape={tuple(self.target_aware_pos_embed.shape)} max_joints={self.max_joints}"
            )
        self.num_surface_samples = int(num_surface_samples)
        self.vertex_samples = int(vertex_samples)
        self.query_tokens = int(query_tokens)
        self.cond_length = int(cond_length)

    def sample_references(self, batch: dict[str, Any], *, generator: torch.Generator | None = None) -> Any:
        return sample_trackable_surface(
            batch["frame_vertices"][:, 0],
            batch["faces"],
            num_samples=self.num_surface_samples,
            vertex_samples=self.vertex_samples,
            query_tokens=self.query_tokens,
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
            generator=generator,
        )

    def build_condition(self, batch: dict[str, Any], *, generator: torch.Generator | None = None) -> torch.Tensor:
        refs = self.sample_references(batch, generator=generator)
        cond = self.conditioner(
            batch["frame_vertices"],
            batch["faces"],
            refs,
            vertex_normals=batch.get("vertex_normals"),
            face_normals=batch.get("face_normals"),
        )
        return self.prefix_projector(cond)

    def _make_token_batch(self, batch: dict[str, Any], device: torch.device) -> PuppeteerTokenBatch:
        return self.tokenizer.make_batch(
            batch["target_joints"].to(device),
            batch["target_parents"].to(device),
            batch["joint_count"].to(device),
            batch["path"],
        )

    def _condition_embeds_from_raw(self, cond: torch.Tensor) -> torch.Tensor:
        cond_type = torch.zeros((cond.shape[0], cond.shape[1]), device=cond.device, dtype=torch.long)
        cond = cond + self._target_aware_condition_bias(cond)
        return cond + self.decoder.model.decoder.cond_embed(cond_type).to(dtype=cond.dtype)

    def _token_embeds(self, token_batch: PuppeteerTokenBatch) -> torch.Tensor:
        dec = self.decoder.model.decoder
        input_ids = token_batch.input_ids.to(dec.embed_tokens.weight.device)
        token_role = token_batch.token_role.to(input_ids.device)
        embeds = dec.embed_tokens(input_ids)
        embeds = embeds + dec.token_embed_positions(bone_ids=token_role)
        type_ids = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
        embeds = embeds + dec.cond_embed(type_ids)
        embeds = embeds + self._target_aware_token_bias(input_ids, embeds.dtype)
        return embeds

    def _target_aware_condition_bias(self, cond: torch.Tensor) -> torch.Tensor:
        if self.target_aware_pos_embed is None:
            return torch.zeros_like(cond)
        target_aware = self.target_aware_pos_embed.to(device=cond.device, dtype=cond.dtype)
        return target_aware[:, 0:1, :].expand(int(cond.shape[0]), int(cond.shape[1]), -1)

    def _target_aware_token_bias(self, input_ids: torch.LongTensor, dtype: torch.dtype) -> torch.Tensor:
        dec_device = self.decoder.model.decoder.embed_tokens.weight.device
        if self.target_aware_pos_embed is None:
            hidden = int(getattr(self.decoder.config, "word_embed_proj_dim", self.decoder.config.hidden_size))
            return torch.zeros((*input_ids.shape, hidden), device=dec_device, dtype=dtype)

        input_ids = input_ids.to(dec_device)
        target_aware = self.target_aware_pos_embed.to(device=dec_device, dtype=dtype)
        hidden = int(target_aware.shape[-1])
        out = torch.zeros((*input_ids.shape, hidden), device=dec_device, dtype=dtype)
        max_target_joints = int(target_aware.shape[1])
        for b in range(int(input_ids.shape[0])):
            payload_index = 0
            for pos in range(int(input_ids.shape[1])):
                token = int(input_ids[b, pos].item())
                if token in {self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}:
                    continue
                joint_index = payload_index // self.tokenizer.bone_per_token
                payload_index += 1
                slot_index = joint_index + 1
                if slot_index >= max_target_joints:
                    raise ValueError(
                        f"Puppeteer target-aware embedding has {max_target_joints} joint slots, "
                        f"but token payload needs joint slot {slot_index}"
                    )
                out[b, pos] = target_aware[0, slot_index]
        return out

    def _prefix_token_roles(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        roles = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        for b in range(int(input_ids.shape[0])):
            payload_index = 0
            for pos in range(int(input_ids.shape[1])):
                token = int(input_ids[b, pos].item())
                if token == self.tokenizer.bos_token_id:
                    roles[b, pos] = self.tokenizer.bos_token_id
                elif token == self.tokenizer.eos_token_id:
                    roles[b, pos] = self.tokenizer.eos_token_id
                elif token == self.tokenizer.pad_token_id:
                    roles[b, pos] = self.tokenizer.pad_token_id
                else:
                    roles[b, pos] = self.tokenizer.offset + (payload_index % self.tokenizer.bone_per_token)
                    payload_index += 1
        return roles

    def _prefix_embeds(self, input_ids: torch.LongTensor) -> torch.Tensor:
        attention = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.float32)
        labels = torch.full(input_ids.shape, -100, device=input_ids.device, dtype=torch.long)
        token_batch = PuppeteerTokenBatch(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
            token_role=self._prefix_token_roles(input_ids),
        )
        return self._token_embeds(token_batch)

    def _condition_embeds(self, batch: dict[str, Any]) -> torch.Tensor:
        cond = self.build_condition(batch)
        return self._condition_embeds_from_raw(cond)

    def _next_token_logits(self, cond: torch.Tensor, prefix_ids: torch.LongTensor) -> torch.Tensor:
        token_embeds = self._prefix_embeds(prefix_ids).to(dtype=cond.dtype)
        inputs_embeds = torch.cat([cond, token_embeds], dim=1)
        attention = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.long)
        out = self.decoder(
            input_ids=prefix_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            use_cache=False,
        )
        return out.logits[:, -1, :]

    def teacher_forced_logits(
        self,
        batch: dict[str, Any],
        *,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PuppeteerTokenBatch, torch.Tensor]:
        if cond is None:
            cond = self._condition_embeds(batch)
        token_batch = self._make_token_batch(batch, cond.device)
        token_embeds = self._token_embeds(token_batch).to(dtype=cond.dtype)
        inputs_embeds = torch.cat([cond, token_embeds], dim=1)
        full_attention = torch.cat(
            [
                torch.ones((cond.shape[0], cond.shape[1]), device=cond.device, dtype=token_batch.attention_mask.dtype),
                token_batch.attention_mask.to(cond.device),
            ],
            dim=1,
        )
        full_labels = torch.cat(
            [
                torch.full((cond.shape[0], cond.shape[1]), -100, device=cond.device, dtype=torch.long),
                token_batch.labels.to(cond.device),
            ],
            dim=1,
        )
        out = self.decoder(
            input_ids=token_batch.input_ids.to(cond.device),
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=full_labels,
            use_cache=False,
        )
        logits = out.logits[:, cond.shape[1] - 1 : cond.shape[1] - 1 + token_batch.labels.shape[1]]
        return logits, token_batch, out.loss

    @torch.no_grad()
    def teacher_forcing_generation_alignment(
        self,
        batch: dict[str, Any],
        *,
        max_positions: int = 32,
    ) -> dict[str, float | int]:
        if int(batch["frame_vertices"].shape[0]) != 1:
            raise ValueError("alignment sanity expects batch_size=1")
        was_training = self.training
        self.eval()
        try:
            cond = self._condition_embeds(batch)
            forced_logits, token_batch, _loss = self.teacher_forced_logits(batch, cond=cond)
            labels = token_batch.labels.to(cond.device)
            valid_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
            checked = 0
            max_abs = 0.0
            mean_abs_total = 0.0
            first_pos = -1
            first_pos_max_abs = 0.0
            for pos in valid_positions[: max(0, int(max_positions))]:
                if pos <= 0:
                    continue
                prefix = token_batch.input_ids[:, :pos].to(cond.device)
                step_logits = self._next_token_logits(cond, prefix)
                diff = (forced_logits[:, pos] - step_logits).float().abs()
                item_max = float(diff.max().detach().cpu())
                item_mean = float(diff.mean().detach().cpu())
                if checked == 0:
                    first_pos = int(pos)
                    first_pos_max_abs = item_max
                max_abs = max(max_abs, item_max)
                mean_abs_total += item_mean
                checked += 1
            return {
                "checked_positions": int(checked),
                "max_abs_logit_diff": float(max_abs),
                "mean_abs_logit_diff": float(mean_abs_total / max(checked, 1)),
                "first_checked_position": int(first_pos),
                "first_position_max_abs_logit_diff": float(first_pos_max_abs),
            }
        finally:
            self.train(was_training)

    def _apply_generation_mask(self, logits: torch.Tensor, generated_count: int, *, max_joints: int) -> torch.Tensor:
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        role = int(generated_count % self.tokenizer.bone_per_token)
        regular_start = self.tokenizer.offset
        regular_end = self.tokenizer.offset + self.tokenizer.n_discrete_size
        if generated_count >= int(max_joints) * self.tokenizer.bone_per_token:
            masked[:, self.tokenizer.eos_token_id] = logits[:, self.tokenizer.eos_token_id]
            return masked
        if role == 0:
            masked[:, regular_start:regular_end] = logits[:, regular_start:regular_end]
            if generated_count >= self.tokenizer.bone_per_token:
                masked[:, self.tokenizer.eos_token_id] = logits[:, self.tokenizer.eos_token_id]
            return masked
        if role in {1, 2}:
            masked[:, regular_start:regular_end] = logits[:, regular_start:regular_end]
            return masked

        child = int(generated_count // self.tokenizer.bone_per_token)
        if child == 0:
            masked[:, regular_start] = logits[:, regular_start]
        else:
            hi = min(child, self.tokenizer.n_discrete_size - 1)
            masked[:, regular_start + 1 : regular_start + hi + 1] = logits[:, regular_start + 1 : regular_start + hi + 1]
        return masked

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> int:
        if not do_sample:
            return int(logits.argmax(dim=-1).item())
        scores = logits.float()
        if temperature is not None:
            scores = scores / max(float(temperature), 1.0e-6)
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), int(scores.shape[-1]))
            threshold = torch.topk(scores, k=k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, torch.finfo(scores.dtype).min)
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            probs = torch.softmax(sorted_scores, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            remove = cumulative > float(top_p)
            remove[:, 0] = False
            sorted_scores = sorted_scores.masked_fill(remove, torch.finfo(sorted_scores.dtype).min)
            filtered = torch.full_like(scores, torch.finfo(scores.dtype).min)
            scores = filtered.scatter(1, sorted_idx, sorted_scores)
        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def generate_skeleton(
        self,
        batch: dict[str, Any],
        *,
        max_new_tokens: int,
        max_joints: int | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(batch["frame_vertices"].shape[0]) != 1:
            raise ValueError("Puppeteer generation currently expects batch_size=1")
        kwargs = dict(generation_kwargs or {})
        for unsupported in ("num_beams", "num_return_sequences", "repetition_penalty", "no_repeat_ngram_size"):
            value = kwargs.get(unsupported)
            if value is not None and float(value) != 1.0 and int(value) != 0:
                raise ValueError(f"Puppeteer manual generation does not support {unsupported}={value}")
        do_sample = bool(kwargs.get("do_sample", False))
        max_joints = int(max_joints or self.max_joints)
        max_joints = min(max_joints, self.tokenizer.n_discrete_size)
        cond = self._condition_embeds(batch)
        prefix = torch.tensor([[self.tokenizer.bos_token_id]], device=cond.device, dtype=torch.long)
        generated: list[int] = []
        has_eos = False
        for _ in range(int(max_new_tokens)):
            logits = self._next_token_logits(cond, prefix)
            logits = self._apply_generation_mask(logits, len(generated), max_joints=max_joints)
            token = self._sample_token(
                logits,
                do_sample=do_sample,
                temperature=kwargs.get("temperature"),
                top_k=kwargs.get("top_k"),
                top_p=kwargs.get("top_p"),
            )
            generated.append(token)
            prefix = torch.cat([prefix, torch.tensor([[token]], device=prefix.device, dtype=torch.long)], dim=1)
            if token == self.tokenizer.eos_token_id:
                has_eos = True
                break
        ids = [self.tokenizer.bos_token_id, *generated]
        decoded = self.tokenizer.decode_ids(ids, require_eos=False)
        decoded["generated_ids"] = np.asarray(ids, dtype=np.int64)
        decoded["steps"] = int(len(generated))
        decoded["has_eos"] = bool(has_eos)
        return decoded

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        logits, token_batch, loss = self.teacher_forced_logits(batch)
        labels = token_batch.labels.to(logits.device)
        pred = logits.argmax(dim=-1)
        valid = labels != -100
        token_acc = (pred[valid] == labels[valid]).float().mean() if valid.any() else logits.new_zeros(())
        eos_mask = labels == self.tokenizer.eos_token_id
        eos_acc = (pred[eos_mask] == labels[eos_mask]).float().mean() if eos_mask.any() else logits.new_zeros(())
        role = token_batch.token_role.to(logits.device)
        coord_mask = valid & (role >= self.tokenizer.offset) & ((role - self.tokenizer.offset) < 3)
        parent_mask = valid & (role == self.tokenizer.offset + 3)
        coord_acc = (pred[coord_mask] == labels[coord_mask]).float().mean() if coord_mask.any() else logits.new_zeros(())
        parent_acc = (pred[parent_mask] == labels[parent_mask]).float().mean() if parent_mask.any() else logits.new_zeros(())
        return {
            "loss": loss,
            "token_acc": token_acc,
            "coord_acc": coord_acc,
            "parent_acc": parent_acc,
            "eos_acc": eos_acc,
        }


def import_puppeteer_decoder(puppeteer_root: str | Path) -> tuple[type[Any], type[nn.Module]]:
    root = Path(puppeteer_root).expanduser().resolve()
    skeleton_root = root / "skeleton"
    if not (skeleton_root / "skeleton_models" / "skeleton_opt.py").exists():
        raise FileNotFoundError(f"cannot find Puppeteer skeleton_models under {skeleton_root}")
    if str(skeleton_root) not in sys.path:
        sys.path.insert(0, str(skeleton_root))
    from skeleton_models.skeleton_opt import SkeletonOPT, SkeletonOPTConfig

    return SkeletonOPTConfig, SkeletonOPT


def load_puppeteer_decoder_state(decoder: nn.Module, checkpoint: str | Path, *, allow_resize_positions: bool = True) -> dict[str, Any]:
    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"unsupported Puppeteer checkpoint object type: {type(obj)!r}")
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("transformer."):
            key = key[len("transformer.") :]
        if key.startswith("decoder."):
            key = "model." + key
        if key.startswith("model.") or key.startswith("lm_head."):
            mapped[key] = value

    current = decoder.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped_shape: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    resized_shape: dict[str, dict[str, Any]] = {}
    for key, value in mapped.items():
        if key not in current:
            continue
        if tuple(value.shape) == tuple(current[key].shape):
            loadable[key] = value
        elif (
            allow_resize_positions
            and key == "model.decoder.embed_positions.weight"
            and value.ndim == 2
            and current[key].ndim == 2
            and int(value.shape[1]) == int(current[key].shape[1])
        ):
            resized = current[key].detach().clone()
            rows = min(int(value.shape[0]), int(resized.shape[0]))
            resized[:rows].copy_(value[:rows])
            loadable[key] = resized
            resized_shape[key] = {
                "source_shape": list(value.shape),
                "target_shape": list(current[key].shape),
                "copied_rows": rows,
                "new_rows": max(0, int(current[key].shape[0]) - rows),
            }
        else:
            skipped_shape[key] = (tuple(value.shape), tuple(current[key].shape))
    if skipped_shape and not allow_resize_positions:
        raise RuntimeError(f"Puppeteer checkpoint shape mismatch: {skipped_shape}")
    missing, unexpected = decoder.load_state_dict(loadable, strict=False)
    loaded_decoder_layers = sum(1 for key in loadable if key.startswith("model.decoder.layers."))
    if loaded_decoder_layers <= 0:
        raise RuntimeError(f"no decoder layer weights were loaded from {ckpt_path}")
    return {
        "checkpoint": str(ckpt_path),
        "source_keys": len(state),
        "mapped_keys": len(mapped),
        "loaded_keys": len(loadable),
        "loaded_decoder_layer_tensors": loaded_decoder_layers,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped_shape": {k: [list(a), list(b)] for k, (a, b) in skipped_shape.items()},
        "resized_shape": resized_shape,
    }


def load_puppeteer_target_aware_pos_embed(checkpoint: str | Path) -> torch.Tensor | None:
    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"unsupported Puppeteer checkpoint object type: {type(obj)!r}")
    for key in ("target_aware_pos_embed", "module.target_aware_pos_embed"):
        value = state.get(key)
        if value is not None:
            return value.detach().to(dtype=torch.float32).clone()
    return None
