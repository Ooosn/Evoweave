"""Online frame selection and surface inputs with optional manifest-bound root RT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import select_mesh_frame_ids
from .frame_batch_sampler import FrameSampleRequest
from .sampling import materialize_trackable_surface, sample_trackable_surface
from .shared_structure_medium_data import load_medium_batch, partition_raw_frames
from .root_rt import BATCH_KEYS


def read_virtual_manifest(path: str | Path, *, expected_count: int, expected_sha256: str) -> list[dict[str, Any]]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"manifest hash differs from the selected immutable data version: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_count or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("manifest count or unique path count differs from the selected dataset")
    for row in rows:
        if not Path(row["path"]).is_absolute() or not row.get("asset_id"):
            raise ValueError("canonical manifest rows require absolute backing paths and asset IDs")
    return rows


def partition_seed(frame_seed: int, asset_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{frame_seed}:{asset_id}".encode()).digest()[:4], "little")


class VirtualFrameDataset(Dataset):
    def __init__(
        self, entries: list[dict[str, Any]], *, frame_seed: int = 20261013,
        sampling_seed: int = 20260529, frames: int = 8, heldout_frames: int = 8,
        mode: str = "train", motion_fps_ratio: float = .75, minimum_random_frames: int = 1,
        motion_vertex_samples: int = 512,
        global_rotation: np.ndarray | None = None,
    ) -> None:
        if mode not in {"train", "seen_eval", "heldout_eval"}:
            raise ValueError("unknown frame partition mode")
        self.entries = entries
        self.frame_seed = frame_seed
        self.sampling_seed = sampling_seed
        self.frames = frames
        self.heldout_frames = heldout_frames
        self.mode = mode
        self.motion_fps_ratio = motion_fps_ratio
        self.minimum_random_frames = minimum_random_frames
        self.motion_vertex_samples = motion_vertex_samples
        self.global_rotation = global_rotation

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, request: int | FrameSampleRequest) -> dict[str, Any]:
        if isinstance(request, FrameSampleRequest):
            index, frames, seed = request.item_index, request.frames, request.sampling_seed
        else:
            index, frames = int(request), self.frames
            seed = partition_seed(self.sampling_seed, self.entries[index]["asset_id"])
        entry = self.entries[index]
        path = Path(entry["path"])
        selection: dict[str, Any] = {}

        def choose(vertices: np.ndarray) -> np.ndarray:
            train, heldout = partition_raw_frames(
                len(vertices), partition_seed(self.frame_seed, entry["asset_id"]), self.heldout_frames)
            eligible = heldout if self.mode == "heldout_eval" else train
            slots = select_mesh_frame_ids(
                vertices[eligible], frame_count=frames, path=path, index=index,
                random_query=True, seed=self.sampling_seed, sampling_seed=seed,
                motion_fps_ratio=self.motion_fps_ratio, motion_vertex_samples=self.motion_vertex_samples,
                minimum_random_frames=self.minimum_random_frames,
            )
            selected = eligible[slots]
            if self.mode != "heldout_eval" and np.intersect1d(selected, heldout).size:
                raise RuntimeError("heldout geometry entered an optimization/seen view")
            selection.update(raw_frame_count=len(vertices), heldout_raw_ids=heldout.tolist())
            return selected

        batch = load_medium_batch(entry, choose, global_rotation=self.global_rotation)
        vertices = batch["frame_vertices"]
        batch["frame_mesh_bbox_side"] = (vertices.amax(dim=2) - vertices.amin(dim=2)).amax(dim=-1)
        batch.update(asset_id=entry["asset_id"], sampling_seed=int(seed), frame_partition=self.mode, **selection)
        return batch


def collate_virtual_targets(batches: list[dict[str, Any]], device: torch.device | str) -> dict[str, Any]:
    if not batches:
        raise ValueError("a microbatch needs at least one asset")
    frames = batches[0]["frame_target_joints"].shape[1]
    if any(x["frame_target_joints"].shape[:2] != (1, frames) for x in batches):
        raise ValueError("all assets in a microbatch must have the same real frame count")
    sizes = [x["frame_target_joints"].shape[2] for x in batches]
    xyz = torch.zeros(len(batches), frames, max(sizes), 3, device=device)
    parents = torch.full((len(batches), max(sizes)), -2, dtype=torch.long, device=device)
    mask = torch.zeros_like(parents, dtype=torch.bool)
    for i, (batch, size) in enumerate(zip(batches, sizes)):
        if not bool(batch["joint_mask"].all()) or not bool(batch["frame_mask"].all()):
            raise ValueError("online source assets must retain complete targets and unpadded frames")
        xyz[i, :, :size] = batch["frame_target_joints"][0].to(device)
        parents[i, :size] = batch["target_parents"][0].to(device)
        mask[i, :size] = True
    result = {
        "frame_target_joints": xyz, "target_joints": xyz[:, 0], "target_parents": parents,
        "joint_mask": mask, "frame_mask": torch.ones(len(batches), frames, dtype=torch.bool, device=device),
        "joint_count": torch.tensor(sizes, dtype=torch.long, device=device),
        "path": [x["path"][0] for x in batches], "asset_id": [x["asset_id"] for x in batches],
        "heldout_raw_ids": [x["heldout_raw_ids"] for x in batches],
    }
    for key in ("selected_frames", "query_center", "query_scale", "frame_mesh_bbox_side"):
        result[key] = torch.cat([x[key] for x in batches], dim=0).to(device)
    if any(any(key in batch for key in BATCH_KEYS) for batch in batches):
        if not all(all(key in batch for key in BATCH_KEYS) for batch in batches):
            raise ValueError("cannot mix root_rt and legacy/incomplete samples in one batch")
        for key in BATCH_KEYS:
            result[key] = torch.cat([batch[key] for batch in batches], dim=0).to(device)
    return result


@torch.no_grad()
def prepare_virtual_surface(
    batches: list[dict[str, Any]], device: torch.device | str, *,
    surface_samples: int = 65536, vertex_samples: int = 8192, query_tokens: int = 1024,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stage geometry, not learned features; the surface network runs inside DDP."""
    device = torch.device(device)
    targets = collate_virtual_targets(batches, device)
    fields = ("dense_points", "dense_normals", "query_points", "query_normals")
    assets: dict[str, list[torch.Tensor]] = {key: [] for key in (*fields, "query_indices")}
    for batch in batches:
        geometry = {key: batch[key].to(device) for key in (
            "frame_vertices", "faces", "vertex_normals", "face_normals", "vertex_count", "face_count")}
        generator = torch.Generator(device=device).manual_seed(batch["sampling_seed"])
        refs = sample_trackable_surface(
            geometry["frame_vertices"][:, 0], geometry["faces"], num_samples=surface_samples,
            vertex_samples=vertex_samples, query_tokens=query_tokens,
            vertex_counts=geometry["vertex_count"], face_counts=geometry["face_count"], generator=generator,
        )
        per_frame = {key: [] for key in fields}
        for frame in range(geometry["frame_vertices"].shape[1]):
            material = materialize_trackable_surface(
                geometry["frame_vertices"][:, frame], geometry["faces"], refs,
                vertex_normals=geometry["vertex_normals"][:, frame],
                face_normals=geometry["face_normals"][:, frame],
            )
            for key in fields:
                per_frame[key].append(getattr(material, key))
        for key in fields:
            assets[key].append(torch.stack(per_frame[key], dim=1))
        assets["query_indices"].append(refs.query_indices)
        del geometry, material, refs, per_frame
    return {key: torch.cat(value, dim=0) for key, value in assets.items()}, targets
