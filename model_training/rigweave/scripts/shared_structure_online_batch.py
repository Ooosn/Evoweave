"""Batch independent assets without changing their sampling or supervision."""
from __future__ import annotations

from typing import Any

import torch

from probe_shared_structure import _cpu_targets
from probe_shared_structure_medium import asset_seed, move_batch, prepare_online_batches
from rigweave.dynamic_rig.sampling import materialize_trackable_surface
from rigweave.dynamic_rig.shared_structure_dynamic import SharedSurfaceSequence


def collate_targets(batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("a physical batch must contain an asset")
    frames = batches[0]["frame_target_joints"].shape[1]
    sizes = []
    for batch in batches:
        xyz, mask = batch["frame_target_joints"], batch["joint_mask"]
        if xyz.ndim != 4 or xyz.shape[0] != 1 or xyz.shape[1] != frames or xyz.shape[-1] != 3:
            raise ValueError("collation requires B1 inputs with a common target frame count")
        if mask.shape != (1, xyz.shape[2]) or not bool(mask.all()):
            raise ValueError("input assets must retain all unpadded joints")
        if batch["target_parents"].shape != mask.shape or not bool(batch["frame_mask"].all()):
            raise ValueError("parents and valid frames must match the input asset")
        sizes.append(xyz.shape[2])
    count = max(sizes)
    xyz = batches[0]["frame_target_joints"].new_zeros(len(batches), frames, count, 3)
    parents = batches[0]["target_parents"].new_full((len(batches), count), -2)
    mask = torch.zeros(len(batches), count, dtype=torch.bool, device=xyz.device)
    for index, (batch, joints) in enumerate(zip(batches, sizes)):
        xyz[index, :, :joints] = batch["frame_target_joints"][0]
        parents[index, :joints] = batch["target_parents"][0]
        mask[index, :joints] = True
    result = {"frame_target_joints": xyz, "target_joints": xyz[:, 0],
              "target_parents": parents, "joint_mask": mask,
              "joint_count": torch.tensor(sizes, dtype=torch.long, device=xyz.device)}
    for key in ("path", "cls"):
        result[key] = [value for batch in batches for value in batch[key]]
    for key in ("frame_mask", "selected_frames", "query_center", "query_scale",
                "target_mesh_centers", "target_mesh_scales", "target_source_centers", "target_source_scales"):
        result[key] = torch.cat([batch[key] for batch in batches], dim=0)
    return result


@torch.no_grad()
def capture_surface_batch(conditioner, batches, references, *, include_motion_geometry=True):
    if not batches or len(batches) != len(references):
        raise ValueError("each asset must have its own sampled references")
    frames = batches[0]["frame_vertices"].shape[1]
    if any(batch["frame_vertices"].shape[:2] != (1, frames) for batch in batches):
        raise ValueError("source batches must have the same actual T; frame padding is not allowed")
    if len({(ref.dense_count, ref.query_count) for ref in references}) != 1:
        raise ValueError("batched surface encoders require equal dense/query sample counts")
    needs_dense = include_motion_geometry and conditioner.motion_encoder.motion_evidence_fusion != "off"
    tokens, points, dense = [], [], []
    for frame in range(frames):
        sampled = [materialize_trackable_surface(
            batch["frame_vertices"][:, frame], batch["faces"], refs,
            vertex_normals=batch["vertex_normals"][:, frame], face_normals=batch["face_normals"][:, frame],
        ) for batch, refs in zip(batches, references)]
        dense_points, dense_normals, query_points, query_normals = (
            torch.cat([getattr(value, key) for value in sampled], dim=0)
            for key in ("dense_points", "dense_normals", "query_points", "query_normals"))
        tokens.append(conditioner.surface_tokenizer(dense_points, dense_normals, query_points, query_normals))
        points.append(query_points)
        if needs_dense:
            dense.append(dense_points)
    return SharedSurfaceSequence(torch.stack(tokens, dim=1), torch.stack(points, dim=1),
        torch.cat([refs.query_indices for refs in references], dim=0),
        torch.stack(dense, dim=1) if needs_dense else None)


@torch.no_grad()
def select_online_batch(original, entries, config, banks, frames, *, training, seeds, device="cuda:0"):
    if not entries or len(entries) != len(banks) or len(entries) != len(seeds):
        raise ValueError("asset, bank and seed lists must have equal nonzero length")
    device = torch.device(device)
    source_batches, target_batches, target_records = [], [], []
    source_refs, target_refs, selections = [], [], []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for entry, bank, seed in zip(entries, banks, seeds):
            source_cpu, target_cpu, reference_cpu, selection = prepare_online_batches(
                entry, config, bank, frames, training=training, seed=seed)
            source, target, reference = (move_batch(batch, device)
                                         for batch in (source_cpu, target_cpu, reference_cpu))
            surface_seed = asset_seed(config, entry["asset_id"])
            torch.manual_seed(surface_seed)
            target_refs.append(original.sample_references(reference))
            torch.manual_seed(surface_seed + {"train_a": 0, "train_b": 1, "heldout": 2}[bank])
            source_refs.append(original.sample_references(source))
            source_batches.append(source)
            target_batches.append(target)
            target_records.append(_cpu_targets(target_cpu))
            selections.append(selection)
        target = capture_surface_batch(original.conditioner, target_batches, target_refs,
                                       include_motion_geometry=False)
        source = capture_surface_batch(original.conditioner, source_batches, source_refs)
    batch = move_batch(collate_targets(target_records), device)
    return source, target, batch, selections
