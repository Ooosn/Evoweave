#!/usr/bin/env python3
"""Build rootless dynamic-rig NPZ files from Pass1 precheck NPZ files.

The source Pass1 NPZ files remain the LBS source of truth.  This derived cache
keeps raw fields unchanged, preserves per-frame raw-root RT as auxiliary data,
and writes a rootless training target that does not add ``__object_root__``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rigweave.data.pose_utils import apply_bone_transforms
from rigweave.dynamic_rig.skeleton_contract import (
    active_descendant_mask,
    active_skin_mask,
    children_from_parents,
    looks_like_dummy_root,
    looks_like_tail_or_end,
    parent_array,
)


SCHEMA_VERSION = "rootless_dynamic_npz_v3"

TAIL_ENDPOINT_FIELDS = {
    "rest_tails",
    "rest_tails_raw",
    "bone_tails",
    "bone_tails_raw",
    "target_rest_tails",
    "posed_tails_rootspace",
    "target_tails_rootspace",
}

CANONICAL_METRIC_KEYS = (
    "raw_joint_count",
    "raw_vertex_count",
    "filtered_vertex_count",
    "raw_face_count",
    "filtered_face_count",
    "mesh_component_count",
    "kept_mesh_component_count",
    "dropped_mesh_component_count",
    "dropped_vertex_count",
    "dropped_face_count",
    "target_joint_count",
    "active_raw_joint_count",
    "target_connector_count",
    "target_tail_or_end_count",
    "dropped_raw_count",
    "raw_root_index",
    "raw_root_count",
    "raw_root_name",
    "raw_root_has_skin",
    "raw_root_skin_weight_max",
    "raw_root_skin_weight_sum",
    "raw_root_child_count",
    "raw_root_looks_like_dummy",
    "recorded_root_count",
    "target_root_count",
    "target_entry_count",
    "raw_root_disposition",
    "rootspace_recon_linf",
    "skin_weight_lost_linf",
)

SUMMARY_COUNT_KEYS = (
    "raw_vertex_count",
    "filtered_vertex_count",
    "raw_face_count",
    "filtered_face_count",
    "mesh_component_count",
    "kept_mesh_component_count",
    "dropped_mesh_component_count",
    "dropped_vertex_count",
    "dropped_face_count",
    "raw_joint_count",
    "target_joint_count",
    "active_raw_joint_count",
    "target_connector_count",
    "target_tail_or_end_count",
    "dropped_raw_count",
)


@dataclass(frozen=True)
class RootlessContract:
    raw_root: int
    raw_root_indices: list[int]
    raw_order: list[int]
    parents: np.ndarray
    names: list[str]
    kept_raw_indices: list[int]
    dropped_raw_indices: list[int]
    active_raw_indices: list[int]
    entry_raw_indices: list[int]
    recorded_root_indices: list[int]
    raw_root_disposition: str


@dataclass(frozen=True)
class ControlledMeshSubset:
    rest_vertices: np.ndarray
    frame_vertices: np.ndarray
    faces: np.ndarray
    skin_weights: np.ndarray
    kept_vertex_indices: np.ndarray
    kept_component_indices: np.ndarray
    dropped_component_indices: np.ndarray
    vertex_component_ids: np.ndarray
    component_controlled_ratios: np.ndarray
    raw_vertex_count: int
    raw_face_count: int
    component_count: int
    kept_component_count: int
    dropped_component_count: int
    dropped_vertex_count: int
    dropped_face_count: int


def _as_rotation(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"root rotation block must have shape (3,3), got {mat.shape}")
    u, _, vt = np.linalg.svd(mat.astype(np.float64), full_matrices=False)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot.astype(np.float32)


def _world_to_frame_roots(points: np.ndarray, root_positions: np.ndarray, root_rotations: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    roots = np.asarray(root_positions, dtype=np.float32)
    rots = np.asarray(root_rotations, dtype=np.float32)
    if pts.ndim < 2 or pts.shape[0] != roots.shape[0] or pts.shape[-1] != 3:
        raise ValueError(f"points shape {pts.shape} does not match root positions {roots.shape}")
    centered = pts - roots.reshape((roots.shape[0],) + (1,) * (pts.ndim - 2) + (3,))
    return np.einsum("t...c,tcd->t...d", centered, rots).astype(np.float32)


def _world_from_rootspace(root_points: np.ndarray, root_positions: np.ndarray, root_rotations: np.ndarray) -> np.ndarray:
    pts = np.asarray(root_points, dtype=np.float32)
    roots = np.asarray(root_positions, dtype=np.float32)
    rots = np.asarray(root_rotations, dtype=np.float32)
    world = np.einsum("t...c,tdc->t...d", pts, rots, optimize=True)
    world = world + roots.reshape((roots.shape[0],) + (1,) * (pts.ndim - 2) + (3,))
    return world.astype(np.float32)


def _transform_mats(root_positions: np.ndarray, root_rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(root_positions, dtype=np.float32)
    r = np.asarray(root_rotations, dtype=np.float32)
    count = int(t.shape[0])
    world_from_root = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    root_from_world = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    world_from_root[:, :3, :3] = r
    world_from_root[:, :3, 3] = t
    root_from_world[:, :3, :3] = np.swapaxes(r, 1, 2)
    root_from_world[:, :3, 3] = -np.einsum("tij,tj->ti", np.swapaxes(r, 1, 2), t)
    return root_from_world.astype(np.float32), world_from_root.astype(np.float32)


def _identity_root_transform(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    roots = np.zeros((int(frame_count), 3), dtype=np.float32)
    rotations = np.tile(np.eye(3, dtype=np.float32), (int(frame_count), 1, 1))
    return roots, rotations


def _recorded_root_transforms(
    recorded_root_indices: Sequence[int],
    posed_joints: np.ndarray,
    bone_transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [int(i) for i in recorded_root_indices]
    frame_count = int(posed_joints.shape[0])
    if not ids:
        return (
            np.zeros((frame_count, 0, 3), dtype=np.float32),
            np.zeros((frame_count, 0, 3, 3), dtype=np.float32),
            np.zeros((frame_count, 0, 3, 3), dtype=np.float32),
        )
    positions = posed_joints[:, ids].astype(np.float32)
    rotations = np.empty((frame_count, len(ids), 3, 3), dtype=np.float32)
    for out_idx, raw_idx in enumerate(ids):
        rotations[:, out_idx] = np.stack(
            [_as_rotation(mat) for mat in bone_transforms[:, raw_idx, :3, :3]],
            axis=0,
        )
    return positions, rotations, np.swapaxes(rotations, 2, 3).astype(np.float32)


def _load_joint_names(src: np.lib.npyio.NpzFile, count: int) -> list[str]:
    if "joint_names" not in src.files:
        return [str(i) for i in range(count)]
    names = np.asarray(src["joint_names"], dtype=object).reshape(-1).tolist()
    if len(names) != count:
        raise ValueError(f"joint_names length {len(names)} != joint count {count}")
    return [str(x) for x in names]


def _copy_npz_fields(src: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {name: src[name] for name in src.files if name not in TAIL_ENDPOINT_FIELDS}


def _json_dumps(data: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(data, ensure_ascii=False, sort_keys=True), dtype=object)


def _first_rootless_entry(
    start: int,
    *,
    children: list[list[int]],
    active: np.ndarray,
    has_active: np.ndarray,
    names: Sequence[str],
) -> int:
    node = int(start)
    while True:
        if bool(active[node]):
            return node
        active_children = [int(child) for child in children[node] if bool(has_active[int(child)])]
        if len(active_children) != 1:
            return node
        node = int(active_children[0])


def _mesh_components(vertex_count: int, faces: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    parent = np.arange(int(vertex_count), dtype=np.int64)
    rank = np.zeros((int(vertex_count),), dtype=np.int8)

    def find(x: int) -> int:
        node = int(x)
        while int(parent[node]) != node:
            parent[node] = parent[int(parent[node])]
            node = int(parent[node])
        return node

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if int(rank[ra]) < int(rank[rb]):
            parent[ra] = rb
        elif int(rank[ra]) > int(rank[rb]):
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    if faces.size:
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape [F,3], got {faces.shape}")
        if int(faces.min()) < 0 or int(faces.max()) >= int(vertex_count):
            raise ValueError("faces contain vertex indices outside the vertex array")
        for tri in np.asarray(faces, dtype=np.int64):
            a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
            union(a, b)
            union(a, c)

    root_to_component: dict[int, int] = {}
    component_ids = np.full((int(vertex_count),), -1, dtype=np.int64)
    groups: list[list[int]] = []
    for vertex in range(int(vertex_count)):
        root = find(vertex)
        comp = root_to_component.get(root)
        if comp is None:
            comp = len(groups)
            root_to_component[root] = comp
            groups.append([])
        component_ids[vertex] = int(comp)
        groups[comp].append(vertex)

    components = [np.asarray(group, dtype=np.int64) for group in groups]
    return components, component_ids


def _controlled_mesh_subset(
    rest_vertices: np.ndarray,
    frame_vertices: np.ndarray,
    faces: np.ndarray,
    skin_weights: np.ndarray,
    target_raw_indices: np.ndarray,
    *,
    active_skin_threshold: float,
    min_component_controlled_ratio: float,
) -> ControlledMeshSubset:
    rest = np.asarray(rest_vertices, dtype=np.float32)
    frames = np.asarray(frame_vertices, dtype=np.float32)
    face_arr = np.asarray(faces, dtype=np.int64)
    skin = np.asarray(skin_weights, dtype=np.float32)
    order = np.asarray(target_raw_indices, dtype=np.int64).reshape(-1)

    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError(f"rest_vertices must have shape [V,3], got {rest.shape}")
    if frames.ndim != 3 or frames.shape[1:] != rest.shape:
        raise ValueError(f"frame_vertices shape {frames.shape} does not match rest_vertices {rest.shape}")
    if skin.ndim != 2 or skin.shape[0] != rest.shape[0]:
        raise ValueError(f"skin_weights shape {skin.shape} does not match vertex count {rest.shape[0]}")
    if order.size == 0:
        raise ValueError("rootless target has no joints for mesh component filtering")
    if int(order.min()) < 0 or int(order.max()) >= skin.shape[1]:
        raise ValueError("target raw joint indices outside skin weight columns")

    components, component_ids = _mesh_components(int(rest.shape[0]), face_arr)
    target_weight = skin[:, order].sum(axis=1)
    controlled = target_weight > float(active_skin_threshold)

    kept_components: list[int] = []
    dropped_components: list[int] = []
    mixed_components: list[dict[str, Any]] = []
    ratios = np.zeros((len(components),), dtype=np.float32)
    for comp_id, vertices in enumerate(components):
        if vertices.size == 0:
            dropped_components.append(int(comp_id))
            continue
        controlled_count = int(controlled[vertices].sum())
        ratio = float(controlled_count / max(1, int(vertices.size)))
        ratios[comp_id] = ratio
        if controlled_count == 0:
            dropped_components.append(int(comp_id))
        elif ratio >= float(min_component_controlled_ratio):
            kept_components.append(int(comp_id))
        else:
            mixed_components.append(
                {
                    "component": int(comp_id),
                    "vertices": int(vertices.size),
                    "controlled_vertices": int(controlled_count),
                    "controlled_ratio": ratio,
                }
            )

    if mixed_components:
        preview = mixed_components[:8]
        raise ValueError(f"mixed_controlled_mesh_components:{preview}")
    if not kept_components:
        raise ValueError("controlled mesh component filter would drop all components")

    keep_vertex_mask = np.isin(component_ids, np.asarray(kept_components, dtype=np.int64))
    kept_vertex_indices = np.flatnonzero(keep_vertex_mask).astype(np.int64)
    remap = np.full((int(rest.shape[0]),), -1, dtype=np.int64)
    remap[kept_vertex_indices] = np.arange(int(kept_vertex_indices.shape[0]), dtype=np.int64)

    if face_arr.size:
        face_keep = keep_vertex_mask[face_arr].all(axis=1)
        filtered_faces = remap[face_arr[face_keep]]
    else:
        face_keep = np.zeros((0,), dtype=bool)
        filtered_faces = face_arr.reshape(0, 3)

    return ControlledMeshSubset(
        rest_vertices=rest[kept_vertex_indices].astype(np.float32, copy=False),
        frame_vertices=frames[:, kept_vertex_indices].astype(np.float32, copy=False),
        faces=filtered_faces.astype(np.int64, copy=False),
        skin_weights=skin[kept_vertex_indices].astype(np.float32, copy=False),
        kept_vertex_indices=kept_vertex_indices,
        kept_component_indices=np.asarray(kept_components, dtype=np.int64),
        dropped_component_indices=np.asarray(dropped_components, dtype=np.int64),
        vertex_component_ids=component_ids.astype(np.int64, copy=False),
        component_controlled_ratios=ratios,
        raw_vertex_count=int(rest.shape[0]),
        raw_face_count=int(face_arr.shape[0]) if face_arr.ndim == 2 else 0,
        component_count=int(len(components)),
        kept_component_count=int(len(kept_components)),
        dropped_component_count=int(len(dropped_components)),
        dropped_vertex_count=int(rest.shape[0] - kept_vertex_indices.shape[0]),
        dropped_face_count=int((~face_keep).sum()) if face_arr.size else 0,
    )


def build_rootless_contract(
    parents: np.ndarray | Sequence[int],
    skin_weights: np.ndarray,
    names: Sequence[str],
    *,
    active_skin_threshold: float,
) -> RootlessContract:
    arr = parent_array(parents)
    n = int(arr.shape[0])
    name_list = [str(x) for x in names]
    if len(name_list) != n:
        raise ValueError(f"names length {len(name_list)} != parent count {n}")

    active = active_skin_mask(skin_weights, active_skin_threshold)
    if active.shape[0] != n:
        raise ValueError(f"skin joint count {active.shape[0]} != parent count {n}")
    active_ids = [int(i) for i in np.flatnonzero(active).tolist()]
    if not active_ids:
        raise ValueError("no active/skinned joints found")

    raw_roots = [int(i) for i in np.flatnonzero(arr < 0).tolist()]
    if not raw_roots:
        raise ValueError("raw skeleton has no parent<0 root")
    children = children_from_parents(arr)
    has_active = active_descendant_mask(arr, active)

    entries: list[int] = []
    recorded_roots: list[int] = []
    for root in raw_roots:
        node = int(root)
        if not bool(has_active[node]):
            recorded_roots.append(node)
            continue
        while not bool(active[node]):
            active_children = [int(child) for child in children[node] if bool(has_active[int(child)])]
            if len(active_children) == 0:
                recorded_roots.append(node)
                node = -1
                break
            if len(active_children) > 1:
                raise ValueError(
                    "rootless target would become a forest after dropping an unskinned top root "
                    f"{node}: active_children={active_children}"
                )
            recorded_roots.append(node)
            node = int(active_children[0])
        if node >= 0:
            entries.append(int(node))
    entries = [int(x) for x in entries]
    if not entries:
        raise ValueError("rootless contract has no active entry")
    if len(entries) != 1:
        raise ValueError(f"rootless target would become a forest with entries={entries}")
    recorded_roots = list(dict.fromkeys(int(x) for x in recorded_roots if int(x) not in entries))
    if recorded_roots:
        raw_root_disposition = "dropped_unskinned_top_roots_record_only"
    else:
        raw_root_disposition = "kept_first_skinned_root"

    entry_set = set(entries)
    keep: set[int] = set()
    for active_id in active_ids:
        path: list[int] = []
        node = int(active_id)
        while node >= 0:
            path.append(node)
            if node in entry_set:
                break
            node = int(arr[node])
        if not path or path[-1] not in entry_set:
            raise ValueError(f"active joint {active_id} is not under a rootless entry")
        keep.update(path)

    raw_order: list[int] = []

    def visit(node: int) -> None:
        if node not in keep:
            return
        raw_order.append(int(node))
        for child in children[int(node)]:
            visit(int(child))

    for entry in entries:
        visit(int(entry))

    old_to_new = {raw: new for new, raw in enumerate(raw_order)}
    target_parents = np.full((len(raw_order),), -1, dtype=np.int64)
    for raw in raw_order:
        p = int(arr[int(raw)])
        while p >= 0 and p not in old_to_new:
            p = int(arr[p])
        target_parents[old_to_new[int(raw)]] = -1 if p < 0 else int(old_to_new[p])

    roots = [idx for idx, parent in enumerate(target_parents.tolist()) if int(parent) < 0]
    if len(roots) != 1:
        raise ValueError(f"rootless target roots are {roots}, expected exactly one")
    for child, parent in enumerate(target_parents.tolist()):
        p = int(parent)
        if p >= child:
            raise ValueError(f"target parent order invalid at child={child} parent={p}")

    dropped_active = sorted(set(active_ids) - keep)
    if dropped_active:
        raise ValueError(f"rootless contract would drop active joints: {dropped_active}")

    kept = sorted(int(x) for x in keep)
    dropped = [int(i) for i in range(n) if i not in keep]
    return RootlessContract(
        raw_root=int(raw_roots[0]),
        raw_root_indices=[int(x) for x in raw_roots],
        raw_order=[int(x) for x in raw_order],
        parents=target_parents,
        names=[name_list[int(i)] for i in raw_order],
        kept_raw_indices=kept,
        dropped_raw_indices=dropped,
        active_raw_indices=active_ids,
        entry_raw_indices=entries,
        recorded_root_indices=recorded_roots,
        raw_root_disposition=raw_root_disposition,
    )


def _canonicalize_one(
    src_path: Path,
    out_path: Path,
    *,
    active_skin_threshold: float,
    min_component_controlled_ratio: float,
    compression: str,
    verify_vertices: int,
) -> dict[str, Any]:
    with np.load(src_path, allow_pickle=True) as src:
        rest_vertices = np.asarray(src["rest_vertices"], dtype=np.float32)
        faces = np.asarray(src["faces"], dtype=np.int64)
        frame_vertices = np.asarray(src["frame_vertices"], dtype=np.float32)
        rest_joints = np.asarray(src["rest_joints"], dtype=np.float32)
        rest_tails = np.asarray(src["rest_tails"], dtype=np.float32)
        parents = np.asarray(src["parents"], dtype=np.int64).reshape(-1)
        skin_weights = np.asarray(src["skin_weights"], dtype=np.float32)
        bone_transforms = np.asarray(src["bone_transforms"], dtype=np.float32)

        if rest_vertices.ndim != 2 or rest_vertices.shape[-1] != 3:
            raise ValueError(f"{src_path}: rest_vertices must be [V,3], got {rest_vertices.shape}")
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError(f"{src_path}: faces must be [F,3], got {faces.shape}")
        if frame_vertices.ndim != 3 or frame_vertices.shape[-1] != 3:
            raise ValueError(f"{src_path}: frame_vertices must be [T,V,3], got {frame_vertices.shape}")
        if frame_vertices.shape[1:] != rest_vertices.shape:
            raise ValueError(
                f"{src_path}: frame_vertices vertex shape {frame_vertices.shape[1:]} != rest_vertices {rest_vertices.shape}"
            )
        if rest_tails.shape != rest_joints.shape:
            raise ValueError(f"{src_path}: rest_tails shape {rest_tails.shape} != rest_joints {rest_joints.shape}")
        if parents.shape != (rest_joints.shape[0],):
            raise ValueError(f"{src_path}: parents shape {parents.shape} != joint count {rest_joints.shape[0]}")
        if skin_weights.ndim != 2 or skin_weights.shape[1] != rest_joints.shape[0]:
            raise ValueError(f"{src_path}: skin_weights shape {skin_weights.shape} != joint count {rest_joints.shape[0]}")
        if bone_transforms.shape[:2] != (frame_vertices.shape[0], rest_joints.shape[0]):
            raise ValueError(
                f"{src_path}: bone_transforms shape {bone_transforms.shape} does not match frames/joints"
            )

        joint_names = _load_joint_names(src, rest_joints.shape[0])
        contract = build_rootless_contract(
            parents,
            skin_weights,
            joint_names,
            active_skin_threshold=active_skin_threshold,
        )
        raw_root = int(contract.raw_root)
        raw_root_indices = np.asarray(contract.raw_root_indices, dtype=np.int64)
        recorded_root_indices = np.asarray(contract.recorded_root_indices, dtype=np.int64)
        order = np.asarray(contract.raw_order, dtype=np.int64)
        raw_children = children_from_parents(parent_array(parents))
        raw_root_skin = skin_weights[:, raw_root].astype(np.float32, copy=False)
        raw_root_name = str(joint_names[raw_root])
        raw_root_names = np.asarray([joint_names[int(i)] for i in raw_root_indices.tolist()], dtype=object)
        recorded_root_names = np.asarray([joint_names[int(i)] for i in recorded_root_indices.tolist()], dtype=object)
        raw_root_has_skin = bool(np.max(raw_root_skin) > float(active_skin_threshold)) if raw_root_skin.size else False
        raw_root_skin_weight_max = float(np.max(raw_root_skin)) if raw_root_skin.size else 0.0
        raw_root_skin_weight_sum = float(np.sum(raw_root_skin)) if raw_root_skin.size else 0.0
        raw_root_child_count = int(len(raw_children[raw_root]))
        raw_root_looks_like_dummy = bool(looks_like_dummy_root(raw_root_name))

        mesh_subset = _controlled_mesh_subset(
            rest_vertices,
            frame_vertices,
            faces,
            skin_weights,
            order,
            active_skin_threshold=active_skin_threshold,
            min_component_controlled_ratio=min_component_controlled_ratio,
        )
        rest_vertices = mesh_subset.rest_vertices
        frame_vertices = mesh_subset.frame_vertices
        faces = mesh_subset.faces
        skin_weights = mesh_subset.skin_weights

        posed_joints = apply_bone_transforms(bone_transforms, rest_joints)
        recorded_root_positions, recorded_root_rotations_world_to_root, recorded_root_rotations_root_to_world = (
            _recorded_root_transforms(recorded_root_indices.tolist(), posed_joints, bone_transforms)
        )
        root_positions, root_rotations = _identity_root_transform(int(frame_vertices.shape[0]))

        frame_vertices_rootspace = frame_vertices.astype(np.float32, copy=True)
        posed_joints_rootspace = posed_joints.astype(np.float32, copy=True)

        target_joints_rootspace = posed_joints_rootspace[:, order]
        target_skin_weights = skin_weights[:, order]

        active_raw = skin_weights.max(axis=0) > float(active_skin_threshold)
        target_has_skin = active_raw[order]
        target_is_synthetic_root = np.zeros((int(order.shape[0]),), dtype=bool)
        target_is_connector = (~target_has_skin) & (~target_is_synthetic_root)
        target_is_tail_or_end = np.asarray([looks_like_tail_or_end(joint_names[int(i)]) for i in order], dtype=bool)
        target_raw_indices = order.astype(np.int64)
        target_joint_names = np.asarray(contract.names, dtype=object)
        rootspace_from_world_mats, world_from_rootspace_mats = _transform_mats(root_positions, root_rotations)

        raw_row_sum = skin_weights.sum(axis=1)
        target_row_sum = target_skin_weights.sum(axis=1)
        skin_weight_lost_linf = float(np.max(np.abs(raw_row_sum - target_row_sum))) if raw_row_sum.size else 0.0
        if not np.isfinite(skin_weight_lost_linf) or skin_weight_lost_linf > 1.0e-4:
            raise ValueError(f"{src_path}: rootless target dropped skinned weight, loss={skin_weight_lost_linf}")

        n_vertices = int(frame_vertices.shape[1])
        if verify_vertices > 0:
            sample_count = min(int(verify_vertices), n_vertices)
            rng = np.random.default_rng(12345)
            sample_ids = rng.choice(n_vertices, size=sample_count, replace=False)
            recon = _world_from_rootspace(frame_vertices_rootspace[:, sample_ids], root_positions, root_rotations)
            recon_err = float(np.max(np.abs(recon - frame_vertices[:, sample_ids])))
            bbox_extent = float(np.max(frame_vertices.max(axis=(0, 1)) - frame_vertices.min(axis=(0, 1))))
        else:
            recon_err = 0.0
            bbox_extent = 1.0
        recon_tol = max(1.0e-4, 1.0e-5 * max(bbox_extent, 1.0e-8))
        if not np.isfinite(recon_err) or recon_err > recon_tol:
            raise ValueError(
                f"{src_path}: rootspace reconstruction error too high: "
                f"{recon_err} > {recon_tol} (bbox_extent={bbox_extent})"
            )

        roots = [int(i) for i, p in enumerate(contract.parents.tolist()) if int(p) < 0]
        if len(roots) != 1:
            raise ValueError(f"{src_path}: rootless target roots are {roots}, expected exactly one")
        if any(str(name) == "__object_root__" for name in target_joint_names.tolist()):
            raise ValueError(f"{src_path}: target still contains __object_root__")

        payload = _copy_npz_fields(src)
        payload.update(
            {
                "canonical_schema_version": np.asarray(SCHEMA_VERSION, dtype=object),
                "canonical_meta_json": _json_dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_path": str(src_path),
                        "active_skin_threshold": float(active_skin_threshold),
                        "rootspace_policy": "identity_training_space_deleted_roots_record_only",
                        "root_rotation_convention": "identity_training_transform; recorded_root_* stores deleted root transforms",
                        "target_root_policy": "rootless_no_synthetic_root",
                        "tail_endpoint_policy": "not_a_training_target_not_written_to_rootless_npz",
                        "mesh_subset_policy": "connected_components_controlled_by_retained_rootless_joints",
                        "min_component_controlled_ratio": float(min_component_controlled_ratio),
                        "raw_root_disposition": contract.raw_root_disposition,
                    }
                ),
                "raw_source_path": np.asarray(str(src_path), dtype=object),
                "raw_vertex_count": np.asarray(mesh_subset.raw_vertex_count, dtype=np.int64),
                "raw_face_count": np.asarray(mesh_subset.raw_face_count, dtype=np.int64),
                "rest_vertices": rest_vertices,
                "faces": faces,
                "frame_vertices": frame_vertices,
                "skin_weights": skin_weights,
                "raw_root_index": np.asarray(raw_root, dtype=np.int64),
                "raw_root_indices": raw_root_indices,
                "raw_root_name": np.asarray(raw_root_name, dtype=object),
                "raw_root_names": raw_root_names,
                "raw_root_has_skin": np.asarray(raw_root_has_skin, dtype=bool),
                "raw_root_skin_weight_max": np.asarray(raw_root_skin_weight_max, dtype=np.float32),
                "raw_root_skin_weight_sum": np.asarray(raw_root_skin_weight_sum, dtype=np.float32),
                "raw_root_child_count": np.asarray(raw_root_child_count, dtype=np.int64),
                "raw_root_looks_like_dummy": np.asarray(raw_root_looks_like_dummy, dtype=bool),
                "raw_root_disposition": np.asarray(contract.raw_root_disposition, dtype=object),
                "recorded_root_indices": recorded_root_indices,
                "recorded_root_names": recorded_root_names,
                "recorded_root_positions": recorded_root_positions,
                "recorded_root_rotations_world_to_root": recorded_root_rotations_world_to_root,
                "recorded_root_rotations_root_to_world": recorded_root_rotations_root_to_world,
                "root_positions": root_positions,
                "root_rotations_world_to_root": root_rotations,
                "root_rotations_root_to_world": np.swapaxes(root_rotations, 1, 2).astype(np.float32),
                "rootspace_from_world_mats": rootspace_from_world_mats,
                "world_from_rootspace_mats": world_from_rootspace_mats,
                "frame_vertices_rootspace": frame_vertices_rootspace,
                "posed_joints_rootspace": posed_joints_rootspace,
                "target_parents": contract.parents.astype(np.int64),
                "target_raw_indices": target_raw_indices,
                "target_joint_names": target_joint_names,
                "target_joints_rootspace": target_joints_rootspace,
                "target_skin_weights": target_skin_weights,
                "target_has_skin": target_has_skin,
                "target_is_connector": target_is_connector,
                "target_is_synthetic_root": target_is_synthetic_root,
                "target_is_tail_or_end": target_is_tail_or_end,
                "target_entry_raw_indices": np.asarray(contract.entry_raw_indices, dtype=np.int64),
                "target_kept_raw_indices": np.asarray(contract.kept_raw_indices, dtype=np.int64),
                "target_dropped_raw_indices": np.asarray(contract.dropped_raw_indices, dtype=np.int64),
                "mesh_kept_vertex_indices": mesh_subset.kept_vertex_indices,
                "mesh_source_component_ids": mesh_subset.vertex_component_ids,
                "mesh_kept_component_indices": mesh_subset.kept_component_indices,
                "mesh_dropped_component_indices": mesh_subset.dropped_component_indices,
                "mesh_component_controlled_ratios": mesh_subset.component_controlled_ratios,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.npz")
    try:
        if compression == "compressed":
            np.savez_compressed(tmp_path, **payload)
        elif compression == "stored":
            np.savez(tmp_path, **payload)
        else:
            raise ValueError(f"unknown compression mode {compression!r}")
        with np.load(tmp_path, allow_pickle=True) as check:
            for key in ("canonical_schema_version", "frame_vertices_rootspace", "target_parents"):
                if key not in check.files:
                    raise RuntimeError(f"temporary output missing {key}: {tmp_path}")
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "source_path": str(src_path),
        "output_path": str(out_path),
        "raw_vertex_count": int(mesh_subset.raw_vertex_count),
        "filtered_vertex_count": int(rest_vertices.shape[0]),
        "raw_face_count": int(mesh_subset.raw_face_count),
        "filtered_face_count": int(faces.shape[0]),
        "mesh_component_count": int(mesh_subset.component_count),
        "kept_mesh_component_count": int(mesh_subset.kept_component_count),
        "dropped_mesh_component_count": int(mesh_subset.dropped_component_count),
        "dropped_vertex_count": int(mesh_subset.dropped_vertex_count),
        "dropped_face_count": int(mesh_subset.dropped_face_count),
        "raw_joint_count": int(parents.shape[0]),
        "target_joint_count": int(contract.parents.shape[0]),
        "active_raw_joint_count": int(active_raw.sum()),
        "target_connector_count": int(target_is_connector.sum()),
        "target_tail_or_end_count": int(target_is_tail_or_end.sum()),
        "dropped_raw_count": int(len(contract.dropped_raw_indices)),
        "raw_root_index": int(raw_root),
        "raw_root_count": int(raw_root_indices.shape[0]),
        "raw_root_name": raw_root_name,
        "raw_root_has_skin": bool(raw_root_has_skin),
        "raw_root_skin_weight_max": float(raw_root_skin_weight_max),
        "raw_root_skin_weight_sum": float(raw_root_skin_weight_sum),
        "raw_root_child_count": int(raw_root_child_count),
        "raw_root_looks_like_dummy": bool(raw_root_looks_like_dummy),
        "recorded_root_count": int(recorded_root_indices.shape[0]),
        "rootspace_recon_linf": float(recon_err),
        "rootspace_recon_tol": float(recon_tol),
        "skin_weight_lost_linf": float(skin_weight_lost_linf),
        "raw_root_disposition": contract.raw_root_disposition,
        "target_root_count": int(len(roots)),
        "target_entry_count": int(len(contract.entry_raw_indices)),
    }


def _iter_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _output_npz_path(row: dict[str, Any], output_root: Path) -> Path:
    rel = row.get("rel")
    if rel:
        rel_path = Path(str(rel))
    else:
        src = Path(row.get("raw_path") or row.get("path"))
        rel_path = Path(src.name)
    if rel_path.suffix != ".npz":
        rel_path = rel_path.with_suffix(".npz")
    return output_root / "npz" / rel_path


def _source_path(row: dict[str, Any]) -> Path:
    return Path(row.get("raw_path") or row["path"])


def _build_one_row(job: dict[str, Any]) -> dict[str, Any]:
    idx = int(job["index"])
    row = dict(job["row"])
    output_root = Path(job["output_root"])
    src_path = _source_path(row)
    asset_id = str(row.get("asset_id") or src_path.stem.replace("_seq0", ""))
    out_path = _output_npz_path(row, output_root)
    try:
        if bool(job["skip_existing"]) and out_path.exists():
            out_path.unlink()
        metrics = _canonicalize_one(
            src_path,
            out_path,
            active_skin_threshold=float(job["active_skin_threshold"]),
            min_component_controlled_ratio=float(job["min_component_controlled_ratio"]),
            compression=str(job["compression"]),
            verify_vertices=int(job["verify_vertices"]),
        )
        out_row = dict(row)
        out_row["raw_path"] = str(src_path)
        out_row["path"] = str(out_path)
        out_row["canonical_schema_version"] = SCHEMA_VERSION
        out_row["canonical_metrics"] = {key: metrics[key] for key in CANONICAL_METRIC_KEYS}
        return {
            "status": "ok",
            "index": idx,
            "asset_id": asset_id,
            "src_path": str(src_path),
            "out_path": str(out_path),
            "out_row": out_row,
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "fail",
            "index": idx,
            "asset_id": asset_id,
            "src_path": str(src_path),
            "error": repr(exc),
        }


def _update_summary_with_metrics(summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    summary["rows"] += 1
    for key in SUMMARY_COUNT_KEYS:
        summary[key] += int(metrics[key])
    summary["max_rootspace_recon_linf"] = max(
        float(summary["max_rootspace_recon_linf"]), float(metrics["rootspace_recon_linf"])
    )
    summary["max_skin_weight_lost_linf"] = max(
        float(summary["max_skin_weight_lost_linf"]), float(metrics["skin_weight_lost_linf"])
    )
    disp = str(metrics["raw_root_disposition"])
    summary["raw_root_disposition"][disp] = int(summary["raw_root_disposition"].get(disp, 0)) + 1


def _write_result(
    result: dict[str, Any],
    *,
    manifest_name: str,
    row_count: int,
    summary: dict[str, Any],
    writer: Any,
    skipped: Any,
) -> None:
    if result["status"] == "ok":
        writer.write(json.dumps(result["out_row"], ensure_ascii=False) + "\n")
        _update_summary_with_metrics(summary, result["metrics"])
        if summary["rows"] % 100 == 0:
            writer.flush()
            skipped.flush()
            print(
                f"[{manifest_name}] wrote {summary['rows']}/{row_count} "
                f"failed={summary['failed']} out={result['out_path']}",
                flush=True,
            )
        return

    summary["failed"] += 1
    failure = {
        "index": int(result["index"]),
        "asset_id": str(result["asset_id"]),
        "path": str(result["src_path"]),
        "error": str(result["error"]),
    }
    summary["failures"].append(failure)
    skipped.write(json.dumps({**failure, "reason": "rootless_build_failed"}, sort_keys=True) + "\n")
    print(f"[FAIL] {failure['index']} {failure['path']}: {failure['error']}", file=sys.stderr, flush=True)


def build_manifest(
    manifest: Path,
    output_root: Path,
    *,
    limit: int,
    active_skin_threshold: float,
    min_component_controlled_ratio: float,
    compression: str,
    verify_vertices: int,
    skip_existing: bool,
    workers: int,
) -> dict[str, Any]:
    rows = _iter_manifest(manifest, limit=limit)
    out_manifest = output_root / manifest.name
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    skipped_manifest = output_root / f"{manifest.stem}.rootless_rejected.jsonl"

    summary: dict[str, Any] = {
        "manifest": str(manifest),
        "output_manifest": str(out_manifest),
        "rows": 0,
        "failed": 0,
        "raw_vertex_count": 0,
        "filtered_vertex_count": 0,
        "raw_face_count": 0,
        "filtered_face_count": 0,
        "mesh_component_count": 0,
        "kept_mesh_component_count": 0,
        "dropped_mesh_component_count": 0,
        "dropped_vertex_count": 0,
        "dropped_face_count": 0,
        "raw_joint_count": 0,
        "target_joint_count": 0,
        "active_raw_joint_count": 0,
        "target_connector_count": 0,
        "target_tail_or_end_count": 0,
        "dropped_raw_count": 0,
        "max_rootspace_recon_linf": 0.0,
        "max_skin_weight_lost_linf": 0.0,
        "raw_root_disposition": {},
        "failures": [],
    }

    jobs: list[dict[str, Any]] = []
    with out_manifest.open("w", encoding="utf-8") as writer, skipped_manifest.open("w", encoding="utf-8") as skipped:
        for idx, row in enumerate(rows):
            jobs.append(
                {
                    "index": idx,
                    "row": row,
                    "output_root": str(output_root),
                    "active_skin_threshold": float(active_skin_threshold),
                    "min_component_controlled_ratio": float(min_component_controlled_ratio),
                    "compression": str(compression),
                    "verify_vertices": int(verify_vertices),
                    "skip_existing": bool(skip_existing),
                }
            )

        if int(workers) <= 1:
            for job in jobs:
                _write_result(
                    _build_one_row(job),
                    manifest_name=manifest.name,
                    row_count=len(rows),
                    summary=summary,
                    writer=writer,
                    skipped=skipped,
                )
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [executor.submit(_build_one_row, job) for job in jobs]
                for future in as_completed(futures):
                    _write_result(
                        future.result(),
                        manifest_name=manifest.name,
                        row_count=len(rows),
                        summary=summary,
                        writer=writer,
                        skipped=skipped,
                    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Input manifest jsonl. Repeatable.")
    parser.add_argument("--output-root", required=True, help="Rootless derived dataset directory.")
    parser.add_argument("--limit", type=int, default=0, help="Rows per manifest; 0 means all.")
    parser.add_argument("--active-skin-threshold", type=float, default=1.0e-4)
    parser.add_argument("--min-component-controlled-ratio", type=float, default=0.95)
    parser.add_argument("--compression", choices=["compressed", "stored"], default="compressed")
    parser.add_argument("--verify-vertices", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1, help="Parallel NPZ build workers per manifest.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-build-rejects",
        action="store_true",
        help="Finish the dataset and record rootless contract/build rejects instead of exiting non-zero.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for manifest_str in args.manifest:
        summary = build_manifest(
            Path(manifest_str),
            output_root,
            limit=int(args.limit),
            active_skin_threshold=float(args.active_skin_threshold),
            min_component_controlled_ratio=float(args.min_component_controlled_ratio),
            compression=str(args.compression),
            verify_vertices=int(args.verify_vertices),
            skip_existing=bool(args.skip_existing),
            workers=max(1, int(args.workers)),
        )
        summaries.append(summary)
    total = {
        "schema_version": SCHEMA_VERSION,
        "output_root": str(output_root),
        "summaries": summaries,
        "rows": int(sum(x["rows"] for x in summaries)),
        "failed": int(sum(x["failed"] for x in summaries)),
    }
    with (output_root / "rootless_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(total, handle, ensure_ascii=False, indent=2)
    print(json.dumps(total, ensure_ascii=False, indent=2), flush=True)
    if total["failed"] and not bool(args.allow_build_rejects):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
