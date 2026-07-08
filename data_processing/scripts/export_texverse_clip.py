#!/usr/bin/env python3
"""Export one TexVerse asset/action into a RigWeave-ready sequence npz.

This is the first Pass1 prototype.  It imports an animated skinned asset with
Blender, selects the main armature and skinned meshes, orders bones with the
UniRig parent-before-child traversal, samples an action into F frames, and exports:

  rest_vertices, faces, frame_vertices, skin_weights, parents, rest_joints,
  rest_tails, bone_transforms, frame_numbers, joint_names, metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--blender", default="/home/wangyy/.local/bin/blender")
    parser.add_argument("--blender-threads", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--min-vertices", type=int, default=1)
    parser.add_argument("--max-joints", type=int, default=256)
    parser.add_argument("--max-vertices", type=int, default=300000)
    parser.add_argument("--max-faces", type=int, default=600000)
    parser.add_argument("--bbox-ratio-hard-min", type=float, default=0.0)
    parser.add_argument("--bbox-ratio-hard-max", type=float, default=0.0)
    parser.add_argument("--min-motion-p95-bbox", type=float, default=0.0)
    parser.add_argument("--topk-weights", type=int, default=8)
    parser.add_argument("--active-skin-threshold", type=float, default=0.0)
    parser.add_argument("--motion-fps-descriptor-vertices", type=int, default=1024)
    parser.add_argument("--timeout-sec", type=int, default=300)
    return parser.parse_args()


def glb_attribute_needed(name: str) -> bool:
    return (
        name in {"POSITION", "NORMAL", "TANGENT"}
        or name.startswith("TEXCOORD_")
        or name.startswith("JOINTS_")
        or name.startswith("WEIGHTS_")
    )


def sanitize_glb_import_source(source: Path, tmp_dir: Path) -> tuple[Path, dict]:
    """Strip Blender-hostile, training-irrelevant GLB vertex attributes."""
    if source.suffix.lower() != ".glb":
        return source, {}

    data = source.read_bytes()
    if len(data) < 20:
        return source, {}
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        return source, {}

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks.append((chunk_type, data[offset : offset + chunk_len]))
        offset += chunk_len
    if not chunks or chunks[0][0] != 0x4E4F534A:
        return source, {}

    gltf = json.loads(chunks[0][1].decode("utf-8").rstrip("\x00 "))
    removed: list[str] = []
    for mesh in gltf.get("meshes", []) or []:
        for prim in mesh.get("primitives", []) or []:
            attrs = prim.get("attributes") or {}
            for name in list(attrs.keys()):
                if not glb_attribute_needed(name):
                    removed.append(str(name))
                    del attrs[name]
    if not removed:
        return source, {}

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_chunk = json_bytes + (b" " * json_pad)
    out = bytearray()
    total_len = 12 + 8 + len(json_chunk) + sum(8 + len(chunk) for _, chunk in chunks[1:])
    out += struct.pack("<III", magic, version, total_len)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A)
    out += json_chunk
    for chunk_type, chunk in chunks[1:]:
        out += struct.pack("<II", len(chunk), chunk_type)
        out += chunk

    sanitized = tmp_dir / f"{source.stem}.rigweave_sanitized.glb"
    sanitized.write_bytes(out)
    return sanitized, {
        "original_source": str(source),
        "glb_import_sanitized": True,
        "glb_removed_vertex_attributes": sorted(set(removed)),
    }


def blender_export_script(
    args: argparse.Namespace,
    import_source: Path,
    import_sanitize_meta: dict,
) -> str:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    template = r'''
import json
import math
import os
import re
import sys
import traceback
import importlib.util

import bpy
import numpy as np
from mathutils import Matrix, Vector

repo_src = __REPO_SRC__
if repo_src and repo_src not in sys.path:
    sys.path.insert(0, repo_src)

contract_path = os.path.join(repo_src, "rigweave", "dynamic_rig", "skeleton_contract.py")
contract_spec = importlib.util.spec_from_file_location("rigweave_skeleton_contract", contract_path)
if contract_spec is None or contract_spec.loader is None:
    raise RuntimeError(f"failed to load skeleton contract module from {contract_path}")
contract_module = importlib.util.module_from_spec(contract_spec)
sys.modules[contract_spec.name] = contract_module
contract_spec.loader.exec_module(contract_module)
build_row_skeleton_contract = contract_module.build_row_skeleton_contract

source = __SOURCE__
original_source = __ORIGINAL_SOURCE__
import_sanitize_meta = __IMPORT_SANITIZE_META__
asset_id = __ASSET_ID__
out_npz = __OUT_NPZ__
out_json = __OUT_JSON__
sample_t = __FRAMES__
min_vertices = __MIN_VERTICES__
max_joints = __MAX_JOINTS__
max_vertices = __MAX_VERTICES__
max_faces = __MAX_FACES__
bbox_ratio_hard_min = __BBOX_RATIO_HARD_MIN__
bbox_ratio_hard_max = __BBOX_RATIO_HARD_MAX__
min_motion_p95_bbox = __MIN_MOTION_P95_BBOX__
topk_weights = __TOPK_WEIGHTS__
active_skin_threshold = __ACTIVE_SKIN_THRESHOLD__
motion_fps_descriptor_vertices = __MOTION_FPS_DESCRIPTOR_VERTICES__

def fail(reason, status="reject", **extra):
    key = "reject_reasons" if status == "reject" else "warnings"
    record = {"asset_id": asset_id, "status": status, key: [reason]}
    if original_source != source:
        record["original_source"] = original_source
    record.update(import_sanitize_meta)
    record.update(extra)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
    print("RIGWEAVE_EXPORT_JSON " + json.dumps(record, ensure_ascii=False))
    sys.exit(2)

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_asset(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        fail("unsupported_source_type", suffix=suffix)

def action_bone_names(action):
    names = set()
    pattern = re.compile(r'pose\.bones\["([^"]+)"\]')
    for curve in action.fcurves:
        match = pattern.search(curve.data_path)
        if match:
            names.add(match.group(1))
    return names

def action_bone_fcurves(action, bone_name_set):
    count = 0
    for curve in action.fcurves:
        if "pose.bones" not in curve.data_path:
            continue
        for name in bone_name_set:
            if f'pose.bones["{name}"]' in curve.data_path:
                count += 1
                break
    return count

def action_frame_count(action):
    return int(round(action.frame_range[1] - action.frame_range[0] + 1))

def action_name_rank(action, armature, assigned_action):
    if assigned_action is not None and action.name == assigned_action.name:
        return 3
    if action.name == f"{armature.name}|Scene" or action.name.startswith(f"{armature.name}|"):
        return 2
    if action.name.startswith(f"{armature.name}."):
        # Usually belongs to a duplicated armature, e.g. RIG.001|Scene.
        return -1
    return 0

def select_best_bone_action(armature, actions, bone_names, sample_frames):
    """Choose a usable bone action, preferring actions long enough to sample.

    Some assets bind a very short/idle action as the armature's assigned action
    while also carrying a valid longer action in the file.  Choosing the assigned
    short clip first caused false `short_action` rejects.  This ranking keeps the
    name/assignment prior, but only after the hard "can provide sample_frames"
    criterion.
    """
    assigned_action = armature.animation_data.action if armature.animation_data and armature.animation_data.action else None
    best = None
    for action in actions:
        curve_count = action_bone_fcurves(action, bone_names)
        if curve_count <= 0:
            continue
        animated_bones = len(action_bone_names(action) & bone_names)
        frames = action_frame_count(action)
        eligible = int(frames >= sample_frames)
        rank = (
            eligible,
            action_name_rank(action, armature, assigned_action),
            animated_bones,
            curve_count,
            frames,
            action.name,
        )
        if best is None or rank > best["rank"]:
            best = {
                "action": action,
                "rank": rank,
                "frames": frames,
                "eligible": bool(eligible),
                "curve_count": int(curve_count),
                "animated_bones": int(animated_bones),
            }
    return best

def armature_meshes(meshes, armatures, armature):
    out = []
    for obj in meshes:
        for mod in obj.modifiers:
            if mod.type != "ARMATURE":
                continue
            if mod.object == armature or (mod.object is None and len(armatures) == 1):
                out.append(obj)
                break
    # Blender does not guarantee that imported scene objects keep a stable
    # iteration order across runs/files.  The concatenated vertex table and
    # skin-weight rows depend on this order, so make it explicit.
    return sorted(out, key=lambda obj: obj.name)

def select_main_armature():
    armatures = sorted([obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"], key=lambda obj: obj.name)
    meshes = sorted([obj for obj in bpy.context.scene.objects if obj.type == "MESH"], key=lambda obj: obj.name)
    actions = sorted(list(bpy.data.actions), key=lambda action: action.name)
    if not armatures:
        fail("no_armature")
    if not meshes:
        fail("no_mesh")
    best = None
    for arm in armatures:
        bone_names = {bone.name for bone in arm.data.bones}
        skinned = armature_meshes(meshes, armatures, arm)
        skinned_vertices = sum(len(obj.data.vertices) for obj in skinned)
        action_info = select_best_bone_action(arm, actions, bone_names, sample_t)
        best_action = None if action_info is None else action_info["action"]
        raw_curve_count = 0 if action_info is None else action_info["curve_count"]
        animated_bones = 0 if action_info is None else action_info["animated_bones"]
        action_frames = 0 if action_info is None else action_info["frames"]
        eligible_action = False if action_info is None else action_info["eligible"]
        score = skinned_vertices + 100 * animated_bones + 10 * len(bone_names)
        item = {
            "armature": arm,
            "meshes": skinned,
            "action": best_action,
            "score": score,
            "skinned_vertices": skinned_vertices,
            "animated_bones": animated_bones,
            "bone_fcurves": max(raw_curve_count, 0),
            "action_frames": int(action_frames),
            "eligible_action": bool(eligible_action),
        }
        if best is None or (int(item["eligible_action"]), item["score"]) > (int(best["eligible_action"]), best["score"]):
            best = item
    if best is None or not best["meshes"]:
        fail("no_skinned_mesh")
    if best["action"] is None or best["bone_fcurves"] <= 0:
        fail("no_bone_action")
    return best

def summarize_armature_candidates():
    armatures = sorted([obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"], key=lambda obj: obj.name)
    meshes = sorted([obj for obj in bpy.context.scene.objects if obj.type == "MESH"], key=lambda obj: obj.name)
    actions = sorted(list(bpy.data.actions), key=lambda action: action.name)
    out = []
    for arm in armatures:
        bone_names = {bone.name for bone in arm.data.bones}
        skinned = armature_meshes(meshes, armatures, arm)
        action_info = select_best_bone_action(arm, actions, bone_names, sample_t)
        best_action = None if action_info is None else action_info["action"]
        raw_curve_count = 0 if action_info is None else action_info["curve_count"]
        animated_bones = 0 if action_info is None else action_info["animated_bones"]
        action_frames = 0 if action_info is None else action_info["frames"]
        eligible_action = False if action_info is None else action_info["eligible"]
        skinned_vertices = sum(len(obj.data.vertices) for obj in skinned)
        score = skinned_vertices + 100 * animated_bones + 10 * len(bone_names)
        out.append(
            {
                "armature": arm.name,
                "bone_count": len(arm.data.bones),
                "root_count": sum(1 for bone in arm.data.bones if bone.parent is None),
                "skinned_meshes": [obj.name for obj in skinned],
                "skinned_vertices": int(skinned_vertices),
                "best_action": None if best_action is None else best_action.name,
                "best_action_frames": int(action_frames),
                "best_action_sample_eligible": bool(eligible_action),
                "animated_bones": int(animated_bones),
                "bone_fcurves": int(raw_curve_count),
                "score": int(score),
            }
        )
    out.sort(key=lambda item: item["score"], reverse=True)
    return out

def unirig_arranged_bones(armature):
    """Mirror UniRig's parent-before-child bone traversal.

    UniRig starts from the first pose bone, walks up to the root, then visits
    children in a deterministic geometry order.  We keep the extra coverage
    check here: a disconnected/multi-root armature is not silently truncated.
    """
    if not armature.pose.bones:
        fail("no_skeleton_root")
    root = armature.pose.bones[0]
    while root.parent is not None:
        root = root.parent

    rot = np.asarray(armature.matrix_world, dtype=np.float32)[:3, :3]
    order = []
    queue = [root]
    seen = set()
    while queue:
        pbone = queue.pop(0)
        if pbone.name in seen:
            fail("skeleton_cycle_or_duplicate", bone=pbone.name)
        seen.add(pbone.name)
        order.append(pbone.bone)
        children = []
        for child in pbone.children:
            head = rot @ np.asarray(pbone.head, dtype=np.float32)
            children.append((child, float(head[0]), float(head[1]), float(head[2]), child.name))
        children = sorted(children, key=lambda item: (item[3], item[1], item[2], item[4]))
        queue = [item[0] for item in children] + queue
    if len(order) != len(armature.data.bones):
        fail("disconnected_or_unvisited_bones", visited=len(order), total=len(armature.data.bones))
    return order

def triangulated_faces(obj, offset):
    faces = []
    for poly in obj.data.polygons:
        verts = list(poly.vertices)
        if len(verts) < 3:
            continue
        if len(verts) == 3:
            faces.append([offset + verts[0], offset + verts[1], offset + verts[2]])
        else:
            for i in range(1, len(verts) - 1):
                faces.append([offset + verts[0], offset + verts[i], offset + verts[i + 1]])
    return faces

def collect_rest_geometry(meshes):
    vertices = []
    faces = []
    mesh_offsets = []
    offset = 0
    for obj in meshes:
        mesh_offsets.append((obj, offset, len(obj.data.vertices)))
        for vertex in obj.data.vertices:
            co = obj.matrix_world @ vertex.co
            vertices.append([co.x, co.y, co.z])
        faces.extend(triangulated_faces(obj, offset))
        offset += len(obj.data.vertices)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64), mesh_offsets

def collect_skin_weights(mesh_offsets, bone_to_index, joint_count):
    weights = np.zeros((sum(count for _, _, count in mesh_offsets), joint_count), dtype=np.float32)
    for obj, offset, _ in mesh_offsets:
        group_names = {group.index: group.name for group in obj.vertex_groups}
        for vertex in obj.data.vertices:
            for group in vertex.groups:
                name = group_names.get(group.group)
                if name in bone_to_index:
                    weights[offset + vertex.index, bone_to_index[name]] += float(group.weight)
    sums = weights.sum(axis=1)
    coverage = float(np.mean(sums > 1e-8)) if weights.shape[0] else 0.0
    if coverage < 0.95:
        fail("low_skin_coverage_after_remap", skin_coverage=coverage)
    if topk_weights > 0 and topk_weights < weights.shape[1]:
        keep = np.argpartition(weights, -topk_weights, axis=1)[:, -topk_weights:]
        mask = np.zeros_like(weights, dtype=bool)
        rows = np.arange(weights.shape[0])[:, None]
        mask[rows, keep] = True
        weights = np.where(mask, weights, 0.0)
        sums = weights.sum(axis=1)
    weights = weights / np.maximum(sums[:, None], 1e-8)
    return weights.astype(np.float32), coverage

def collect_frame_vertices(mesh_offsets, frame):
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    coords = []
    for obj, _, rest_count in mesh_offsets:
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        try:
            if len(eval_mesh.vertices) != rest_count:
                fail(
                    "topology_changed_under_evaluation",
                    mesh=obj.name,
                    rest_vertices=rest_count,
                    eval_vertices=len(eval_mesh.vertices),
                )
            for vertex in eval_mesh.vertices:
                co = eval_obj.matrix_world @ vertex.co
                coords.append([co.x, co.y, co.z])
        finally:
            eval_obj.to_mesh_clear()
    return np.asarray(coords, dtype=np.float32)

def matrix_to_np(mat):
    return np.asarray([[float(mat[r][c]) for c in range(4)] for r in range(4)], dtype=np.float32)

def process_unirig_tails(rest_joints, raw_tails, parents):
    """Mirror UniRig TailConfig(copy_joint_to_tail=False, connect_tail_to_unique_son=True)."""
    processed = np.asarray(raw_tails, dtype=np.float32).copy()
    children = [[] for _ in range(len(parents))]
    for child, parent in enumerate(parents):
        if int(parent) >= 0:
            children[int(parent)].append(child)
    for parent, kids in enumerate(children):
        if len(kids) == 1:
            processed[parent] = rest_joints[kids[0]]
    return processed.astype(np.float32)

def collect_bone_data(armature, ordered_bones, frames):
    joint_names = [bone.name for bone in ordered_bones]
    bone_to_index = {name: i for i, name in enumerate(joint_names)}
    parents = []
    rest_joints = []
    rest_tails_raw = []
    rest_mats = []
    bone_use_connect = []
    bone_use_deform = []
    for bone in ordered_bones:
        parents.append(-1 if bone.parent is None else bone_to_index[bone.parent.name])
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        rest_joints.append([head.x, head.y, head.z])
        rest_tails_raw.append([tail.x, tail.y, tail.z])
        rest_mats.append(matrix_to_np(armature.matrix_world @ bone.matrix_local))
        bone_use_connect.append(bool(bone.use_connect))
        bone_use_deform.append(bool(bone.use_deform))
    rest_joints = np.asarray(rest_joints, dtype=np.float32)
    rest_tails_raw = np.asarray(rest_tails_raw, dtype=np.float32)
    parents_arr = np.asarray(parents, dtype=np.int64)
    rest_tails = process_unirig_tails(rest_joints, rest_tails_raw, parents_arr)
    transforms = []
    for frame in frames:
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_arm = armature.evaluated_get(depsgraph)
        frame_transforms = []
        for bone in ordered_bones:
            pose_bone = eval_arm.pose.bones.get(bone.name)
            if pose_bone is None:
                fail("missing_pose_bone", bone=bone.name)
            rest_world = armature.matrix_world @ bone.matrix_local
            pose_world = eval_arm.matrix_world @ pose_bone.matrix
            frame_transforms.append(matrix_to_np(pose_world @ rest_world.inverted()))
        transforms.append(frame_transforms)
    return (
        parents_arr,
        rest_joints,
        rest_tails,
        rest_tails_raw,
        np.asarray(rest_mats, dtype=np.float32),
        np.asarray(transforms, dtype=np.float32),
        np.asarray(joint_names, dtype=object),
        np.asarray(bone_use_connect, dtype=bool),
        np.asarray(bone_use_deform, dtype=bool),
        bone_to_index,
    )

def apply_row_skeleton_contract(
    parents,
    rest_joints,
    rest_tails,
    rest_tails_raw,
    rest_bone_mats,
    bone_transforms,
    joint_names,
    bone_use_connect,
    bone_use_deform,
    skin_weights,
):
    contract = build_row_skeleton_contract(
        parents,
        skin_weights,
        names=[str(x) for x in np.asarray(joint_names, dtype=object).tolist()],
        active_skin_threshold=active_skin_threshold,
        drop_unweighted_tail_leaf=True,
    )
    order = np.asarray(contract.raw_order, dtype=np.int64)
    new_parents = contract.parents.astype(np.int64)
    new_rest_joints = rest_joints[order].astype(np.float32)
    new_rest_tails_raw = rest_tails_raw[order].astype(np.float32)
    new_rest_tails = process_unirig_tails(new_rest_joints, new_rest_tails_raw, new_parents)
    return (
        new_parents,
        new_rest_joints,
        new_rest_tails,
        new_rest_tails_raw,
        rest_bone_mats[order].astype(np.float32),
        bone_transforms[:, order].astype(np.float32),
        np.asarray(contract.names, dtype=object),
        bone_use_connect[order].astype(bool),
        bone_use_deform[order].astype(bool),
        skin_weights[:, order].astype(np.float32),
        contract,
    )

def bbox_diag(vertices):
    if len(vertices) == 0:
        return 0.0
    ext = vertices.max(axis=0) - vertices.min(axis=0)
    return float(np.linalg.norm(ext))

def farthest_frame_indices(features, count):
    total = int(features.shape[0])
    count = min(int(count), total)
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    center = features.mean(axis=0, keepdims=True)
    first = int(np.sum((features - center) ** 2, axis=1).argmax())
    chosen = [first]
    min_dist = np.sum((features - features[first:first + 1]) ** 2, axis=1)
    min_dist[first] = -np.inf
    for _ in range(1, count):
        nxt = int(min_dist.argmax())
        chosen.append(nxt)
        dist = np.sum((features - features[nxt:nxt + 1]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -np.inf
    return chosen

def select_motion_diverse_frame_numbers(start, end, sample_count, mesh_offsets):
    action_frames = int(round(end - start + 1))
    if action_frames == int(sample_count):
        frame_numbers = np.arange(int(round(start)), int(round(end)) + 1, dtype=np.int64)
        return frame_numbers, {
            "frame_sampling_policy": "all_available_action_frames",
            "candidate_frame_count": int(frame_numbers.shape[0]),
            "motion_fps_descriptor_vertices": 0,
        }
    candidate_count = min(action_frames, max(int(sample_count), int(sample_count) * 4))
    candidate_numbers = np.unique(np.linspace(start, end, candidate_count).round().astype(np.int64))
    if int(candidate_numbers.shape[0]) < int(sample_count):
        fail(
            "too_few_unique_candidate_frames",
            action_frames=int(action_frames),
            candidate_frame_count=int(candidate_numbers.shape[0]),
            sample_t=int(sample_count),
        )
    candidate_frames = np.stack([collect_frame_vertices(mesh_offsets, int(frame)) for frame in candidate_numbers], axis=0)
    vertex_count = int(candidate_frames.shape[1])
    descriptor_vertices = min(int(motion_fps_descriptor_vertices), vertex_count)
    if descriptor_vertices <= 0:
        fail("no_vertices_for_motion_fps", vertex_count=vertex_count)
    if descriptor_vertices == vertex_count:
        vertex_ids = np.arange(vertex_count, dtype=np.int64)
    else:
        vertex_ids = np.linspace(0, vertex_count - 1, descriptor_vertices).round().astype(np.int64)
    sampled = candidate_frames[:, vertex_ids].astype(np.float32)
    sampled = sampled - sampled.mean(axis=1, keepdims=True)
    features = sampled.reshape(sampled.shape[0], -1)
    chosen = farthest_frame_indices(features, int(sample_count))
    chosen_numbers = np.asarray([int(candidate_numbers[i]) for i in chosen], dtype=np.int64)
    frame_numbers = chosen_numbers[np.argsort(chosen_numbers)].astype(np.int64)
    return frame_numbers, {
        "frame_sampling_policy": "dense_candidate_then_mesh_motion_fps",
        "candidate_frame_count": int(candidate_numbers.shape[0]),
        "motion_fps_descriptor_vertices": int(descriptor_vertices),
    }

try:
    reset_scene()
    import_asset(source)
    candidates = summarize_armature_candidates()
    selection = select_main_armature()
    armature = selection["armature"]
    action = selection["action"]
    meshes = selection["meshes"]
    armature.animation_data_create()
    armature.animation_data.action = action

    ordered_bones = unirig_arranged_bones(armature)
    (
        parents,
        rest_joints,
        rest_tails,
        rest_tails_raw,
        rest_bone_mats,
        _,
        joint_names,
        bone_use_connect,
        bone_use_deform,
        bone_to_index,
    ) = collect_bone_data(armature, ordered_bones, [int(action.frame_range[0])])

    rest_vertices, faces, mesh_offsets = collect_rest_geometry(meshes)
    skin_weights, skin_coverage = collect_skin_weights(mesh_offsets, bone_to_index, len(ordered_bones))
    if min_vertices > 0 and rest_vertices.shape[0] < min_vertices:
        fail("too_few_selected_vertices", vertices=int(rest_vertices.shape[0]), min_vertices=int(min_vertices))
    if max_vertices > 0 and rest_vertices.shape[0] > max_vertices:
        fail("too_many_vertices_for_direct_export", status="needs_simplify", vertices=int(rest_vertices.shape[0]), max_vertices=max_vertices)
    if max_faces > 0 and faces.shape[0] > max_faces:
        fail("too_many_faces_for_direct_export", status="needs_simplify", faces=int(faces.shape[0]), max_faces=max_faces)

    start, end = action.frame_range
    action_frames = int(round(end - start + 1))
    if action_frames < sample_t:
        fail(
            "short_action",
            armature=armature.name,
            action=action.name,
            action_frames=action_frames,
            sample_t=sample_t,
            selected_action_sample_eligible=bool(selection["eligible_action"]),
            selected_bone_fcurves=int(selection["bone_fcurves"]),
            selected_animated_bones=int(selection["animated_bones"]),
            armature_candidates_first5=candidates[:5],
        )
    frame_numbers, frame_sampling_meta = select_motion_diverse_frame_numbers(start, end, sample_t, mesh_offsets)
    frame_vertices = np.stack([collect_frame_vertices(mesh_offsets, int(frame)) for frame in frame_numbers], axis=0)
    (
        parents,
        rest_joints,
        rest_tails,
        rest_tails_raw,
        rest_bone_mats,
        bone_transforms,
        joint_names,
        bone_use_connect,
        bone_use_deform,
        _,
    ) = collect_bone_data(armature, ordered_bones, frame_numbers)
    (
        parents,
        rest_joints,
        rest_tails,
        rest_tails_raw,
        rest_bone_mats,
        bone_transforms,
        joint_names,
        bone_use_connect,
        bone_use_deform,
        skin_weights,
        row_contract,
    ) = apply_row_skeleton_contract(
        parents,
        rest_joints,
        rest_tails,
        rest_tails_raw,
        rest_bone_mats,
        bone_transforms,
        joint_names,
        bone_use_connect,
        bone_use_deform,
        skin_weights,
    )
    if max_joints > 0 and parents.shape[0] > max_joints:
        fail(
            "too_many_joints",
            joints=int(parents.shape[0]),
            raw_joints=int(row_contract.raw_count),
            max_joints=max_joints,
        )

    rest_diag = bbox_diag(rest_vertices)
    if not np.isfinite(rest_vertices).all() or not np.isfinite(frame_vertices).all() or not np.isfinite(bone_transforms).all():
        fail("non_finite_export")
    if rest_diag <= 1e-8:
        fail("zero_size_rest_mesh")

    disp_from_first = np.linalg.norm(frame_vertices - frame_vertices[:1], axis=-1)
    motion_p50 = float(np.percentile(disp_from_first, 50))
    motion_p95 = float(np.percentile(disp_from_first, 95))
    motion_p99 = float(np.percentile(disp_from_first, 99))
    motion_p95_bbox = motion_p95 / max(rest_diag, 1e-8)
    if min_motion_p95_bbox > 0.0 and motion_p95_bbox < min_motion_p95_bbox:
        fail("low_vertex_motion", motion_p95=motion_p95, motion_p95_bbox=motion_p95_bbox, rest_bbox_diag=rest_diag)

    bbox_ratios = []
    for verts in frame_vertices:
        bbox_ratios.append(bbox_diag(verts) / rest_diag)
    bbox_ratios = np.asarray(bbox_ratios, dtype=np.float32)
    bbox_ratio_outside_hard_range = (
        (bbox_ratio_hard_min > 0.0 and float(bbox_ratios.min()) < bbox_ratio_hard_min)
        or (bbox_ratio_hard_max > 0.0 and float(bbox_ratios.max()) > bbox_ratio_hard_max)
    )
    if bbox_ratio_outside_hard_range:
        fail("bbox_explosion_or_collapse", bbox_ratio_min=float(bbox_ratios.min()), bbox_ratio_max=float(bbox_ratios.max()))

    meta = {
        "asset_id": asset_id,
        "source": source,
        "original_source": original_source,
        "status": "clean",
        "armature": armature.name,
        "action": action.name,
        "action_frames": int(action_frames),
        "selected_action_sample_eligible": bool(selection["eligible_action"]),
        "selected_meshes": [obj.name for obj in meshes],
        "raw_joint_count": int(row_contract.raw_count),
        "joint_count": int(parents.shape[0]),
        "row_contract_schema": "pass1_row_skeleton_v1",
        "row_contract_policy": "drop_unweighted_tail_end_leaf_rows",
        "row_contract_kept_raw_count": int(len(row_contract.kept_raw_indices)),
        "row_contract_dropped_raw_count": int(len(row_contract.dropped_raw_indices)),
        "row_contract_dropped_tail_leaf_count": int(len(row_contract.dropped_tail_leaf_indices)),
        "row_contract_dropped_raw_indices_first20": row_contract.dropped_raw_indices[:20],
        "row_contract_dropped_raw_names_first20": [str(ordered_bones[i].name) for i in row_contract.dropped_raw_indices[:20]],
        "row_contract_active_raw_count": int(len(row_contract.active_raw_indices)),
        "vertex_count": int(rest_vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "skin_coverage": skin_coverage,
        "frame_numbers": frame_numbers.tolist(),
        "sequence_frame_count": int(frame_numbers.shape[0]),
        **frame_sampling_meta,
        "motion_p50": motion_p50,
        "motion_p95": motion_p95,
        "motion_p99": motion_p99,
        "motion_p95_bbox": motion_p95_bbox,
        "min_motion_p95_bbox": float(min_motion_p95_bbox),
        "rest_bbox_diag": rest_diag,
        "bbox_ratio_min": float(bbox_ratios.min()),
        "bbox_ratio_max": float(bbox_ratios.max()),
        "bbox_ratio_pass1_warning": bool(float(bbox_ratios.min()) < 0.25 or float(bbox_ratios.max()) > 4.0),
        "bbox_ratio_hard_min": float(bbox_ratio_hard_min),
        "bbox_ratio_hard_max": float(bbox_ratio_hard_max),
        "bone_fcurves": selection["bone_fcurves"],
        "animated_bones": selection["animated_bones"],
        "bone_order_policy": "unirig_arranged_pose_bones",
        "tail_policy": "unirig_connect_tail_to_unique_son",
        "target_policy": "pass1_row_skeleton_contract",
        "connected_bone_count": int(np.asarray(bone_use_connect, dtype=bool).sum()),
        "deform_bone_count": int(np.asarray(bone_use_deform, dtype=bool).sum()),
        "armature_candidate_count": int(len([c for c in candidates if c["skinned_vertices"] > 0 and c["bone_fcurves"] > 0])),
        "armature_candidates_first5": candidates[:5],
    }
    meta.update(import_sanitize_meta)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        rest_vertices=rest_vertices,
        faces=faces,
        frame_vertices=frame_vertices.astype(np.float32),
        delta_vertices=(frame_vertices - frame_vertices[:1]).astype(np.float32),
        skin_weights=skin_weights,
        # Raw Blender-bone table.  These arrays stay aligned with
        # skin_weights/bone_transforms and are needed for LBS audits.
        parents=parents,
        rest_joints=rest_joints,
        rest_tails=rest_tails,
        rest_tails_raw=rest_tails_raw,
        rest_bone_mats=rest_bone_mats,
        bone_transforms=bone_transforms,
        bone_use_connect=bone_use_connect,
        bone_use_deform=bone_use_deform,
        bone_parents=parents,
        bone_heads=rest_joints,
        bone_tails=rest_tails,
        bone_tails_raw=rest_tails_raw,
        row_contract_kept_raw_indices=np.asarray(row_contract.kept_raw_indices, dtype=np.int64),
        row_contract_dropped_raw_indices=np.asarray(row_contract.dropped_raw_indices, dtype=np.int64),
        row_contract_active_raw_indices=np.asarray(row_contract.active_raw_indices, dtype=np.int64),
        row_contract_schema_version=np.asarray("pass1_row_skeleton_v1", dtype=object),
        frame_numbers=frame_numbers,
        joint_names=joint_names,
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    if out_json:
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
    print("RIGWEAVE_EXPORT_JSON " + json.dumps(meta, ensure_ascii=False))
except SystemExit:
    raise
except Exception as exc:
    traceback.print_exc()
    fail("export_exception", error=repr(exc))
'''
    return (
        template.replace("__REPO_SRC__", repr(str(repo_src)))
        .replace("__SOURCE__", repr(str(import_source)))
        .replace("__ORIGINAL_SOURCE__", repr(str(args.source)))
        .replace("__IMPORT_SANITIZE_META__", repr(import_sanitize_meta))
        .replace("__ASSET_ID__", repr(args.asset_id))
        .replace("__OUT_NPZ__", repr(str(args.out_npz)))
        .replace("__OUT_JSON__", repr(str(args.out_json)) if args.out_json else "None")
        .replace("__FRAMES__", str(args.frames))
        .replace("__MIN_VERTICES__", str(args.min_vertices))
        .replace("__MAX_JOINTS__", str(args.max_joints))
        .replace("__MAX_VERTICES__", str(args.max_vertices))
        .replace("__MAX_FACES__", str(args.max_faces))
        .replace("__BBOX_RATIO_HARD_MIN__", repr(float(args.bbox_ratio_hard_min)))
        .replace("__BBOX_RATIO_HARD_MAX__", repr(float(args.bbox_ratio_hard_max)))
        .replace("__MIN_MOTION_P95_BBOX__", repr(float(args.min_motion_p95_bbox)))
        .replace("__TOPK_WEIGHTS__", str(args.topk_weights))
        .replace("__ACTIVE_SKIN_THRESHOLD__", repr(float(args.active_skin_threshold)))
        .replace("__MOTION_FPS_DESCRIPTOR_VERTICES__", str(int(args.motion_fps_descriptor_vertices)))
    )


def main() -> None:
    args = parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)

    def write_wrapper_reject(reason: str, **extra: object) -> None:
        if not args.out_json:
            return
        record = {
            "asset_id": args.asset_id,
            "source": str(args.source),
            "status": "reject",
            "reject_reasons": [reason],
        }
        record.update(extra)
        with args.out_json.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)

    with tempfile.TemporaryDirectory(prefix="rigweave_export_") as tmp_name:
        tmp_dir = Path(tmp_name)
        import_source, import_sanitize_meta = sanitize_glb_import_source(args.source, tmp_dir)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(blender_export_script(args, import_source, import_sanitize_meta))
            script = Path(handle.name)
        try:
            cmd = [args.blender, "--background"]
            if int(args.blender_threads) > 0:
                cmd.extend(["--threads", str(int(args.blender_threads))])
            cmd.extend(["--python", str(script)])
            env = os.environ.copy()
            if int(args.blender_threads) > 0:
                env.setdefault("OMP_NUM_THREADS", str(int(args.blender_threads)))
                env.setdefault("OPENBLAS_NUM_THREADS", str(int(args.blender_threads)))
                env.setdefault("MKL_NUM_THREADS", str(int(args.blender_threads)))
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_sec,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            write_wrapper_reject("blender_export_timeout", timeout_sec=int(args.timeout_sec))
            raise SystemExit(f"Blender export timed out after {args.timeout_sec}s")
        finally:
            script.unlink(missing_ok=True)

    print(proc.stdout)
    if proc.returncode not in (0,):
        if args.out_json is not None and not args.out_json.exists():
            write_wrapper_reject("blender_export_failed_no_json", returncode=int(proc.returncode))
        raise SystemExit(proc.returncode)
    if not args.out_npz.exists() or (args.out_json is not None and not args.out_json.exists()):
        write_wrapper_reject(
            "blender_export_missing_output",
            has_npz=bool(args.out_npz.exists()),
            has_json=bool(args.out_json is not None and args.out_json.exists()),
        )
        raise SystemExit("Blender export finished without writing expected npz/json")


if __name__ == "__main__":
    main()
