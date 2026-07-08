#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

BLENDER_CODE = r'''
import argparse
import json
import math
import os
import sys
import traceback

import bpy
import numpy as np


def parse_blender_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--armature", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--selected-mesh", action="append", default=[])
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
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
        raise RuntimeError("unsupported source type: " + suffix)
    bpy.context.view_layer.update()


def arr3(v):
    return np.asarray([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)


def bbox(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return {"count": 0, "diag": 0.0, "min": [], "max": [], "span": []}
    pts = pts.reshape(-1, 3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] == 0:
        return {"count": 0, "diag": 0.0, "min": [], "max": [], "span": []}
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    span = hi - lo
    return {"count": int(finite.shape[0]), "diag": float(np.linalg.norm(span)), "min": lo.tolist(), "max": hi.tolist(), "span": span.tolist()}


def mesh_vertices_world(meshes, evaluated=False):
    pts = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        if evaluated:
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            try:
                for v in eval_mesh.vertices:
                    co = eval_obj.matrix_world @ v.co
                    pts.append([co.x, co.y, co.z])
            finally:
                eval_obj.to_mesh_clear()
        else:
            for v in obj.data.vertices:
                co = obj.matrix_world @ v.co
                pts.append([co.x, co.y, co.z])
    return np.asarray(pts, dtype=np.float64)


def mesh_vertices_local(meshes):
    pts = []
    for obj in meshes:
        for v in obj.data.vertices:
            pts.append([float(v.co.x), float(v.co.y), float(v.co.z)])
    return np.asarray(pts, dtype=np.float64)


def armature_meshes(meshes, armatures, armature):
    out = []
    for obj in meshes:
        for mod in obj.modifiers:
            if mod.type != "ARMATURE":
                continue
            if mod.object == armature or (mod.object is None and len(armatures) == 1):
                out.append(obj)
                break
    return sorted(out, key=lambda obj: obj.name)


def choose_armature(name):
    arms = sorted([obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"], key=lambda o: o.name)
    if not arms:
        raise RuntimeError("no armatures")
    if name:
        for arm in arms:
            if arm.name == name:
                return arm
    return arms[0]


def bone_world_arrays(arm):
    bones = list(arm.data.bones)
    heads = []
    tails = []
    matrix_trans = []
    pose_heads = []
    pose_tails = []
    names = []
    parents = []
    use_connect = []
    use_deform = []
    name_to_idx = {b.name: i for i, b in enumerate(bones)}
    for b in bones:
        names.append(b.name)
        parents.append(-1 if b.parent is None else name_to_idx.get(b.parent.name, -2))
        use_connect.append(bool(b.use_connect))
        use_deform.append(bool(b.use_deform))
        h = arm.matrix_world @ b.head_local
        t = arm.matrix_world @ b.tail_local
        heads.append([h.x, h.y, h.z])
        tails.append([t.x, t.y, t.z])
        mt = arm.matrix_world @ b.matrix_local
        matrix_trans.append([float(mt[0][3]), float(mt[1][3]), float(mt[2][3])])
        pb = arm.pose.bones.get(b.name)
        if pb is not None:
            ph = arm.matrix_world @ pb.head
            pt = arm.matrix_world @ pb.tail
            pose_heads.append([ph.x, ph.y, ph.z])
            pose_tails.append([pt.x, pt.y, pt.z])
        else:
            pose_heads.append([math.nan, math.nan, math.nan])
            pose_tails.append([math.nan, math.nan, math.nan])
    return {
        "names": names,
        "parents": parents,
        "use_connect": use_connect,
        "use_deform": use_deform,
        "heads": np.asarray(heads, dtype=np.float64),
        "tails": np.asarray(tails, dtype=np.float64),
        "matrix_trans": np.asarray(matrix_trans, dtype=np.float64),
        "pose_heads": np.asarray(pose_heads, dtype=np.float64),
        "pose_tails": np.asarray(pose_tails, dtype=np.float64),
    }


def bone_transform_arrays(arm):
    bones = list(arm.data.bones)
    world = []
    local = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_arm = arm.evaluated_get(depsgraph)
    for b in bones:
        pb = eval_arm.pose.bones.get(b.name)
        if pb is None:
            continue
        rest_world = arm.matrix_world @ b.matrix_local
        pose_world = eval_arm.matrix_world @ pb.matrix
        world.append(np.asarray(pose_world @ rest_world.inverted(), dtype=np.float64))
        local.append(np.asarray(pb.matrix @ b.matrix_local.inverted(), dtype=np.float64))
    return np.asarray(world, dtype=np.float64), np.asarray(local, dtype=np.float64)


def skin_active_by_bone(meshes, names):
    counts = np.zeros((len(names),), dtype=np.int64)
    weights = np.zeros((len(names),), dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(names)}
    for obj in meshes:
        group_names = {g.index: g.name for g in obj.vertex_groups}
        for v in obj.data.vertices:
            for g in v.groups:
                n = group_names.get(g.group)
                i = name_to_idx.get(n)
                if i is not None and float(g.weight) > 1e-8:
                    counts[i] += 1
                    weights[i] += float(g.weight)
    return counts, weights


def skin_matrix(meshes, names):
    total = sum(len(obj.data.vertices) for obj in meshes)
    out = np.zeros((total, len(names)), dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(names)}
    offset = 0
    for obj in meshes:
        group_names = {g.index: g.name for g in obj.vertex_groups}
        for v in obj.data.vertices:
            for g in v.groups:
                n = group_names.get(g.group)
                i = name_to_idx.get(n)
                if i is not None:
                    out[offset + v.index, i] += float(g.weight)
        offset += len(obj.data.vertices)
    sums = out.sum(axis=1, keepdims=True)
    return out / np.maximum(sums, 1.0e-8)


def lbs_error(rest_vertices, frame_vertices, skin, transforms):
    if rest_vertices.shape[0] == 0 or transforms.shape[0] == 0:
        return {"max": 0.0, "p95": 0.0}
    homog = np.concatenate([rest_vertices.astype(np.float64), np.ones((rest_vertices.shape[0], 1), dtype=np.float64)], axis=1)
    posed = np.einsum("jbc,vc->vjb", transforms.astype(np.float64), homog, optimize=True)[..., :3]
    pred = np.einsum("vj,vjb->vb", skin.astype(np.float64), posed, optimize=True)
    err = np.linalg.norm(pred - frame_vertices.astype(np.float64), axis=1)
    return {
        "max": float(err.max()) if err.size else 0.0,
        "p95": float(np.percentile(err, 95)) if err.size else 0.0,
        "bbox_p95": float(np.percentile(err, 95) / max(bbox(frame_vertices)["diag"], 1.0e-12)) if err.size else 0.0,
    }


def main():
    args = parse_blender_args()
    reset_scene()
    import_asset(args.source)
    arms = sorted([obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"], key=lambda o: o.name)
    meshes_all = sorted([obj for obj in bpy.context.scene.objects if obj.type == "MESH"], key=lambda o: o.name)
    arm = choose_armature(args.armature)
    if args.action:
        for action in bpy.data.actions:
            if action.name == args.action:
                arm.animation_data_create()
                arm.animation_data.action = action
                break
    if args.frame is not None:
        bpy.context.scene.frame_set(int(args.frame))
        bpy.context.view_layer.update()
    selected = [m for m in meshes_all if not args.selected_mesh or m.name in set(args.selected_mesh)]
    if not selected:
        selected = armature_meshes(meshes_all, arms, arm)
    data = bone_world_arrays(arm)
    heads = data["heads"]
    tails = data["tails"]
    pose_heads = data["pose_heads"]
    pose_tails = data["pose_tails"]
    endpoints = np.concatenate([heads, tails], axis=0)
    skin_counts, skin_sums = skin_active_by_bone(selected, data["names"])
    active = skin_sums > 1e-8
    # parent tail to child head distances in world coordinates.
    parent_tail_child_head = []
    parent_head_child_head = []
    child_head_parent_tail = []
    for i, p in enumerate(data["parents"]):
        if p >= 0:
            parent_tail_child_head.append(float(np.linalg.norm(tails[p] - heads[i])))
            parent_head_child_head.append(float(np.linalg.norm(heads[p] - heads[i])))
            child_head_parent_tail.append({"child": data["names"][i], "parent": data["names"][p], "dist": float(np.linalg.norm(tails[p] - heads[i])), "use_connect": bool(data["use_connect"][i])})
    mesh_rest = mesh_vertices_world(selected, evaluated=False)
    mesh_eval = mesh_vertices_world(selected, evaluated=True)
    mesh_local = mesh_vertices_local(selected)
    mesh_eval_local = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in selected:
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        try:
            for v in eval_mesh.vertices:
                mesh_eval_local.append([float(v.co.x), float(v.co.y), float(v.co.z)])
        finally:
            eval_obj.to_mesh_clear()
    mesh_eval_local = np.asarray(mesh_eval_local, dtype=np.float64)
    skin = skin_matrix(selected, data["names"])
    world_transforms, local_transforms = bone_transform_arrays(arm)
    mesh_summaries = []
    for obj in selected:
        local = np.asarray([[float(v.co.x), float(v.co.y), float(v.co.z)] for v in obj.data.vertices], dtype=np.float64)
        world = mesh_vertices_world([obj], evaluated=False)
        mesh_summaries.append(
            {
                "name": obj.name,
                "vertex_count": int(len(obj.data.vertices)),
                "matrix_world": [[float(obj.matrix_world[r][c]) for c in range(4)] for r in range(4)],
                "local_bbox": bbox(local),
                "world_bbox": bbox(world),
                "modifiers": [
                    {
                        "name": mod.name,
                        "type": mod.type,
                        "object": "" if getattr(mod, "object", None) is None else mod.object.name,
                    }
                    for mod in obj.modifiers
                ],
            }
        )
    out = {
        "source": args.source,
        "armature": arm.name,
        "armature_matrix_world": [[float(arm.matrix_world[r][c]) for c in range(4)] for r in range(4)],
        "armature_count": len(arms),
        "mesh_count_all": len(meshes_all),
        "selected_meshes": [m.name for m in selected],
        "bone_count": len(data["names"]),
        "root_count": int(sum(1 for p in data["parents"] if p < 0)),
        "use_connect_count": int(sum(data["use_connect"])),
        "use_deform_count": int(sum(data["use_deform"])),
        "active_skin_bone_count": int(active.sum()),
        "mesh_rest_bbox": bbox(mesh_rest),
        "mesh_eval_bbox": bbox(mesh_eval),
        "mesh_local_bbox": bbox(mesh_local),
        "mesh_eval_local_bbox": bbox(mesh_eval_local),
        "lbs_error_world": lbs_error(mesh_rest, mesh_eval, skin, world_transforms),
        "lbs_error_local": lbs_error(mesh_local, mesh_eval_local, skin, local_transforms),
        "mesh_object_summaries": mesh_summaries,
        "head_bbox": bbox(heads),
        "tail_bbox": bbox(tails),
        "endpoint_bbox": bbox(endpoints),
        "pose_head_bbox": bbox(pose_heads),
        "pose_tail_bbox": bbox(pose_tails),
        "matrix_translation_bbox": bbox(data["matrix_trans"]),
        "active_head_bbox": bbox(heads[active]) if active.any() else bbox(np.zeros((0,3))),
        "active_tail_bbox": bbox(tails[active]) if active.any() else bbox(np.zeros((0,3))),
        "active_endpoint_bbox": bbox(np.concatenate([heads[active], tails[active]], axis=0)) if active.any() else bbox(np.zeros((0,3))),
        "parent_tail_child_head_stats": {
            "count": len(parent_tail_child_head),
            "p50": float(np.percentile(parent_tail_child_head, 50)) if parent_tail_child_head else 0.0,
            "p95": float(np.percentile(parent_tail_child_head, 95)) if parent_tail_child_head else 0.0,
            "max": float(max(parent_tail_child_head)) if parent_tail_child_head else 0.0,
        },
        "parent_head_child_head_stats": {
            "count": len(parent_head_child_head),
            "p50": float(np.percentile(parent_head_child_head, 50)) if parent_head_child_head else 0.0,
            "p95": float(np.percentile(parent_head_child_head, 95)) if parent_head_child_head else 0.0,
            "max": float(max(parent_head_child_head)) if parent_head_child_head else 0.0,
        },
        "worst_parent_tail_child_head_first20": sorted(child_head_parent_tail, key=lambda x: x["dist"], reverse=True)[:20],
        "bones_first80": [
            {
                "i": i,
                "name": data["names"][i],
                "parent": int(data["parents"][i]),
                "use_connect": bool(data["use_connect"][i]),
                "use_deform": bool(data["use_deform"][i]),
                "skin_vertex_count": int(skin_counts[i]),
                "skin_weight_sum": float(skin_sums[i]),
                "head": heads[i].tolist(),
                "tail": tails[i].tolist(),
                "pose_head": pose_heads[i].tolist(),
                "pose_tail": pose_tails[i].tolist(),
                "matrix_translation": data["matrix_trans"][i].tolist(),
            }
            for i in range(min(80, len(data["names"])))
        ],
    }
    mesh_diag = max(out["mesh_rest_bbox"]["diag"], 1e-12)
    for key in ["head_bbox", "tail_bbox", "endpoint_bbox", "active_head_bbox", "active_tail_bbox", "active_endpoint_bbox", "pose_head_bbox", "pose_tail_bbox", "matrix_translation_bbox"]:
        out[key + "_to_mesh"] = float(out[key]["diag"] / mesh_diag)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

try:
    main()
except Exception as exc:
    traceback.print_exc()
    with open(parse_blender_args().out_json, "w", encoding="utf-8") as f:
        json.dump({"error": repr(exc)}, f, indent=2)
    sys.exit(2)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--armature", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--selected-mesh", action="append", default=[])
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "probe.py"
        script.write_text(BLENDER_CODE, encoding="utf-8")
        cmd = [args.blender, "--background", "--python", str(script), "--", "--source", str(args.source), "--out-json", str(args.out_json)]
        if args.armature:
            cmd += ["--armature", args.armature]
        if args.action:
            cmd += ["--action", args.action]
        if args.frame is not None:
            cmd += ["--frame", str(int(args.frame))]
        for mesh in args.selected_mesh:
            cmd += ["--selected-mesh", mesh]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        log = args.out_json.with_suffix(".log")
        log.write_text(proc.stdout, encoding="utf-8")
        return proc.returncode

if __name__ == "__main__":
    raise SystemExit(main())
