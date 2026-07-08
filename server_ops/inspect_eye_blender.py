import json
import sys
from collections import Counter, defaultdict

import bpy


def mat_rows(mat):
    return [[float(mat[r][c]) for c in range(4)] for r in range(4)]


def vec3(v):
    return [float(v.x), float(v.y), float(v.z)]


def group_weight_counts(obj):
    group_names = {g.index: g.name for g in obj.vertex_groups}
    counts = Counter()
    sums = defaultdict(float)
    maxw = defaultdict(float)
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            name = group_names.get(group.group, "")
            counts[name] += 1
            sums[name] += float(group.weight)
            maxw[name] = max(maxw[name], float(group.weight))
    rows = []
    for name, count in counts.most_common():
        rows.append(
            {
                "name": name,
                "count": int(count),
                "weight_sum": float(sums[name]),
                "max_weight": float(maxw[name]),
            }
        )
    return rows


def object_bbox_world(obj, evaluated=False):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph) if evaluated else obj
    mesh = eval_obj.to_mesh() if evaluated else obj.data
    try:
        pts = [eval_obj.matrix_world @ v.co for v in mesh.vertices]
        if not pts:
            return None
        mn = [min(float(p[i]) for p in pts) for i in range(3)]
        mx = [max(float(p[i]) for p in pts) for i in range(3)]
        return {"min": mn, "max": mx}
    finally:
        if evaluated:
            eval_obj.to_mesh_clear()


def main():
    argv = sys.argv
    src = argv[argv.index("--") + 1]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=src)

    arm = bpy.data.objects.get("Body_skeleton")
    actions = sorted(bpy.data.actions, key=lambda a: a.name)
    action = bpy.data.actions.get("Body_skeleton|mixamo.com|Layer0") or (actions[0] if actions else None)
    if arm is not None and action is not None:
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = action

    out = {
        "source": src,
        "armature": None if arm is None else {
            "name": arm.name,
            "matrix_world": mat_rows(arm.matrix_world),
            "scale": [float(x) for x in arm.scale],
            "rotation_mode": arm.rotation_mode,
            "action": None if action is None else action.name,
            "action_frame_range": None if action is None else [float(action.frame_range[0]), float(action.frame_range[1])],
        },
        "objects": [],
        "bones": {},
    }

    if arm is not None:
        for name in ["Eye.L", "Eye.R", "ValveBiped.Bip01_Head"]:
            bone = arm.data.bones.get(name)
            pbone = arm.pose.bones.get(name)
            if bone is None:
                continue
            out["bones"][name] = {
                "head_local": vec3(bone.head_local),
                "tail_local": vec3(bone.tail_local),
                "matrix_local": mat_rows(bone.matrix_local),
                "use_deform": bool(bone.use_deform),
                "parent": None if bone.parent is None else bone.parent.name,
                "pose_matrix_frame0": None if pbone is None else mat_rows(pbone.matrix),
            }

    for name in ["Eye", "EyeG", "Body", "Endo"]:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        item = {
            "name": obj.name,
            "type": obj.type,
            "vertex_count": len(obj.data.vertices) if obj.type == "MESH" else None,
            "matrix_world": mat_rows(obj.matrix_world),
            "scale": [float(x) for x in obj.scale],
            "parent": None if obj.parent is None else obj.parent.name,
            "parent_type": obj.parent_type,
            "modifiers": [],
            "top_vertex_groups": group_weight_counts(obj)[:20] if obj.type == "MESH" else [],
            "rest_bbox_world": object_bbox_world(obj, evaluated=False) if obj.type == "MESH" else None,
            "evaluated_bbox_by_frame": {},
        }
        for mod in obj.modifiers:
            md = {
                "name": mod.name,
                "type": mod.type,
                "show_viewport": bool(mod.show_viewport),
            }
            if mod.type == "ARMATURE":
                md.update(
                    {
                        "object": None if mod.object is None else mod.object.name,
                        "use_vertex_groups": bool(mod.use_vertex_groups),
                        "use_bone_envelopes": bool(mod.use_bone_envelopes),
                        "vertex_group": str(mod.vertex_group),
                        "invert_vertex_group": bool(mod.invert_vertex_group),
                        "use_deform_preserve_volume": bool(mod.use_deform_preserve_volume),
                    }
                )
            item["modifiers"].append(md)
        if obj.type == "MESH":
            for frame in [10, 45, 311]:
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()
                item["evaluated_bbox_by_frame"][str(frame)] = object_bbox_world(obj, evaluated=True)
        out["objects"].append(item)

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
