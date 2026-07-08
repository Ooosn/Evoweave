#!/usr/bin/env python3
"""Audit TexVerse animated assets for RigWeave data usability.

The goal is not to decide whether an FBX is visually nice.  It is to filter for
assets that can produce supervised dynamic rigged-mesh sequences:

  mesh geometry + armature hierarchy + vertex skin weights + bone animation.

The script samples/downloads TexVerse zip files, imports candidate files with
Blender in background mode, and writes one JSONL record per imported asset with
metrics and reject reasons.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from texverse_archive_utils import expand_nested_archives, find_import_candidates, safe_extract_zip


TEXVERSE_REPO = "YiboZhang2001/TexVerse-Skeleton-Animation"
TEXVERSE_URL = f"https://huggingface.co/datasets/{TEXVERSE_REPO}/resolve/main"


@dataclass
class ZipRecord:
    asset_id: str
    zip_path: str
    source: str
    status: str
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-paths", type=Path, required=True)
    parser.add_argument("--animation-ids", type=Path)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument("--blender-threads", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-expand-nested-archives", action="store_true")
    parser.add_argument("--nested-archive-depth", type=int, default=2)
    parser.add_argument("--max-nested-archives", type=int, default=12)
    parser.add_argument("--max-nested-archive-mb", type=float, default=0.0)
    parser.add_argument("--nested-archive-timeout-sec", type=int, default=120)
    parser.add_argument("--target-usable", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-candidates-per-zip", type=int, default=4)
    parser.add_argument("--min-bones", type=int, default=4)
    parser.add_argument("--min-skinned-vertices", type=int, default=100)
    parser.add_argument("--min-skin-coverage", type=float, default=0.0)
    parser.add_argument("--min-action-frames", type=int, default=16)
    parser.add_argument("--min-bone-fcurves", type=int, default=0)
    parser.add_argument("--flat-ratio", type=float, default=0.015)
    return parser.parse_args()


def read_model_paths(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        asset_id = Path(rel).stem
        mapping[asset_id] = rel
    return mapping


def read_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def choose_ids(args: argparse.Namespace, model_paths: dict[str, str]) -> list[str]:
    if args.ids_file:
        ids = read_ids(args.ids_file)
        if args.ids:
            ids.extend(args.ids)
    elif args.ids:
        ids = args.ids
    else:
        pool = read_ids(args.animation_ids)
        if not pool:
            pool = list(model_paths)
        pool = [asset_id for asset_id in pool if asset_id in model_paths]
        rng = random.Random(args.seed)
        if args.sample_size > 0:
            ids = rng.sample(pool, min(args.sample_size, len(pool)))
        else:
            ids = pool
    missing = [asset_id for asset_id in ids if asset_id not in model_paths]
    if missing:
        raise SystemExit(f"Missing {len(missing)} ids in model_paths, first={missing[:5]}")
    return ids


def download_zip(asset_id: str, rel_path: str, out_dir: Path, redownload: bool) -> ZipRecord:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{asset_id}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0 and not redownload:
        if zipfile.is_zipfile(zip_path):
            return ZipRecord(asset_id, str(zip_path), "cache", "ok")
        zip_path.unlink(missing_ok=True)
    url = f"{TEXVERSE_URL}/{urllib.parse.quote(rel_path)}"
    tmp_path = zip_path.with_suffix(".zip.tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if not zipfile.is_zipfile(tmp_path):
            tmp_path.unlink(missing_ok=True)
            return ZipRecord(asset_id, str(zip_path), "download", "error", "downloaded_file_is_not_a_zip")
        tmp_path.replace(zip_path)
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        return ZipRecord(asset_id, str(zip_path), "download", "error", str(exc))
    return ZipRecord(asset_id, str(zip_path), "download", "ok")


def blender_audit_script(
    candidate: Path,
    asset_id: str,
    thresholds: dict[str, float | int],
) -> str:
    template = r"""
import json
import math
import os
import sys
import traceback

import bpy
from mathutils import Vector

asset_id = __ASSET_ID__
candidate = __CANDIDATE__
thresholds = __THRESHOLDS__

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
        raise RuntimeError("unsupported file type: " + suffix)

def mesh_bbox(meshes):
    points = []
    for obj in meshes:
        for vertex in obj.data.vertices:
            points.append(obj.matrix_world @ vertex.co)
    if not points:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    mins = [min(getattr(p, axis) for p in points) for axis in ("x", "y", "z")]
    maxs = [max(getattr(p, axis) for p in points) for axis in ("x", "y", "z")]
    extents = [maxs[i] - mins[i] for i in range(3)]
    return mins, maxs, extents

def is_armature_mesh(obj):
    if len(obj.vertex_groups) == 0:
        return False
    return any(mod.type == "ARMATURE" for mod in obj.modifiers)

try:
    reset_scene()
    import_asset(candidate)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    skinned_meshes = [obj for obj in meshes if is_armature_mesh(obj)]
    weighted_meshes = [obj for obj in meshes if len(obj.vertex_groups) > 0]

    skinned_vertex_count = sum(len(obj.data.vertices) for obj in skinned_meshes)
    weighted_vertex_count = sum(len(obj.data.vertices) for obj in weighted_meshes)
    total_vertex_count = sum(len(obj.data.vertices) for obj in meshes)
    total_face_count = sum(len(obj.data.polygons) for obj in meshes)

    vertices_with_groups = 0
    group_assignments = 0
    for obj in weighted_meshes:
        for vertex in obj.data.vertices:
            if vertex.groups:
                vertices_with_groups += 1
                group_assignments += len(vertex.groups)
    skin_coverage = vertices_with_groups / max(1, weighted_vertex_count)

    _, _, all_extents = mesh_bbox(meshes)
    _, _, skin_extents = mesh_bbox(skinned_meshes if skinned_meshes else weighted_meshes)
    sorted_extents = sorted([float(v) for v in skin_extents])
    flatness = sorted_extents[0] / max(sorted_extents[-1], 1e-8)

    actions = list(bpy.data.actions)
    action_summaries = []
    max_action_frames = 0
    max_bone_fcurves = 0
    for action in actions:
        start, end = action.frame_range
        frames = max(0, int(round(end - start + 1)))
        bone_fcurves = sum(1 for curve in action.fcurves if "pose.bones" in curve.data_path)
        max_action_frames = max(max_action_frames, frames)
        max_bone_fcurves = max(max_bone_fcurves, bone_fcurves)
        action_summaries.append({
            "name": action.name,
            "start": float(start),
            "end": float(end),
            "frames": frames,
            "fcurves": len(action.fcurves),
            "bone_fcurves": bone_fcurves,
        })

    bone_counts = [len(obj.data.bones) for obj in armatures]
    max_bones = max(bone_counts or [0])
    lower_name = os.path.basename(candidate).lower()
    path_lower = candidate.lower()

    reject_reasons = []
    if "2d" in lower_name or "background" in path_lower:
        reject_reasons.append("filename_or_path_suggests_2d_background")
    if not meshes:
        reject_reasons.append("no_mesh")
    if not armatures:
        reject_reasons.append("no_armature")
    elif max_bones < thresholds["min_bones"]:
        reject_reasons.append("too_few_bones")
    if skinned_vertex_count < thresholds["min_skinned_vertices"]:
        reject_reasons.append("too_few_skinned_vertices")
    if thresholds["min_skin_coverage"] > 0 and skin_coverage < thresholds["min_skin_coverage"]:
        reject_reasons.append("low_skin_coverage")
    if max_action_frames < thresholds["min_action_frames"]:
        reject_reasons.append("short_or_missing_action")
    if thresholds["min_bone_fcurves"] > 0 and max_bone_fcurves < thresholds["min_bone_fcurves"]:
        reject_reasons.append("action_not_on_bones")
    if flatness < thresholds["flat_ratio"]:
        reject_reasons.append("flat_or_card_like_geometry")

    record = {
        "asset_id": asset_id,
        "candidate": candidate,
        "status": "ok",
        "usable": len(reject_reasons) == 0,
        "reject_reasons": reject_reasons,
        "mesh_count": len(meshes),
        "skinned_mesh_count": len(skinned_meshes),
        "weighted_mesh_count": len(weighted_meshes),
        "armature_count": len(armatures),
        "action_count": len(actions),
        "bone_counts": bone_counts,
        "max_bones": max_bones,
        "total_vertices": total_vertex_count,
        "total_faces": total_face_count,
        "skinned_vertices": skinned_vertex_count,
        "weighted_vertices": weighted_vertex_count,
        "vertices_with_groups": vertices_with_groups,
        "group_assignments": group_assignments,
        "skin_coverage": skin_coverage,
        "all_bbox_extents": all_extents,
        "skin_bbox_extents": skin_extents,
        "flatness": flatness,
        "max_action_frames": max_action_frames,
        "max_bone_fcurves": max_bone_fcurves,
        "actions": action_summaries[:16],
        "mesh_summaries": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "faces": len(obj.data.polygons),
                "vertex_groups": len(obj.vertex_groups),
                "has_armature_modifier": is_armature_mesh(obj),
            }
            for obj in meshes[:32]
        ],
    }
except Exception as exc:
    record = {
        "asset_id": asset_id,
        "candidate": candidate,
        "status": "error",
        "usable": False,
        "reject_reasons": ["import_error"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }

print("RIGWEAVE_AUDIT_JSON " + json.dumps(record, ensure_ascii=False))
"""
    return (
        template.replace("__ASSET_ID__", repr(asset_id))
        .replace("__CANDIDATE__", repr(str(candidate)))
        .replace("__THRESHOLDS__", json.dumps(thresholds))
    )


def run_blender_audit(
    blender: str,
    candidate: Path,
    asset_id: str,
    args: argparse.Namespace,
) -> dict:
    thresholds: dict[str, float | int] = {
        "min_bones": args.min_bones,
        "min_skinned_vertices": args.min_skinned_vertices,
        "min_skin_coverage": args.min_skin_coverage,
        "min_action_frames": args.min_action_frames,
        "min_bone_fcurves": args.min_bone_fcurves,
        "flat_ratio": args.flat_ratio,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(blender_audit_script(candidate, asset_id, thresholds))
        script_path = handle.name
    try:
        cmd = [blender, "--background"]
        if args.blender_threads > 0:
            cmd.extend(["--threads", str(args.blender_threads)])
        cmd.extend(["--python", script_path])
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "asset_id": asset_id,
            "candidate": str(candidate),
            "status": "timeout",
            "usable": False,
            "reject_reasons": ["blender_timeout"],
        }
    finally:
        Path(script_path).unlink(missing_ok=True)

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RIGWEAVE_AUDIT_JSON "):
            return json.loads(line[len("RIGWEAVE_AUDIT_JSON ") :])
    return {
        "asset_id": asset_id,
        "candidate": str(candidate),
        "status": "error",
        "usable": False,
        "reject_reasons": ["no_audit_output"],
        "returncode": proc.returncode,
        "blender_output_tail": proc.stdout.splitlines()[-80:],
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_existing_audit(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0
    seen: set[str] = set()
    usable_assets: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            asset_id = row.get("asset_id")
            if asset_id:
                asset_id = str(asset_id)
                transient = (
                    (row.get("source") == "download" and row.get("status") == "error")
                    or "zip_or_extract_error" in (row.get("reject_reasons") or [])
                )
                if not transient:
                    seen.add(asset_id)
                if row.get("usable") is True:
                    usable_assets.add(asset_id)
    return seen, len(usable_assets)


def audit_one_asset(asset_id: str, rel_path: str, args: argparse.Namespace) -> tuple[str, list[dict], bool]:
    records: list[dict] = []
    zip_record = download_zip(asset_id, rel_path, args.download_dir, args.redownload)
    if zip_record.status != "ok":
        records.append(asdict(zip_record) | {"usable": False})
        return asset_id, records, False

    extract_dir = args.extract_root / asset_id
    asset_usable = False
    try:
        safe_extract_zip(Path(zip_record.zip_path), extract_dir)
        nested_archives = expand_nested_archives(
            extract_dir,
            no_expand=args.no_expand_nested_archives,
            max_depth=args.nested_archive_depth,
            max_archives=args.max_nested_archives,
            max_archive_mb=args.max_nested_archive_mb,
            timeout_sec=args.nested_archive_timeout_sec,
        )
        candidates, candidate_count = find_import_candidates(extract_dir, args.max_candidates_per_zip)
        if not candidates:
            records.append(
                asdict(zip_record)
                | {
                    "usable": False,
                    "reject_reasons": ["no_importable_3d_file"],
                    "candidate_count": candidate_count,
                    "candidates_imported": 0,
                    "nested_archives": nested_archives,
                }
            )
            return asset_id, records, False
        for candidate in candidates:
            record = run_blender_audit(args.blender, candidate, asset_id, args)
            record["zip_path"] = zip_record.zip_path
            record["texverse_rel_path"] = rel_path
            record["candidate_count"] = candidate_count
            record["candidates_imported"] = len(candidates)
            if nested_archives:
                record["nested_archives"] = nested_archives
            records.append(record)
            if record.get("usable") is True:
                asset_usable = True
        return asset_id, records, asset_usable
    except Exception as exc:  # noqa: BLE001
        records.append(
            asdict(zip_record)
            | {"usable": False, "reject_reasons": ["zip_or_extract_error"], "error": repr(exc)}
        )
        return asset_id, records, False
    finally:
        if not args.keep_extracted:
            shutil.rmtree(extract_dir, ignore_errors=True)


def print_progress(processed: int, total: int, usable_count: int, output_jsonl: Path) -> None:
    print(
        json.dumps(
            {
                "event": "pass0_progress",
                "processed_assets": processed,
                "remaining_assets": max(0, total - processed),
                "usable_assets": usable_count,
                "audit_jsonl": str(output_jsonl),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    model_paths = read_model_paths(args.model_paths)
    asset_ids = choose_ids(args, model_paths)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.extract_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    usable_count = 0
    if args.resume:
        seen_ids, usable_count = read_existing_audit(args.output_jsonl)
        if seen_ids:
            asset_ids = [asset_id for asset_id in asset_ids if asset_id not in seen_ids]

    total_assets = len(asset_ids)
    processed = 0
    max_workers = max(1, int(args.workers))
    if max_workers == 1:
        for asset_id in asset_ids:
            if args.target_usable > 0 and usable_count >= args.target_usable:
                break
            _, records, asset_usable = audit_one_asset(asset_id, model_paths[asset_id], args)
            for record in records:
                append_jsonl(args.output_jsonl, record)
            processed += 1
            if asset_usable:
                usable_count += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print_progress(processed, total_assets, usable_count, args.output_jsonl)
        return

    pending: set[concurrent.futures.Future] = set()
    asset_iter = iter(asset_ids)

    def submit_until_full(executor: concurrent.futures.ThreadPoolExecutor) -> None:
        while len(pending) < max_workers:
            try:
                asset_id = next(asset_iter)
            except StopIteration:
                return
            pending.add(executor.submit(audit_one_asset, asset_id, model_paths[asset_id], args))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submit_until_full(executor)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                _, records, asset_usable = future.result()
                for record in records:
                    append_jsonl(args.output_jsonl, record)
                processed += 1
                if asset_usable:
                    usable_count += 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print_progress(processed, total_assets, usable_count, args.output_jsonl)
            if args.target_usable > 0 and usable_count >= args.target_usable:
                for future in pending:
                    future.cancel()
                break
            submit_until_full(executor)


if __name__ == "__main__":
    main()
