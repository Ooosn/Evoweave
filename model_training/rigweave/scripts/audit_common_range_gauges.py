"""CPU-only range and invertibility audit; never changes source NPZ geometry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from common_range_geometry import coordinate_summary, mesh_gauge, root_gauge, transform_sequence
from prepare_common_range_probe import sha256


def inspect_case(case, corpus):
    path = corpus / case["local_relative_path"]
    if sha256(path) != case["npz_sha256"]:
        raise ValueError("diagnostic input changed after corpus preparation")
    with np.load(path, allow_pickle=False) as archive:
        raw_vertices = np.asarray(archive["frame_vertices_rootspace"], dtype=np.float32)
        raw_joints = np.asarray(archive["target_joints_rootspace"], dtype=np.float32)
        root_index = int(np.asarray(archive["raw_root_index"]).item())
        deltas = np.asarray(archive["bone_transforms"])
    reference = int(case["selected_frames"][0])
    order = [reference] + [index for index in range(len(raw_vertices)) if index != reference]
    positions = [order.index(index) for index in case["selected_frames"]]
    lo, hi = raw_vertices[reference].min(0), raw_vertices[reference].max(0)
    center = ((lo + hi) * np.float32(.5)).astype(np.float32)
    scale = float(np.max((hi - lo) * np.float32(.5)))
    vertices = ((raw_vertices[order] - center) / scale).astype(np.float32)
    joints = ((raw_joints[order] - center) / scale).astype(np.float32)
    output = {**case, "order": order, "selected_positions": positions, "query_center": center.tolist(), "query_scale": scale,
              "original": {"mesh_all_frames": coordinate_summary(vertices), "joints_all_frames": coordinate_summary(joints),
                           "mesh_selected": coordinate_summary(vertices[positions]), "joints_selected": coordinate_summary(joints[positions])},
              "gauges": {}}
    for method in ("translation", "rigid", "robust_rigid", "raw_root_oracle"):
        if method == "raw_root_oracle":
            try:
                rotations, translations = root_gauge(deltas, root_index, order, center, scale)
            except ValueError as exc:
                output["gauges"][method] = {"available": False, "reason": str(exc)}
                continue
            report = {"method": method, "raw_root_index": root_index,
                      "requires_gt_skeleton": True, "gauge_not_authoritative_global_rt": True}
        else:
            rotations, translations, report = mesh_gauge(vertices, method)
        aligned_vertices = transform_sequence(vertices, rotations, translations)
        aligned_joints = transform_sequence(joints, rotations, translations)
        restored = np.einsum("tji,tnj->tni", rotations, aligned_vertices - translations[:, None])
        inverse_error = float(np.max(np.abs(restored - vertices)) / 2.)
        if inverse_error > 1e-4 or not np.array_equal(aligned_vertices[0], vertices[0]):
            raise ValueError(f"coordinate-gauge invertibility or reference identity failed: {method}")
        output["gauges"][method] = {"available": True, **report,
            "mesh_all_frames": coordinate_summary(aligned_vertices), "joints_all_frames": coordinate_summary(aligned_joints),
            "mesh_selected": coordinate_summary(aligned_vertices[positions]), "joints_selected": coordinate_summary(aligned_joints[positions]),
            "inverse_max_error_bbox": inverse_error,
            "rotations": rotations.tolist(), "translations_common": translations.tolist()}
    return output


def summarize(rows):
    groups = {}
    for stratum in sorted({row["stratum"] for row in rows}):
        group = [row for row in rows if row["stratum"] == stratum]
        result = {"assets": len(group), "methods": {}}
        for method in ("original", "translation", "rigid", "robust_rigid", "raw_root_oracle"):
            values = [row["original"] if method == "original" else row["gauges"][method] for row in group]
            available = [value for value in values if value.get("available", True)]
            ranges = np.array([value["mesh_selected"]["maxabs"] for value in available])
            stats = {"available": len(available), "median_mesh_maxabs": float(np.median(ranges)) if len(ranges) else None,
                     "maximum_mesh_maxabs": float(ranges.max()) if len(ranges) else None,
                     "assets_over_2": int(np.sum(ranges > 2.)), "assets_over_4": int(np.sum(ranges > 4.))}
            fits = [frame for value in available for frame in value.get("frames", []) if frame["frame_position"]]
            if fits:
                for key in ("fit_rms_bbox", "split_rotation_degrees", "split_translation_bbox", "split_alignment_rms_bbox"):
                    stats[key] = {"median": float(np.median([item[key] for item in fits])),
                                  "maximum": max(item[key] for item in fits)}
                stats["ill_conditioned_frames"] = sum(item["second_to_first_singular_ratio"] is not None
                    and item["second_to_first_singular_ratio"] < 1e-3 for item in fits)
            result["methods"][method] = stats
        groups[stratum] = result
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases_path = args.corpus / "cases.json"
    cases = json.loads(cases_path.read_text())["cases"]
    selected = set(range(len(cases))) if args.indices == "all" else {int(x) for x in args.indices.split(",")}
    if not selected.issubset(range(len(cases))):
        raise ValueError("case index out of range")
    report = {"complete": False, "started_at": datetime.now(timezone.utc).isoformat(), "cases_sha256": sha256(cases_path),
              "script_sha256": sha256(__file__), "geometry_script_sha256": sha256(Path(__file__).with_name("common_range_geometry.py")),
              "rows": [], "selection": "read-only targeted matched subset; all stored frames retained for feasibility audit",
              "note": "Mesh-only fits define a coordinate gauge, not uniquely identifiable true object-global RT; root oracle is unavailable at mesh-only inference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for case in cases:
        if case["case_index"] not in selected:
            continue
        row = inspect_case(case, args.corpus)
        report["rows"].append(row)
        args.output.write_text(json.dumps(report) + "\n")
        print(json.dumps({"completed": len(report["rows"]), "total": len(selected), "asset_id": case["asset_id"],
                          "original_max": row["original"]["mesh_selected"]["maxabs"],
                          "rigid_max": row["gauges"]["rigid"]["mesh_selected"]["maxabs"]}), flush=True)
    report.update(complete=True, summary=summarize(report["rows"]), elapsed_seconds=time.monotonic() - started)
    args.output.write_text(json.dumps(report) + "\n")
    print(json.dumps({"complete": True, "assets": len(report["rows"]), "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
