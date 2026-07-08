#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_npz_root(run: Path, source: str) -> Path:
    return run / source / "pass1_motionfps_sequence40" / "npz"


def oracle_npz_path(oracle: Path, source: str, asset_id: str) -> Path:
    return oracle / source / "pass1_motionfps_sequence40" / "npz" / f"{asset_id}_seq0.npz"


def npz_path(root: Path, asset_id: str) -> Path:
    return root / f"{asset_id}_seq0.npz"


def meta(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta_json" not in data.files:
        return {}
    value = data["meta_json"]
    try:
        text = str(value.item())
    except Exception:
        text = str(value)
    try:
        return json.loads(text)
    except Exception:
        return {}


def safe_shape(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        return "missing"
    return "x".join(str(x) for x in data[key].shape)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare_pair(source: str, asset_id: str, new_path: Path, old_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "asset_id": asset_id,
        "new_path": str(new_path),
        "oracle_path": str(old_path),
        "new_exists": new_path.exists(),
        "oracle_exists": old_path.exists(),
    }
    if not new_path.exists() or not old_path.exists():
        row["status"] = "missing"
        return row
    with np.load(new_path, allow_pickle=True) as new, np.load(old_path, allow_pickle=True) as old:
        nmeta = meta(new)
        ometa = meta(old)
        row.update(
            {
                "status": "compared",
                "new_frame_policy": nmeta.get("frame_sampling_policy", ""),
                "oracle_frame_policy": ometa.get("frame_sampling_policy", ""),
                "new_candidate_frame_count": nmeta.get("candidate_frame_count", ""),
                "new_motion_fps_descriptor_vertices": nmeta.get("motion_fps_descriptor_vertices", ""),
                "new_row_contract_policy": nmeta.get("row_contract_policy", ""),
                "oracle_row_contract_policy": ometa.get("row_contract_policy", ""),
                "new_joint_count": int(new["parents"].shape[0]) if "parents" in new.files else -1,
                "oracle_joint_count": int(old["parents"].shape[0]) if "parents" in old.files else -1,
                "new_vertex_count": int(new["rest_vertices"].shape[0]) if "rest_vertices" in new.files else -1,
                "oracle_vertex_count": int(old["rest_vertices"].shape[0]) if "rest_vertices" in old.files else -1,
            }
        )
        for key in [
            "rest_vertices",
            "faces",
            "frame_vertices",
            "skin_weights",
            "parents",
            "rest_joints",
            "rest_tails",
            "bone_transforms",
            "frame_numbers",
        ]:
            row[f"{key}_new_shape"] = safe_shape(new, key)
            row[f"{key}_oracle_shape"] = safe_shape(old, key)
            if key in new.files and key in old.files:
                diff = max_abs_diff(new[key], old[key])
                row[f"{key}_max_abs_diff"] = "" if diff is None else diff
                row[f"{key}_same_shape"] = bool(new[key].shape == old[key].shape)
        if "parents" in new.files and "parents" in old.files and new["parents"].shape == old["parents"].shape:
            row["parents_exact"] = bool(np.array_equal(new["parents"], old["parents"]))
        if "frame_numbers" in new.files and "frame_numbers" in old.files:
            row["frame_numbers_equal"] = bool(np.array_equal(new["frame_numbers"], old["frame_numbers"]))
            row["new_frame_numbers"] = " ".join(map(str, new["frame_numbers"].astype(int).tolist()))
            row["oracle_frame_numbers"] = " ".join(map(str, old["frame_numbers"].astype(int).tolist()))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_manifests = {
        "texverse": args.run / "texverse" / "pass1_motionfps_sequence40" / "manifest.jsonl",
        "objaverse_xl": args.run / "objaverse_xl" / "pass1_motionfps_sequence40" / "manifest.jsonl",
        "objaverse_xl_more_anim": args.run / "objaverse_xl_more_anim" / "pass1_motionfps_sequence40" / "manifest.jsonl",
    }
    for source, manifest in source_manifests.items():
        for record in load_jsonl(manifest):
            asset_id = str(record.get("asset_id", ""))
            row = {
                "source": source,
                "asset_id": asset_id,
                "export_status": record.get("status", ""),
                "reject_reasons": " ".join(record.get("reject_reasons", []) or []),
            }
            if record.get("status") == "clean":
                row.update(
                    compare_pair(
                        source,
                        asset_id,
                        npz_path(source_npz_root(args.run, source), asset_id),
                        oracle_npz_path(args.oracle, source, asset_id),
                    )
                )
            rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.out / "pass1_oracle_compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {"rows": len(rows), "by_status": {}, "shape_mismatches": {}, "frame_mismatch_assets": []}
    for row in rows:
        status = str(row.get("export_status", ""))
        summary["by_status"][status] = int(summary["by_status"].get(status, 0)) + 1
        if row.get("frame_numbers_equal") is False:
            summary["frame_mismatch_assets"].append({"source": row["source"], "asset_id": row["asset_id"]})
        for key, value in row.items():
            if key.endswith("_same_shape") and value is False:
                summary["shape_mismatches"][key] = int(summary["shape_mismatches"].get(key, 0)) + 1
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
