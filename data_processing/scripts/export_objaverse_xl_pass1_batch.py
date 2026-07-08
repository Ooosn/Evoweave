#!/usr/bin/env python3
"""Export downloaded Objaverse-XL files into RigWeave sequence NPZ files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Blender sequence export for downloaded Objaverse-XL files.")
    parser.add_argument("--download-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--blender-threads", type=int, default=2)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--motion-fps-descriptor-vertices", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--max-joints", type=int, default=256)
    parser.add_argument("--min-vertices", type=int, default=1)
    parser.add_argument("--max-vertices", type=int, default=300000)
    parser.add_argument("--max-faces", type=int, default=600000)
    parser.add_argument("--bbox-ratio-hard-min", type=float, default=0.0)
    parser.add_argument("--bbox-ratio-hard-max", type=float, default=0.0)
    parser.add_argument("--min-motion-p95-bbox", type=float, default=0.0)
    parser.add_argument("--active-skin-threshold", type=float, default=0.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_existing_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_one(row: dict, args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    asset_id = row["asset_id"]
    source = Path(row["path"])
    out_npz = args.out_root / "npz" / f"{asset_id}_seq0.npz"
    out_json = args.out_root / "json" / f"{asset_id}_seq0.json"
    existing = load_existing_json(out_json)
    if existing is not None and (out_npz.exists() or existing.get("status") != "clean"):
        return {
            **existing,
            "asset_id": asset_id,
            "source_file": str(source),
            "npz_path": str(out_npz) if out_npz.exists() else "",
            "json_path": str(out_json),
            "batch_status": "skipped_existing",
        }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/export_texverse_clip.py",
        "--source",
        str(source),
        "--asset-id",
        asset_id,
        "--out-npz",
        str(out_npz),
        "--out-json",
        str(out_json),
        "--blender",
        args.blender,
        "--blender-threads",
        str(args.blender_threads),
        "--frames",
        str(args.frames),
        "--motion-fps-descriptor-vertices",
        str(args.motion_fps_descriptor_vertices),
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-joints",
        str(args.max_joints),
        "--min-vertices",
        str(args.min_vertices),
        "--max-vertices",
        str(args.max_vertices),
        "--max-faces",
        str(args.max_faces),
        "--bbox-ratio-hard-min",
        str(args.bbox_ratio_hard_min),
        "--bbox-ratio-hard-max",
        str(args.bbox_ratio_hard_max),
        "--min-motion-p95-bbox",
        str(args.min_motion_p95_bbox),
        "--active-skin-threshold",
        str(args.active_skin_threshold),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_sec + 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["batch_export_timeout"],
            "source_file": str(source),
        }
    record = load_existing_json(out_json) or {
        "asset_id": asset_id,
        "status": "reject",
        "reject_reasons": ["missing_export_json"],
    }
    combined_stdout = proc.stdout
    record.update(
        {
            "asset_id": asset_id,
            "source_file": str(source),
            "download_manifest": row,
            "npz_path": str(out_npz) if out_npz.exists() else "",
            "json_path": str(out_json),
            "returncode": proc.returncode,
            "batch_status": "done",
        }
    )
    if proc.returncode != 0 and record.get("status") == "clean":
        record["status"] = "reject"
        record["reject_reasons"] = ["nonzero_export_returncode"]
    if proc.returncode != 0:
        log_path = args.out_root / "logs" / f"{asset_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined_stdout, encoding="utf-8")
        record["log_path"] = str(log_path)
    return record


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.download_manifest_jsonl) if row.get("status") == "ok" and row.get("path")]
    if args.limit > 0:
        rows = rows[: args.limit]
    done = {row.get("asset_id") for row in read_jsonl(args.manifest_jsonl)}
    rows = [row for row in rows if row.get("asset_id") not in done]
    if not rows:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for i, record in enumerate(executor.map(lambda row: run_one(row, args), rows), start=1):
            append_jsonl(args.manifest_jsonl, record)
            if i % 25 == 0:
                print(json.dumps({"event": "objaverse_pass1_progress", "processed": i}, sort_keys=True))


if __name__ == "__main__":
    main()
