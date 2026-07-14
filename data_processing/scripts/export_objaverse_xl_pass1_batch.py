#!/usr/bin/env python3
"""Export downloaded Objaverse-XL files into RigWeave sequence NPZ files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

from process_utils import run_process_group


RETRYABLE_REJECT_REASONS = frozenset(
    {
        "batch_export_timeout",
        "blender_export_timeout",
    }
)


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
    parser.add_argument("--timeout-retries", type=int, default=1)
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


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_existing_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def reject_reasons(record: dict) -> list[str]:
    reasons = record.get("reject_reasons") or record.get("reasons") or []
    return [str(reason) for reason in reasons]


def is_retryable_record(record: dict) -> bool:
    return any(reason in RETRYABLE_REJECT_REASONS for reason in reject_reasons(record))


def prepare_manifest_for_resume(path: Path) -> tuple[set[str], list[str]]:
    rows = read_jsonl(path)
    if not rows:
        return set(), []

    order: list[str] = []
    latest: dict[str, dict] = {}
    for row in rows:
        asset_id = row.get("asset_id")
        if not asset_id:
            raise ValueError(f"Pass1 manifest row is missing asset_id: {row}")
        asset_id = str(asset_id)
        if asset_id not in latest:
            order.append(asset_id)
        latest[asset_id] = row

    retry_asset_ids = [asset_id for asset_id in order if is_retryable_record(latest[asset_id])]
    retained = [latest[asset_id] for asset_id in order if asset_id not in retry_asset_ids]
    if len(retained) != len(rows):
        write_jsonl_atomic(path, retained)
    return {str(row["asset_id"]) for row in retained}, retry_asset_ids


def run_one(row: dict, args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    asset_id = row["asset_id"]
    source = Path(row["path"])
    out_npz = args.out_root / "npz" / f"{asset_id}_seq0.npz"
    out_json = args.out_root / "json" / f"{asset_id}_seq0.json"
    existing = load_existing_json(out_json)
    if existing is not None and existing.get("status") == "clean" and out_npz.exists():
        return {
            **existing,
            "asset_id": asset_id,
            "source_file": str(source),
            "npz_path": str(out_npz) if out_npz.exists() else "",
            "json_path": str(out_json),
            "batch_status": "skipped_existing",
        }
    if existing is not None and existing.get("status") != "clean" and not is_retryable_record(existing):
        return {
            **existing,
            "asset_id": asset_id,
            "source_file": str(source),
            "npz_path": str(out_npz) if out_npz.exists() else "",
            "json_path": str(out_json),
            "batch_status": "skipped_existing",
        }
    if existing is not None and is_retryable_record(existing):
        out_json.unlink(missing_ok=True)
        out_npz.unlink(missing_ok=True)
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
        proc = run_process_group(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_sec + 30,
            check=False,
        )
        returncode = int(proc.returncode)
        combined_stdout = proc.stdout
        record = load_existing_json(out_json) or {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["missing_export_json"],
        }
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        combined_stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        out_json.unlink(missing_ok=True)
        out_npz.unlink(missing_ok=True)
        record = {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["batch_export_timeout"],
            "source_file": str(source),
        }
    record.update(
        {
            "asset_id": asset_id,
            "source_file": str(source),
            "download_manifest": row,
            "npz_path": str(out_npz) if out_npz.exists() else "",
            "json_path": str(out_json),
            "returncode": returncode,
            "batch_status": "done",
            "batch_attempts": 1,
        }
    )
    if returncode != 0 and record.get("status") == "clean":
        record["status"] = "reject"
        record["reject_reasons"] = ["nonzero_export_returncode"]
    if returncode != 0:
        log_path = args.out_root / "logs" / f"{asset_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined_stdout, encoding="utf-8")
        record["log_path"] = str(log_path)
    return record


def retry_serially(row: dict, args: argparse.Namespace, initial_record: dict) -> dict:
    record = initial_record
    attempts = int(record.get("batch_attempts", 1))
    transient_attempt_reasons: list[str] = []
    while is_retryable_record(record) and attempts <= int(args.timeout_retries):
        transient_attempt_reasons.extend(reject_reasons(record))
        record = run_one(row, args)
        attempts += 1

    record["batch_attempts"] = attempts
    if transient_attempt_reasons:
        record["transient_attempt_reasons"] = list(dict.fromkeys(transient_attempt_reasons))
    return record


def main() -> None:
    args = parse_args()
    if args.timeout_retries < 0:
        raise ValueError("--timeout-retries must be >= 0")
    rows = [row for row in read_jsonl(args.download_manifest_jsonl) if row.get("status") == "ok" and row.get("path")]
    if args.limit > 0:
        rows = rows[: args.limit]
    done, retry_asset_ids = prepare_manifest_for_resume(args.manifest_jsonl)
    if retry_asset_ids:
        print(
            json.dumps(
                {
                    "event": "objaverse_pass1_resume_transient_retries",
                    "asset_count": len(retry_asset_ids),
                    "asset_ids": retry_asset_ids,
                },
                sort_keys=True,
            )
        )
    rows = [row for row in rows if row.get("asset_id") not in done]
    if not rows:
        return
    retry_queue: list[tuple[dict, dict]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(lambda row: (row, run_one(row, args)), rows)
        for i, (row, record) in enumerate(results, start=1):
            if is_retryable_record(record) and args.timeout_retries > 0:
                retry_queue.append((row, record))
            else:
                append_jsonl(args.manifest_jsonl, record)
            if i % 25 == 0:
                print(json.dumps({"event": "objaverse_pass1_progress", "processed": i}, sort_keys=True))

    for row, initial_record in retry_queue:
        record = retry_serially(row, args, initial_record)
        append_jsonl(args.manifest_jsonl, record)
        print(
            json.dumps(
                {
                    "event": "objaverse_pass1_serial_retry_complete",
                    "asset_id": row["asset_id"],
                    "attempts": record["batch_attempts"],
                    "status": record.get("status"),
                    "reject_reasons": reject_reasons(record),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
