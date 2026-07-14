#!/usr/bin/env python3
"""Batch Pass1 export for TexVerse asset-sequence candidates.

Input is one or more Pass0 audit JSONL files.  For each usable Pass0 record,
this script re-extracts the cached TexVerse zip, reconstructs the audited source
path, and calls `export_texverse_clip.py` to export a multi-frame asset
sequence. The exported sequence is not a fixed training clip; training samples
are still drawn online from it.
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
from dataclasses import dataclass
from pathlib import Path

from process_utils import run_process_group
from texverse_archive_utils import (
    expand_nested_archives,
    find_import_candidates,
    relative_source_from_candidate,
    safe_extract_zip,
)


@dataclass
class Candidate:
    asset_id: str
    rel_source: str | None
    pass0_record: dict


SOURCE_SELECTION_POLICY = "pass0_best_usable_rig_motion_score_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--zip-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--skip-asset-id-file", type=Path, action="append", default=[])
    parser.add_argument("--skip-npz-dir", type=Path, action="append", default=[])
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument("--blender-threads", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--motion-fps-descriptor-vertices", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-clean", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--max-joints", type=int, default=256)
    parser.add_argument("--min-vertices", type=int, default=1)
    parser.add_argument("--max-vertices", type=int, default=300000)
    parser.add_argument("--max-faces", type=int, default=600000)
    parser.add_argument("--bbox-ratio-hard-min", type=float, default=0.0)
    parser.add_argument("--bbox-ratio-hard-max", type=float, default=0.0)
    parser.add_argument("--min-motion-p95-bbox", type=float, default=0.0)
    parser.add_argument("--active-skin-threshold", type=float, default=0.0)
    parser.add_argument("--no-expand-nested-archives", action="store_true")
    parser.add_argument("--nested-archive-depth", type=int, default=2)
    parser.add_argument("--max-nested-archives", type=int, default=12)
    parser.add_argument("--max-nested-archive-mb", type=float, default=0.0)
    parser.add_argument("--nested-archive-timeout-sec", type=int, default=120)
    parser.add_argument("--max-source-candidates", type=int, default=32)
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--include-nonusable-pass0", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_asset_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def asset_id_from_npz(path: Path) -> str:
    stem = path.stem
    if "_seq" in stem:
        return stem.split("_seq", 1)[0]
    return stem.split("_clip", 1)[0]


def load_skip_asset_ids(args: argparse.Namespace) -> set[str]:
    skip: set[str] = set()
    for ids_path in args.skip_asset_id_file:
        skip.update(read_asset_ids(ids_path))
    for npz_dir in args.skip_npz_dir:
        if npz_dir.exists():
            skip.update(asset_id_from_npz(path) for path in npz_dir.glob("*.npz"))
    return skip


def _float_metric(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_selection_key(row: dict, order: int) -> tuple[float, float, float, float, float, int, str]:
    return (
        _float_metric(row, "max_action_frames"),
        _float_metric(row, "max_bone_fcurves"),
        _float_metric(row, "skinned_vertices"),
        _float_metric(row, "skin_coverage"),
        _float_metric(row, "max_bones"),
        -order,
        str(row.get("candidate") or ""),
    )


def select_pass0_rows(rows: list[tuple[int, dict]], include_nonusable: bool) -> list[tuple[int, dict]]:
    by_asset: dict[str, list[tuple[int, dict]]] = {}
    for order, row in rows:
        if not include_nonusable and row.get("usable") is not True:
            continue
        asset_id = row.get("asset_id")
        if not asset_id:
            continue
        by_asset.setdefault(str(asset_id), []).append((order, row))

    selected: list[tuple[int, dict]] = []
    for asset_id in sorted(by_asset):
        asset_rows = by_asset[asset_id]
        best_order, best = max(asset_rows, key=lambda item: _candidate_selection_key(item[1], item[0]))
        best = dict(best)
        best["source_selection_policy"] = SOURCE_SELECTION_POLICY
        best["source_selection_candidate_count"] = len(asset_rows)
        best["source_selection_score"] = list(_candidate_selection_key(best, best_order)[:5])
        selected.append((best_order, best))
    selected.sort(key=lambda item: item[0])
    return selected


def load_candidates(audit_paths: list[Path], include_nonusable: bool) -> list[Candidate]:
    rows_with_order: list[tuple[int, dict]] = []
    order = 0
    for audit_path in audit_paths:
        for row in read_jsonl(audit_path):
            rows_with_order.append((order, row))
            order += 1

    candidates = []
    for _, row in select_pass0_rows(rows_with_order, include_nonusable):
        asset_id = str(row["asset_id"])
        candidates.append(
            Candidate(
                asset_id=asset_id,
                rel_source=relative_source_from_candidate(asset_id, row.get("candidate")),
                pass0_record=row,
            )
        )
    return candidates


def find_zip(asset_id: str, zip_dirs: list[Path]) -> Path | None:
    for zip_dir in zip_dirs:
        path = zip_dir / f"{asset_id}.zip"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def load_existing_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def remove_outputs(out_npz: Path, out_json: Path) -> None:
    for path in (out_npz, out_json):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def ordered_import_sources(
    extract_dir: Path,
    rel_source: str | None,
    max_candidates: int,
) -> tuple[list[Path], int | None]:
    sources, source_count = find_import_candidates(extract_dir, max_candidates)
    ordered: list[Path] = []
    seen: set[Path] = set()

    if rel_source:
        pass0_source = extract_dir / rel_source
        if pass0_source.exists() and pass0_source.is_file():
            resolved = pass0_source.resolve()
            ordered.append(pass0_source)
            seen.add(resolved)

    for source in sources:
        resolved = source.resolve()
        if resolved not in seen:
            ordered.append(source)
            seen.add(resolved)
    return ordered, source_count


def export_cmd(
    source: Path,
    asset_id: str,
    out_npz: Path,
    out_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
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


def run_export_attempt(
    source: Path,
    asset_id: str,
    out_npz: Path,
    out_json: Path,
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[dict, str]:
    remove_outputs(out_npz, out_json)
    cmd = export_cmd(source, asset_id, out_npz, out_json, args)
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
    except subprocess.TimeoutExpired as exc:
        record = {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["batch_export_timeout"],
            "returncode": None,
            "error": repr(exc),
        }
        return record, exc.stdout or ""

    record = load_existing_json(out_json)
    combined_stdout = proc.stdout

    if record is None:
        record = {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["missing_export_json"],
        }
    record["returncode"] = proc.returncode
    if proc.returncode != 0 and record.get("status") == "clean":
        record["status"] = "reject"
        record["reject_reasons"] = ["nonzero_export_returncode"]
    return record, combined_stdout


def run_one(candidate: Candidate, args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    asset_id = candidate.asset_id
    out_npz = args.out_root / "npz" / f"{asset_id}_seq0.npz"
    out_json = args.out_root / "json" / f"{asset_id}_seq0.json"
    existing = load_existing_json(out_json)
    if existing is not None and existing.get("status") == "clean" and out_npz.exists():
        existing["asset_id"] = asset_id
        existing["npz_path"] = str(out_npz) if out_npz.exists() else ""
        existing["json_path"] = str(out_json)
        existing["batch_status"] = "skipped_existing"
        return existing
    if existing is not None and existing.get("status") != "clean":
        remove_outputs(out_npz, out_json)

    zip_path = find_zip(asset_id, args.zip_dir)
    if zip_path is None:
        return {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["missing_cached_zip"],
            "batch_status": "missing_zip",
        }

    extract_dir = args.extract_root / asset_id
    try:
        safe_extract_zip(zip_path, extract_dir)
        nested_archives = expand_nested_archives(
            extract_dir,
            no_expand=args.no_expand_nested_archives,
            max_depth=args.nested_archive_depth,
            max_archives=args.max_nested_archives,
            max_archive_mb=args.max_nested_archive_mb,
            timeout_sec=args.nested_archive_timeout_sec,
        )
        sources, source_count = ordered_import_sources(
            extract_dir,
            candidate.rel_source,
            args.max_source_candidates,
        )
        if not sources:
            return {
                "asset_id": asset_id,
                "status": "reject",
                "reject_reasons": ["no_importable_source_after_extract"],
                "zip_path": str(zip_path),
                "nested_archives": nested_archives,
            }

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        attempts: list[dict] = []
        failure_logs: list[str] = []
        last_record: dict | None = None
        for attempt_idx, source in enumerate(sources):
            record, stdout = run_export_attempt(source, asset_id, out_npz, out_json, args, repo_root)
            attempt = {
                "index": attempt_idx,
                "source_file": str(source),
                "status": record.get("status"),
                "reject_reasons": record.get("reject_reasons", []),
                "returncode": record.get("returncode"),
            }
            attempts.append(attempt)
            last_record = record
            if record.get("status") == "clean" and out_npz.exists():
                record.update(
                    {
                        "asset_id": asset_id,
                        "source_file": str(source),
                        "zip_path": str(zip_path),
                        "npz_path": str(out_npz),
                        "json_path": str(out_json),
                        "batch_status": "done",
                        "pass0_candidate": candidate.pass0_record.get("candidate"),
                        "pass0_rel_source": candidate.rel_source,
                        "source_selection_policy": candidate.pass0_record.get("source_selection_policy"),
                        "source_selection_score": candidate.pass0_record.get("source_selection_score"),
                        "source_selection_candidate_count": candidate.pass0_record.get(
                            "source_selection_candidate_count"
                        ),
                        "source_candidate_count": source_count,
                        "source_attempt_count": len(attempts),
                        "candidate_attempts": attempts,
                        "nested_archives": nested_archives,
                    }
                )
                return record
            if stdout:
                failure_logs.append(
                    f"--- source attempt {attempt_idx}: {source} ---\n{stdout}"
                )

        record = last_record or {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["missing_export_json"],
        }
        remove_outputs(out_npz, out_json)
        record.update(
            {
                "asset_id": asset_id,
                "source_file": str(sources[-1]),
                "zip_path": str(zip_path),
                "npz_path": "",
                "json_path": str(out_json),
                "batch_status": "done",
                "pass0_candidate": candidate.pass0_record.get("candidate"),
                "pass0_rel_source": candidate.rel_source,
                "source_selection_policy": candidate.pass0_record.get("source_selection_policy"),
                "source_selection_score": candidate.pass0_record.get("source_selection_score"),
                "source_selection_candidate_count": candidate.pass0_record.get(
                    "source_selection_candidate_count"
                ),
                "source_candidate_count": source_count,
                "source_attempt_count": len(attempts),
                "candidate_attempts": attempts,
                "nested_archives": nested_archives,
            }
        )
        log_path = args.out_root / "logs" / f"{asset_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(failure_logs), encoding="utf-8")
        record["log_path"] = str(log_path)
        return record
    except Exception as exc:  # noqa: BLE001
        return {
            "asset_id": asset_id,
            "status": "reject",
            "reject_reasons": ["batch_exception"],
            "error": repr(exc),
            "zip_path": str(zip_path),
        }
    finally:
        if not args.keep_extracted:
            shutil.rmtree(extract_dir, ignore_errors=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_count(path: Path) -> int:
    rows = read_jsonl(path)
    return sum(row.get("status") == "clean" for row in rows)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.extract_root.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.audit_jsonl, args.include_nonusable_pass0)
    skip_ids = load_skip_asset_ids(args)
    if skip_ids:
        candidates = [candidate for candidate in candidates if candidate.asset_id not in skip_ids]
    if args.shuffle:
        random.Random(args.seed).shuffle(candidates)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    done = {row.get("asset_id") for row in read_jsonl(args.manifest_jsonl)}
    candidates = [candidate for candidate in candidates if candidate.asset_id not in done]
    if not candidates:
        return

    current_clean = clean_count(args.manifest_jsonl)
    max_workers = max(1, args.workers)
    pending: set[concurrent.futures.Future] = set()
    candidate_iter = iter(candidates)

    def submit_until_full(executor: concurrent.futures.ThreadPoolExecutor) -> None:
        while len(pending) < max_workers:
            try:
                candidate = next(candidate_iter)
            except StopIteration:
                return
            pending.add(executor.submit(run_one, candidate, args))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submit_until_full(executor)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                record = future.result()
                append_jsonl(args.manifest_jsonl, record)
                if record.get("status") == "clean":
                    current_clean += 1
            if args.target_clean > 0 and current_clean >= args.target_clean:
                for future in pending:
                    future.cancel()
                break
            submit_until_full(executor)


if __name__ == "__main__":
    main()
