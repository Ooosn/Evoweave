#!/usr/bin/env python3
"""Download Objaverse-XL proxy candidates as raw source files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Objaverse-XL candidate files.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-file-mb", type=float, default=256.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Do not treat previous non-ok manifest rows as done; append a new attempt record for them.",
    )
    parser.add_argument(
        "--retry-reason-contains",
        action="append",
        default=[],
        help="With --retry-errors, retry only previous non-ok rows whose latest reason contains this string.",
    )
    parser.add_argument("--redownload", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_path(row: dict, download_dir: Path) -> Path:
    ext = str(row.get("fileType") or Path(str(row["fileIdentifier"])).suffix.lstrip(".")).lower()
    if not ext:
        ext = "bin"
    return download_dir / ext / f"{row['asset_id']}.{ext}"


def download_one(row: dict, args: argparse.Namespace) -> dict:
    asset_id = row["asset_id"]
    out = target_path(row, args.download_dir)
    tmp = out.with_suffix(out.suffix + ".tmp")
    base = {
        "asset_id": asset_id,
        "source": row.get("source"),
        "file_type": row.get("fileType"),
        "sha256_expected": row.get("sha256"),
        "fileIdentifier": row.get("fileIdentifier"),
        "download_url": row.get("download_url"),
        "path": str(out),
    }
    if out.exists() and out.stat().st_size > 0 and not args.redownload:
        actual = sha256_file(out)
        return {
            **base,
            "status": "ok" if not row.get("sha256") or actual == row.get("sha256") else "error",
            "reason": "" if not row.get("sha256") or actual == row.get("sha256") else "sha256_mismatch_cached",
            "bytes": out.stat().st_size,
            "sha256_actual": actual,
            "source_state": "cache",
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)
    last_error = ""
    attempts = max(1, int(args.attempts))
    for attempt in range(1, attempts + 1):
        tmp.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(str(row["download_url"]), headers={"User-Agent": "evoweave-objaverse-xl"})
            with urllib.request.urlopen(request, timeout=args.timeout_sec) as response:
                length = response.headers.get("Content-Length")
                if length is not None and args.max_file_mb > 0 and int(length) > args.max_file_mb * 1024 * 1024:
                    return {
                        **base,
                        "status": "reject",
                        "reason": "file_too_large_by_header",
                        "bytes": int(length),
                        "attempt": attempt,
                    }
                with tmp.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if args.max_file_mb > 0 and tmp.stat().st_size > args.max_file_mb * 1024 * 1024:
                tmp.unlink(missing_ok=True)
                return {**base, "status": "reject", "reason": "file_too_large_after_download", "attempt": attempt}
            actual = sha256_file(tmp)
            if row.get("sha256") and actual != row.get("sha256"):
                tmp.unlink(missing_ok=True)
                last_error = "sha256_mismatch_download"
            else:
                tmp.replace(out)
                return {
                    **base,
                    "status": "ok",
                    "reason": "",
                    "bytes": out.stat().st_size,
                    "sha256_actual": actual,
                    "source_state": "download",
                    "attempt": attempt,
                }
        except HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        finally:
            tmp.unlink(missing_ok=True)

        if attempt < attempts:
            # Retry 404 as well.  In practice most GitHub 404s are persistent
            # source/path failures, but transient raw/LFS responses do happen.
            # 429 needs longer backoff to avoid making the rate limit worse.
            sleep = float(args.retry_sleep_sec) * attempt
            if "429" in last_error:
                sleep *= 5.0
            time.sleep(sleep)
    return {**base, "status": "error", "reason": last_error, "attempt": attempts}


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.candidate_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]
    previous = []
    if args.manifest_jsonl.exists():
        previous = read_jsonl(args.manifest_jsonl)
        if args.retry_errors:
            done = {row.get("asset_id") for row in previous if row.get("status") == "ok"}
        else:
            done = {row.get("asset_id") for row in previous}
    else:
        done = set()
    rows = [row for row in rows if row.get("asset_id") not in done]
    if args.retry_errors and args.retry_reason_contains:
        latest = {row.get("asset_id"): row for row in previous}
        needles = tuple(args.retry_reason_contains)
        rows = [
            row
            for row in rows
            if any(needle in str(latest.get(row.get("asset_id"), {}).get("reason", "")) for needle in needles)
        ]
    if not rows:
        return
    start = time.time()
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for i, record in enumerate(executor.map(lambda row: download_one(row, args), rows), start=1):
            write_jsonl(args.manifest_jsonl, record)
            ok += int(record.get("status") == "ok")
            if i % 100 == 0:
                print(json.dumps({"event": "download_progress", "processed": i, "ok": ok, "elapsed_sec": time.time() - start}))


if __name__ == "__main__":
    main()
