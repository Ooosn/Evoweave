#!/usr/bin/env python3
"""Create and verify a short-lived receipt for model-agent context loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "model_training" / "state" / "current.json"
RECEIPT_PATH = REPO_ROOT / ".agents" / "model_training_context_receipt.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _context_path(state: dict[str, Any]) -> Path:
    raw = state.get("human_context")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("state is missing human_context")
    path = (REPO_ROOT / raw).resolve()
    if not path.is_file():
        raise RuntimeError(f"human context does not exist: {path}")
    return path


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_id": state.get("state_id"),
        "updated_at": state.get("updated_at"),
        "reference_runs": state.get("reference_runs"),
        "verified_fixed_causes": state.get("verified_fixed_causes"),
        "verified_open_facts": state.get("verified_open_facts"),
        "unknowns": state.get("unknowns"),
        "allowed_operations": state.get("allowed_operations"),
        "blocked_operations": state.get("blocked_operations"),
        "next_required_result": state.get("next_required_result"),
        "active_operation": state.get("active_operation"),
        "last_operation": state.get("last_operation"),
    }


def begin() -> int:
    state = _load_json(STATE_PATH)
    context_path = _context_path(state)
    required = state.get("required_context")
    if not isinstance(required, list) or not required:
        raise RuntimeError("state is missing required_context")
    missing = [raw for raw in required if not (REPO_ROOT / str(raw)).is_file()]
    if missing:
        raise RuntimeError(f"required context files are missing: {missing}")

    now = datetime.now(timezone.utc)
    receipt = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "state_id": state.get("state_id"),
        "state_sha256": _sha256(STATE_PATH),
        "context_sha256": _sha256(context_path),
        "git_head": _git_head(),
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_snapshot(state), indent=2, ensure_ascii=False, sort_keys=True))
    print(f"receipt={RECEIPT_PATH}")
    return 0


def check(operation: str) -> int:
    state = _load_json(STATE_PATH)
    context_path = _context_path(state)
    if not RECEIPT_PATH.is_file():
        raise RuntimeError("missing context receipt; run `agent_work_guard.py begin`")
    receipt = _load_json(RECEIPT_PATH)

    expected = {
        "state_id": state.get("state_id"),
        "state_sha256": _sha256(STATE_PATH),
        "context_sha256": _sha256(context_path),
        "git_head": _git_head(),
    }
    mismatches = {
        key: {"receipt": receipt.get(key), "current": value}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "stale context receipt; run `agent_work_guard.py begin`: "
            + json.dumps(mismatches, sort_keys=True)
        )

    created_at = receipt.get("created_at")
    if not isinstance(created_at, str):
        raise RuntimeError("receipt is missing created_at")
    created = datetime.fromisoformat(created_at)
    age_minutes = (datetime.now(timezone.utc) - created).total_seconds() / 60.0
    max_age = float(state.get("receipt_max_age_minutes", 240))
    if age_minutes > max_age:
        raise RuntimeError(
            f"context receipt is {age_minutes:.1f} minutes old; maximum is {max_age:.1f}"
        )

    blocked = set(str(value) for value in state.get("blocked_operations", []))
    allowed = set(str(value) for value in state.get("allowed_operations", []))
    if operation in blocked:
        raise RuntimeError(
            f"operation {operation!r} is blocked by state {state.get('state_id')}; "
            f"next required result: {state.get('next_required_result')}"
        )
    if operation not in allowed:
        raise RuntimeError(
            f"operation {operation!r} is not explicitly allowed; allowed={sorted(allowed)}"
        )

    print(
        json.dumps(
            {
                "allowed": True,
                "operation": operation,
                "state_id": state.get("state_id"),
                "git_head": expected["git_head"],
                "receipt_age_minutes": round(age_minutes, 2),
                "active_operation": state.get("active_operation"),
                "next_required_result": state.get("next_required_result"),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("begin")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--operation", required=True)
    args = parser.parse_args()
    if args.command == "begin":
        return begin()
    return check(args.operation)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"agent-work-guard: {exc}", file=sys.stderr)
        raise SystemExit(2)
