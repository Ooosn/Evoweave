from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from gpu_node_time import login, make_session, request_json, token_cache_path


DEFAULT_IMAGE = "192.168.108.1:5000/pytorch/liuhaohan_w-liuhaohan:deps20260602"
DEFAULT_PROJECT_ID = "1d6af889b9834b219584b9cdb4f06932"
# The nominal CPU group currently has no schedulable nodes. This train group
# accepts CPU-only jobs when acceleratorCardKind=CPU and acceleratorCardNum=0.
DEFAULT_CPU_GROUP_ID = "359ccff0-e7ae-48f5-8cf0-7917f7b3d497"
DEFAULT_LOG_DIR = ""

STRICT_ROOT = "/ssdwork/liuhaohan/evorig/evoweave_combined_dynamic_strict_20260613_more_anim"
CANONICAL_ROOT = "/ssdwork/liuhaohan/evorig/evoweave_combined_dynamic_canonical_rootspace_20260630"
REPO_ROOT = "/ssdwork/liuhaohan/evorig/evoweave_repo/rigweave"
PYTHON_BIN = "/opt/conda/envs/evoweave/bin/python"
TASK_LOG = "/ssdwork/liuhaohan/evorig/logs/canonical_rootspace_cpu_20260630.log"


def default_command() -> str:
    manifests = [
        f"{STRICT_ROOT}/train_manifest.westlake.jsonl",
        f"{STRICT_ROOT}/val_manifest.westlake.jsonl",
        f"{STRICT_ROOT}/test_manifest.westlake.jsonl",
    ]
    manifest_args = " ".join(f"--manifest {path}" for path in manifests)
    return (
        "bash -lc 'set -euo pipefail; "
        "mkdir -p /ssdwork/liuhaohan/evorig/logs; "
        f"cd {REPO_ROOT}; "
        "echo canonical_rootspace_cpu_start $(date -Is); "
        f"{PYTHON_BIN} scripts/build_canonical_dynamic_npz.py "
        f"{manifest_args} "
        f"--output-root {CANONICAL_ROOT} "
        "--verify-vertices 64 "
        "--compression stored "
        "--skip-existing "
        f"2>&1 | tee {TASK_LOG}; "
        "status=${PIPESTATUS[0]}; "
        "echo canonical_rootspace_cpu_end $(date -Is) status=${status}; "
        "exit ${status}'"
    )


def load_connection() -> tuple[str, str | None, str | None]:
    base_url = os.environ.get("GPU_PLATFORM_BASE_URL")
    account = os.environ.get("GPU_PLATFORM_ACCOUNT")
    token = None
    cache_path = Path(token_cache_path())
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        base_url = base_url or data.get("base_url")
        account = account or data.get("account")
        token = data.get("token")
    if not base_url:
        raise SystemExit("Missing GPU_PLATFORM_BASE_URL and no cached base_url found.")
    return base_url.rstrip("/"), account, token


def make_authenticated_session(base_url: str, account: str | None, token: str | None):
    if account:
        return login(
            base_url,
            account,
            os.environ.get("GPU_PLATFORM_PASSWORD"),
            os.environ.get("GPU_PLATFORM_CAPTCHA"),
        )
    if not token:
        raise SystemExit("No account and no cached token available.")
    return make_session(token)


def build_payload(args: argparse.Namespace) -> dict:
    config = {
        "worker": {
            "nodeNum": 1,
            "cpuNum": args.cpu,
            "acceleratorCardNum": 0,
            "memory": 0,
            "minNodeNum": -1,
        }
    }
    return {
        "name": args.name,
        "description": "",
        "projectId": args.project_id,
        "imageType": "pytorch",
        "resGroupId": args.group_id,
        "acceleratorCardKind": "CPU",
        "image": args.image,
        "mountDir": "",
        "startScript": "",
        "logOut": args.log_out,
        "distFlag": False,
        "enUpdateDataSet": 0,
        "ports": "",
        "param": "",
        "execDir": "",
        "nodeName": args.node_name,
        "mpiFlag": False,
        "type": "pytorch",
        "shmSize": args.shm_size,
        "datasetId": None,
        "emergencyFlag": False,
        "imageFlag": 0,
        "switchType": "ib",
        "isElastic": False,
        "models": [],
        "config": json.dumps(config, separators=(",", ":")),
        "command": args.command,
        "commandScriptList": [],
        "jobVolume": [],
    }


def check_resources(base_url: str, session, payload: dict) -> dict:
    body = {
        "type": payload["type"],
        "acceleratorCardKind": payload["acceleratorCardKind"],
        "distFlag": payload["distFlag"],
        "mpiFlag": payload["mpiFlag"],
        "resGroupId": payload["resGroupId"],
        "config": payload["config"],
    }
    data, _ = request_json(
        session,
        "POST",
        f"{base_url}/api/iresource/v1/train/check-resources",
        data=json.dumps(body),
    )
    return data


def recent_tasks(base_url: str, session, page_size: int = 10) -> list[dict]:
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/train",
        params={"page": 1, "pageSize": page_size, "statusFlag": 0},
    )
    return data.get("resData", {}).get("data", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument(
        "--name",
        default="evoweave_canonical_rootspace_cpu_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--group-id", default=DEFAULT_CPU_GROUP_ID)
    parser.add_argument("--node-name", default="")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--shm-size", type=int, default=8)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--command", default=default_command())
    parser.add_argument("--log-out", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    base_url, account, token = load_connection()
    session = make_authenticated_session(base_url, account, token)
    payload = build_payload(args)
    print("Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\nResource check:")
    check = check_resources(base_url, session, payload)
    print(json.dumps(check, ensure_ascii=False, indent=2))

    if not args.submit:
        print("\nDry run only. Add --submit to create the CPU training task.")
        return 0

    print("\nSubmitting CPU training task...")
    data, _ = request_json(
        session,
        "POST",
        f"{base_url}/api/iresource/v1/train",
        data=json.dumps(payload),
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))

    if args.poll:
        expected = args.name
        for _ in range(24):
            time.sleep(5)
            rows = recent_tasks(base_url, session)
            for row in rows:
                if row.get("name") == expected:
                    summary = {
                        "id": row.get("id"),
                        "name": row.get("name"),
                        "status": row.get("status"),
                        "statusReason": row.get("statusReason"),
                        "createTime": row.get("createTime"),
                        "startTime": row.get("startTime"),
                    }
                    print("\nMatched task:")
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                    return 0
        print("\nSubmitted, but the task was not visible in the first polling window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
