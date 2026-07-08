import argparse
import json
import os
import posixpath
import random
import re
import string
import subprocess
import time
from datetime import datetime
from pathlib import Path

from gpu_node_time import (
    encrypt_password,
    fmt_seconds,
    get_current_ssh_pod,
    login,
    request_json,
    require_env,
)


def env_session():
    base_url = require_env("GPU_PLATFORM_BASE_URL").rstrip("/")
    account = require_env("GPU_PLATFORM_ACCOUNT")
    password = os.environ.get("GPU_PLATFORM_PASSWORD")
    captcha = os.environ.get("GPU_PLATFORM_CAPTCHA")
    return base_url, login(base_url, account, password, captcha)


def get_dev_env_rows(base_url, session):
    base_data, _ = request_json(
        session, "GET", f"{base_url}/api/iresource/v1/work-platform/base"
    )
    status_data, _ = request_json(
        session, "GET", f"{base_url}/api/iresource/v1/work-platform/status"
    )
    usage_data, _ = request_json(
        session, "GET", f"{base_url}/api/iresource/v1/work-platform/usage"
    )

    by_id = {
        item["wpId"]: dict(item) for item in base_data["resData"].get("data", [])
    }
    for source in (status_data, usage_data):
        for item in source["resData"].get("data", []):
            by_id.setdefault(item["wpId"], {}).update(item)
    return list(by_id.values())


def pod_names(row):
    pods = row.get("workPlatformPodSummaryList") or []
    return [p.get("podName") for p in pods if p.get("podName")]


def require_confirm(args, action):
    if not getattr(args, "confirm", False):
        raise SystemExit(f"Refusing to {action} without --confirm.")


def select_source(rows, source_wp_id=None, source_name=None, ssh_alias=None):
    if source_wp_id:
        for row in rows:
            if row.get("wpId") == source_wp_id:
                return row
        raise SystemExit(f"No development environment with wpId: {source_wp_id}")

    if source_name:
        matches = [row for row in rows if row.get("wpName") == source_name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise SystemExit(f"No development environment named: {source_name}")
        raise SystemExit(f"Multiple development environments named: {source_name}")

    current_pod = get_current_ssh_pod(ssh_alias or "my-gpu-server")
    if current_pod:
        for row in rows:
            if current_pod in pod_names(row):
                return row

    running = [row for row in rows if row.get("wpStatus") == "Running"]
    if len(running) == 1:
        return running[0]
    if running:
        running.sort(key=lambda row: row.get("startTime") or "", reverse=True)
        return running[0]
    if rows:
        rows.sort(key=lambda row: row.get("startTime") or "", reverse=True)
        return rows[0]
    raise SystemExit("No development environments found.")


def cmd_list(args):
    base_url, session = env_session()
    rows = get_dev_env_rows(base_url, session)
    current_pod = get_current_ssh_pod(args.ssh_alias)
    rows.sort(key=lambda row: (row.get("wpStatus") != "Running", row.get("startTime") or ""))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"Development environments: {len(rows)}")
    print(f"Current SSH pod: {current_pod or '-'}")
    print("")
    for row in rows:
        names = pod_names(row)
        marker = " *current*" if current_pod and current_pod in names else ""
        print(f"- {row.get('wpName', '-')} [{row.get('wpStatus', '-')}] {marker}")
        print(f"  wpId: {row.get('wpId', '-')}")
        print(f"  pod: {', '.join(names) if names else '-'}")
        print(f"  start: {row.get('startTime', '-')}")
        print(f"  remaining: {fmt_seconds(row.get('remainTime'))}")
        print(
            "  spec: "
            f"cpu={row.get('cpu', '-')} "
            f"gpu={row.get('acceleratorCard', '-')} "
            f"gpuType={row.get('acceleratorCardType', '-')}"
        )
        print(f"  image: {row.get('image', '-')}")
        print("")


def get_template(base_url, session, wp_id):
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/work-platform/{wp_id}/rebuild",
    )
    return data["resData"]


def cmd_template(args):
    base_url, session = env_session()
    rows = get_dev_env_rows(base_url, session)
    source = select_source(rows, args.source_wp_id, args.source_name, args.ssh_alias)
    template = get_template(base_url, session, source["wpId"])
    print(json.dumps(template, ensure_ascii=False, indent=2))


def normalize_create_payload(template):
    payload = dict(template)
    if payload.get("ports") is None:
        payload["ports"] = []
    if payload.get("volumes") is None:
        payload["volumes"] = []
    if payload.get("nodeList") is None:
        payload["nodeList"] = []
    return payload


def select_resource_group_for_gpu_type(base_url, session, gpu_type, gpu_count):
    if not gpu_type:
        return None
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/node",
        params={"page": 1, "pageSize": 100},
    )
    groups = {}
    for row in data.get("resData", {}).get("data", []):
        if row.get("nodeStatus") != "ready":
            continue
        if row.get("cardType") != gpu_type and row.get("nodeCardType") != gpu_type:
            continue
        group_id = row.get("groupId")
        if not group_id:
            continue
        total = int(row.get("acceleratorCard") or 0)
        used = int(row.get("acceleratorCardUsage") or 0)
        free = max(total - used, 0)
        item = groups.setdefault(
            group_id,
            {
                "groupId": group_id,
                "groupName": row.get("groupName") or "",
                "free": 0,
                "nodes": 0,
            },
        )
        item["free"] += free
        item["nodes"] += 1
    candidates = [g for g in groups.values() if g["free"] >= int(gpu_count or 1)]
    if not candidates:
        return None

    def sort_key(item):
        name = item["groupName"]
        upper_name = name.upper()
        if "40GB" in gpu_type:
            standard_penalty = 0 if "A100_40" in upper_name else 2
        elif "80GB" in gpu_type:
            standard_penalty = 0 if ("A100_80" in upper_name or "A800" in upper_name) else 2
        else:
            standard_penalty = 0 if "A100" in upper_name else 2
        preferred_penalty = 0
        if "A800" in gpu_type and "dev" not in name.lower():
            preferred_penalty = 1
        if "large" in name.lower():
            preferred_penalty = 1
        if "dev" in name.lower() and "A800" not in gpu_type:
            preferred_penalty = 2
        return (standard_penalty, preferred_penalty, -int(item["free"]), name)

    candidates.sort(key=sort_key)
    return candidates[0]


def apply_env_overrides(base_url, session, payload, args, default_new_name=None):
    if default_new_name is not None:
        payload["wpName"] = args.new_name or default_new_name
    if not args.keep_node_list:
        payload["nodeList"] = []
    if args.gpu_count is not None:
        payload["acceleratorCard"] = args.gpu_count
    if args.gpu_type:
        payload["acceleratorCardType"] = args.gpu_type
        if not getattr(args, "group_id", None):
            group = select_resource_group_for_gpu_type(
                base_url,
                session,
                args.gpu_type,
                args.gpu_count if args.gpu_count is not None else payload.get("acceleratorCard"),
            )
            if group is not None:
                payload["groupId"] = group["groupId"]
                payload["groupName"] = group["groupName"]
    if getattr(args, "group_id", None):
        payload["groupId"] = args.group_id
    if getattr(args, "group_name", None):
        payload["groupName"] = args.group_name
    if args.cpu is not None:
        payload["cpu"] = args.cpu
    if args.memory is not None:
        payload["memory"] = args.memory
    if args.image:
        payload["image"] = args.image
    elif args.image_name:
        payload["image"] = resolve_image(
            base_url, session, args.image_name, args.image_tag, args.image_type
        )
        payload["frameWork"] = args.image_type
    if args.shm_size is not None:
        payload["shmSize"] = args.shm_size
    if args.command is not None:
        payload["command"] = args.command
    return payload


def cmd_plan(args):
    base_url, session = env_session()
    rows = get_dev_env_rows(base_url, session)
    source = select_source(rows, args.source_wp_id, args.source_name, args.ssh_alias)
    payload = normalize_create_payload(get_template(base_url, session, source["wpId"]))
    apply_env_overrides(
        base_url,
        session,
        payload,
        args,
        default_new_name=datetime.now().strftime("%Y%m%d%H%M%S"),
    )

    print("Dry run only. No development environment was created.")
    print("")
    print("POST /api/iresource/v1/work-platform/")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def get_work_platform_pods(base_url, session, wp_id):
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/work-platform/{wp_id}/shell",
    )
    return data["resData"] or []


def get_jupyter_pods(base_url, session, wp_id):
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/work-platform/{wp_id}/jupyter",
    )
    return data["resData"] or []


def selected_wp(base_url, session, args):
    rows = get_dev_env_rows(base_url, session)
    return select_source(
        rows,
        getattr(args, "source_wp_id", None),
        getattr(args, "source_name", None),
        getattr(args, "ssh_alias", None),
    )


def get_first_pod_id(base_url, session, wp_id):
    for pod in get_jupyter_pods(base_url, session, wp_id):
        if pod.get("podId"):
            return pod["podId"]
    for pod in get_work_platform_pods(base_url, session, wp_id):
        if pod.get("podId"):
            return pod["podId"]
    raise SystemExit(f"No podId found for work platform: {wp_id}")


def cmd_commit_image(args):
    require_confirm(args, "commit image")
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    pod_id = args.pod_id or get_first_pod_id(base_url, session, source["wpId"])

    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/images/check",
        params={
            "imageName": args.image_name,
            "imageType": args.image_type,
            "imageTag": args.image_tag,
        },
    )
    if data["resData"] and not args.overwrite:
        raise SystemExit(
            "Image already exists. Re-run with --overwrite if replacement is intended."
        )

    payload = {
        "imageName": args.image_name,
        "imageTag": args.image_tag,
        "imageComment": args.memo or "",
        "wpId": source["wpId"],
        "podId": pod_id,
        "imageType": args.image_type,
    }
    result, _ = request_json(
        session,
        "POST",
        f"{base_url}/api/iresource/v1/work-platform/commit-image",
        data=json.dumps(payload),
    )
    print("Commit image submitted.")
    print(json.dumps({"payload": payload, "response": result["resData"]}, ensure_ascii=False, indent=2))

    if args.wait:
        wait_for_image(
            base_url,
            session,
            args.image_name,
            args.image_tag,
            args.wait_seconds,
            args.max_seconds,
        )


def get_image_progress_rows(base_url, session):
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/images/progress",
        params={"page": 1, "pageSize": 50},
    )
    return data["resData"].get("data", [])


def image_name_matches(row_name, image_name):
    if row_name == image_name:
        return True
    return bool(
        row_name
        and (
            row_name.endswith("/" + image_name)
            or row_name.endswith("/" + image_name + "-liuhaohan")
        )
    )


def latest_image_progress(base_url, session, image_name, image_tag):
    rows = [
        row
        for row in get_image_progress_rows(base_url, session)
        if image_name_matches(row.get("imageName"), image_name)
        and row.get("imageTag") == image_tag
        and row.get("operationType") == 4
    ]
    rows.sort(key=lambda row: row.get("createTime") or "", reverse=True)
    return rows[0] if rows else None


def wait_deadline(wait_seconds, max_seconds):
    now = time.time()
    idle_deadline = now + wait_seconds
    max_deadline = now + max_seconds if max_seconds and max_seconds > 0 else None
    return idle_deadline, max_deadline


def wait_expired(idle_deadline, max_deadline):
    now = time.time()
    return now >= idle_deadline or (max_deadline is not None and now >= max_deadline)


def extend_idle_wait(wait_seconds, max_deadline):
    deadline = time.time() + wait_seconds
    if max_deadline is not None:
        return min(deadline, max_deadline)
    return deadline


def image_wait_key(row):
    if not row:
        return None
    return (
        row.get("id"),
        row.get("imageStatus"),
        row.get("imageProgress"),
        row.get("queueLocation"),
        row.get("updateTime"),
    )


def wait_for_image(base_url, session, image_name, image_tag, wait_seconds, max_seconds=0):
    idle_deadline, max_deadline = wait_deadline(wait_seconds, max_seconds)
    last = None
    last_key = object()
    while not wait_expired(idle_deadline, max_deadline):
        row = latest_image_progress(base_url, session, image_name, image_tag)
        key = image_wait_key(row)
        if key != last_key:
            idle_deadline = extend_idle_wait(wait_seconds, max_deadline)
            last_key = key
        if row:
            last = row
            status = row.get("imageStatus")
            progress = row.get("imageProgress")
            print(f"Image progress: status={status} progress={progress}%")
            if status == 2:
                print("Image commit succeeded.")
                return row
            if status == 3:
                raise SystemExit(
                    "Image commit failed: " + str(row.get("exceptionReason") or "-")
                )
        else:
            print("Image progress: waiting for task row...")
        time.sleep(10)
    raise SystemExit(
        "Timed out waiting for image commit to change or finish. Last row: "
        + json.dumps(last, ensure_ascii=False)
    )


def cmd_image_progress(args):
    base_url, session = env_session()
    if args.wait:
        wait_for_image(
            base_url,
            session,
            args.image_name,
            args.image_tag,
            args.wait_seconds,
            args.max_seconds,
        )
        return
    row = latest_image_progress(base_url, session, args.image_name, args.image_tag)
    print(json.dumps(row or {}, ensure_ascii=False, indent=2))


def resolve_image(base_url, session, image_name, image_tag, image_type):
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/images",
        params={
            "imageType": image_type,
            "page": 1,
            "pageSize": 100,
            "share": "",
            "tag": image_name,
        },
    )
    for row in data["resData"].get("data", []):
        if image_name_matches(row.get("imageName"), image_name) and row.get("imageTag") == image_tag:
            return row.get("imageName") + ":" + row.get("imageTag")
    return image_name + ":" + image_tag


def cmd_create(args):
    require_confirm(args, "create development environment")
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    payload = normalize_create_payload(get_template(base_url, session, source["wpId"]))
    apply_env_overrides(
        base_url,
        session,
        payload,
        args,
        default_new_name=datetime.now().strftime("%Y%m%d%H%M%S"),
    )

    result, _ = request_json(
        session,
        "POST",
        f"{base_url}/api/iresource/v1/work-platform/",
        data=json.dumps(payload),
    )
    print("Create development environment submitted.")
    print(json.dumps({"payload": payload, "response": result["resData"]}, ensure_ascii=False, indent=2))

    if args.wait:
        row = wait_for_wp(
            base_url,
            session,
            payload["wpName"],
            args.wait_seconds,
            args.max_seconds,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))


def submit_existing_wp_action(args, action, method, path, wait_for_running=False):
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    payload = normalize_create_payload(get_template(base_url, session, source["wpId"]))
    apply_env_overrides(base_url, session, payload, args)
    url_path = path.format(wp_id=source["wpId"])

    if getattr(args, "dry_run", False) or not getattr(args, "confirm", False):
        print(f"Dry run only. Re-run with --confirm to {action}.")
        print("")
        print(f"{method.upper()} {url_path}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return None

    result, _ = request_json(
        session,
        method.upper(),
        f"{base_url}{url_path}",
        data=json.dumps(payload),
    )
    print(f"{action.capitalize()} submitted.")
    print(json.dumps({"wpId": source["wpId"], "wpName": source.get("wpName"), "response": result["resData"]}, ensure_ascii=False, indent=2))

    if getattr(args, "wait", False):
        wait_for_wp_id(
            base_url,
            session,
            source["wpId"],
            args.wait_seconds,
            args.max_seconds,
            success_statuses={"Running"} if wait_for_running else None,
        )
    return result


def cmd_restart(args):
    submit_existing_wp_action(
        args,
        "restart development environment",
        "POST",
        "/api/iresource/v1/work-platform/{wp_id}",
        wait_for_running=True,
    )


def cmd_resume(args):
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/work-platform/restore/{source['wpId']}",
        params={"wpId": source["wpId"]},
    )
    payload = normalize_create_payload(data["resData"])
    apply_env_overrides(base_url, session, payload, args)

    if args.dry_run or not args.confirm:
        print("Dry run only. Re-run with --confirm to resume development environment.")
        print("")
        print(f"PUT /api/iresource/v1/work-platform/restore/{source['wpId']}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    result, _ = request_json(
        session,
        "PUT",
        f"{base_url}/api/iresource/v1/work-platform/restore/{source['wpId']}",
        data=json.dumps(payload),
    )
    print("Resume development environment submitted.")
    print(json.dumps({"wpId": source["wpId"], "wpName": source.get("wpName"), "response": result["resData"]}, ensure_ascii=False, indent=2))

    if args.wait:
        wait_for_wp_id(
            base_url,
            session,
            source["wpId"],
            args.wait_seconds,
            args.max_seconds,
            success_statuses={"Running"},
        )


def cmd_resource_change(args):
    submit_existing_wp_action(
        args,
        "resource change",
        "POST",
        "/api/iresource/v1/work-platform/{wp_id}/reconfig",
        wait_for_running=True,
    )


def wp_wait_key(row):
    if not row:
        return None
    pods = tuple(
        (pod.get("podName"), pod.get("podStatus"))
        for pod in row.get("workPlatformPodSummaryList") or []
    )
    return (
        row.get("wpId"),
        row.get("wpStatus"),
        pods,
        row.get("datasetPullProgress"),
        row.get("startTime"),
        row.get("deleteTime"),
    )


def wait_for_wp(base_url, session, wp_name, wait_seconds, max_seconds=0):
    idle_deadline, max_deadline = wait_deadline(wait_seconds, max_seconds)
    last = None
    last_key = object()
    while not wait_expired(idle_deadline, max_deadline):
        rows = get_dev_env_rows(base_url, session)
        matches = [row for row in rows if row.get("wpName") == wp_name]
        if matches:
            matches.sort(key=lambda row: row.get("createDateTime") or "", reverse=True)
            last = matches[0]
            key = wp_wait_key(last)
            if key != last_key:
                idle_deadline = extend_idle_wait(wait_seconds, max_deadline)
                last_key = key
            print(
                f"Development environment: {wp_name} "
                f"status={last.get('wpStatus')} pods={','.join(pod_names(last)) or '-'}"
            )
            if last.get("wpStatus") == "Running":
                return last
            if last.get("wpStatus") in {"Failed", "InitError", "Halt"}:
                raise SystemExit(
                    "Development environment failed or halted: "
                    + json.dumps(last, ensure_ascii=False)
                )
        else:
            if last_key is not None:
                idle_deadline = extend_idle_wait(wait_seconds, max_deadline)
                last_key = None
            print(f"Development environment: {wp_name} waiting for row...")
        time.sleep(10)
    raise SystemExit(
        "Timed out waiting for development environment to change or finish. Last row: "
        + json.dumps(last, ensure_ascii=False)
    )


def wait_for_wp_id(
    base_url,
    session,
    wp_id,
    wait_seconds,
    max_seconds=0,
    success_statuses=None,
    failure_statuses=None,
):
    success_statuses = success_statuses or {"Running", "Halt", "Pause"}
    failure_statuses = failure_statuses or {"Failed", "InitError", "Error"}
    idle_deadline, max_deadline = wait_deadline(wait_seconds, max_seconds)
    last = None
    last_key = object()
    while not wait_expired(idle_deadline, max_deadline):
        rows = get_dev_env_rows(base_url, session)
        matches = [row for row in rows if row.get("wpId") == wp_id]
        last = matches[0] if matches else None
        key = wp_wait_key(last)
        if key != last_key:
            idle_deadline = extend_idle_wait(wait_seconds, max_deadline)
            last_key = key
        if not last:
            print("Development environment no longer listed.")
            return None
        print(
            f"Development environment: {last.get('wpName', wp_id)} "
            f"status={last.get('wpStatus')} pods={','.join(pod_names(last)) or '-'}"
        )
        if last.get("wpStatus") in success_statuses:
            return last
        if last.get("wpStatus") in failure_statuses:
            raise SystemExit(
                "Development environment failed: "
                + json.dumps(last, ensure_ascii=False)
            )
        time.sleep(10)
    raise SystemExit(
        "Timed out waiting for development environment to change or finish. Last row: "
        + json.dumps(last, ensure_ascii=False)
    )


def cmd_terminate(args):
    require_confirm(args, "terminate development environment")
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    result, _ = request_json(
        session,
        "DELETE",
        f"{base_url}/api/iresource/v1/work-platform/{source['wpId']}",
    )
    print("Terminate development environment submitted.")
    print(json.dumps({"wpId": source["wpId"], "wpName": source.get("wpName"), "response": result["resData"]}, ensure_ascii=False, indent=2))

    if args.wait:
        wait_for_wp_terminated(
            base_url,
            session,
            source["wpId"],
            args.wait_seconds,
            args.max_seconds,
        )


def cmd_pause(args):
    require_confirm(args, "pause development environment")
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    result, _ = request_json(
        session,
        "PUT",
        f"{base_url}/api/iresource/v1/work-platform/pause/{source['wpId']}",
    )
    print("Pause development environment submitted.")
    print(json.dumps({"wpId": source["wpId"], "wpName": source.get("wpName"), "response": result["resData"]}, ensure_ascii=False, indent=2))

    if args.wait:
        wait_for_wp_id(
            base_url,
            session,
            source["wpId"],
            args.wait_seconds,
            args.max_seconds,
            success_statuses={"Pause", "Halt"},
        )


def wait_for_wp_terminated(base_url, session, wp_id, wait_seconds, max_seconds=0):
    idle_deadline, max_deadline = wait_deadline(wait_seconds, max_seconds)
    last = None
    last_key = object()
    while not wait_expired(idle_deadline, max_deadline):
        rows = get_dev_env_rows(base_url, session)
        matches = [row for row in rows if row.get("wpId") == wp_id]
        last = matches[0] if matches else None
        key = wp_wait_key(last)
        if key != last_key:
            idle_deadline = extend_idle_wait(wait_seconds, max_deadline)
            last_key = key
        if not last:
            print("Development environment no longer listed.")
            return None
        print(f"Development environment terminate wait: status={last.get('wpStatus')}")
        if last.get("wpStatus") == "Halt":
            return last
        time.sleep(10)
    raise SystemExit(
        "Timed out waiting for termination to change or finish. Last row: "
        + json.dumps(last, ensure_ascii=False)
    )


def aes_decrypt_hex(cipher_hex, key):
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    cipher = Cipher(
        algorithms.AES(key.encode("utf-8")),
        modes.CBC(b"87EEA6BC8A6383B3"),
    )
    decryptor = cipher.decryptor()
    padded = decryptor.update(bytes.fromhex(cipher_hex)) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode("utf-8")


def random_aes_key():
    alphabet = "ABCDEFGHIJKLMNUPQRSTUVWXYZabcdefghijklmnupqrstuvwxyz0123456789"
    return "".join(random.choice(alphabet) for _ in range(16))


def get_decrypted_ssh_info(base_url, session, wp_id, pod_id):
    secret, _ = request_json(session, "GET", f"{base_url}/api/ibase/v1/system/secret")
    key = random_aes_key()
    cipher_key = encrypt_password(key, secret["resData"])
    data, _ = request_json(
        session,
        "GET",
        f"{base_url}/api/iresource/v1/work-platform/{wp_id}/pod/{pod_id}/shell",
        params={"k": cipher_key},
    )
    info = data["resData"]
    info["decryptedPassword"] = aes_decrypt_hex(info["password"], key)
    return info


def parse_ssh_user(info):
    command = info.get("sshCommand") or ""
    match = re.search(r"ssh\s+([^@\s]+)@", command)
    return match.group(1) if match else "root"


def ensure_ssh_key(key_path):
    key_path = Path(key_path).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if not key_path.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", "codex-my-gpu-server"],
            check=True,
        )
    pub_path = Path(str(key_path) + ".pub")
    if not pub_path.exists():
        raise SystemExit(f"Public key not found: {pub_path}")
    return key_path, pub_path.read_text(encoding="utf-8").strip()


def install_pubkey(host, port, user, password, pubkey):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=int(port),
        username=user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
    )
    try:
        _, stdout, _ = client.exec_command('printf %s "$HOME"')
        home = stdout.read().decode("utf-8", errors="replace").strip()
        if not home:
            home = "/root" if user == "root" else "/home/" + user

        sftp = client.open_sftp()
        try:
            ssh_dir = posixpath.join(home, ".ssh")
            auth_keys = posixpath.join(ssh_dir, "authorized_keys")
            try:
                sftp.mkdir(ssh_dir, mode=0o700)
            except OSError:
                pass
            try:
                existing = sftp.file(auth_keys, "r").read().decode("utf-8", errors="replace")
            except OSError:
                existing = ""
            if pubkey not in existing:
                with sftp.file(auth_keys, "a") as f:
                    if existing and not existing.endswith("\n"):
                        f.write("\n")
                    f.write(pubkey + "\n")
            sftp.chmod(ssh_dir, 0o700)
            sftp.chmod(auth_keys, 0o600)
        finally:
            sftp.close()
    finally:
        client.close()


def update_ssh_config(alias, host, port, user, key_path):
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    config_path = ssh_dir / "config"
    lines = config_path.read_text(encoding="utf-8").splitlines() if config_path.exists() else []

    filtered = []
    skip = False
    for line in lines:
        match = re.match(r"^\s*Host\s+(.+?)\s*$", line)
        if match:
            skip = alias in match.group(1).split()
        if not skip:
            filtered.append(line)

    while filtered and not filtered[-1].strip():
        filtered.pop()
    if filtered:
        filtered.append("")

    key_path_text = str(Path(key_path).expanduser()).replace("\\", "/")
    filtered.extend(
        [
            f"Host {alias}",
            f"  HostName {host}",
            f"  User {user}",
            f"  Port {port}",
            f"  IdentityFile {key_path_text}",
            "  IdentitiesOnly yes",
            "  PreferredAuthentications publickey,password",
            "  StrictHostKeyChecking accept-new",
            "  ServerAliveInterval 60",
            "  ServerAliveCountMax 3",
        ]
    )
    config_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def cmd_ssh_config(args):
    require_confirm(args, "configure SSH")
    base_url, session = env_session()
    source = selected_wp(base_url, session, args)
    pod_id = args.pod_id or get_first_pod_id(base_url, session, source["wpId"])
    info = get_decrypted_ssh_info(base_url, session, source["wpId"], pod_id)

    host = info["sshIp"]
    port = str(info["sshPort"])
    user = args.user or parse_ssh_user(info)
    key_path, pubkey = ensure_ssh_key(args.key_path)
    install_pubkey(host, port, user, info["decryptedPassword"], pubkey)
    update_ssh_config(args.host_alias, host, port, user, key_path)

    subprocess.run(["ssh-keygen", "-R", f"[{host}]:{port}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ssh-keygen", "-R", host], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    test = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host_alias, "hostname; whoami; pwd"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if test.returncode != 0:
        raise SystemExit(test.stderr.strip() or "SSH test failed.")
    print("SSH configured.")
    print(json.dumps({
        "hostAlias": args.host_alias,
        "host": host,
        "port": port,
        "user": user,
        "podId": pod_id,
        "test": test.stdout.strip().splitlines(),
    }, ensure_ascii=False, indent=2))


def cmd_resources(args):
    base_url, session = env_session()
    result = get_resource_snapshot(base_url, session)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def get_resource_snapshot(base_url, session):
    endpoints = [
        ("work-platform config", "/api/iresource/v1/work-platform/config"),
        ("work-platform summary", "/api/iresource/v1/work-platform/summary"),
        ("user quota", "/api/iresource/v1/user-quota"),
        ("nodes", "/api/iresource/v1/node"),
        ("vgpu usage", "/api/iresource/v1/node/vgpu-usage"),
    ]

    result = {}
    for label, path in endpoints:
        try:
            data, _ = request_json(session, "GET", f"{base_url}{path}")
            result[label] = data["resData"]
        except Exception as exc:
            result[label] = {"error": str(exc)}
    return result


def is_scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


CARD_TERMS = ("gpu", "vgpu", "card", "accelerator")
RESOURCE_TERMS = (
    "alloc",
    "available",
    "card",
    "count",
    "free",
    "gpu",
    "idle",
    "kind",
    "memory",
    "name",
    "quota",
    "remain",
    "total",
    "type",
    "used",
    "vgpu",
)


def compact_resource_dict(value):
    compact = {}
    for key, item in value.items():
        key_text = str(key).lower()
        if is_scalar(item) and any(term in key_text for term in RESOURCE_TERMS):
            compact[key] = item
    return compact


def find_card_candidates(value, path="root"):
    candidates = []
    if isinstance(value, dict):
        key_text = " ".join(str(key).lower() for key in value.keys())
        compact = compact_resource_dict(value)
        if compact and any(term in key_text for term in CARD_TERMS):
            candidates.append({"path": path, "data": compact})
        for key, item in value.items():
            candidates.extend(find_card_candidates(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            candidates.extend(find_card_candidates(item, f"{path}[{index}]"))
    return candidates


def unique_candidates(candidates):
    seen = set()
    result = []
    for item in candidates:
        key = json.dumps(item["data"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def cmd_cards(args):
    base_url, session = env_session()
    snapshot = get_resource_snapshot(base_url, session)
    candidates = unique_candidates(find_card_candidates(snapshot))

    if args.json:
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
        return

    if not candidates:
        print("No GPU/card resource fields found. Run `resources` for raw data.")
        return

    print("GPU/card resource candidates:")
    for item in candidates:
        print(f"- {item['path']}")
        for key, value in item["data"].items():
            print(f"  {key}: {value}")
        print("")


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect GPU platform development environments.")
    parser.add_argument("--ssh-alias", default=os.environ.get("GPU_NODE_SSH_ALIAS", "my-gpu-server"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List development environments.")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_template = sub.add_parser("template", help="Print a rebuild template.")
    p_template.add_argument("--source-wp-id")
    p_template.add_argument("--source-name")
    p_template.set_defaults(func=cmd_template)

    p_plan = sub.add_parser("plan", help="Print a dry-run create payload.")
    p_plan.add_argument("--source-wp-id")
    p_plan.add_argument("--source-name")
    p_plan.add_argument("--new-name")
    p_plan.add_argument("--gpu-count", type=int)
    p_plan.add_argument("--gpu-type")
    p_plan.add_argument("--group-id")
    p_plan.add_argument("--group-name")
    p_plan.add_argument("--cpu", type=int)
    p_plan.add_argument("--memory", type=int, default=0, help="Use 0 for platform default memory; do not set node physical memory.")
    p_plan.add_argument("--image")
    p_plan.add_argument("--image-name")
    p_plan.add_argument("--image-tag", default="latest")
    p_plan.add_argument("--image-type", default="pytorch")
    p_plan.add_argument("--shm-size", type=int)
    p_plan.add_argument("--command")
    p_plan.add_argument("--keep-node-list", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_resources = sub.add_parser("resources", help="Print read-only resource information.")
    p_resources.set_defaults(func=cmd_resources)

    p_cards = sub.add_parser("cards", help="Print GPU/card resource candidates.")
    p_cards.add_argument("--json", action="store_true")
    p_cards.set_defaults(func=cmd_cards)

    p_commit = sub.add_parser("commit-image", help="Commit a development environment image.")
    p_commit.add_argument("--source-wp-id")
    p_commit.add_argument("--source-name")
    p_commit.add_argument("--pod-id")
    p_commit.add_argument("--image-name", required=True)
    p_commit.add_argument("--image-tag", default="latest")
    p_commit.add_argument("--image-type", default="pytorch")
    p_commit.add_argument("--memo")
    p_commit.add_argument("--overwrite", action="store_true")
    p_commit.add_argument("--wait", action="store_true")
    p_commit.add_argument("--wait-seconds", type=int, default=900)
    p_commit.add_argument("--max-seconds", type=int, default=0)
    p_commit.add_argument("--confirm", action="store_true")
    p_commit.set_defaults(func=cmd_commit_image)

    p_progress = sub.add_parser("image-progress", help="Print or wait for image commit progress.")
    p_progress.add_argument("--image-name", required=True)
    p_progress.add_argument("--image-tag", default="latest")
    p_progress.add_argument("--wait", action="store_true")
    p_progress.add_argument("--wait-seconds", type=int, default=900)
    p_progress.add_argument("--max-seconds", type=int, default=0)
    p_progress.set_defaults(func=cmd_image_progress)

    p_create = sub.add_parser("create", help="Create a development environment.")
    p_create.add_argument("--source-wp-id")
    p_create.add_argument("--source-name")
    p_create.add_argument("--new-name")
    p_create.add_argument("--image")
    p_create.add_argument("--image-name")
    p_create.add_argument("--image-tag", default="latest")
    p_create.add_argument("--image-type", default="pytorch")
    p_create.add_argument("--gpu-count", type=int)
    p_create.add_argument("--gpu-type")
    p_create.add_argument("--group-id")
    p_create.add_argument("--group-name")
    p_create.add_argument("--cpu", type=int)
    p_create.add_argument("--memory", type=int, default=0, help="Use 0 for platform default memory; do not set node physical memory.")
    p_create.add_argument("--shm-size", type=int)
    p_create.add_argument("--command")
    p_create.add_argument("--keep-node-list", action="store_true")
    p_create.add_argument("--wait", action="store_true")
    p_create.add_argument("--wait-seconds", type=int, default=900)
    p_create.add_argument("--max-seconds", type=int, default=0)
    p_create.add_argument("--confirm", action="store_true")
    p_create.set_defaults(func=cmd_create)

    p_restart = sub.add_parser("restart", help="Restart an existing development environment with an updated template.")
    p_restart.add_argument("--source-wp-id")
    p_restart.add_argument("--source-name")
    p_restart.add_argument("--image")
    p_restart.add_argument("--image-name")
    p_restart.add_argument("--image-tag", default="latest")
    p_restart.add_argument("--image-type", default="pytorch")
    p_restart.add_argument("--gpu-count", type=int)
    p_restart.add_argument("--gpu-type")
    p_restart.add_argument("--cpu", type=int)
    p_restart.add_argument("--memory", type=int, default=0, help="Use 0 for platform default memory; do not set node physical memory.")
    p_restart.add_argument("--shm-size", type=int)
    p_restart.add_argument("--command")
    p_restart.add_argument("--keep-node-list", action="store_true")
    p_restart.add_argument("--dry-run", action="store_true")
    p_restart.add_argument("--wait", action="store_true")
    p_restart.add_argument("--wait-seconds", type=int, default=900)
    p_restart.add_argument("--max-seconds", type=int, default=0)
    p_restart.add_argument("--confirm", action="store_true")
    p_restart.set_defaults(func=cmd_restart)

    p_resume = sub.add_parser("resume", help="Resume a halted/paused development environment.")
    p_resume.add_argument("--source-wp-id")
    p_resume.add_argument("--source-name")
    p_resume.add_argument("--image")
    p_resume.add_argument("--image-name")
    p_resume.add_argument("--image-tag", default="latest")
    p_resume.add_argument("--image-type", default="pytorch")
    p_resume.add_argument("--gpu-count", type=int)
    p_resume.add_argument("--gpu-type")
    p_resume.add_argument("--cpu", type=int)
    p_resume.add_argument("--memory", type=int, default=0, help="Use 0 for platform default memory; do not set node physical memory.")
    p_resume.add_argument("--shm-size", type=int)
    p_resume.add_argument("--command")
    p_resume.add_argument("--keep-node-list", action="store_true")
    p_resume.add_argument("--dry-run", action="store_true")
    p_resume.add_argument("--wait", action="store_true")
    p_resume.add_argument("--wait-seconds", type=int, default=900)
    p_resume.add_argument("--max-seconds", type=int, default=0)
    p_resume.add_argument("--confirm", action="store_true")
    p_resume.set_defaults(func=cmd_resume)

    p_resource_change = sub.add_parser("resource-change", aliases=["reconfig"], help="Change resources on an existing development environment.")
    p_resource_change.add_argument("--source-wp-id")
    p_resource_change.add_argument("--source-name")
    p_resource_change.add_argument("--image")
    p_resource_change.add_argument("--image-name")
    p_resource_change.add_argument("--image-tag", default="latest")
    p_resource_change.add_argument("--image-type", default="pytorch")
    p_resource_change.add_argument("--gpu-count", type=int)
    p_resource_change.add_argument("--gpu-type")
    p_resource_change.add_argument("--cpu", type=int)
    p_resource_change.add_argument("--memory", type=int, default=0, help="Use 0 for platform default memory; do not set node physical memory.")
    p_resource_change.add_argument("--shm-size", type=int)
    p_resource_change.add_argument("--command")
    p_resource_change.add_argument("--keep-node-list", action="store_true")
    p_resource_change.add_argument("--dry-run", action="store_true")
    p_resource_change.add_argument("--wait", action="store_true")
    p_resource_change.add_argument("--wait-seconds", type=int, default=900)
    p_resource_change.add_argument("--max-seconds", type=int, default=0)
    p_resource_change.add_argument("--confirm", action="store_true")
    p_resource_change.set_defaults(func=cmd_resource_change)

    p_pause = sub.add_parser("pause", help="Pause a running development environment.")
    p_pause.add_argument("--source-wp-id")
    p_pause.add_argument("--source-name")
    p_pause.add_argument("--wait", action="store_true")
    p_pause.add_argument("--wait-seconds", type=int, default=300)
    p_pause.add_argument("--max-seconds", type=int, default=0)
    p_pause.add_argument("--confirm", action="store_true")
    p_pause.set_defaults(func=cmd_pause)

    p_terminate = sub.add_parser("terminate", help="Terminate a development environment.")
    p_terminate.add_argument("--source-wp-id")
    p_terminate.add_argument("--source-name")
    p_terminate.add_argument("--wait", action="store_true")
    p_terminate.add_argument("--wait-seconds", type=int, default=300)
    p_terminate.add_argument("--max-seconds", type=int, default=0)
    p_terminate.add_argument("--confirm", action="store_true")
    p_terminate.set_defaults(func=cmd_terminate)

    p_ssh = sub.add_parser("ssh-config", help="Configure local SSH alias for an environment.")
    p_ssh.add_argument("--source-wp-id")
    p_ssh.add_argument("--source-name")
    p_ssh.add_argument("--pod-id")
    p_ssh.add_argument("--host-alias", default=os.environ.get("GPU_NODE_SSH_ALIAS", "my-gpu-server"))
    p_ssh.add_argument("--user")
    p_ssh.add_argument("--key-path", default=str(Path.home() / ".ssh" / "id_ed25519_my_gpu_server"))
    p_ssh.add_argument("--confirm", action="store_true")
    p_ssh.set_defaults(func=cmd_ssh_config)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
