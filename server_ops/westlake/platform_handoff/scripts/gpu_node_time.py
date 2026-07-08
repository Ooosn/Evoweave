import base64
import getpass
import json
import os
import subprocess
import time
from datetime import timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
import urllib3
from gmssl import sm2


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def fmt_seconds(value):
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "-"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_current_ssh_pod(alias):
    if not alias:
        return None
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, "hostname"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def encrypt_password(password, public_key):
    public_key = public_key[-128:]
    crypt = sm2.CryptSM2(public_key=public_key, private_key="", mode=0)
    return "04" + crypt.encrypt(password.encode("utf-8")).hex()


def token_cache_path():
    return os.environ.get("GPU_PLATFORM_TOKEN_CACHE") or os.path.join(
        os.path.dirname(__file__), ".cache", "gpu_platform_token.json"
    )


def make_session(token=None):
    session = requests.Session()
    session.verify = False
    session.headers.update(
        {
            "X-Accept-Language": "zh_CN",
            "Content-Type": "application/json;charset=UTF-8",
        }
    )
    if token:
        session.headers.update({"X-Auth-Token": token})
    return session


def load_cached_token(base_url, account):
    path = token_cache_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        return None
    if data.get("base_url") != base_url or data.get("account") != account:
        return None
    return data.get("token")


def save_cached_token(base_url, account, token):
    path = token_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_url": base_url,
                "account": account,
                "token": token,
                "saved_at": int(time.time()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def validate_session(session, base_url):
    try:
        data, _ = request_json(session, "GET", f"{base_url}/api/ibase/v1/login")
    except Exception:
        return False
    return bool(data.get("resData"))


def request_json(session, method, url, **kwargs):
    response = session.request(method, url, timeout=20, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not data.get("flag"):
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    return data, response


def login(base_url, account, password=None, captcha_code=None):
    cached = load_cached_token(base_url, account)
    if cached:
        session = make_session(cached)
        if validate_session(session, base_url):
            return session

    if not password:
        password = getpass.getpass("Platform password: ")

    session = make_session()

    secret, _ = request_json(session, "GET", f"{base_url}/api/ibase/v1/system/secret")
    encrypted = encrypt_password(password, secret["resData"])

    captcha, _ = request_json(session, "GET", f"{base_url}/api/ibase/v1/captcha")
    captcha_image = captcha.get("resData")
    if captcha_image and not captcha_code:
        path = os.environ.get("GPU_PLATFORM_CAPTCHA_PATH") or os.path.join(
            os.getcwd(), "captcha.png"
        )
        with open(path, "wb") as f:
            f.write(base64.b64decode(captcha_image))
        print(f"Captcha required. Image saved to: {path}")
        captcha_code = input("Captcha code: ").strip()

    payload = {"account": account, "password": encrypted}
    if captcha_image:
        payload["captcha"] = captcha_code

    data, _ = request_json(
        session,
        "POST",
        f"{base_url}/api/ibase/v1/login",
        data=json.dumps(payload),
    )
    token = data["resData"]["token"]
    session.headers.update({"X-Auth-Token": token})
    save_cached_token(base_url, account, token)
    return session


def main():
    base_url = require_env("GPU_PLATFORM_BASE_URL").rstrip("/")
    account = require_env("GPU_PLATFORM_ACCOUNT")
    password = os.environ.get("GPU_PLATFORM_PASSWORD")
    captcha_code = os.environ.get("GPU_PLATFORM_CAPTCHA")
    ssh_alias = os.environ.get("GPU_NODE_SSH_ALIAS", "my-gpu-server")

    session = login(base_url, account, password, captcha_code)
    current_pod = get_current_ssh_pod(ssh_alias)

    base_data, _ = request_json(
        session, "GET", f"{base_url}/api/iresource/v1/work-platform/base"
    )
    status_data, status_response = request_json(
        session, "GET", f"{base_url}/api/iresource/v1/work-platform/status"
    )

    server_utc = parsedate_to_datetime(status_response.headers["Date"])
    if server_utc.tzinfo is None:
        server_utc = server_utc.replace(tzinfo=timezone.utc)
    cst = timezone(timedelta(hours=8))
    jst = timezone(timedelta(hours=9))

    by_id = {
        item["wpId"]: dict(item) for item in base_data["resData"].get("data", [])
    }
    for item in status_data["resData"].get("data", []):
        by_id.setdefault(item["wpId"], {}).update(item)

    print(f"Account: {account}")
    print(f"Server time UTC: {server_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current SSH pod: {current_pod or '-'}")
    print("")

    rows = list(by_id.values())
    rows.sort(key=lambda x: (x.get("wpStatus") != "Running", x.get("startTime") or ""), reverse=False)
    for row in rows:
        pods = row.get("workPlatformPodSummaryList") or []
        pod_names = [p.get("podName") for p in pods if p.get("podName")]
        marker = " *current*" if current_pod and current_pod in pod_names else ""
        remain = row.get("remainTime")
        try:
            expire_utc = server_utc + timedelta(seconds=int(float(remain)))
            expire_text = (
                f"{expire_utc.astimezone(cst).strftime('%Y-%m-%d %H:%M:%S')} CST / "
                f"{expire_utc.astimezone(jst).strftime('%Y-%m-%d %H:%M:%S')} JST"
            )
        except (TypeError, ValueError):
            expire_text = "-"

        print(f"- {row.get('wpName', '-')} [{row.get('wpStatus', '-')}] {marker}")
        print(f"  wpId: {row.get('wpId', '-')}")
        print(f"  pod: {', '.join(pod_names) if pod_names else '-'}")
        print(f"  start: {row.get('startTime', '-')}")
        print(f"  runtime: {fmt_seconds(row.get('runTime'))}")
        print(f"  remaining: {fmt_seconds(remain)}")
        print(f"  expires: {expire_text}")
        print("")


if __name__ == "__main__":
    main()
