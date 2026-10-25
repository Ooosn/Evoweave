"""Admission checks for the explicitly authorized, read-only checkpoint study."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from train_legacy_single_target import file_sha256, require_job_membership, verify_exception_keeper

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = "model_training/rigweave/scripts/diagnose_id_xyz_checkpoint.py"
LAUNCHER = "model_training/jobs/launch_query_diagnostics.py"
SCOPE = "frozen_query_decoder_diagnostics_20261025_v1"
FAMILY = "frozen_id_xyz_checkpoint_diagnostics_v1"
EXCEPTION_ENV = "EVOWEAVE_QUERY_DIAGNOSTICS_SSH_EXCEPTION"
CPUS = [58, 63, 64, 65]


def record(state):
    value = state.get("active_operation", {}).get("query_decoder_diagnostics", {})
    if value.get("label") != SCOPE or value.get("explicit_human_authorized") is not True:
        raise RuntimeError("this inference study lacks its own human authorization")
    return value


def validate_ssh_exception(state, config):
    value = record(state)
    exception, runtime = value.get("ssh_exception", {}), config["runtime"]
    expected = {"scope": SCOPE, "explicit_human_authorized": True,
        "kernel_scheduler_membership": False, "allocation": 130309880,
        "host": "pcg01i", "physical_gpus": [5, 6], "cpu_slots": 1,
        "execution_cpu_budget": 4, "cpu_affinity": CPUS,
        "rss_limit_bytes": 68719476736, "gpu_memory_fraction_limit": .85,
        "keeper_pid": 1090013}
    for key, expected_value in expected.items():
        if type(exception.get(key)) is not type(expected_value) or exception[key] != expected_value:
            raise RuntimeError(f"SSH exception changed {key}")
        if key not in ("scope", "explicit_human_authorized", "kernel_scheduler_membership"):
            if type(runtime.get(key)) is not type(expected_value) or runtime[key] != expected_value:
                raise RuntimeError(f"resource binding changed {key}")
    if (config.get("family") != FAMILY or runtime.get("execution_mode") != "explicit_ssh_exception"
            or runtime.get("require_scheduler_membership") is not False
            or not exception.get("authorization")):
        raise RuntimeError("diagnostics must retain the declared bounded SSH provenance")
    return exception


def execution_cpu_binding(config, exception, allowed_cpus):
    if not exception or not set(CPUS) <= set(allowed_cpus):
        raise RuntimeError("the approved four-CPU binding is unavailable")
    return set(CPUS)


def check_deadline(config):
    deadline = datetime.fromisoformat(config["runtime"]["stop_before_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("the bounded diagnostic time window ended")


def validate_repeatability_protocol(config, op):
    protocol = config["diagnostics"]
    if protocol.get("require_historical_token_equality", True) is True:
        return
    evidence = config.get("repeatability_receipt", {})
    if (op.get("native_repeatability_accepted") is not True or protocol.get("native_repeats") != 3
            or protocol.get("require_fixed_condition_decoder_equality") is not True
            or file_sha256(evidence["path"]) != evidence["sha256"]):
        raise RuntimeError("paired repetition protocol requires accepted native-variation evidence")


def verify_launch(args, config):
    runtime = config["runtime"]
    state = json.loads((ROOT / "model_training/state/current.json").read_text())
    op = record(state)
    exception = validate_ssh_exception(state, config)
    verify_exception_keeper(exception)
    if os.environ.get(EXCEPTION_ENV) != "1":
        raise RuntimeError("SSH exception must be enabled by the supervisor")
    supervisor = os.environ.get("EVOWEAVE_RESOURCE_SUPERVISOR_PID", "")
    if not supervisor.isdigit():
        raise RuntimeError("missing resource supervisor")
    command = (Path("/proc") / supervisor / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if str(ROOT / LAUNCHER) not in command:
        raise RuntimeError("the resource supervisor is not the reviewed launcher")
    pid = os.getpid()
    while pid > 1 and pid != int(supervisor):
        status = (Path("/proc") / str(pid) / "status").read_text()
        pid = int(next(line.split(":", 1)[1] for line in status.splitlines() if line.startswith("PPid:")))
    if pid != int(supervisor):
        raise RuntimeError("payload escaped its supervisor")
    cpu = args.mode == "audit"
    if (socket.gethostname().split(".")[0] != "pcg01"
            or os.sched_getaffinity(0) != set(CPUS)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ("" if cpu else "5,6")
            or runtime.get("world_size") != 2):
        raise RuntimeError("host, CPU, GPU or world-size binding differs")
    import resource
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise RuntimeError("CUDA requires unlimited virtual address space")
    grant = subprocess.check_output(["qstat", "-j", "130309880"], text=True, timeout=30)
    if ("pcg01i: /dev/nvidia5 /dev/nvidia6" not in grant
            or not re.search(r"job_state\s+\d+:\s+r\b", grant)
            or not re.search(r"^owner:\s+wangyy\s*$", grant, re.MULTILINE)
            or not re.search(r"exec_host_list\s+\d+:\s+pcg01i:1\s*$", grant, re.MULTILINE)):
        raise RuntimeError("the recorded two-GPU allocation changed")
    uuids = subprocess.check_output(["nvidia-smi", "-i", "5,6", "--query-gpu=uuid",
        "--format=csv,noheader"], text=True, timeout=30).split()
    if uuids != runtime["gpu_uuids"]:
        raise RuntimeError("GPU identity changed")
    if not cpu and os.getpid() == int(supervisor):
        applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader"], text=True, timeout=30)
        for line in applications.splitlines():
            pid, gpu = (part.strip() for part in line.split(","))
            if gpu in uuids:
                raise RuntimeError(f"approved GPU is already occupied by PID {pid}; no process was stopped")
    if (ROOT.resolve() != Path(runtime["repo"]).resolve()
            or args.config.resolve().parent != ROOT / "model_training/experiments"
            or args.output.resolve().parent != Path(runtime["output_root"]).resolve()):
        raise RuntimeError("diagnostic paths escaped their isolated scope")
    matches = [run for run in op.get("runs", []) if run.get("mode") == args.mode
        and Path(run["config"]).resolve() == args.config.resolve()
        and Path(run["output"]).resolve() == args.output.resolve()]
    if (len(matches) != 1 or matches[0].get("approved") is not True
            or matches[0]["config_sha256"] != file_sha256(args.config)
            or (not cpu and op.get("GPU_allowed") is not True)
            or (args.mode == "diagnose" and op.get("smoke_accepted") is not True)):
        raise RuntimeError("this exact diagnostic stage is not approved")
    if op.get("complete") is not False:
        raise RuntimeError("the diagnostic study is already closed")
    validate_repeatability_protocol(config, op)
    check_deadline(config)
    subprocess.run([sys.executable, str(ROOT / "model_training/tools/agent_work_guard.py"),
        "check", "--operation", "report" if cpu else "preflight", "--entrypoint", ENTRYPOINT],
        check=True, stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("diagnostic source must be clean and committed")
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", op["code_commit"], source], check=True)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", op["code_commit"], source, "--",
        "model_training/rigweave", "model_training/jobs", "model_training/tools"], check=True)
    return source
