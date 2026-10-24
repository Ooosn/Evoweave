"""Standard-library-only launch checks before model or CUDA imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from train_legacy_single_target import require_job_membership, verify_exception_keeper


ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = "model_training/rigweave/scripts/train_id_xyz_consistency.py"
SCOPE = "real_frame_id_xyz_consistency_20261024_v1"
FAMILY = "real_frame_id_xyz_consistency_v1"
EXCEPTION_ENV = "EVOWEAVE_ID_XYZ_SSH_EXCEPTION"
CPU_OVERRIDE_FIELDS = ("execution_cpu_budget", "cpu_affinity")


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _configured_cpu_binding(config):
    runtime = config["runtime"]
    if type(runtime.get("cpu_slots")) is not int or runtime["cpu_slots"] != 1:
        raise RuntimeError("the actual scheduler CPU grant must remain one slot")
    if "resource_transition" not in config:
        if any(key in runtime for key in CPU_OVERRIDE_FIELDS):
            raise RuntimeError("CPU override fields require a declared resource transition")
        return None
    transition = config["resource_transition"]
    affinity = runtime.get("cpu_affinity")
    if (not isinstance(transition, dict) or transition.get("kind") != "cpu_budget_only_v1"
            or config.get("family") != FAMILY
            or runtime.get("execution_mode") != "explicit_ssh_exception"
            or runtime.get("require_scheduler_membership") is not False
            or type(runtime.get("execution_cpu_budget")) is not int
            or runtime["execution_cpu_budget"] != 4
            or type(affinity) is not list or len(affinity) != 4
            or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
            or affinity != sorted(set(affinity))):
        raise RuntimeError("CPU transition requires an explicit sorted four-CPU SSH binding")
    return set(affinity)


def validate_ssh_exception(state, config):
    op = state.get("active_operation", {})
    exception = op.get("ssh_exception") if isinstance(op, dict) else None
    runtime = config["runtime"]
    if (not isinstance(exception, dict) or op.get("label") != SCOPE
            or exception.get("scope") != SCOPE or exception.get("explicit_human_authorized") is not True
            or exception.get("kernel_scheduler_membership") is not False
            or runtime.get("execution_mode") != "explicit_ssh_exception"
            or runtime.get("require_scheduler_membership") is not False
            or not isinstance(exception.get("authorization"), str) or not exception["authorization"].strip()):
        raise RuntimeError("the new experiment lacks its own explicit SSH authorization")
    expected = {"allocation": 130309880, "host": "pcg01i", "physical_gpus": [5, 6],
                "cpu_slots": 1, "rss_limit_bytes": 68719476736,
                "gpu_memory_fraction_limit": .85, "keeper_pid": 1090013}
    for key, value in expected.items():
        if (type(runtime.get(key)) is not type(value) or runtime[key] != value
                or type(exception.get(key)) is not type(value) or exception[key] != value):
            raise RuntimeError(f"SSH exception changed {key}")
    binding = _configured_cpu_binding(config)
    if binding is None:
        if any(key in exception for key in CPU_OVERRIDE_FIELDS):
            raise RuntimeError("SSH exception has undeclared CPU override fields")
    else:
        for key in CPU_OVERRIDE_FIELDS:
            if (type(exception.get(key)) is not type(runtime[key])
                    or exception[key] != runtime[key]):
                raise RuntimeError(f"SSH exception changed {key}")
        if any(type(cpu) is not int for cpu in exception["cpu_affinity"]):
            raise RuntimeError("SSH CPU affinity must contain integer CPU IDs")
    return exception


def execution_cpu_binding(config, exception, allowed_cpus):
    binding = _configured_cpu_binding(config)
    allowed = set(allowed_cpus)
    if exception is not None:
        validate_ssh_exception(
            {"active_operation": {"label": SCOPE, "ssh_exception": exception}}, config)
    if binding is not None:
        if not exception or not binding <= allowed:
            raise RuntimeError("four-CPU execution requires authorization and allowed CPUs")
        return binding
    if exception:
        allowed &= os.sched_getaffinity(exception["keeper_pid"])
    if not allowed:
        raise RuntimeError("insufficient allowed CPUs")
    return {min(allowed)}


def verify_supervisor():
    value = os.environ.get("EVOWEAVE_RESOURCE_SUPERVISOR_PID", "")
    if not value.isdigit():
        raise RuntimeError("an independent resource supervisor is required")
    supervisor = int(value)
    command = (Path("/proc") / value / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "/model_training/jobs/launch_id_xyz_consistency.py" not in command:
        raise RuntimeError("the supervisor is not the declared new experiment launcher")
    pid = os.getpid()
    while pid > 1 and pid != supervisor:
        status = (Path("/proc") / str(pid) / "status").read_text()
        pid = int(next(line.split(":", 1)[1] for line in status.splitlines() if line.startswith("PPid:")))
    if pid != supervisor:
        raise RuntimeError("the payload did not inherit its supervisor")


def verify_launch(args, config):
    runtime = config["runtime"]
    state = json.loads((ROOT / "model_training/state/current.json").read_text())
    exception = None
    try:
        require_job_membership(Path("/proc/self/cgroup").read_text(), runtime["allocation"])
        in_job = True
    except RuntimeError:
        exception = validate_ssh_exception(state, config)
        if os.environ.get(EXCEPTION_ENV) != "1":
            raise RuntimeError("SSH exception was not enabled by the bounded supervisor")
        verify_exception_keeper(exception)
        verify_supervisor()
        in_job = False
    configured_binding = _configured_cpu_binding(config)
    if configured_binding is not None and (in_job or not exception):
        raise RuntimeError("the execution CPU override is restricted to authorized SSH")
    affinity = os.sched_getaffinity(0)
    binding_matches = (affinity == configured_binding if configured_binding is not None
                       else len(affinity) == 1)
    cpu = args.mode == "audit"
    visible = "" if cpu else "5,6"
    if (config.get("family") != FAMILY or socket.gethostname().split(".")[0] != "pcg01"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != visible
            or not binding_matches
            or (in_job and (os.environ.get("JOB_ID") != str(runtime["allocation"])
                            or int(os.environ.get("NSLOTS", "0")) != runtime["cpu_slots"]))):
        raise RuntimeError("host, devices, CPU binding or scheduler provenance differs")
    import resource
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise RuntimeError("CUDA requires unlimited virtual address space")
    grant = subprocess.check_output(["qstat", "-j", str(runtime["allocation"])], text=True, timeout=30)
    if ("pcg01i: /dev/nvidia5 /dev/nvidia6" not in grant
            or not re.search(r"job_state\s+\d+:\s+r\b", grant)
            or not re.search(r"^owner:\s+wangyy\s*$", grant, re.MULTILINE)
            or not re.search(r"exec_host_list\s+\d+:\s+pcg01i:1\s*$", grant, re.MULTILINE)):
        raise RuntimeError("the two-card grant is no longer valid")
    actual = subprocess.check_output(["nvidia-smi", "-i", "5,6", "--query-gpu=uuid",
                                     "--format=csv,noheader"], text=True, timeout=30).split()
    if actual != runtime["gpu_uuids"]:
        raise RuntimeError("the physical GPU identities changed")
    if (ROOT.resolve() != Path(runtime["repo"]).resolve()
            or args.config.resolve().parent != (ROOT / "model_training/experiments").resolve()
            or args.output.resolve().parent != Path(runtime["output_root"]).resolve()):
        raise RuntimeError("source/config/output escaped the isolated experiment")
    op = state["active_operation"]
    if (state["architecture_contract"]["id"] != FAMILY or op.get("label") != SCOPE
            or op.get("required_entrypoint") != ENTRYPOINT or op.get("complete") is not False
            or (not cpu and op.get("gpu_work_allowed") is not True)
            or (args.mode == "train" and op.get("formal_training_allowed") is not True)):
        raise RuntimeError("the exact experiment stage is not released")
    matches = [run for run in op.get("runs", []) if run.get("mode") == args.mode
               and Path(run["config"]).resolve() == args.config.resolve()
               and Path(run["output"]).resolve() == args.output.resolve()]
    if len(matches) != 1:
        raise RuntimeError("the stage is not uniquely declared")
    run = matches[0]
    if (run.get("approved") is not True or run["config_sha256"] != file_sha256(args.config)
            or run["allocation"] != runtime["allocation"] or run["physical_gpus"] != [5, 6]):
        raise RuntimeError("config/output/resource binding changed")
    operation = "report" if cpu else ("train" if args.mode == "train" else "preflight")
    if int(os.environ.get("RANK", "0")) == 0:
        subprocess.run([sys.executable, str(ROOT / "model_training/tools/agent_work_guard.py"),
                        "check", "--operation", operation, "--entrypoint", ENTRYPOINT], check=True)
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("execution requires clean committed source")
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", op["code_commit"], source], check=True)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", op["code_commit"], source, "--",
                    "model_training/rigweave", "model_training/jobs", "model_training/tools"], check=True)
    return source
