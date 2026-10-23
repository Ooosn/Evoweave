#!/usr/bin/env python3
"""Supervise the exact approved allocation, including its run-scoped SSH exception."""
import argparse
import ctypes
import importlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from supervise_structure_xyz_resources import descendants


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def cpu_ticks(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    return int(fields[11]) + int(fields[12])


def process_identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    return int(fields[19]), fields[0]


def reap_exited_children():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def cleanup_payload(child, keeper_pid):
    # Elastic workers use separate sessions. A subreaper retains orphan ownership.
    known = {}
    terminated = set()
    for sig, duration in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        until = time.monotonic() + duration
        while True:
            for pid in descendants(os.getpid()):
                if pid in (os.getpid(), keeper_pid):
                    continue
                try:
                    known.setdefault(pid, process_identity(pid)[0])
                except (FileNotFoundError, ProcessLookupError):
                    pass
            alive = []
            for pid, identity in known.items():
                try:
                    started, status = process_identity(pid)
                    if started == identity and status != 'Z':
                        alive.append(pid)
                        terminated.add(pid)
                        os.kill(pid, sig)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            child.poll()
            if not alive:
                reap_exited_children()
                return sorted(terminated)
            if time.monotonic() >= until:
                break
            time.sleep(.2)
    raise RuntimeError(f'payload children survived cleanup: {alive}')


def resource_cgroups(text):
    result = {}
    for line in text.splitlines():
        _, controllers, path = line.split(':', 2)
        for controller in controllers.split(','):
            if controller in ('devices', 'cpuset'):
                result[controller] = path
    return result


def verify_child_cgroups(pid, expected):
    if process_identity(pid)[1] == 'Z':
        return False
    actual = resource_cgroups((Path('/proc') / str(pid) / 'cgroup').read_text())
    if actual != expected:
        # Linux moves exited tasks to root cgroups before their parent reaps them.
        if process_identity(pid)[1] == 'Z':
            return False
        raise RuntimeError(f'live child {pid} changed resource cgroups: {actual} != {expected}')
    return True


def main(*, trainer_module_name='train_legacy_single_target',
         exception_env_name='EVOWEAVE_LEGACY_SINGLE_TARGET_SSH_EXCEPTION',
         trainer_script='train_legacy_single_target.py',
         modes=('preflight', 'capacity', 'train'), cpu_modes=()):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=modes, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = config['runtime']
    root = Path(runtime['repo'])
    sys.path.insert(0, str(root / 'model_training/rigweave/scripts'))
    checks = importlib.import_module(trainer_module_name)
    require_job_membership = checks.require_job_membership
    validate_ssh_exception = checks.validate_ssh_exception
    verify_exception_keeper = checks.verify_exception_keeper
    verify_launch = checks.verify_launch
    # Read actual kernel provenance before setting CUDA visibility or importing torch.
    actual_cgroup = Path('/proc/self/cgroup').read_text()
    exception = None
    try:
        require_job_membership(actual_cgroup, runtime['allocation'])
        in_job = True
    except RuntimeError:
        state = json.loads((root / 'model_training/state/current.json').read_text())
        exception = validate_ssh_exception(state, config)
        verify_exception_keeper(exception)
        in_job = False
        os.environ[exception_env_name] = '1'
    if in_job and (os.environ.get('JOB_ID') != str(runtime['allocation'])
            or int(os.environ.get('NSLOTS', '0')) != runtime['cpu_slots']):
        raise RuntimeError('the launcher must inherit the real job ID and CPU grant')
    allowed = os.sched_getaffinity(0)
    if exception:
        allowed &= os.sched_getaffinity(exception['keeper_pid'])
    allowed = sorted(allowed)
    if len(allowed) < runtime['cpu_slots']:
        raise RuntimeError('insufficient allowed CPUs')
    binding = set(allowed[:runtime['cpu_slots']])
    os.sched_setaffinity(0, binding)
    os.environ['EVOWEAVE_RESOURCE_SUPERVISOR_PID'] = str(os.getpid())
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.mode in cpu_modes else ','.join(map(str, runtime['physical_gpus']))
    os.environ.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'RIGWEAVE_DISABLE_OPEN3D',
        'RIGWEAVE_DISABLE_LIGHTNING_IMPORT')})
    os.environ.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_MODULE_LOADING='LAZY',
        EVOWEAVE_UNIRIG_ROOT=runtime['unirig_root'], EVOWEAVE_OPT_CONFIG_ROOT=runtime['opt_config_root'])
    os.environ['PYTHONPATH'] = os.pathsep.join((str(root / 'model_training/rigweave/src'),
        str(root / 'model_training/rigweave/scripts'), runtime['unirig_root'], runtime['extra_pythonpath']))
    source = verify_launch(args, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_name(args.output.name + '_resources.json')
    if receipt_path.exists() or args.output.exists():
        raise RuntimeError('refuse to overwrite a prior stage or resource receipt')
    command = [runtime['python']]
    if args.mode not in cpu_modes:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2']
    command += [str(root / 'model_training/rigweave/scripts' / trainer_script),
                '--config', str(args.config), '--output', str(args.output), '--mode', args.mode]
    receipt = {'complete': False, 'passed': False, 'source': source, 'mode': args.mode,
        'allocation': runtime['allocation'], 'kernel_scheduler_membership': in_job,
        'execution_mode': runtime['execution_mode'], 'actual_cgroup': actual_cgroup,
        'ssh_exception': exception,
        'physical_gpus': runtime['physical_gpus'], 'cpu_affinity': sorted(binding),
        'rss_limit_bytes': runtime['rss_limit_bytes'], 'peak_tree_rss_bytes': 0,
        'aggregate_cpu_seconds': 0., 'supervisor_pid': os.getpid()}
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot retain ownership of orphaned rank workers')
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    receipt['child_pid'] = child.pid
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    last_ticks, total_ticks, next_grant = {}, 0, 0.
    try:
        while child.poll() is None:
            if interrupted:
                raise RuntimeError(f'supervisor interrupted by signal {interrupted[0]}')
            now = time.monotonic()
            if now >= next_grant:
                grant = subprocess.check_output(['qstat', '-j', str(runtime['allocation'])], text=True, timeout=30)
                if (f'pcg01i: /dev/nvidia5 /dev/nvidia6' not in grant
                        or not re.search(r'job_state\s+\d+:\s+r\b', grant)
                        or not re.search(r'^owner:\s+wangyy\s*$', grant, re.MULTILINE)
                        or not re.search(r'exec_host_list\s+\d+:\s+pcg01i:1\s*$', grant, re.MULTILINE)):
                    raise RuntimeError('the authorized GPU grant changed')
                if exception:
                    verify_exception_keeper(exception)
                next_grant = now + 60
            processes = descendants(child.pid)
            rss = sum(value[1] for value in processes.values())
            if rss > runtime['rss_limit_bytes']:
                raise RuntimeError('the training process tree exceeded64GiB RAM')
            for pid in processes:
                try:
                    if not verify_child_cgroups(pid, resource_cgroups(actual_cgroup)):
                        continue
                    for task in (Path('/proc') / str(pid) / 'task').iterdir():
                        if not os.sched_getaffinity(int(task.name)) <= binding:
                            raise RuntimeError('a child thread escaped its CPU affinity')
                    ticks = cpu_ticks(pid)
                    total_ticks += max(0, ticks - last_ticks.get(pid, 0))
                    last_ticks[pid] = ticks
                except (FileNotFoundError, ProcessLookupError):
                    pass
            text = subprocess.check_output(['nvidia-smi', '-i', '5,6',
                '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                text=True, timeout=30)
            memory = [[int(v.strip()) for v in line.split(',')] for line in text.strip().splitlines()]
            if sorted(row[0] for row in memory) != [5, 6] or any(used > total*.85 for _, used, total in memory):
                raise RuntimeError('the authorized GPUs exceeded85percent memory or changed identity')
            if runtime.get('gpu_uuids'):
                applications = subprocess.check_output(['nvidia-smi',
                    '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True, timeout=30)
                for line in applications.splitlines():
                    pid, gpu = (value.strip() for value in line.split(','))
                    if int(pid) in processes and gpu not in runtime['gpu_uuids']:
                        raise RuntimeError('a payload process used an unapproved GPU')
            receipt.update(last_observed_utc=datetime.now(timezone.utc).isoformat(), tree_rss_bytes=rss,
                peak_tree_rss_bytes=max(rss, receipt['peak_tree_rss_bytes']), process_count=len(processes),
                aggregate_cpu_seconds=total_ticks/os.sysconf('SC_CLK_TCK'), gpu_memory_mib=memory)
            write_json(receipt_path, receipt)
            for _ in range(5):
                if child.poll() is not None or interrupted:
                    break
                time.sleep(1)
        receipt.update(complete=True, passed=child.returncode == 0, exit_code=child.returncode)
    except BaseException as exc:
        receipt['error'] = repr(exc)
        receipt.update(complete=True, passed=False)
        raise
    finally:
        try:
            receipt['cleaned_payload_pids'] = cleanup_payload(child, exception['keeper_pid'] if exception else None)
            receipt['payload_cleanup_complete'] = True
            if receipt['cleaned_payload_pids']:
                receipt.update(passed=False, error=receipt.get('error', 'orphan workers remained after torchrun exit'))
        except BaseException as exc:
            receipt.update(passed=False, payload_cleanup_complete=False, cleanup_error=repr(exc))
            raise
        finally:
            receipt['exit_code'] = child.poll()
            receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(receipt_path, receipt)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
