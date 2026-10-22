#!/usr/bin/env python3
"""Bound an explicitly approved SSH child without claiming scheduler membership."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def read_process(pid):
    status = (Path('/proc') / str(pid) / 'status').read_text()
    values = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
    return int(values['PPid']), int(values.get('VmRSS', '0 kB').strip().split()[0]) * 1024


def descendants(root):
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                processes[int(path.name)] = read_process(int(path.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    selected = {root} if root in processes else set()
    while True:
        expanded = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if expanded == selected:
            return {pid: processes[pid] for pid in selected}
        selected = expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    config = json.loads(args.config.read_text())
    runtime = config['runtime']
    state = json.loads((Path(runtime['repo']) / 'model_training/state/current.json').read_text())
    exception = state['active_operation']['ssh_exception']
    assert exception['explicit_human_authorized'] is True and exception['kernel_scheduler_membership'] is False
    assert os.environ.get('EVOWEAVE_EXPLICIT_SSH_EXCEPTION') == '1'
    affinity = os.sched_getaffinity(0)
    assert len(affinity) == exception['cpu_slots']
    job = str(runtime['allocation'])
    devices = ','.join(map(str, runtime['physical_gpus']))
    limit = exception['rss_limit_bytes']
    receipt = {'kernel_scheduler_membership': False, 'explicit_human_authorized': True,
               'allocation': runtime['allocation'], 'physical_gpus': runtime['physical_gpus'],
               'cpu_affinity': sorted(affinity), 'rss_limit_bytes': limit, 'peak_tree_rss_bytes': 0,
               'supervisor_pid': os.getpid(), 'complete': False, 'passed': False}
    child = subprocess.Popen(command, start_new_session=True)
    receipt['child_pid'] = child.pid
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    next_grant = 0.
    try:
        while child.poll() is None:
            if interrupted:
                raise RuntimeError(f'supervisor interrupted by signal {interrupted[0]}')
            now = time.monotonic()
            if now >= next_grant:
                grant = subprocess.check_output(['qstat', '-j', job], text=True, timeout=30)
                lines = [line for line in grant.splitlines() if line.startswith('granted_devices')]
                assert len(lines) == 1 and runtime['host'] + ':' in lines[0], 'allocation host changed'
                assert sorted(map(int, re.findall(r'/dev/nvidia(\d+)', lines[0]))) == sorted(runtime['physical_gpus']), 'GPU grant changed'
                assert re.search(r'job_state\s+\d+:\s+r\b', grant), 'allocation is no longer running'
                keeper = Path('/proc') / str(exception['allocation_keeper_pid']) / 'cgroup'
                assert '/AGE/' + job + '.1' in keeper.read_text(), 'allocation keeper exited'
                next_grant = now + 60
            processes = descendants(child.pid)
            total = sum(rss for _, rss in processes.values())
            receipt['peak_tree_rss_bytes'] = max(receipt['peak_tree_rss_bytes'], total)
            assert total <= limit, 'SSH process tree exceeded the declared RAM cap'
            for pid in processes:
                try:
                    for task in (Path('/proc') / str(pid) / 'task').iterdir():
                        assert os.sched_getaffinity(int(task.name)) <= affinity, 'a child thread escaped the CPU affinity cap'
                except (FileNotFoundError, ProcessLookupError):
                    pass
            gpu = subprocess.check_output(['nvidia-smi', '-i', devices,
                '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=30)
            memory = [[int(value.strip()) for value in line.split(',')] for line in gpu.strip().splitlines()]
            assert len(memory) == runtime['world_size']
            assert all(used < total * .90 for _, used, total in memory), 'GPU device memory exceeded90percent'
            receipt.update(last_observed_utc=datetime.now(timezone.utc).isoformat(),
                           tree_rss_bytes=total, process_count=len(processes), gpu_memory_mib=memory)
            temporary = args.receipt.with_suffix('.tmp')
            temporary.write_text(json.dumps(receipt, indent=2) + '\n')
            temporary.replace(args.receipt)
            for _ in range(10):
                if child.poll() is not None or interrupted:
                    break
                time.sleep(1)
        receipt.update(complete=True, passed=child.returncode == 0, exit_code=child.returncode)
    except BaseException as exc:
        receipt['error'] = repr(exc)
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        except ProcessLookupError:
            pass
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
