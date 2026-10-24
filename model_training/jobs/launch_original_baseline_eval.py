#!/usr/bin/env python3
"""Supervise local inference while preserving all unrelated GPU processes."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from original_baseline_eval_guard import verify_stage, validate_runtime


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def gpu_snapshot():
    data = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    uuid, used, total = next(csv.reader(io.StringIO(data)))
    return {'uuid': uuid.strip(), 'used_mib': int(used), 'total_mib': int(total)}


def other_gpu_python_pids(ignore=()):
    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True, timeout=20)
    return [int(pid) for pid, name in csv.reader(io.StringIO(data))
            if 'python' in Path(name.strip()).name.lower() and int(pid) not in ignore]


def stop_payload(child):
    if child.poll() is not None:
        return []
    process = psutil.Process(child.pid)
    owned = process.children(recursive=True) + [process]
    for proc in reversed(owned):
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=8)
    for proc in alive:
        proc.kill()
    child.wait(timeout=15)
    return [proc.pid for proc in owned]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'smoke', 'evaluate'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    runtime = config['runtime']
    validate_runtime(runtime)
    psutil.Process().cpu_affinity(runtime['cpu_affinity'])
    os.environ['EVOWEAVE_LOCAL_BASELINE_SUPERVISOR_PID'] = str(os.getpid())
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.mode == 'audit' else '0'
    source = verify_stage(args, config)
    receipt_path = args.output.with_name(args.output.name + '_resources.json')
    if args.output.exists() or receipt_path.exists():
        raise RuntimeError('refuse to overwrite a prior evaluation stage')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'complete': False, 'passed': False, 'source': source, 'mode': args.mode,
               'supervisor_pid': os.getpid(), 'peak_tree_rss_bytes': 0,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime}
    write(receipt_path, receipt)
    if args.mode != 'audit':
        deadline = time.monotonic() + 10800
        while True:
            busy = other_gpu_python_pids((os.getpid(),))
            gpu = gpu_snapshot()
            if gpu['uuid'] != runtime['gpu_uuid']:
                raise RuntimeError('the local GPU identity changed')
            if not busy and gpu['total_mib'] - gpu['used_mib'] >= 18000:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('local GPU remains occupied; no unrelated process was stopped')
            receipt.update(waiting_for_gpu=True, conflicting_python_pids=busy, gpu=gpu,
                           last_observed_utc=datetime.now(timezone.utc).isoformat())
            write(receipt_path, receipt)
            time.sleep(15)
    env = os.environ.copy()
    env.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                     'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'HF_HUB_OFFLINE',
                                     'TRANSFORMERS_OFFLINE', 'RIGWEAVE_DISABLE_OPEN3D',
                                     'RIGWEAVE_DISABLE_LIGHTNING_IMPORT', 'PYTHONUNBUFFERED')})
    env['EVOWEAVE_UNIRIG_ROOT'] = runtime['unirig_root']
    env['EVOWEAVE_OPT_CONFIG_ROOT'] = runtime['opt_config_root']
    cache = Path(runtime['cache_root'])
    env.update(HF_HOME=str(cache / 'huggingface'), HF_HUB_CACHE=str(cache / 'huggingface/hub'),
               TRANSFORMERS_CACHE=str(cache / 'transformers'), TORCH_HOME=str(cache / 'torch'))
    env['PYTHONPATH'] = os.pathsep.join([runtime['extra_pythonpath'], str(ROOT / 'model_training/rigweave/src'),
                                        str(ROOT / 'model_training/rigweave/scripts'), runtime['unirig_root']])
    command = [runtime['python'], '-B', str(ROOT / 'model_training/rigweave/scripts/eval_original_baseline_matched.py'),
               '--config', str(args.config), '--output', str(args.output), '--mode', args.mode]
    child = subprocess.Popen(command, cwd=ROOT, env=env, shell=False)
    receipt.update(child_pid=child.pid, waiting_for_gpu=False)
    try:
        while child.poll() is None:
            tree = [psutil.Process(child.pid)]
            tree += tree[0].children(recursive=True)
            rss = 0
            for proc in tree:
                try:
                    rss += proc.memory_info().rss
                    if not set(proc.cpu_affinity()) <= set(runtime['cpu_affinity']):
                        raise RuntimeError('local payload CPU affinity drifted')
                except psutil.NoSuchProcess:
                    pass
            if rss > runtime['rss_limit_bytes']:
                raise RuntimeError('local evaluation exceeded its32GiB RAM limit')
            receipt.update(tree_rss_bytes=rss, peak_tree_rss_bytes=max(rss, receipt['peak_tree_rss_bytes']),
                           last_observed_utc=datetime.now(timezone.utc).isoformat())
            if args.mode != 'audit':
                gpu = gpu_snapshot()
                receipt['gpu'] = gpu
                if gpu['uuid'] != runtime['gpu_uuid'] or gpu['used_mib'] / gpu['total_mib'] > runtime['aggregate_gpu_fraction_limit']:
                    raise RuntimeError('local aggregate GPU budget exceeded; stopping only this evaluation')
            write(receipt_path, receipt)
            time.sleep(2)
        receipt.update(complete=True, passed=child.returncode == 0, exit_code=child.returncode)
    except BaseException as error:
        receipt.update(complete=True, passed=False, error=repr(error))
        raise
    finally:
        receipt['cleaned_payload_pids'] = stop_payload(child)
        receipt.update(exit_code=child.poll(), finished_utc=datetime.now(timezone.utc).isoformat(),
                       payload_cleanup_complete=True)
        write(receipt_path, receipt)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
