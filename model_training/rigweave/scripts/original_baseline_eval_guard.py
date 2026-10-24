"""Local-only admission for the explicitly requested matched checkpoint evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil


ROOT = Path(__file__).resolve().parents[3]
SCOPE = "original_single_query_matched_eval_20261025_v1"
FAMILY = "original_single_query_matched_eval_v1"
ENTRYPOINT = "model_training/rigweave/scripts/eval_original_baseline_matched.py"
LAUNCHER = "model_training/jobs/launch_original_baseline_eval.py"
GPU_UUID = "GPU-7907eaad-32b2-c0d1-236e-e7b84cee8bbb"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def operation_record(state):
    record = state.get('active_operation', {}).get('baseline_matched_evaluation', {})
    if record.get('label') != SCOPE or record.get('explicit_human_authorized') is not True:
        raise RuntimeError('the local baseline evaluation is not explicitly recorded')
    return record


def validate_runtime(runtime):
    expected = {'execution_mode': 'local_windows', 'world_size': 1, 'device_index': 0,
                'gpu_uuid': GPU_UUID, 'cpu_affinity': list(range(8, 16)),
                'rss_limit_bytes': 32 * 2**30, 'gpu_memory_fraction_limit': .45,
                'aggregate_gpu_fraction_limit': .85}
    for key, value in expected.items():
        if type(runtime.get(key)) is not type(value) or runtime[key] != value:
            raise RuntimeError(f'local resource budget differs: {key}')
    if any(type(cpu) is not int for cpu in runtime['cpu_affinity']):
        raise RuntimeError('CPU IDs must be integers')
    if Path(runtime['repo']).resolve() != ROOT:
        raise RuntimeError('the local evaluator checkout differs')


def verify_stage(args, config):
    if sys.platform != 'win32' or config.get('family') != FAMILY:
        raise RuntimeError('this evaluation is restricted to the authorized local Windows host')
    runtime = config['runtime']
    validate_runtime(runtime)
    state = json.loads((ROOT / 'model_training/state/current.json').read_text(encoding='utf-8'))
    record = operation_record(state)
    matches = [run for run in record.get('runs', []) if run.get('mode') == args.mode
               and Path(run['config']).resolve() == args.config.resolve()
               and Path(run['output']).resolve() == args.output.resolve()]
    if len(matches) != 1 or matches[0].get('approved') is not True:
        raise RuntimeError('this exact local stage is not approved')
    if file_sha256(args.config) != matches[0]['config_sha256']:
        raise RuntimeError('the approved local configuration changed')
    if args.mode == 'evaluate' and record.get('smoke_accepted') is not True:
        raise RuntimeError('full evaluation requires primary smoke acceptance')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if args.mode == 'audit' else '0'):
        raise RuntimeError('unexpected local CUDA visibility')
    if set(psutil.Process().cpu_affinity()) != set(runtime['cpu_affinity']):
        raise RuntimeError('the local CPU budget was not applied')
    supervisor = int(os.environ.get('EVOWEAVE_LOCAL_BASELINE_SUPERVISOR_PID', '0'))
    if supervisor <= 0:
        raise RuntimeError('missing local resource supervisor')
    parent = psutil.Process(supervisor)
    command = parent.cmdline()
    if not any(Path(part).resolve() == ROOT / LAUNCHER for part in command[1:] if part.endswith('.py')):
        raise RuntimeError('unexpected local resource supervisor')
    if supervisor != os.getpid() and supervisor not in {proc.pid for proc in psutil.Process().parents()}:
        raise RuntimeError('evaluation is outside its resource supervisor')
    operation = 'inspect' if args.mode == 'audit' else 'matched_eval'
    subprocess.check_call([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                           'check', '--operation', operation, '--entrypoint', ENTRYPOINT],
                          stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('the evaluator checkout must be clean')
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only',
                                      record['code_commit'], head], text=True).splitlines()
    allowed = {'model_training/state/current.json', 'model_training/docs/CURRENT_MODEL_CONTEXT.md',
               'model_training/docs/ORIGINAL_BASELINE_MATCHED_20261025.md',
               'model_training/experiments/original_baseline_matched_20261025_v1.json'}
    if set(changed) - allowed:
        raise RuntimeError('evaluation computation changed after review')
    return head
