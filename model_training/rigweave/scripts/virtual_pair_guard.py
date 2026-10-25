"""Admission for the fixed-5k, matched virtual-path training pair."""
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
ENTRYPOINT = 'model_training/rigweave/scripts/train_virtual_pair.py'
LAUNCHER = 'model_training/jobs/launch_virtual_pair.py'
SCOPE = 'virtual_query_matched_5k_20261025_v1'
EXCEPTION_ENV = 'EVOWEAVE_VIRTUAL_PAIR_SSH_EXCEPTION'
CPUS = [58, 63, 64, 65]


def operation(state):
    op = state.get('active_operation', {})
    if op.get('label') != SCOPE or op.get('explicit_human_authorized') is not True:
        raise RuntimeError('matched pair lacks explicit human authorization')
    if op.get('required_entrypoint') != ENTRYPOINT:
        raise RuntimeError('state is not bound to the matched-pair trainer')
    return op


def validate_ssh_exception(state, config):
    op, runtime = operation(state), config['runtime']
    exception = op.get('ssh_exception', {})
    expected = {'scope': SCOPE, 'explicit_human_authorized': True,
        'kernel_scheduler_membership': False, 'allocation': 130309880,
        'host': 'pcg01i', 'physical_gpus': [5, 6], 'cpu_slots': 1,
        'execution_cpu_budget': 4, 'cpu_affinity': CPUS, 'rss_limit_bytes': 68719476736,
        'gpu_memory_fraction_limit': .85, 'keeper_pid': 1090013}
    for key, value in expected.items():
        if type(exception.get(key)) is not type(value) or exception[key] != value:
            raise RuntimeError(f'SSH exception differs: {key}')
        if key not in ('scope', 'explicit_human_authorized', 'kernel_scheduler_membership'):
            if type(runtime.get(key)) is not type(value) or runtime[key] != value:
                raise RuntimeError(f'resource binding differs: {key}')
    if (config.get('family') != SCOPE or runtime.get('execution_mode') != 'explicit_ssh_exception'
            or runtime.get('require_scheduler_membership') is not False or not exception.get('authorization')):
        raise RuntimeError('undeclared execution provenance')
    return exception


def execution_cpu_binding(config, exception, allowed_cpus):
    if not exception or not set(CPUS) <= allowed_cpus:
        raise RuntimeError('approved four-CPU binding unavailable')
    return set(CPUS)


def deadline_due(config):
    deadline = datetime.fromisoformat(config['runtime']['stop_before_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('deadline needs a timezone')
    return datetime.now(timezone.utc) >= deadline


def validate_stage(state, config, args):
    op = operation(state)
    matches = [row for row in op.get('runs', []) if row.get('mode') == args.mode
        and Path(row['config']).resolve() == args.config.resolve()
        and Path(row['output']).resolve() == args.output.resolve()]
    if (len(matches) != 1 or not matches[0].get('approved')
            or matches[0]['config_sha256'] != file_sha256(args.config) or op.get('complete') is not False):
        raise RuntimeError('exact stage/config/output is not approved')
    if args.mode != 'audit' and op.get('gpu_work_allowed') is not True:
        raise RuntimeError('GPU stage not released')
    if args.mode == 'train' and (op.get('formal_training_allowed') is not True
            or op.get('capacity_accepted') is not True or op.get('CPU_audit_accepted') is not True):
        raise RuntimeError('training needs primary CPU/capacity acceptance')
    return op


def verify_launch(args, config):
    runtime = config['runtime']
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    op = validate_stage(state, config, args)
    exception = validate_ssh_exception(state, config)
    verify_exception_keeper(exception)
    supervisor = os.environ.get('EVOWEAVE_RESOURCE_SUPERVISOR_PID', '')
    if os.environ.get(EXCEPTION_ENV) != '1' or not supervisor.isdigit():
        raise RuntimeError('reviewed resource supervisor is required')
    command = (Path('/proc') / supervisor / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if str(ROOT / LAUNCHER) not in command:
        raise RuntimeError('unexpected supervisor entrypoint')
    pid = os.getpid()
    while pid > 1 and pid != int(supervisor):
        status = (Path('/proc') / str(pid) / 'status').read_text()
        pid = int(next(x.split(':', 1)[1] for x in status.splitlines() if x.startswith('PPid:')))
    if pid != int(supervisor):
        raise RuntimeError('payload escaped supervisor')
    cpu = args.mode == 'audit'
    if (socket.gethostname().split('.')[0] != 'pcg01' or os.sched_getaffinity(0) != set(CPUS)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if cpu else '5,6') or runtime['world_size'] != 2):
        raise RuntimeError('host/CPU/GPU binding differs')
    import resource
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise RuntimeError('unlimited CUDA virtual address space required')
    grant = subprocess.check_output(['qstat', '-j', str(runtime['allocation'])], text=True, timeout=30)
    if ('pcg01i: /dev/nvidia5 /dev/nvidia6' not in grant
            or not re.search(r'job_state\s+\d+:\s+r\b', grant)
            or not re.search(r'^owner:\s+wangyy\s*$', grant, re.MULTILINE)
            or not re.search(r'exec_host_list\s+\d+:\s+pcg01i:1\s*$', grant, re.MULTILINE)):
        raise RuntimeError('live allocation differs')
    uuids = subprocess.check_output(['nvidia-smi', '-i', '5,6', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=30).split()
    if uuids != runtime['gpu_uuids']:
        raise RuntimeError('GPU UUIDs changed')
    if not cpu and os.getpid() == int(supervisor):
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader'], text=True, timeout=30)
        if any(line.split(',')[1].strip() in uuids for line in apps.splitlines()):
            raise RuntimeError('approved GPU already occupied; no process was stopped')
    if (ROOT.resolve() != Path(runtime['repo']).resolve()
            or args.config.resolve().parent != ROOT / 'model_training/experiments'
            or args.output.resolve().parent != Path(runtime['output_root']).resolve()):
        raise RuntimeError('paths escaped isolated run scope')
    if deadline_due(config):
        raise RuntimeError('allocation handoff deadline reached')
    requested = 'report' if cpu else 'train' if args.mode == 'train' else 'preflight'
    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
        'check', '--operation', requested, '--entrypoint', ENTRYPOINT], check=True, stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('source must be clean and committed')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', op['code_commit'], source], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', op['code_commit'], source, '--',
        'model_training/rigweave', 'model_training/jobs', 'model_training/tools'], check=True)
    return source
