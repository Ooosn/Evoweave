"""Admission for the fixed-5k, matched virtual-path training pair."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import shutil
import subprocess
import sys

from train_legacy_single_target import file_sha256, verify_exception_keeper
from single_query_protocol import SINGLE_SCOPE, TWO_SCOPE, DENSITY_SCOPE
from reference_xyz_protocol import SCOPE as REFERENCE_SCOPE, ENTRYPOINT as REFERENCE_ENTRYPOINT

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = 'model_training/rigweave/scripts/train_virtual_pair.py'
LAUNCHER = 'model_training/jobs/launch_virtual_pair.py'
SCOPE = 'virtual_query_matched_5k_20261025_v1'
EXCEPTION_ENV = 'EVOWEAVE_VIRTUAL_PAIR_SSH_EXCEPTION'
CPUS = [58, 63, 64, 65]


def require_job_membership(text, job):
    membership = {}
    for line in text.splitlines():
        try:
            _, controllers, path = line.split(':', 2)
        except ValueError as error:
            raise RuntimeError('invalid scheduler cgroup record') from error
        for controller in controllers.split(','):
            if controller in ('devices', 'cpuset'):
                if controller in membership:
                    raise RuntimeError('duplicate scheduler cgroup controller')
                membership[controller] = path
    root = f'/AGE/{job}.1'
    # qrsh places this allocation's payload in /master under the job root.
    if (membership.get('devices') not in (root, root + '/master')
            or membership.get('cpuset') != membership.get('devices')):
        raise RuntimeError('launcher is outside the granted scheduler job')


def operation(state):
    op = state.get('active_operation', {})
    if op.get('label') not in (SCOPE, SINGLE_SCOPE, TWO_SCOPE, DENSITY_SCOPE, REFERENCE_SCOPE) or op.get('explicit_human_authorized') is not True:
        raise RuntimeError('matched pair lacks explicit human authorization')
    expected_entrypoint = REFERENCE_ENTRYPOINT if op.get('label') == REFERENCE_SCOPE else ENTRYPOINT
    if op.get('required_entrypoint') != expected_entrypoint:
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
    if config['runtime'].get('execution_mode') == 'scheduler_job':
        runtime = scheduler_runtime(config)
        binding = set(runtime['cpu_affinity'])
        if exception is not None or not binding <= allowed_cpus:
            raise RuntimeError('approved scheduler CPU binding unavailable')
        return binding
    if not exception or not set(CPUS) <= allowed_cpus:
        raise RuntimeError('approved four-CPU binding unavailable')
    return set(CPUS)


def scheduler_runtime(config):
    runtime = config['runtime']
    if (config.get('family') not in (SCOPE, SINGLE_SCOPE, TWO_SCOPE, DENSITY_SCOPE, REFERENCE_SCOPE) or runtime.get('execution_mode') != 'scheduler_job'
            or runtime.get('require_scheduler_membership') is not True):
        raise RuntimeError('scheduler-owned execution must require job membership')
    for name, expected in (('cpu_slots', 16), ('execution_cpu_budget', 16), ('world_size', 2)):
        if type(runtime.get(name)) is not int or runtime[name] != expected:
            raise RuntimeError(f'scheduler resource binding differs: {name}')
    cpus, gpus, uuids = (runtime.get(name) for name in ('cpu_affinity', 'physical_gpus', 'gpu_uuids'))
    for name, values, count in (('CPU affinity', cpus, 16), ('physical GPUs', gpus, 2)):
        if (type(values) is not list or len(values) != count
                or any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != count):
            raise RuntimeError(f'invalid scheduler {name}')
    if (type(uuids) is not list or len(uuids) != 2
            or any(not isinstance(value, str) or not value.strip() for value in uuids)
            or len(set(uuids)) != 2):
        raise RuntimeError('invalid scheduler GPU UUID binding')
    if (type(runtime.get('allocation')) is not int or runtime['allocation'] <= 0
            or not isinstance(runtime.get('host'), str) or not runtime['host'].strip()
            or ('hostname' in runtime and (not isinstance(runtime['hostname'], str)
                                          or not runtime['hostname'].strip()))):
        raise RuntimeError('invalid scheduler allocation/host binding')
    limit = runtime.get('gpu_memory_fraction_limit')
    if type(limit) not in (int, float) or not 0 < limit <= 1:
        raise RuntimeError('invalid scheduler GPU memory limit')
    return runtime


def gpu_selection(config):
    runtime = config['runtime']
    if runtime.get('execution_mode') == 'scheduler_job':
        scheduler_runtime(config)
        return ','.join(runtime['gpu_uuids'])
    return ','.join(map(str, runtime['physical_gpus']))


def verify_allocation(config, grant):
    if config['runtime'].get('execution_mode') == 'scheduler_job':
        runtime = scheduler_runtime(config)
        fields = {
            'job': re.findall(r'^job_number:[ \t]*(\d+)[ \t]*$', grant, re.MULTILINE),
            'owner': re.findall(r'^owner:[ \t]*(\S+)[ \t]*$', grant, re.MULTILINE),
            'state': re.findall(r'^job_state[ \t]+\d+:[ \t]*(\S+)[ \t]*$', grant, re.MULTILINE),
            'host': re.findall(r'^exec_host_list[ \t]+\d+:[ \t]*(.+?)[ \t]*$', grant, re.MULTILINE),
            'devices': re.findall(r'^granted_devices\b[^\r\n]*$', grant, re.MULTILINE),
        }
        if (fields['job'] != [str(runtime['allocation'])] or fields['owner'] != ['wangyy']
                or fields['state'] != ['r']
                or fields['host'] != [f"{runtime['host']}:{runtime['cpu_slots']}"]
                or len(fields['devices']) != runtime['cpu_slots']):
            raise RuntimeError('live scheduler allocation differs')
        # This scheduler reports the same device grant once for each task-1 CPU slot.
        records = [re.fullmatch(r'granted_devices[ \t]+(\d+),(\d+):[ \t]*(.+?)[ \t]*', row)
                   for row in fields['devices']]
        if (any(row is None for row in records)
                or {(int(row[1]), int(row[2])) for row in records}
                != {(1, slot) for slot in range(runtime['cpu_slots'])}):
            raise RuntimeError('live scheduler GPU slot records differ')
        expected = {f'/dev/nvidia{gpu}' for gpu in runtime['physical_gpus']}
        for row in records:
            host, separator, devices = row[3].partition(':')
            if (not separator or host != runtime['host'] or len(devices.split()) != len(expected)
                    or set(devices.split()) != expected):
                raise RuntimeError('live scheduler GPU devices differ')
        required_time = runtime.get('required_h_rt_seconds')
        if required_time is not None:
            limits = re.findall(r'\bh_rt=(\d+)\b', grant)
            if (type(required_time) is not int or required_time <= 0
                    or len(limits) != 1 or int(limits[0]) < required_time):
                raise RuntimeError('live scheduler lifetime is shorter than the declared pilot')
        return
    if ('pcg01i: /dev/nvidia5 /dev/nvidia6' not in grant
            or not re.search(r'job_state\s+\d+:\s+r\b', grant)
            or not re.search(r'^owner:\s+wangyy\s*$', grant, re.MULTILINE)
            or not re.search(r'exec_host_list\s+\d+:\s+pcg01i:1\s*$', grant, re.MULTILINE)):
        raise RuntimeError('live allocation differs')


def deadline_due(config):
    deadline = datetime.fromisoformat(config['runtime']['stop_before_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('deadline needs a timezone')
    return datetime.now(timezone.utc) >= deadline


def validate_stage(state, config, args):
    op = operation(state)
    if op['label'] != config.get('family'):
        raise RuntimeError('active operation and experiment family differ')
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
    scheduler = runtime.get('execution_mode') == 'scheduler_job'
    exception = None
    if scheduler:
        scheduler_runtime(config)
        require_job_membership(Path('/proc/self/cgroup').read_text(), runtime['allocation'])
        if (os.environ.get('JOB_ID') != str(runtime['allocation'])
                or os.environ.get('NSLOTS') != str(runtime['cpu_slots'])
                or os.environ.get(EXCEPTION_ENV) == '1'):
            raise RuntimeError('scheduler execution needs inherited job/slots without an SSH exception')
    else:
        exception = validate_ssh_exception(state, config)
        verify_exception_keeper(exception)
    supervisor = os.environ.get('EVOWEAVE_RESOURCE_SUPERVISOR_PID', '')
    if (not scheduler and os.environ.get(EXCEPTION_ENV) != '1') or not supervisor.isdigit():
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
    expected_host = (runtime.get('hostname', 'pcg01' if runtime['host'] == 'pcg01i' else runtime['host'])
                     if scheduler else 'pcg01')
    affinity = os.sched_getaffinity(0)
    binding = execution_cpu_binding(config, exception, affinity)
    devices = gpu_selection(config)
    if (socket.gethostname().split('.')[0] != expected_host or affinity != binding
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if cpu else devices) or runtime['world_size'] != 2):
        raise RuntimeError('host/CPU/GPU binding differs')
    import resource
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise RuntimeError('unlimited CUDA virtual address space required')
    grant = subprocess.check_output(['qstat', '-j', str(runtime['allocation'])], text=True, timeout=30)
    verify_allocation(config, grant)
    try:
        uuids = subprocess.check_output(['nvidia-smi', '-i', devices, '--query-gpu=uuid',
            '--format=csv,noheader'], text=True, timeout=30, stderr=subprocess.STDOUT).split()
    except subprocess.CalledProcessError as error:
        context = {'uid': os.getuid(), 'euid': os.geteuid(), 'nvidia_smi': shutil.which('nvidia-smi'),
            'environment': {name: os.environ.get(name) for name in ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'JOB_ID', 'NSLOTS', 'SGE_HGR_h100')},
            'cgroups': Path('/proc/self/cgroup').read_text()}
        for name, command in [('inventory', ['nvidia-smi', '-L']), ('version', ['nvidia-smi', '--version'])]:
            diagnostic = subprocess.run(command, text=True, capture_output=True, timeout=30)
            context[name] = {'returncode': diagnostic.returncode, 'stdout': diagnostic.stdout[-4000:],
                             'stderr': diagnostic.stderr[-4000:]}
        if shutil.which('strace'):
            diagnostic = subprocess.run(['strace', '-f', '-e', 'trace=openat,ioctl', '-s', '120',
                'nvidia-smi', '-i', devices, '--query-gpu=uuid', '--format=csv,noheader'],
                text=True, capture_output=True, timeout=30)
            context['access_trace'] = diagnostic.stderr[-12000:]
        print('GPU_ADMISSION_DIAGNOSTIC ' + json.dumps(context), file=sys.stderr, flush=True)
        raise RuntimeError(f'GPU identity query failed rc={error.returncode}: {error.output}') from error
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
        'check', '--operation', requested, '--entrypoint', op['required_entrypoint']], check=True, stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('source must be clean and committed')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', op['code_commit'], source], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', op['code_commit'], source, '--',
        'model_training/rigweave', 'model_training/jobs', 'model_training/tools'], check=True)
    return source
