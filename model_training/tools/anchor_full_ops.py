#!/usr/bin/env python3
"""CPU-only planning and native-keeper dispatch for the approved full-data run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT.parent
ALLOCATION = RUN/'allocation_binding_native_20261027'
CONFIG = ROOT/'model_training/experiments/anchor_full_20261030_v1.json'


def command(*args):
    return subprocess.check_output(list(args), text=True, timeout=60).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('refuse to overwrite an existing admission artifact')
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def setup(config):
    runtime = config['runtime']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.sched_setaffinity(0, set(runtime['cpu_affinity'][:4]))
    sys.path[:0] = [str(ROOT/'model_training/rigweave/scripts'), str(ROOT/'model_training/rigweave/src'),
        runtime['extra_pythonpath'], runtime['unirig_root']]


def snapshot(config, *, idle):
    from virtual_pair_guard import require_job_membership, verify_allocation
    runtime = config['runtime']
    grant = command('qstat', '-j', str(runtime['allocation']))
    verify_allocation(config, grant)
    proc = Path('/proc')/str(runtime['keeper_pid'])
    membership = (proc/'cgroup').read_text()
    require_job_membership(membership, runtime['allocation'])
    children = (proc/'task'/str(runtime['keeper_pid'])/'children').read_text().split()
    if idle and children:
        raise RuntimeError('native keeper already has an active payload')
    for path in (proc/'task').iterdir():
        if sorted(os.sched_getaffinity(int(path.name))) != runtime['cpu_affinity']:
            raise RuntimeError('native keeper CPU binding changed')
    pending = []
    for request in (ALLOCATION/'requests').glob('*.json'):
        receipt = ALLOCATION/(request.stem+'_launch.json')
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            pending.append(request.name)
    if idle and pending:
        raise RuntimeError('another native stage is pending')
    apps = command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader')
    occupied = [line for line in apps.splitlines() if line.split(',')[1].strip() in runtime['gpu_uuids']]
    if idle and occupied:
        raise RuntimeError('approved GPUs are already occupied')
    deadline = datetime.fromisoformat(runtime['stop_before_utc'].replace('Z', '+00:00'))
    left = (deadline-datetime.now(timezone.utc)).total_seconds()
    if idle and left < 3600:
        raise RuntimeError('less than one hour before the mandatory saved stop')
    return {'observed_utc': datetime.now(timezone.utc).isoformat(), 'job_id': runtime['allocation'],
        'native_keeper_membership_verified': True, 'keeper_children': children, 'pending_stages': pending,
        'approved_GPU_processes': occupied, 'seconds_before_saved_stop': left,
        'free_SSD_bytes': shutil.disk_usage(RUN).free,
        'GPU_memory': command('nvidia-smi', '-i', ','.join(runtime['gpu_uuids']),
            '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('legacy-plan', 'plan', 'execute', 'observe'))
    parser.add_argument('--stage')
    parser.add_argument('--source')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    setup(config)
    from anchor_full_protocol import scientific_sha, source_hashes, validate_config
    from virtual_pair_guard import validate_stage
    validate_config(config)
    live = snapshot(config, idle=args.action != 'observe')
    if args.action == 'observe':
        if args.stage:
            output = RUN/'outputs'/args.stage
            for name in ('progress_rank_0.json', 'progress_rank_1.json', 'cache_progress.json',
                         'cleanup_progress.json', 'result.json', 'failure_rank_0.json', 'failure_rank_1.json'):
                path = output/name
                if path.exists():
                    value = json.loads(path.read_text())
                    if name == 'result.json':
                        value = {k: value.get(k) for k in ('complete', 'passed', 'samples', 'optimizer_updates',
                            'representative_mean_update_seconds', 'maximum_train_joints', 'cuda_initialized', 'legacy_cleanup')}
                    if isinstance(value, dict) and 'cases' in value:
                        value = {'completed_cases': len(value['cases']), 'last_case': value['cases'][-1] if value['cases'] else None}
                    live[name] = value
            for path in (ALLOCATION/(args.stage+'_launch.json'), RUN/'outputs'/(args.stage+'_resources.json')):
                if path.exists():
                    value = json.loads(path.read_text())
                    live[path.name] = {k: value.get(k) for k in ('complete', 'exit_code', 'passed',
                        'payload_cleanup_complete', 'error', 'supervisor_pid', 'peak_tree_rss_bytes', 'finished_utc')}
        print(json.dumps(live, indent=2))
        return
    source = command('git', '-C', str(ROOT), 'rev-parse', 'HEAD')
    if source != args.source or command('git', '-C', str(ROOT), 'status', '--porcelain'):
        raise RuntimeError('native workspace must be the exact clean committed source')
    if args.action == 'legacy-plan':
        from anchor_full_cleanup import plan_legacy
        plan = plan_legacy()
        path = RUN/'outputs/legacy_retirement_plan_20261030.json'
        write_new(path, plan)
        print(json.dumps({'path': str(path), 'sha256': digest(path), 'counts': plan['counts'],
            'full_GiB_before': plan['full_bytes_before_both_copies']/1024**3,
            'full_GiB_to_replace_or_remove': plan['full_bytes_selected_for_removal_both_copies']/1024**3,
            'kept_full': [r['relative'] for r in plan['rows'] if r['action'] == 'keep_full'], 'live': live}, indent=2))
        return
    mode = args.stage.split('_')[0] if args.stage else None
    if mode not in ('audit', 'capacity', 'train'):
        raise ValueError('a native audit/capacity/train stage is required')
    output = RUN/'outputs'/args.stage
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    op = validate_stage(state, config, SimpleNamespace(mode=mode, config=CONFIG, output=output))
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', op['code_commit'], source, '--',
        'model_training/rigweave', 'model_training/jobs', 'model_training/tools'], check=True)
    if live['free_SSD_bytes'] < config['data_cache']['reserve_free_bytes']:
        raise RuntimeError('SSD safety reserve unavailable')
    if mode == 'audit':
        spec = config['execution']['cleanup_plan']
        if digest(spec['path']) != spec['sha256']:
            raise RuntimeError('legacy cleanup plan changed')
    else:
        for key in ('CPU_audit',) if mode == 'capacity' else ('CPU_audit', 'capacity'):
            spec = config['execution'][key]
            if not spec or digest(spec['path']) != spec['sha256']:
                raise RuntimeError('primary admission result changed: '+key)
            result = json.loads(Path(spec['path']).read_text())
            resources = json.loads(Path(spec['resources_path']).read_text())
            if (not result.get('complete') or not result.get('passed')
                    or result['scientific_sha256'] != scientific_sha(config)
                    or result['source_sha256'] != source_hashes()
                    or not resources.get('passed') or resources.get('exit_code') != 0
                    or not resources.get('payload_cleanup_complete')):
                raise RuntimeError('primary admission receipt is not accepted: '+key)
    request = {'stage': args.stage, 'mode': mode, 'config': str(CONFIG), 'output': str(output), 'source': source}
    target = ALLOCATION/'requests'/(args.stage+'.json')
    if any(path.exists() for path in (output, target, RUN/(args.stage+'.log'), ALLOCATION/(args.stage+'_launch.json'))):
        raise RuntimeError('native stage identifier was already used')
    plan_path = RUN/'outputs'/('full_'+args.stage+'_admission.json')
    if args.action == 'plan':
        plan = {'passed': True, 'request': request, 'config_sha256': digest(CONFIG), 'live': live,
            'inspection_provenance': 'read-only SSH CPU inspection,4CPU affinity, no CUDA; not scheduler membership'}
        write_new(plan_path, plan)
        print(json.dumps(plan, indent=2))
        return
    plan = json.loads(plan_path.read_text())
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(plan['live']['observed_utc'])).total_seconds()
    if not plan['passed'] or plan['request'] != request or plan['config_sha256'] != digest(CONFIG) or not 0 <= age < 1200:
        raise RuntimeError('native stage plan is stale or changed')
    guard = ROOT/'model_training/tools/agent_work_guard.py'
    subprocess.run([sys.executable, str(guard), 'begin'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    operation = 'report' if mode == 'audit' else 'train' if mode == 'train' else 'preflight'
    subprocess.run([sys.executable, str(guard), 'check', '--operation', operation,
        '--entrypoint', op['required_entrypoint']], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    temporary = target.with_suffix('.tmp')
    write_new(temporary, request)
    temporary.replace(target)
    print(json.dumps({'queued': request, 'plan_sha256': digest(plan_path), 'live': live}, indent=2))


if __name__ == '__main__':
    main()
