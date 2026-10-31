#!/usr/bin/env python3
"""Read-only admission checks and native keeper dispatch for fixed full-model visuals."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from anchor_full_ops import ROOT, RUN, ALLOCATION, command, digest, setup, write_new


def snapshot(config, *, idle):
    from virtual_pair_guard import scheduler_runtime, verify_allocation, require_job_membership
    runtime = scheduler_runtime(config)
    verify_allocation(config, command('qstat', '-j', str(runtime['allocation'])))
    proc = Path('/proc')/str(runtime['keeper_pid'])
    require_job_membership((proc/'cgroup').read_text(), runtime['allocation'])
    assert sorted(os.sched_getaffinity(runtime['keeper_pid'])) == runtime['allocated_cpu_affinity']
    children = (proc/'task'/str(runtime['keeper_pid'])/'children').read_text().split()
    pending = [p.name for p in (ALLOCATION/'requests').glob('*.json')
        if not (ALLOCATION/(p.stem+'_launch.json')).exists()
        or not json.loads((ALLOCATION/(p.stem+'_launch.json')).read_text())['complete']]
    apps = command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader')
    selected, other = [], []
    for line in apps.splitlines():
        pid, uuid = line.split(',')
        if uuid.strip() in runtime['gpu_uuids']:
            selected.append(line)
        elif uuid.strip() == 'GPU-1fb4221b-91b0-0df6-8615-b36fbdd55bf7':
            threads = [set(os.sched_getaffinity(int(p.name))) for p in (Path('/proc')/pid.strip()/'task').iterdir()]
            if any(cpus & set(runtime['cpu_affinity']) for cpus in threads):
                raise RuntimeError('other allocated-GPU workload overlaps the visual CPU subset')
            other.append({'pid': int(pid), 'GPU': uuid.strip(), 'thread_CPU_union': sorted(set.union(*threads))})
    if idle and (children or pending or selected):
        raise RuntimeError('native keeper or selected GPU4 is occupied')
    left = (datetime.fromisoformat(runtime['stop_before_utc'].replace('Z', '+00:00'))-datetime.now(timezone.utc)).total_seconds()
    if idle and left < 1800:
        raise RuntimeError('insufficient allocation time for visual inference')
    return {'observed_utc': datetime.now(timezone.utc).isoformat(), 'job_id': runtime['allocation'],
        'native_keeper_membership_verified': True, 'keeper_children': children, 'pending_stages': pending,
        'selected_GPU_processes': selected, 'other_GPU_workloads_not_modified': other,
        'seconds_before_saved_stop': left, 'scheduler_CPU_slots': 16, 'execution_CPU_budget': 8,
        'physical_GPU_used': 4, 'GPU_memory': command('nvidia-smi', '-i', runtime['gpu_uuids'][0],
            '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader')}


def main():
    if os.environ.get('EVOWEAVE_VISUAL_SITE_READY') != '1':
        script = ('set -e\nsource /home/geadmin/N1GE/default/common/settings.sh\nsource ~/1.sh\n'
                  'export EVOWEAVE_VISUAL_SITE_READY=1 CUDA_VISIBLE_DEVICES=""\nexec "$@"')
        os.execv('/bin/bash', ['/bin/bash', '-c', script, 'visual-site', sys.executable,
            '-B', str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'execute', 'observe'))
    parser.add_argument('--stage', choices=('audit_v27', 'capacity_v27'), required=True)
    args = parser.parse_args()
    path = ROOT/'model_training/experiments/full_model_visual12_20261031.json'
    config = json.loads(path.read_text())
    setup(config)
    from predicted_reference_protocol import scientific_sha, validate_config
    from virtual_pair_guard import validate_stage
    mode = args.stage.split('_')[0]
    validate_config(config, mode)
    live = snapshot(config, idle=args.action != 'observe')
    output = RUN/'outputs'/args.stage
    if args.action == 'observe':
        for label, item in [('result', output/'result.json'),
                ('resources', RUN/'outputs'/(args.stage+'_resources.json')),
                ('launch', ALLOCATION/(args.stage+'_launch.json'))]:
            if item.exists():
                value = json.loads(item.read_text())
                live[label] = {k: value.get(k) for k in ('complete', 'passed', 'exit_code',
                    'payload_cleanup_complete', 'error', 'finished_utc', 'optimizer_updates', 'summaries')}
        for item in sorted(output.glob('*rank_*.json')):
            if item.name.startswith(('progress_', 'failure_', 'beam_capacity_')):
                live[item.name] = json.loads(item.read_text())
        print(json.dumps(live, indent=2))
        return
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    op = validate_stage(state, config, SimpleNamespace(mode=mode, config=path, output=output))
    if op['formal_training_allowed'] or config['evaluation']['optimizer_updates'] != 0:
        raise RuntimeError('visual scope must not authorize training')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('dirty deployment checkout')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', op['code_commit'], source,
        '--', 'model_training/rigweave', 'model_training/jobs', 'model_training/tools'], check=True)
    if mode == 'capacity':
        spec = config['audit_result']
        if digest(spec['path']) != spec['sha256']:
            raise RuntimeError('CPU audit changed')
        audit, resources = (json.loads(Path(spec[k]).read_text()) for k in ('path', 'resources_path'))
        if not (audit['passed'] and audit['complete'] and audit['scientific_sha256'] == scientific_sha(config)
                and resources['passed'] and resources['exit_code'] == 0 and resources['payload_cleanup_complete']):
            raise RuntimeError('CPU visual admission did not pass')
    request = {'stage': args.stage, 'mode': mode, 'config': str(path), 'output': str(output), 'source': source}
    target = ALLOCATION/'requests'/(args.stage+'.json')
    if any(p.exists() for p in (output, target, ALLOCATION/(args.stage+'_launch.json'))):
        raise RuntimeError('visual stage already used')
    plan_path = RUN/'outputs'/('full_visual_'+args.stage+'_admission.json')
    plan = {'passed': True, 'request': request, 'config_sha256': digest(path), 'live': live}
    if args.action == 'plan':
        write_new(plan_path, plan)
    else:
        previous = json.loads(plan_path.read_text())
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(previous['live']['observed_utc'])).total_seconds()
        if not previous['passed'] or previous['request'] != request or previous['config_sha256'] != digest(path) or not 0 <= age < 1200:
            raise RuntimeError('visual admission plan stale or changed')
        guard = ROOT/'model_training/tools/agent_work_guard.py'
        subprocess.run([sys.executable, str(guard), 'begin'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(guard), 'check', '--operation', 'report' if mode == 'audit' else 'preflight',
            '--entrypoint', op['required_entrypoint']], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        write_new(target.with_suffix('.tmp'), request)
        target.with_suffix('.tmp').replace(target)
        plan['queued'] = True
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
