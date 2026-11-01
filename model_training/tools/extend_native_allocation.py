#!/usr/bin/env python3
"""Plan and extend the retained native allocation without replacing its GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def command(*argv):
    return subprocess.check_output(list(argv), text=True, timeout=45).strip()

def resources(grant):
    line = re.search(r'^hard_resource_list:\s*(.+)$', grant, re.M)
    if not line:
        raise RuntimeError('missing scheduler resource list')
    return dict(field.strip().split('=', 1) for field in line[1].split(','))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'execute'))
    args = parser.parse_args()
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    allocation = state['compute_allocation']
    op = state['active_operation']
    approved = op['allocation_lifetime_extension']
    if (approved.get('approved') is not True or approved['allocation'] != allocation['qlogin_job_id']
            or approved['requested_h_rt_seconds'] != 345600 or op['gpu_work_allowed']
            or op['formal_training_allowed']):
        raise RuntimeError('exact resource-only96h extension is not admitted')
    os.sched_setaffinity(0, set(allocation['cpu_affinity'][:4]))
    source = command('git', '-C', str(ROOT), 'rev-parse', 'HEAD')
    if command('git', '-C', str(ROOT), 'status', '--porcelain'):
        raise RuntimeError('dirty source checkout')
    job, keeper = allocation['qlogin_job_id'], allocation['keeper_pid']
    proc = Path('/proc')/str(keeper)
    groups = (proc/'cgroup').read_text()
    if any(':'+name+':/AGE/'+str(job)+'.1/master' not in groups for name in ('cpuset', 'devices')):
        raise RuntimeError('native keeper membership differs')
    if (proc/'task'/str(keeper)/'children').read_text().strip():
        raise RuntimeError('review allocation extension between payloads')
    grant = command('qstat', '-j', str(job))
    before = resources(grant)
    if (before not in ({'h100':'2','h_rt':'86400','mem_req':'4G'},
                       {'h100':'2','h_rt':'345600','mem_req':'4G'})
            or not re.search(r'job_state\s+1:\s+r\b', grant)
            or not re.search(r'exec_host_list\s+1:\s+pcg01i:16\b', grant)):
        raise RuntimeError('grant identity or unchanged GPU/CPU/memory contract differs')
    guard = ROOT/'model_training/tools/agent_work_guard.py'
    entry = 'model_training/tools/extend_native_allocation.py'
    for arguments in (['begin'], ['check','--operation','preflight','--entrypoint',entry]):
        subprocess.run([sys.executable, '-B', str(guard), *arguments], cwd=ROOT,
            check=True, stdout=subprocess.DEVNULL)
    directory = Path(allocation['allocation_directory'])
    path = directory/'lifetime_extension_96h_plan.json'
    argv = ['qalter','-when','now','-l','h100=2,h_rt=96:00:00,mem_req=4G',str(job)]
    plan = {'source':source,'job':job,'keeper':keeper,'before':before,'argv':argv,
        'state_sha256':hashlib.sha256((ROOT/'model_training/state/current.json').read_bytes()).hexdigest(),
        'observed_utc':datetime.now(timezone.utc).isoformat(),'no_GPU_or_CPU_count_change':True}
    if args.action == 'plan':
        with path.open('x') as handle:
            json.dump(plan,handle,indent=2,sort_keys=True)
        print(json.dumps(plan))
        return
    saved = json.loads(path.read_text())
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(saved['observed_utc'])).total_seconds()
    if (not 0 <= age < 1200 or any(saved[k] != plan[k] for k in ('source','job','keeper','argv','state_sha256'))):
        raise RuntimeError('extension plan is stale or changed')
    response = ('already96h' if before['h_rt'] == '345600' else command(*argv))
    after_text = command('qstat','-j',str(job))
    after = resources(after_text)
    if (after != {'h100':'2','h_rt':'345600','mem_req':'4G'}
            or not re.search(r'exec_host_list\s+1:\s+pcg01i:16\b',after_text)
            or not proc.exists() or (proc/'cgroup').read_text() != groups):
        raise RuntimeError('scheduler did not retain the requested96h native grant')
    result = {'complete':True,'passed':True,'plan':saved,'scheduler_response':response,
        'after':after,'keeper_preserved':True,'native_cgroups_unchanged':True,
        'observed_utc':datetime.now(timezone.utc).isoformat()}
    result_path=directory/'lifetime_extension_96h_result.json'
    with result_path.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
