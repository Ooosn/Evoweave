#!/usr/bin/env python3
"""Plan, dispatch and record the approved hardcase stages in the native keeper."""
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
ALLOCATION = RUN/'allocation_binding_native_20261031'
CONFIG = ROOT/'model_training/experiments/hardcase3608_20261101_v1.json'
STATE = ROOT/'model_training/state/current.json'


def command(*args):
    return subprocess.check_output(list(args), text=True, timeout=60).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    content = json.dumps(value, indent=2, sort_keys=True)+'\n'
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError('refuse to overwrite an admission artifact: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(content)


def snapshot(config, *, idle_gpu):
    from virtual_pair_guard import require_job_membership, verify_allocation
    runtime = config['runtime']
    verify_allocation(config, command('qstat','-j',str(runtime['allocation'])))
    proc=Path('/proc')/str(runtime['keeper_pid'])
    require_job_membership((proc/'cgroup').read_text(),runtime['allocation'])
    children=(proc/'task'/str(runtime['keeper_pid'])/'children').read_text().split()
    pending=[]
    for request in (ALLOCATION/'requests').glob('*.json'):
        receipt=ALLOCATION/(request.stem+'_launch.json')
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            pending.append(request.name)
    apps=command('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    occupied=[line for line in apps.splitlines() if line.split(',')[1].strip() in runtime['gpu_uuids']]
    if children or pending or (idle_gpu and occupied):
        raise RuntimeError('keeper/stage/GPU already occupied; preserve all existing processes')
    left=(datetime.fromisoformat(runtime['stop_before_utc'].replace('Z','+00:00'))
        -datetime.now(timezone.utc)).total_seconds()
    if left < 3600 or shutil.disk_usage(RUN).free < config['data_cache']['reserve_free_bytes']:
        raise RuntimeError('allocation time or SSD reserve is insufficient')
    return {'observed_utc':datetime.now(timezone.utc).isoformat(),'job_id':runtime['allocation'],
        'keeper_membership_verified':True,'keeper_children':children,'pending_stages':pending,
        'approved_GPU_processes':occupied,'seconds_before_saved_stop':left,
        'free_SSD_bytes':shutil.disk_usage(RUN).free,
        'keeper_affinity':sorted(os.sched_getaffinity(runtime['keeper_pid']))}


def record(config, mode, stage):
    from hardcase_adaptation_protocol import SCOPE
    from anchor_full_protocol import scientific_sha, source_hashes
    output=RUN/'outputs'/stage
    result_path, resources_path=output/'result.json', output.with_name(output.name+'_resources.json')
    result, resources=(json.loads(p.read_text()) for p in (result_path,resources_path))
    if (not result.get('complete') or not result.get('passed')
            or result['scientific_sha256']!=scientific_sha(config)
            or result['source_sha256']!=source_hashes()
            or not resources.get('complete') or not resources.get('passed')
            or resources.get('exit_code')!=0 or not resources.get('payload_cleanup_complete')
            or resources.get('kernel_scheduler_membership') is not True
            or resources.get('allocation')!=config['runtime']['allocation']):
        raise RuntimeError('primary result/resource evidence is not accepted')
    if mode=='audit' and (result.get('cuda_initialized') is not False
            or result['plan']['assets']!=30000 or result['plan']['updates']!=628
            or result['plan']['unique_assets']!=18002 or result['validation_panel']!=112
            or result['initialization'].get('project_optimizer_loaded') is not True):
        raise RuntimeError('CPU audit coverage, weighted plan or Adam import is incomplete')
    if mode=='capacity' and (len(result['ranks'])!=2
            or any(not r.get('resume_roundtrip_exact') for r in result['ranks'])):
        raise RuntimeError('two-rank capacity did not restore model/Adam/RNG/sampler exactly')
    state=json.loads(STATE.read_text())
    op=state['active_operation']
    if op['label']!=SCOPE:
        raise RuntimeError('another active operation owns the current state')
    spec={'path':str(result_path),'sha256':digest(result_path),
        'resources_path':str(resources_path),'resources_sha256':digest(resources_path)}
    key='CPU_audit' if mode=='audit' else 'capacity'
    config['execution'][key]=spec
    config_text=json.dumps(config,indent=2,sort_keys=True)+'\n'
    config_sha=hashlib.sha256(config_text.encode()).hexdigest()
    op[key+'_accepted']=True
    op[key]=spec
    for row in op['runs']:
        if row['mode']==mode:
            row.update(approved=False,complete=True,status='completed_passed',exit_code=0,result=spec)
        elif not row.get('complete'):
            row['config_sha256']=config_sha
            row['approved']=row['mode']==('capacity' if mode=='audit' else 'train')
    if mode=='capacity':
        op['formal_training_allowed']=True
        state['allowed_operations']=['inspect','report','preflight','train']
        state['blocked_operations']=['submit','matched_eval']
    op['phase']='capacity_ready_at_scheduled_time' if mode=='audit' else 'formal_training_ready'
    state['state_id']='hardcase3608_'+('CPU_ready' if mode=='audit' else 'capacity_ready')+'_20261101'
    state['updated_at']=datetime.now(timezone.utc).isoformat()
    state['next_required_result']=('At the authorized scheduled time, freshly plan and run capacity_v30, '
        'review its receipts, then record and commit admission before train_v30.' if mode=='audit'
        else 'Freshly plan and start train_v30 from full300k; never use the capacity checkpoint as initialization.')
    for path, content in ((CONFIG,config_text),(STATE,json.dumps(state,indent=2,sort_keys=True)+'\n')):
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps({'recorded':key,'evidence':spec,'config_sha256':config_sha,
        'commit_and_push_state_and_config_before_next_dispatch':True},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('snapshot','bind','plan','execute','record'))
    parser.add_argument('--stage')
    parser.add_argument('--source')
    args=parser.parse_args()
    config=json.loads(CONFIG.read_text())
    runtime=config['runtime']
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]='1'
    os.sched_setaffinity(0,set(runtime['cpu_affinity'][:4]))
    sys.path[:0]=[str(ROOT/'model_training/rigweave/scripts'),str(ROOT/'model_training/rigweave/src'),
        runtime['extra_pythonpath'],runtime['unirig_root']]
    from anchor_full_protocol import scientific_sha, source_hashes, validate_config
    from virtual_pair_guard import validate_stage
    validate_config(config)
    mode=args.stage.split('_')[0] if args.stage else None
    live=snapshot(config,idle_gpu=mode in ('capacity','train'))
    if args.action=='snapshot':
        print(json.dumps(live,indent=2)); return
    source=command('git','-C',str(ROOT),'rev-parse','HEAD')
    if source!=args.source or command('git','-C',str(ROOT),'status','--porcelain'):
        raise RuntimeError('exact clean source commit required')
    if args.action=='bind':
        binding=set(runtime['cpu_affinity'])
        for task in (Path('/proc')/str(runtime['keeper_pid'])/'task').iterdir():
            pid=int(task.name)
            if not binding <= os.sched_getaffinity(pid):
                raise RuntimeError('approved16-CPU binding is unavailable')
            os.sched_setaffinity(pid,binding)
        write_new(ALLOCATION/'hardcase3608_cpu_binding.json',{'job':runtime['allocation'],
            'keeper_pid':runtime['keeper_pid'],'cpu_affinity':sorted(binding),'cgroups_verified':True})
        print(json.dumps({'bound_keeper_CPU_affinity':sorted(binding)})); return
    if mode not in ('audit','capacity','train'):
        raise ValueError('expected audit/capacity/train stage')
    if live['keeper_affinity']!=runtime['cpu_affinity']:
        raise RuntimeError('bind the idle native keeper to the approved16CPU subset first')
    if args.action=='record':
        if mode=='train': raise ValueError('record training completion through the full result review')
        record(config,mode,args.stage); return
    output=RUN/'outputs'/args.stage
    op=validate_stage(json.loads(STATE.read_text()),config,
        SimpleNamespace(mode=mode,config=CONFIG,output=output))
    for key in (() if mode=='audit' else ('CPU_audit',) if mode=='capacity' else ('CPU_audit','capacity')):
        from anchor_full_runtime import checked_evidence
        checked_evidence(config,key)
    request={'stage':args.stage,'mode':mode,'config':str(CONFIG),'output':str(output),'source':source}
    target=ALLOCATION/'requests'/(args.stage+'.json')
    if any(p.exists() for p in (output,target,RUN/(args.stage+'.log'),ALLOCATION/(args.stage+'_launch.json'))):
        raise RuntimeError('stage already exists; inspect it instead of launching again')
    admission=RUN/'outputs'/('hardcase_'+args.stage+'_admission.json')
    if args.action=='plan':
        plan={'passed':True,'request':request,'config_sha256':digest(CONFIG),
            'scientific_sha256':scientific_sha(config),'source_sha256':source_hashes(),'live':live}
        write_new(admission,plan)
        print(json.dumps(plan,indent=2)); return
    plan=json.loads(admission.read_text())
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(plan['live']['observed_utc'])).total_seconds()
    if (not plan['passed'] or plan['request']!=request or plan['config_sha256']!=digest(CONFIG)
            or plan['source_sha256']!=source_hashes() or not 0<=age<1200):
        raise RuntimeError('stage plan is stale or changed')
    guard=ROOT/'model_training/tools/agent_work_guard.py'
    subprocess.run([sys.executable,str(guard),'begin'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    operation='report' if mode=='audit' else 'train' if mode=='train' else 'preflight'
    subprocess.run([sys.executable,str(guard),'check','--operation',operation,
        '--entrypoint',op['required_entrypoint']],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    temporary=target.with_suffix('.tmp')
    write_new(temporary,request)
    temporary.replace(target)
    print(json.dumps({'queued':request,'plan_sha256':digest(admission)},indent=2))


if __name__=='__main__':
    main()
