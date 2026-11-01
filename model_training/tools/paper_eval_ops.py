"""Use the existing native keeper for a separately recorded paper inference stage."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from anchor_full_ops import ROOT, RUN, setup, digest, write_new
from full_visual_ops import snapshot


def main():
    if os.environ.get('EVOWEAVE_VISUAL_SITE_READY') != '1':
        script = ('set -e\nsource /home/geadmin/N1GE/default/common/settings.sh\nsource ~/1.sh\n'
                  'export EVOWEAVE_VISUAL_SITE_READY=1 CUDA_VISIBLE_DEVICES=""\nexec "$@"')
        import sys
        os.execv('/bin/bash', ['/bin/bash', '-c', script, 'paper-site', sys.executable,
            '-B', str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan','execute','observe'))
    parser.add_argument('--stage', required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.stage not in ('audit_v36','audit_v37','audit_v38','capacity_v36','capacity_v38'):
        raise ValueError('unrecorded paper stage')
    config = json.loads(args.config.read_text())
    setup(config)
    from paper_benchmark_protocol import validate_config, scientific_sha, shared_gpu_admission
    from virtual_pair_guard import validate_stage
    from types import SimpleNamespace
    mode = args.stage.split('_')[0]
    validate_config(config, mode)
    live = snapshot(config, idle=False, allow_shared_cpu=True)
    if args.action != 'observe':
        if live['keeper_children'] or live['pending_stages']:
            raise RuntimeError('a native payload is already active')
        if live['seconds_before_saved_stop'] < config['evaluation']['payload_limit_seconds'] + 60:
            raise RuntimeError('insufficient retained allocation time')
        live['shared_GPU_admission'] = (shared_gpu_admission(config) if mode!='audit'
            else {'CPU_only':True,'GPU_memory_admission_deferred':True})
    allocation = Path(config['runtime']['allocation_directory'])
    output = RUN/'outputs'/args.stage
    if args.action == 'observe':
        for label, path in [('launch',allocation/(args.stage+'_launch.json')),
            ('resources',RUN/'outputs'/(args.stage+'_resources.json')),
            ('result',output/'result.json'),('progress',output/'progress.json'),('failure',output/'failure.json')]:
            if path.exists():
                value = json.loads(path.read_text())
                live[label] = {k:v for k,v in value.items() if k in ('complete','passed','exit_code',
                    'error','method','assets','rows','failures','edge_f1_all','summaries','payload_cleanup_complete','seconds')}
        print(json.dumps(live,indent=2))
        return
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    op = validate_stage(state,config,SimpleNamespace(mode=mode,config=args.config,output=output))
    if op.get('formal_training_allowed'):
        raise ValueError('training must remain closed')
    source = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('source must be clean')
    if mode == 'capacity':
        spec = config['audit_result']
        accepted = json.loads(Path(spec['path']).read_text())
        resources = json.loads(Path(spec['resources_path']).read_text())
        if (digest(spec['path']) != spec['sha256'] or not accepted['passed'] or not accepted['complete']
                or accepted['scientific_sha256'] != scientific_sha(config)
                or not resources['passed'] or not resources['payload_cleanup_complete']):
            raise ValueError('paper CPU audit has not passed')
    request = {'stage':args.stage,'mode':mode,'config':str(args.config),'output':str(output),'source':source}
    target = allocation/'requests'/(args.stage+'.json')
    if any(p.exists() for p in (target,output,allocation/(args.stage+'_launch.json'))):
        raise ValueError('paper stage already exists; inspect it instead of restarting')
    plan_path = RUN/'outputs'/('paper_'+args.stage+'_admission.json')
    plan = {'passed':True,'request':request,'config_sha256':digest(args.config),'live':live}
    if args.action == 'plan':
        write_new(plan_path,plan)
    else:
        previous = json.loads(plan_path.read_text())
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(previous['live']['observed_utc'])).total_seconds()
        if (not previous['passed'] or previous['request']!=request
                or previous['config_sha256']!=digest(args.config) or not 0 <= age < 1200):
            raise ValueError('paper plan stale or changed')
        write_new(target.with_suffix('.tmp'),request)
        target.with_suffix('.tmp').replace(target)
        plan['queued'] = True
    print(json.dumps(plan,indent=2))


if __name__ == '__main__':
    main()
