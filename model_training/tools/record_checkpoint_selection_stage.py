"""Record one reviewed native stage; admit only the next frozen selection stage."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT.parent
STAGES=('audit_v34','audit_v35','capacity_v34','capacity_v35')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def command(*args): return subprocess.check_output(list(args),text=True).strip()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=STAGES,required=True); args=parser.parse_args()
    stage=args.stage; version=stage.split('_')[-1]; mode=stage.split('_')[0]
    config_path=ROOT/('model_training/experiments/hardcase'+('380' if version=='v34' else '390')+'k_visual12_20261102.json')
    config=read(config_path)
    from anchor_full_ops import setup
    setup(config)
    from predicted_reference_protocol import scientific_sha
    state_path=ROOT/'model_training/state/current.json'; state=read(state_path); op=state['active_operation']
    if op['experiment_id']!='hardcase_checkpoint_selection_20261102' or op['formal_training_allowed'] or op['complete']:
        raise RuntimeError('not the active frozen selection operation')
    if Path('/proc/2268188/task/2268188/children').read_text().strip() or command('git','-C',str(ROOT),'status','--porcelain'):
        raise RuntimeError('payload active or checkout dirty; source update prohibited')
    entry=next(r for r in op['runs'] if r['stage']==stage)
    result_path=RUN/'outputs'/stage/'result.json'; resources_path=RUN/'outputs'/(stage+'_resources.json')
    result,resources=read(result_path),read(resources_path)
    launch=read(RUN/'allocation_binding_native_20261031'/(stage+'_launch.json'))
    if entry['complete']:
        if entry['result']['sha256']!=sha(result_path): raise RuntimeError('recorded result changed')
        print(json.dumps({'already_recorded':True,'stage':stage})); return
    if not entry['approved'] or result['source']!=command('git','-C',str(ROOT),'rev-parse','HEAD'):
        raise RuntimeError('unapproved or unexpected stage source')
    if not (result['complete'] and result['passed'] and result['optimizer_updates']==0
            and result['scientific_sha256']==scientific_sha(config)
            and resources['complete'] and resources['passed'] and resources['exit_code']==0
            and resources['payload_cleanup_complete'] and resources['kernel_scheduler_membership']
            and resources['allocation']==131170166 and resources['scheduler_cpu_slots']==16
            and resources['execution_cpu_budget']==8 and launch['complete'] and launch['exit_code']==0):
        raise RuntimeError('actual stage result/native resource gate failed')
    for pid in (resources['supervisor_pid'],resources['child_pid']):
        if Path('/proc',str(pid)).exists(): raise RuntimeError('own payload still exists')
    if mode=='audit':
        if result['references']!=12 or result['cuda_initialized'] or not result['frozen_baseline_verified']['same_inputs_and_official_seed']:
            raise RuntimeError('candidate CPU identity audit failed')
        if result['inherited_official_capacity']['0']['stress_repeated']: raise RuntimeError('capacity stress was repeated')
        config['audit_result']={'path':str(result_path),'sha256':sha(result_path),'resources_path':str(resources_path)}
        if scientific_sha(config)!=result['scientific_sha256']: raise RuntimeError('scientific audit binding changed')
        write(config_path,config)
        op.setdefault('CPU_audits',{})[version]={**config['audit_result'],'resources_sha256':sha(resources_path)}
    else:
        if result['checkpoints']!=config['checkpoints'] or resources['physical_gpus']!=[4]: raise RuntimeError('actual candidate or GPU differs')
        rows=refs=0
        for artifact in result['artifacts']:
            if artifact['beam_capacity']['stress_repeated']: raise RuntimeError('new stress forbidden')
            for key in ('rows','references','examples'):
                item=artifact[key]
                if sha(item['path'])!=item['sha256'] or sha(item['durable_path'])!=item['sha256']: raise RuntimeError('candidate artifact/durable hash differs')
            rows+=len(read(artifact['rows']['path'])); refs+=len(read(artifact['references']['path']))
        if rows!=336 or refs!=12: raise RuntimeError('candidate output denominator differs')
    entry.update(approved=False,complete=True,exit_code=0,status='passed_pending_independent_case_review' if mode=='capacity' else 'CPU_audit_passed',
        result={'path':str(result_path),'sha256':sha(result_path),'resources_path':str(resources_path),'resources_sha256':sha(resources_path),'source':result['source']})
    for r in op['runs']:
        if r['config']==str(config_path) and not r['complete']: r['config_sha256']=sha(config_path)
    index=STAGES.index(stage)
    if index<3: next(r for r in op['runs'] if r['stage']==STAGES[index+1])['approved']=True
    cpu_done=all(r['complete'] for r in op['runs'] if r['mode']=='audit')
    op.update(CPU_audit_accepted=cpu_done,capacity_accepted=cpu_done,gpu_work_allowed=cpu_done and index<3,
        phase='independent_review_and_final_working_choice' if index==3 else 'native_'+STAGES[index+1],
        status=stage+'_recorded',new_capacity_stress_permitted=False)
    state['allowed_operations']=['inspect','report']+(['preflight'] if cpu_done and index<3 else [])
    state['blocked_operations']=['train','submit','matched_eval']+(['preflight'] if not op['gpu_work_allowed'] else [])
    state['state_id']='hardcase_checkpoint_selection_'+stage+'_recorded_20261102'
    state['updated_at']=datetime.now(timezone.utc).isoformat()
    state['next_required_result']=('Independently rescore both candidates and inspect all12cases; apply frozen working-choice rule then record one model, fallback330k. No more GPU work or training.' if index==3 else 'Plan and execute '+STAGES[index+1]+' once with current exact config/native resource proof. No training, repeated stress, baseline regeneration or retries.')
    write(state_path,state)
    subprocess.run(['git','-C',str(ROOT),'diff','--check'],check=True)
    subprocess.run(['git','-C',str(ROOT),'add','--',str(state_path.relative_to(ROOT)),str(config_path.relative_to(ROOT))],check=True)
    subprocess.run(['git','-C',str(ROOT),'commit','-m','Record frozen checkpoint selection '+stage+' and admit next stage only'],check=True,stdout=subprocess.DEVNULL)
    source=command('git','-C',str(ROOT),'rev-parse','HEAD')
    archive=RUN/'source_deployments'/('source_'+source+'.tar.gz')
    durable=Path('/home/wangyy/evorig/outputs/hardcase3608_20261101/source_deployments')/archive.name
    if archive.exists() or durable.exists(): raise RuntimeError('source snapshot already exists')
    subprocess.run(['git','-C',str(ROOT),'archive','--format=tar.gz','--prefix=source/','-o',str(archive),source],check=True)
    with tarfile.open(archive,'r:gz') as handle:
        if handle.extractfile('source/model_training/state/current.json').read()!=state_path.read_bytes(): raise RuntimeError('source archive state differs')
    shutil.copyfile(archive,durable); os.chmod(durable,0o600)
    if sha(archive)!=sha(durable): raise RuntimeError('durable source archive differs')
    subprocess.run(['git','-C',str(ROOT),'push','origin','HEAD:refs/heads/codex/virtual-fusion-20261013'],check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'model_training/tools/agent_work_guard.py'),'begin'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    print(json.dumps({'recorded':stage,'source':source,'result':entry['result'],'state_id':state['state_id'],
        'next_stage':None if index==3 else STAGES[index+1],'GPU_work_allowed':op['gpu_work_allowed'],
        'source_archive_sha256':sha(archive),'formal_training_allowed':False},indent=2))

if __name__=='__main__': main()
