"""One native, fail-closed sequence after the already-running continuation."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parents[2]
ROOT = CODE.parent
PY = sys.executable
EXP = CODE/'model_training/experiments'
STATE = CODE/'model_training/state/current.json'
CONT = 'tokenrig_native_cont400k_20261107'
VIS = 'tokenrig_native_cont400k_visual12_20261107'
CONTROL = 'tokenrig_native_frame_control_20261108'
OUT = ROOT/'outputs_native_cont400k_20261107'
HOME = Path('/home/wangyy/evorig/outputs/tokenrig_alignment_20261105')
DUR = HOME/'native_cont400k_20261107'
STATUS = DUR/'followup_status.json'
STATE_ARCHIVE = DUR/'receipts/followup_source'
SCRIPT = CODE/'model_training/rigweave/scripts'
sys.path.insert(0, str(SCRIPT))

from tokenrig_comparison_matrix import digest, read
from tokenrig_native_frame_control import recipe, same_native_capacity
from virtual_pair_guard import require_job_membership


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(['git','-C',str(CODE),*args],text=True).strip()


def commit(label, *paths):
    names = [str(p.relative_to(CODE)) for p in paths]
    git('add','--',*names)
    git('-c','user.name=Evoweave native runner','-c','user.email=evoweave-runner@localhost',
        'commit','--only','-m',label,'--',*names)
    return git('rev-parse','HEAD')


def snapshot(label, base, path):
    STATE_ARCHIVE.mkdir(parents=True, exist_ok=True)
    for source, name in ((path,'config.json'),(STATE,'state.json')):
        destination = STATE_ARCHIVE/f'{label}_{name}'
        if destination.exists():
            raise RuntimeError('refuse to replace phase provenance')
        shutil.copyfile(source,destination)
    git('bundle','create',str(STATE_ARCHIVE/f'{label}.bundle'),'HEAD','^'+base)


def stage(config, path, stage_mode, state, source, label, *, audited=None):
    out = Path(config['runtime']['output_root'])
    durable = Path(config['artifacts']['durable_root'])/'receipts'
    durable.mkdir(parents=True, exist_ok=True)
    entry = 'evaluate_tokenrig_visual12.py' if label == VIS else 'train_anchor_full.py'
    entry = 'model_training/rigweave/scripts/'+entry
    op = {'label':config['family'], 'experiment':label, 'complete':False,
        'explicit_human_authorized':True, 'code_commit':source, 'required_entrypoint':entry,
        'operation_entrypoints':{'preflight':entry,'train':entry,'matched_eval':entry},
        'formal_training_allowed':stage_mode=='train', 'gpu_work_allowed':stage_mode!='audit',
        'CPU_audit_accepted':bool(audited),'capacity_accepted':stage_mode=='train',
        'automatic_retry_allowed':False, 'working_model_change_allowed':False,
        'status':'exact_native_'+stage_mode+'_admitted'}
    if label == VIS:
        op['optimizer_updates_allowed'] = 0
    else:
        op['arm'] = config['native_frame_control']['arm']
        op['capacity_basis'] = 'unchanged native training resources; per-step loss and 78 GiB runtime guards'
    if audited:
        op['CPU_audit'] = audited
    write(path, config)
    op['runs'] = [{'mode':stage_mode,'config':str(path),'config_sha256':digest(path),
        'output':str(out/stage_mode),'approved':True}]
    state.update(active_operation=op,state_id=label+'_'+path.stem+'_'+stage_mode,updated_at=now(),
        allowed_operations=['inspect','report','preflight']+(['train'] if stage_mode=='train' else []),
        blocked_operations=['submit','matched_eval']+([] if stage_mode=='train' else ['train']),
        next_required_result='Run only the exact native phase and verify its results before proceeding.')
    write(STATE,state)
    head = commit('Admit '+path.stem+' '+stage_mode,path,STATE)
    snapshot(path.stem+'_'+stage_mode,source,path)
    write(STATUS,{'complete':False,'phase':label,'arm':op.get('arm'),'mode':stage_mode,
        'source':head,'config':str(path),'config_sha256':digest(path),'started_utc':now(),
        'output':str(out/stage_mode)})
    subprocess.run([PY,'-B',str(CODE/'model_training/tools/agent_work_guard.py'),'begin'],
        cwd=CODE,check=True,stdout=subprocess.DEVNULL)
    log = durable/f'{stage_mode}.log'
    try:
        with log.open('xb') as handle:
            subprocess.run([PY,'-B',str(CODE/'model_training/jobs/launch_virtual_pair.py'),
                '--mode',stage_mode,'--config',str(path),'--output',str(out/stage_mode)],
                cwd=CODE,check=True,stdout=handle,stderr=subprocess.STDOUT)
    finally:
        resources = out/f'{stage_mode}_resources.json'
        if resources.is_file():
            shutil.copyfile(resources,durable/resources.name)
    result_path = out/stage_mode/'result.json'
    result, receipt = read(result_path),read(resources)
    if (not result['complete'] or not result['passed'] or not receipt['complete'] or not receipt['passed']
            or receipt['exit_code'] != 0 or not receipt['kernel_scheduler_membership']
            or not receipt['payload_cleanup_complete'] or receipt['source'] != head
            or result['source'] != head or receipt['allocation'] != 131942123):
        raise RuntimeError('native stage did not pass and clean up: '+str(out/stage_mode))
    shutil.copyfile(result_path,durable/f'{stage_mode}_result.json')
    return result, {'path':str(result_path),'sha256':digest(result_path),
        'resources_path':str(resources),'resources_sha256':digest(resources)}


def visual_config(training, screening, report):
    original = read(EXP/'tokenrig_native_visual12_20261107.json')
    original.pop('development_panels')
    original.pop('matrix_scopes')
    return {**original,'execution':{'phase':VIS},
        'runtime':{**original['runtime'],**{k:training['runtime'][k] for k in (
            'allocation','cpu_affinity','physical_gpus','gpu_uuids','stop_before_utc')},
            'repo':str(CODE),'output_root':str(ROOT/'outputs_native_cont400k_visual12_20261107')},
        'candidates':report['selected_candidates'],
        'screening':{'path':str(screening),'sha256':digest(screening)},
        'retained_native_baselines':{'path':str(HOME/'native_visual12_20261107/result.json'),
            'sha256':'929288030609ae9ccaad8d5ac36c4c36226438355d3c32f38898481cd12653ec'},
        'audit_result':None,'artifacts':{'durable_root':str(DUR/'visual12')}}


def run_control(arm, training, state, source, full_audit):
    config = recipe(arm)
    path = EXP/f'{CONTROL}_{arm}.json'
    out = ROOT/f'outputs_native_frame_control_20261108/{arm}'
    durable = HOME/f'native_frame_control_20261108/{arm}'
    config['runtime'] = {**training['runtime'],'repo':str(CODE),'output_root':str(out)}
    config['execution'] = {'phase':CONTROL,'CPU_audit':None}
    config['artifacts']['durable_root'] = str(durable)
    audit, receipt = stage(config,path,'audit',state,source,CONTROL)
    if (audit['cuda_initialized'] or audit['train_assets'] != 5000
            or audit['plan']['assets'] != 50000 or audit['plan']['updates'] != 1045
            or audit['plan']['complete_passes'] != 10):
        raise RuntimeError('5k/50k CPU admission differs')
    if full_audit is not None and any(audit[k] != full_audit[k] for k in ('plan','initialization','cache')):
        raise RuntimeError('frame-control initialization, sample plan or cache changed across arms')
    same_native_capacity(config)
    config['execution']['CPU_audit'] = receipt
    result, receipt = stage(config,path,'train',state,source,CONTROL,audited=receipt)
    if (result['samples'] != 50000 or result['optimizer_updates'] != 1045
            or result['input_frame_visits'] != audit['plan']['input_frame_visits']):
        raise RuntimeError('5k/50k training ended before the fixed budget')
    for rank in (0,1):
        rank_result = read(out/f'train/result_rank_{rank}.json')
        if any(rank_result[k] != result[k] for k in ('complete','passed','samples',
                'optimizer_updates','input_frame_visits','initialization')):
            raise RuntimeError('two 5k ranks did not finish together')
        shutil.copyfile(out/f'train/updates_rank_{rank}.jsonl',durable/f'receipts/updates_rank_{rank}.jsonl')
    checkpoint = result['last_checkpoint']
    if (not checkpoint['both_copies_verified'] or not checkpoint['optimizer_and_rank_RNG_retained']
            or checkpoint['samples'] != 50000):
        raise RuntimeError('5k endpoint full recovery did not pass')
    panel = durable/'evaluation_050000'
    summary = read(panel/'summary.json')
    if not summary['complete'] or summary['stage1']['views'] != 1024:
        raise RuntimeError('5k endpoint matched validation is incomplete')
    state.setdefault('native_frame_control_completed',{})[arm] = {**receipt,
        'checkpoint':checkpoint,'summary':str(panel/'summary.json'),'summary_sha256':digest(panel/'summary.json')}
    return audit, panel


def main():
    require_job_membership(Path('/proc/self/cgroup').read_text(),131942123)
    if STATUS.exists() or git('status','--porcelain'):
        raise RuntimeError('refuse to overwrite a prior followup or dirty source')
    base = git('rev-parse','HEAD')
    state = read(STATE)
    training = read(EXP/(CONT+'.json'))
    write(STATUS,{'complete':False,'phase':'verify_completed_continuation','started_utc':now()})
    subprocess.run([PY,'-B',str(CODE/'server_ops/hgc/review_tokenrig_native_cont400k_20261107.py')],
        check=True,cwd=CODE)
    screen_path = DUR/'screening.json'
    screening = read(screen_path)
    state['last_completed_native_cont400k'] = {**state['active_operation'],'complete':True,
        'formal_training_allowed':False,'gpu_work_allowed':False,'status':'training_and_development_complete',
        'screening':str(screen_path),'screening_sha256':digest(screen_path)}
    visual = visual_config(training,screen_path,screening)
    path = EXP/(VIS+'.json')
    _, receipt = stage(visual,path,'audit',state,base,VIS)
    visual['audit_result'] = receipt
    _, receipt = stage(visual,path,'capacity',state,base,VIS,audited=receipt)
    subprocess.run([PY,'-B',str(CODE/'server_ops/hgc/review_tokenrig_visual12_20261107.py'),'--continuation'],
        check=True,cwd=CODE)
    state['last_completed_native_cont400k_visual12'] = {**state['active_operation'],'complete':True,
        'gpu_work_allowed':False,'quality_accepted':False,'working_model_unchanged':True,**receipt}
    full_audit, full_panel = run_control('full',training,state,base,None)
    _, local_panel = run_control('frame_local',training,state,base,full_audit)
    from tokenrig_comparison_matrix import matrix
    result = matrix({'development_panels':{'full':str(full_panel),'frame_local':str(local_panel)},
        'matrix_scopes':{'matched_5k_50k':['full','frame_local']}})
    if result['strictly_paired_scopes']['matched_5k_50k']['assets'] != 128:
        raise RuntimeError('matched frame-control panels differ')
    comparison = HOME/'native_frame_control_20261108/comparison.json'
    write(comparison,result)
    state['active_operation'].update(complete=True,formal_training_allowed=False,gpu_work_allowed=False,
        status='matched_frame_control_complete',comparison=str(comparison),comparison_sha256=digest(comparison))
    state.update(state_id='tokenrig_native_frame_control_completed_20261108',updated_at=now(),
        allowed_operations=['inspect','report'],blocked_operations=['submit','train','preflight','matched_eval'],
        next_required_result='Report matched 5k/50k and 12-case results; no new training is admitted.')
    write(STATE,state)
    commit('Record completed native 40w and 5k frame controls',STATE)
    snapshot('completed',base,EXP/f'{CONTROL}_frame_local.json')
    write(STATUS,{'complete':True,'passed':True,'finished_utc':now(),'comparison':str(comparison),
        'screening':str(screen_path),'visual_review':str(DUR/'visual12/comparison_review.json')})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if STATUS.exists() and not read(STATUS).get('complete'):
            write(STATUS,{**read(STATUS),'complete':True,'passed':False,'failed_utc':now(),'error':repr(exc)})
        raise
