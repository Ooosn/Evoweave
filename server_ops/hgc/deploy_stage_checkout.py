"""Preserve the old scratch checkout and deploy a fresh fixed-commit SSD stage."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib,json,os,re,shutil,subprocess,tarfile
from pathlib import Path

RUN=Path('/work/wangyy/virtual_query_matched_5k_20261025')
DURABLE=Path('/home/wangyy/evorig/outputs/hardcase3608_20261101/source_deployments')

def command(*args):
    return subprocess.check_output(list(args),text=True,timeout=180).strip()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def inventory(root):
    rows={}
    for folder,dirs,files in os.walk(root,followlinks=False):
        for name in list(dirs)+files:
            path=Path(folder)/name; relative=path.relative_to(root).as_posix()
            if path.is_symlink(): rows[relative]={'link':os.readlink(path)}
            elif path.is_file(): rows[relative]={'bytes':path.stat().st_size,'sha256':sha(path)}
    return rows

def verify_archive(path,expected,prefix='code/'):
    found={}
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            if member.isdir(): continue
            if not member.name.startswith(prefix): raise ValueError('archive member escaped source prefix')
            relative=member.name[len(prefix):]
            if relative in found or relative not in expected: raise ValueError('unexpected archive member')
            if member.issym(): found[relative]={'link':member.linkname}
            elif member.isfile() or member.islnk():
                h=hashlib.sha256(); size=0
                with archive.extractfile(member) as handle:
                    for block in iter(lambda:handle.read(8*1024**2),b''): h.update(block); size+=len(block)
                found[relative]={'bytes':size,'sha256':h.hexdigest()}
            else: raise ValueError('unsupported source archive member')
    if found!=expected: raise ValueError('source archive content mismatch')

def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw: raise RuntimeError('refuse to overwrite deployment evidence')
    else:
        with path.open('xb') as handle: handle.write(raw)

def snapshot(shared_inference=False):
    repo=RUN/'code'; keeper=Path('/proc/2268188')
    if repo.is_symlink() or not (repo/'.git').is_dir(): raise RuntimeError('expected a standalone canonical Git checkout')
    groups=(keeper/'cgroup').read_text()
    if not all(x in groups for x in ('cpuset:/AGE/131170166.1/master','devices:/AGE/131170166.1/master')): raise RuntimeError('native keeper changed')
    if (keeper/'task/2268188/children').read_text().strip(): raise RuntimeError('payload active; source replacement prohibited')
    grant=command('qstat','-j','131170166')
    if 'h100=2,h_rt=345600,mem_req=4G' not in grant or 'pcg01i:16' not in grant: raise RuntimeError('retained96h grant differs')
    apps=command('nvidia-smi','-i','3,4','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    shared=None
    if shared_inference:
        import sys
        operator=Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(operator/'model_training/rigweave/scripts'))
        from paper_benchmark_protocol import shared_gpu_admission,SCOPE
        authorization=json.loads((operator/'model_training/state/current.json').read_text())['active_operation']
        if authorization.get('label')!=SCOPE or authorization.get('explicit_human_authorized') is not True or authorization.get('formal_training_allowed'):
            raise RuntimeError('shared deployment lacks exact inference-only authorization')
        config=json.loads((operator/'model_training/experiments/paper_benchmark_20261102_audit.json').read_text())
        shared=shared_gpu_admission(config)
    elif apps: raise RuntimeError('GPU3/4occupied; nothing stopped')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: cwd=(proc/'cwd').resolve(strict=True)
        except (PermissionError,FileNotFoundError,ProcessLookupError): continue
        if cwd==repo or repo in cwd.parents: raise RuntimeError('process still uses source checkout: '+proc.name)
    source=command('findmnt','-T',str(repo),'-n','-o','SOURCE')
    if set(command('lsblk','-s','-n','-r','-o','ROTA',source).split())!={'0'}: raise RuntimeError('target SSD media not verified')
    state=json.loads((repo/'model_training/state/current.json').read_text())
    if not state['active_operation']['complete']: raise RuntimeError('old operation not completed')
    stop=datetime.fromisoformat(state['compute_allocation']['stop_before_utc'].replace('Z','+00:00'))
    if not datetime.now(timezone.utc)+timedelta(hours=1)<stop<datetime.now(timezone.utc)+timedelta(days=5): raise RuntimeError('time window exceeds fresh-source retention')
    return {'observed_utc':datetime.now(timezone.utc).isoformat(),'head':command('git','-C',str(repo),'rev-parse','HEAD'),
        'status':subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).splitlines(),
        'keeper_code_sha256':sha(repo/'model_training/jobs/hold_virtual_pair_allocation.py'),
        'mount_source':source,'SSD_verified':True,'keeper_pid':2268188,'job_id':131170166,
        'keeper_affinity':sorted(os.sched_getaffinity(2268188)),
        'shared_inference':shared_inference,'shared_GPU_admission':shared,'existing_GPU_processes_not_modified':apps}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('plan','execute')); parser.add_argument('--source',required=True)
    parser.add_argument('--deployment-id',required=True)
    parser.add_argument('--shared-inference',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.source): raise ValueError('exact commit required')
    if not re.fullmatch('[a-z][a-z0-9_]{2,80}',args.deployment_id): raise ValueError('safe unique deployment id required')
    os.sched_setaffinity(0,{6,7,9,10}); os.environ['GIT_TERMINAL_PROMPT']='0'
    evidence=RUN/'source_deployments'; plan_path=evidence/(args.deployment_id+'.plan.json'); receipt_path=evidence/(args.deployment_id+'.json')
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['complete'] and command('git','-C',str(RUN/'code'),'rev-parse','HEAD')==args.source:
            print(json.dumps(receipt)); return
        raise RuntimeError('deployment already recorded; inspect actual progress')
    if args.shared_inference and command('git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD')!=args.source:
        raise RuntimeError('shared operator must itself be the exact target source')
    live=snapshot(args.shared_inference)
    if args.action=='plan':
        write_new(plan_path,{'complete':True,'source':args.source,'live':live,'tool_sha256':sha(__file__)})
        print(json.dumps({'planned_source':args.source,'live':live},indent=2)); return
    plan=json.loads(plan_path.read_text()); old=RUN/'code'
    if plan['source']!=args.source or plan['tool_sha256']!=sha(__file__) or plan['live'].get('shared_inference',False)!=args.shared_inference: raise RuntimeError('deployment plan changed')
    if (datetime.now(timezone.utc)-datetime.fromisoformat(plan['live']['observed_utc'])).total_seconds()>1200: raise RuntimeError('deployment plan stale')
    if any(live[k]!=plan['live'][k] for k in ('head','status','keeper_code_sha256','mount_source','keeper_affinity')): raise RuntimeError('old checkout/resources changed')
    staged=RUN/('code_staged_'+args.source[:12]); preserved=RUN/('code_preserved_'+live['head'][:12]+'_before_'+args.source[:12])
    if staged.exists() or preserved.exists(): raise RuntimeError('deployment paths already exist; inspect before continuing')
    DURABLE.mkdir(parents=True,exist_ok=True); os.chmod(DURABLE,0o700); evidence.mkdir(exist_ok=True)
    if command('git','-C',str(old),'remote','get-url','origin')!='https://github.com/Ooosn/Evoweave.git': raise RuntimeError('unexpected Git origin')
    created=datetime.now(timezone.utc)
    subprocess.run(['git','clone','--single-branch','--branch','codex/virtual-fusion-20261013','https://github.com/Ooosn/Evoweave.git',str(staged)],check=True,timeout=180,stdout=subprocess.DEVNULL)
    if command('git','-C',str(staged),'rev-parse','HEAD')!=args.source or command('git','-C',str(staged),'status','--porcelain'): raise RuntimeError('new source is not exact and clean')
    if sha(staged/'server_ops/hgc/deploy_stage_checkout.py')!=sha(__file__): raise RuntimeError('deployment tool is not from pinned source')
    if sha(staged/'model_training/jobs/hold_virtual_pair_allocation.py')!=live['keeper_code_sha256']: raise RuntimeError('keeper implementation changed')
    subprocess.run(['git','-C',str(staged),'fsck','--no-dangling'],check=True,timeout=120,stdout=subprocess.DEVNULL)
    before=inventory(old); archive=evidence/('old_checkout_before_'+args.source[:12]+'.tar.gz')
    with tarfile.open(archive,'x:gz',dereference=False) as handle: handle.add(old,arcname='code')
    verify_archive(archive,before)
    if inventory(old)!=before: raise RuntimeError('old checkout changed during preservation')
    durable_old=DURABLE/archive.name
    if durable_old.exists(): raise RuntimeError('durable snapshot already exists')
    shutil.copyfile(archive,durable_old); os.chmod(durable_old,0o600)
    if sha(archive)!=sha(durable_old): raise RuntimeError('old durable archive differs')
    source_archive=evidence/('source_'+args.source+'.tar.gz')
    if source_archive.exists() or (DURABLE/source_archive.name).exists(): raise RuntimeError('new source archive already exists')
    subprocess.run(['git','-C',str(staged),'archive','--format=tar.gz','--prefix=source/','-o',str(source_archive),args.source],check=True)
    with tarfile.open(source_archive,'r:gz') as handle:
        if not any(x.name=='source/AGENTS.md' for x in handle): raise RuntimeError('new source archive incomplete')
    durable_source=DURABLE/source_archive.name; shutil.copyfile(source_archive,durable_source); os.chmod(durable_source,0o600)
    if sha(source_archive)!=sha(durable_source): raise RuntimeError('new durable source archive differs')
    if old.stat().st_dev!=staged.stat().st_dev or old.parent.resolve()!=RUN.resolve(): raise RuntimeError('source replacement escaped same SSD')
    latest=snapshot(args.shared_inference)
    if any(latest[k]!=live[k] for k in ('head','status','keeper_code_sha256','keeper_affinity')): raise RuntimeError('source/resources changed before cutover')
    os.rename(old,preserved)
    os.rename(staged,old)
    if inventory(preserved)!=before or command('git','-C',str(old),'status','--porcelain'): raise RuntimeError('source cutover verification failed')
    receipt={'complete':True,'observed_utc':datetime.now(timezone.utc).isoformat(),'source':args.source,
        'canonical_path':str(old),'preserved_old_directory':str(preserved),'preserved_old_status':live['status'],
        'preserved_old_inventory':before,'old_archive':{'path':str(archive),'durable_path':str(durable_old),'sha256':sha(archive)},
        'source_archive':{'path':str(source_archive),'durable_path':str(durable_source),'sha256':sha(source_archive)},
        'new_worktree_clean':True,'SSD_verified':True,'training_outputs_left_in_place':True,
        'old_files_restored':0,'native_keeper_preserved':Path('/proc/2268188').exists(),'job_id':131170166,
        'deployment_id':args.deployment_id,'source_created_utc':created.isoformat(),
        'valid_until_before_UTC':(created+timedelta(days=5)).isoformat()}
    write_new(receipt_path,receipt); write_new(DURABLE/receipt_path.name,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='preserved_old_inventory'},indent=2))

if __name__=='__main__': main()
