"""Bind the documented adapter correction; retain unaffected completed outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=Path('D:/evoweave/server_ops/outputs/paper_v1_20261102/evidence')
RUN='/work/wangyy/virtual_query_matched_5k_20261025'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-audit',type=Path);args=parser.parse_args()
    state_path=ROOT/'model_training/state/current.json';state=read(state_path);op=state['active_operation']
    audit_path=ROOT/'model_training/experiments/paper_adapter_repair_20261102_audit.json'
    gpu_path=ROOT/'model_training/experiments/paper_adapter_repair_20261102.json'
    if args.record_audit:
        result=read(args.record_audit);resources=read(EVIDENCE/'audit_v39_resources.json')
        if not all((result['complete'],result['passed'],result['prepared_inputs_inherited_without_resampling'],
                    resources['complete'],resources['passed'],resources['payload_cleanup_complete'])):
            raise ValueError('new metadata-only compatibility binding did not complete safely')
        config=read(audit_path)
        config['audit_result']={'path':RUN+'/outputs/audit_v39/result.json','sha256':sha(args.record_audit),
            'resources_path':RUN+'/outputs/audit_v39_resources.json'}
        write(gpu_path,config)
        next(r for r in op['runs'] if r['stage']=='audit_v39').update(complete=True,approved=False,status='passed')
        op['CPU_audit_accepted']=True;op['gpu_work_allowed']=True
        op['runs'].append({'stage':'capacity_v39','mode':'capacity','approved':True,'complete':False,
            'config':RUN+'/code/model_training/experiments/'+gpu_path.name,'config_sha256':sha(gpu_path),
            'output':RUN+'/outputs/capacity_v39'})
        state['state_id']='paper_adapter_correction_CPU_accepted_20261102'
    else:
        if audit_path.exists():raise ValueError('repair preparation exists; inspect actual status')
        resources=read(EVIDENCE/'capacity_v38_resources.json');result=read(EVIDENCE/'capacity_v38/result.json')
        if not resources['complete'] or not resources['payload_cleanup_complete'] or resources['exit_code']!=1:
            raise ValueError('partial native payload has not safely finished')
        if 'query and key must have the same dtype' not in read(EVIDENCE/'capacity_v38/failure_skintokens_official.json')['error']:
            raise ValueError('unexpected SkinTokens failure')
        if result['summaries']['unirig_official']['rows']!=24 or result['summaries']['unirig_official']['failures']:
            raise ValueError('cannot inherit an incomplete UniRig run')
        config=read(ROOT/'model_training/experiments/paper_baseline_completion_20261102.json')
        config['baseline_adapter_revision']='official_puppeteer_unit_cube_skin_eager_bf16_20261102_v2'
        config['runtime']['paper_completed_baselines']={'unirig_official':{
            'rows_path':RUN+'/outputs/capacity_v38/unirig_official_rows.json',
            'rows_sha256':sha(EVIDENCE/'capacity_v38/unirig_official_rows.json'),
            'resources_path':RUN+'/outputs/capacity_v38_resources.json',
            'resources_sha256':sha(EVIDENCE/'capacity_v38_resources.json')}}
        config['inherit_prepared_inputs']=config['audit_result'];config['audit_result']=None
        write(audit_path,config)
        next(r for r in op['runs'] if r['stage']=='capacity_v38').update(complete=True,approved=False,
            status='partial_unirig_retained_puppeteer_invalidated_skin_setup_failed',
            reason='Puppeteer adapter used half the documented input scale; invalidate those scores rather than interpret them. SkinTokens failed before generating any row due to attention dtype mismatch. Preserve all raw evidence; retain unchanged UniRig and Evoweave predictions.')
        op.update(code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            CPU_audit_accepted=False,gpu_work_allowed=False)
        op['runs'].append({'stage':'audit_v39','mode':'audit','approved':True,'complete':False,
            'config':RUN+'/code/model_training/experiments/'+audit_path.name,'config_sha256':sha(audit_path),
            'output':RUN+'/outputs/audit_v39'})
        op['acceptance_test']='Correct only documented Puppeteer input scaling and SkinTokens BF16/eager compatibility. Reuse all fixed input data and all completed Evoweave and UniRig generations. Greedy generation, no GT count or skeleton, no failed-generation retries, no training or checkpoint search. Keep original adapter-invalid and setup-failure evidence.'
        state['state_id']='paper_adapter_correction_metadata_binding_20261102'
    write(state_path,state);print(state['state_id'])


if __name__=='__main__':main()
