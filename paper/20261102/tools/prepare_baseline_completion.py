"""Record missing baselines while inheriting the completed frozen-cohort work."""
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
    p=argparse.ArgumentParser();p.add_argument('--record-audit',type=Path);args=p.parse_args()
    state_path=ROOT/'model_training/state/current.json';state=read(state_path);op=state['active_operation']
    audit_path=ROOT/'model_training/experiments/paper_baseline_completion_20261102_audit.json'
    gpu_path=ROOT/'model_training/experiments/paper_baseline_completion_20261102.json'
    if args.record_audit:
        result=read(args.record_audit)
        if not result['complete'] or not result['passed'] or not result['prepared_inputs_inherited_without_resampling']:
            raise ValueError('metadata-only inheritance audit did not pass')
        config=read(audit_path)
        config['audit_result']={'path':RUN+'/outputs/audit_v38/result.json','sha256':sha(args.record_audit),
            'resources_path':RUN+'/outputs/audit_v38_resources.json'}
        write(gpu_path,config)
        next(r for r in op['runs'] if r['stage']=='audit_v38').update(complete=True,approved=False,status='passed')
        op['CPU_audit_accepted']=True;op['gpu_work_allowed']=True
        op['runs'].append({'stage':'capacity_v38','mode':'capacity','approved':True,'complete':False,
            'config':RUN+'/code/model_training/experiments/'+gpu_path.name,
            'config_sha256':sha(gpu_path),'output':RUN+'/outputs/capacity_v38'})
        state['state_id']='paper_remaining_baselines_CPU_accepted_20261102'
    else:
        if audit_path.exists():raise ValueError('completion preparation exists; inspect it')
        resource=read(EVIDENCE/'capacity_v36_resources.json')
        if not resource['complete'] or not resource['payload_cleanup_complete'] or resource['exit_code']!=1:
            raise ValueError('original payload not safely finished')
        if "unexpected keyword argument 'method'" not in (EVIDENCE/'capacity_v36/unirig_official.log').read_text():
            raise ValueError('unexpected original baseline failure')
        config=read(ROOT/'model_training/experiments/paper_benchmark_20261102.json')
        retained={name:{'path':RUN+'/outputs/capacity_v36/'+name,'sha256':sha(EVIDENCE/'capacity_v36'/name)}
            for name in ('evoweave330k_rows.json','dynamic_rows.json')}
        config['runtime']['paper_completed_evoweave']={'source':resource['source'],
            'resources_path':RUN+'/outputs/capacity_v36_resources.json',
            'resources_sha256':sha(EVIDENCE/'capacity_v36_resources.json'),'files':retained}
        config['release_extension']='skintokens_skeleton_prefix_20261102_v1'
        config['evaluation']['methods'].append('skintokens_official')
        config['baseline_sources']['skintokens_official']={'path':'/work/wangyy/paper_baselines_20261102/sources/SkinTokens',
            'commit':'273b691d35989d71cd17ff2895fdc735097b92d1'}
        root='/work/wangyy/paper_baselines_20261102/weights/SkinTokens'
        config['baseline_weights'].update({
            'skintokens_official':{'path':root+'/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt',
                'sha256':'f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692'},
            'skintokens_vae':{'path':root+'/experiments/skin_vae_2_10_32768/last.ckpt',
                'sha256':'4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4'}})
        config['runtime']['paper_skin_qwen_config']=root+'/Qwen3Config'
        config['runtime']['paper_skin_pythonpath']=':'.join([RUN+'/code/model_training/rigweave/src',
            RUN+'/code/model_training/rigweave/scripts','/work/wangyy/paper_baselines_20261102/runtime_skin',
            '/work/wangyy/paper_baselines_20261102/runtime_unirig','/home/wangyy/evorig/runtime/puppeteer_py310_hgc'])
        config['inherit_prepared_inputs']=config['audit_result'];config['audit_result']=None
        write(audit_path,config)
        op.update(code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            CPU_audit_accepted=False,gpu_work_allowed=False)
        next(r for r in op['runs'] if r['stage']=='capacity_v36').update(complete=True,approved=False,
            status='partial_completed_evoweave_baseline_adapter_failure',
            reason='UniRig tokenizer factory requires TokenizerConfig, not keyword expansion. It failed before any baseline generation. Retain all24completed Evoweave references and336target rows; no generation retries.')
        op['runs'].append({'stage':'audit_v38','mode':'audit','approved':True,'complete':False,
            'config':RUN+'/code/model_training/experiments/'+audit_path.name,
            'config_sha256':sha(audit_path),'output':RUN+'/outputs/audit_v38'})
        op['acceptance_test']='Finish only UniRig, Puppeteer and releasedSkinTokens skeleton-prefix predictions on the existing24fixed canonical inputs. Metadata audit inherits existing prepared cache without resampling; completed Evoweave predictions must never regenerate. No GT count, retries, training, checkpoint search or other-task changes.'
        state['state_id']='paper_remaining_baselines_metadata_preparation_20261102'
    write(state_path,state)
    print(state['state_id'])


if __name__=='__main__':main()
