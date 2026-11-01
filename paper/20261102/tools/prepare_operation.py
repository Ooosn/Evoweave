"""Build exact inference records without changing the selected model or old evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RUN = '/work/wangyy/virtual_query_matched_5k_20261025'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-audit', type=Path)
    args = parser.parse_args()
    state_path = ROOT/'model_training/state/current.json'
    state = json.loads(state_path.read_text())
    if args.record_audit:
        result = json.loads(args.record_audit.read_text())
        if not result['passed'] or not result['complete'] or result['optimizer_updates'] != 0:
            raise ValueError('paper CPU audit did not pass')
        audit_config = ROOT/'model_training/experiments/paper_benchmark_20261102_audit.json'
        gpu = json.loads(audit_config.read_text())
        gpu['audit_result'] = {'path': RUN+'/outputs/audit_v36/result.json', 'sha256':sha(args.record_audit)}
        gpu_path = ROOT/'model_training/experiments/paper_benchmark_20261102.json'
        write(gpu_path,gpu)
        op = state['active_operation']
        if op['label'] != 'paper_benchmark_20261102_v1':
            raise ValueError('unexpected active operation')
        op['CPU_audit_accepted'] = True
        op['gpu_work_allowed'] = True
        op['runs'][0].update(approved=False,complete=True,status='passed',result=gpu['audit_result'])
        op['runs'].append({'stage':'capacity_v36','mode':'capacity','approved':True,'complete':False,
            'config':RUN+'/code/model_training/experiments/'+gpu_path.name,
            'config_sha256':sha(gpu_path),'output':RUN+'/outputs/capacity_v36'})
        state['state_id']='paper_benchmark_CPU_accepted_20261102'
    else:
        archive = ROOT/'model_training/state/archive/before_paper_draft_20261102.json'
        if archive.exists():
            raise ValueError('paper preparation already exists; inspect before modifying it')
        write(archive,state)
        prior_path=ROOT/'model_training/experiments/hardcase390k_visual12_20261102.json'
        prior=json.loads(prior_path.read_text())
        parent=ROOT/'model_training/experiments/anchor_full_20261030_v1.json'
        backbone=json.loads((ROOT/'model_training/experiments/anchor_full_backbone_20261030_v1.json').read_text())
        working=json.loads((ROOT/'model_training/state/working_model.json').read_text())
        selected=state['current_working_model']['checkpoint']
        config={'schema_version':1,'family':'paper_benchmark_20261102_v1','runtime':prior['runtime'],
            'parent_config':{'relative_path':str(parent.relative_to(ROOT)).replace('\\','/'),'sha256':sha(parent)},
            'full_model_inputs':prior['full_model_inputs'],
            'working_model':{'path':selected['path'],'sha256':selected['sha256'],'layout':'anchor'},
            'baseline_sources':{
                'unirig_official':{'path':'/home/wangyy/evorig/model_assets/UniRig','commit':'6793c6640ff01c8fb389f3993434124bb43d2933'},
                'puppeteer_official':{'path':'/work/wangyy/paper_baselines_20261102/sources/Puppeteer','commit':'1c0f9fc6ad209667a0ec5ceac9b59964938a8b51'}},
            'baseline_weights':{
                'unirig_official':{'path':backbone['initialization']['checkpoint'],'sha256':backbone['initialization']['sha256']},
                'puppeteer_official':{'path':'/home/wangyy/evorig/model_assets/Puppeteer/skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth',
                    'sha256':'2af9a75294b34adde314e20a3142e025925377f31637d232456fd09acb714b43',
                    'official_repository':'Seed3D/Puppeteer','official_revision':'d83b1b73cd2e1507968d3eade2d6314082c82763'}},
            'evaluation':{'assets':24,'input_frames':8,'reference_slot':0,'selection_seed':20261102,
                'optimizer_updates':0,'payload_limit_seconds':10800,
                'methods':['evoweave330k','unirig_official','puppeteer_official'],
                'retry_failed_generation':False,'use_GT_reference':False,
                'case_selection':'hash_order_excluding_development128'},'audit_result':None}
        config['runtime']['paper_shared_inference']={'explicit_human_authorized':True,
            'scope':config['family'],'own_memory_fraction':.30,'minimum_free_mib':32768,
            'preserve_existing_processes':True}
        config_path=ROOT/'model_training/experiments/paper_benchmark_20261102_audit.json'
        write(config_path,config)
        code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        state.update(state_id='paper_benchmark_CPU_preparation_20261102',
            allowed_operations=['inspect','report','preflight'],
            blocked_operations=['train','submit','matched_eval'])
        state['active_operation']={'label':config['family'],'complete':False,
            'explicit_human_authorized':True,'authorization':'User2026-11-02 authorizes autonomous paper writing, baseline deployments, and bounded H100 inference, including sharing with other training without stopping it. No further training or checkpoint search.',
            'required_entrypoint':'model_training/rigweave/scripts/evaluate_paper_benchmark.py',
            'operation_entrypoints':{'preflight':'model_training/rigweave/scripts/evaluate_paper_benchmark.py'},
            'code_commit':code_commit,'formal_training_allowed':False,'gpu_work_allowed':False,
            'CPU_audit_accepted':False,'capacity_accepted':False,'automatic_retry_allowed':False,
            'working_model_unchanged':True,'optimizer_updates':0,
            'resource_policy':'Reuse native131170166 GPU4 CPU8subset. Preserve keeper, other jobs, all results, and96h expiry. Actual free memory and native membership required. Shared timing is not isolated throughput.',
            'acceptance_test':'Freeze24hash-selected validation assets outside the128development panel before generation. Same8pose canonical inputs, reference0. Greedy all methods. No GT skeleton enters predictors. Report all failures, raw Puppeteer joint-token output, different input pose counts, inherited data overlap, and root-RT metadata. No test-set or production-rig claim.',
            'runs':[{'stage':'audit_v36','mode':'audit','approved':True,'complete':False,
                'config':RUN+'/code/model_training/experiments/'+config_path.name,
                'config_sha256':sha(config_path),'output':RUN+'/outputs/audit_v36'}]}
        state['next_required_result']='Complete paper inference and record its actual outcome, then finalize the manuscript and author evidence report. Keep330kselected, no further training or checkpoint search. Do not reuse historical launch authorizations.'
    state['updated_at']=datetime.now(timezone.utc).isoformat()
    write(state_path,state)
    print(json.dumps({'state_id':state['state_id'],'runs':[(r['stage'],r['approved']) for r in state['active_operation']['runs']]}))


if __name__ == '__main__': main()
