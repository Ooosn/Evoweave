"""Fixed final-30k case diagnosis, separate from the rejected shortlist."""
import copy
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

from reference_flow_protocol import sha256

RUN = '/work/wangyy/virtual_query_matched_5k_20261025'
SPEC = {
    'candidate_additional_visits': 30000,
    'posthoc_case_diagnostic': True,
    'promote_model': False,
    'previous_full105_replay_unresolved': True,
    'source_training': {
        'result': {'path': RUN+'/outputs/train_v30/result.json',
            'sha256': 'a38d2fa9114c9d49e3334aefef4356323f70f27d4893be0d09243e4ef8b77e86'},
        'resources': {'path': RUN+'/outputs/train_v30_resources.json',
            'sha256': 'd5e059981376c9c13215619a408f5901a20a499f7aaae68442a79597b419170b'}},
    'baseline': {
        'result': {'path': RUN+'/outputs/capacity_v28/result.json',
            'sha256': 'fa2268f03518276c7aab5a5b545b977b99c9d71a297e721f3407b02bb12dad3a'},
        'resources': {'path': RUN+'/outputs/capacity_v28_resources.json',
            'sha256': '08d32ea7fc2f26ad056a46a71eded720f8ab6a56a2fd6e6b78da0a23cb75a748'}}}

FINAL400K = copy.deepcopy(SPEC)
FINAL400K.update(kind='hardcase400k_fixed12_20261102',candidate_additional_visits=100000,
    candidate_total_visits=400000,source_training={
        'result':{'path':RUN+'/outputs/train_v32/result.json','sha256':'0d46a418784d16123ee128a587e432dcfe2980c8bcaba6d7060dd375689460f8'},
        'resources':{'path':RUN+'/outputs/train_v32_resources.json','sha256':'5ca1d4c23d6f9100120fb212ee507cdf053306f164110497bf2429ed82c465dc'}},
    previous_adaptation={
        'result':{'path':RUN+'/outputs/capacity_v31/result.json','sha256':'964cba9ad856de3f1c8f8e4c9e050302f5e9e3d045adaa9c978da098224441fa'},
        'resources':{'path':RUN+'/outputs/capacity_v31_resources.json','sha256':'388895ab2fbbe562d78e5f21ce7351614ba3c74e09e2973dbf53edc8bac50735'}})


def is_final400k(config):
    return config.get('hardcase_visual',{}).get('kind')=='hardcase400k_fixed12_20261102'


def validate_visual(config):
    final=is_final400k(config)
    if config['hardcase_visual'] != (FINAL400K if final else SPEC) or 'inference_diagnosis' in config:
        raise ValueError('unreviewed hardcase visual comparison or resumed replay diagnosis')
    runtime = config['runtime']
    if (runtime['allocation'] != 131170166 or runtime['keeper_pid'] != 2268188
            or runtime.get('allocation_directory') != RUN+'/allocation_binding_native_20261031'
            or runtime['stop_before_utc'] != ('2026-11-04T11:40:04Z' if final else '2026-11-01T11:40:04Z')):
        raise ValueError('hardcase visual allocation differs')
    if final and (runtime['required_h_rt_seconds']!=345600
            or config['checkpoints']['stage1']['path']!=RUN+'/outputs/train_v32/weights_100000.pt'
            or config['official_smoke'].get('inherited_official_capacity')!=FINAL400K['previous_adaptation']):
        raise ValueError('400k endpoint or inherited native beam capacity differs')
    if final and config.get('source_deployment')!={'receipt':RUN+'/source_deployments/hardcase400k_visual12_20261102.json'}:
        raise ValueError('fresh400k stage source provenance required')


def verify_evidence(config, records):
    from stage1_official_inference import reference_seed
    from stage1_surface_repair_runtime import canonical_identity
    validate_visual(config)
    results = {}
    final=is_final400k(config)
    for label in (('source_training','baseline','previous_adaptation') if final else ('source_training','baseline')):
        values = {}
        for name, spec in config['hardcase_visual'][label].items():
            if sha256(spec['path']) != spec['sha256']:
                raise ValueError('retained hardcase visual evidence changed')
            values[name] = json.loads(Path(spec['path']).read_text())
        result, resource = values['result'], values['resources']
        if (not result['complete'] or not result['passed'] or not resource['complete']
                or not resource['passed'] or resource['exit_code'] != 0
                or not resource['payload_cleanup_complete']):
            raise ValueError('source inference/training did not complete cleanly')
        results[label] = result
    training, baseline = results['source_training'], results['baseline']
    if training['samples'] != (100000 if final else 30000) or training['optimizer_updates'] != (2091 if final else 628) or baseline['optimizer_updates'] != 0:
        raise ValueError('wrong final endpoint or nonzero baseline updates')
    policy = baseline['policy']
    if any(policy[k] != config['official_smoke'][k] for k in ('case_indices', 'generation_kwargs', 'base_seed')):
        raise ValueError('baseline official policy differs')
    references = {}
    for artifact in baseline['artifacts']:
        spec = artifact['references']
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('baseline reference artifact changed')
        references.update(json.loads(Path(spec['path']).read_text()))
    if set(references) != {r['asset_id'] for r in records}:
        raise ValueError('baseline fixed cases differ')
    for record in records:
        row = references[record['asset_id']]
        if any(row.get(k) != value for k, value in canonical_identity(record, 0).items()):
            raise ValueError('baseline canonical mesh/GT/frame/normalization differs')
        if row['seed'] != reference_seed(record['asset_id'], record['raw_frames'][0], policy['base_seed']):
            raise ValueError('baseline random seed differs')
    previous_valid=None
    if final:
        previous=results['previous_adaptation']
        if previous['optimizer_updates']!=0 or previous['checkpoints']['stage1']['sha256']!='abaa87aa7b377f17093850981024a4d63529449f68f082e7c6502ea7a5571018':
            raise ValueError('retained330k arm differs')
        if any(previous['policy'][k]!=policy[k] for k in ('case_indices','generation_kwargs','base_seed')):
            raise ValueError('retained330k official policy differs')
        prior={}
        for artifact in previous['artifacts']:
            spec=artifact['references']
            if sha256(spec['path'])!=spec['sha256']: raise ValueError('retained330k references changed')
            prior.update(json.loads(Path(spec['path']).read_text()))
        if set(prior)!=set(references): raise ValueError('retained330k cases differ')
        for record in records:
            row=prior[record['asset_id']]
            if any(row.get(k)!=v for k,v in canonical_identity(record,0).items()) or row['seed']!=references[record['asset_id']]['seed']:
                raise ValueError('retained330k canonical identity or seed differs')
        previous_valid=sum(r['success'] for r in prior.values())
    return {'assets': len(records), 'references_valid': sum(r['success'] for r in references.values()),
        'previous330k_references_valid':previous_valid,'previous330k_recomputed':False,
        'same_inputs_and_official_seed': True, 'baseline_recomputed': False,
        'full105_replay_unresolved': True, 'model_promotion_allowed': False}


def inherited_final400k_capacity(config):
    validate_visual(config)
    if not is_final400k(config): raise ValueError('400k-only inheritance')
    deployment=verify_stage_deployment(config)
    spec=FINAL400K['previous_adaptation']
    for item in spec.values():
        if sha256(item['path'])!=item['sha256']: raise ValueError('native15beam evidence changed')
    result,resources=(json.loads(Path(spec[k]['path']).read_text()) for k in ('result','resources'))
    if (not result['complete'] or not result['passed'] or result['optimizer_updates']!=0
            or not resources['complete'] or not resources['passed'] or resources['exit_code']!=0
            or not resources['payload_cleanup_complete'] or not resources['kernel_scheduler_membership']
            or resources['allocation']!=131170166
            or resources['physical_gpus']!=[4] or resources['execution_cpu_budget']!=8
            or result['policy']['generation_kwargs']!=config['official_smoke']['generation_kwargs']):
        raise ValueError('native15beam source was not accepted')
    from predicted_reference_protocol import ROOT
    current=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    paths=['model_training/rigweave/src','model_training/rigweave/scripts/ar_only_runtime.py',
        'model_training/rigweave/scripts/stage1_official_inference.py','model_training/rigweave/scripts/stage1_feedback.py',
        'model_training/rigweave/scripts/predicted_reference_runtime.py','model_training/rigweave/scripts/virtual_query_xyz_runtime.py']
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet',result['source'],current,'--',*paths],check=True)
    profiles={}
    for artifact in result['artifacts']:
        p=artifact['beam_capacity']
        if not p['passed'] or (p['new_tokens'],p['num_beams'],p['input_frames'])!=(2048,15,8):
            raise ValueError('native15beam maximum decode proof differs')
        profiles[artifact['rank']]=dict(p,inherited_from=spec['result'],inherited_across_same_architecture_checkpoint=True,
            current_checkpoint_sha256=config['checkpoints']['stage1']['sha256'],stress_repeated=False,
            source_deployment=deployment)
    if set(profiles)!={0}: raise ValueError('single-native-GPU4 proof required')
    return profiles


def verify_stage_deployment(config):
    from predicted_reference_protocol import ROOT
    path=Path(config['source_deployment']['receipt']); receipt=json.loads(path.read_text())
    if not (receipt['complete'] and receipt['new_worktree_clean'] and receipt['SSD_verified']
            and receipt['training_outputs_left_in_place'] and receipt['old_files_restored']==0
            and receipt['native_keeper_preserved'] and receipt['job_id']==131170166
            and Path(receipt['canonical_path']).resolve()==ROOT.resolve()):
        raise ValueError('stage deployment/preservation not verified')
    expires=datetime.fromisoformat(receipt['valid_until_before_UTC'])
    stop=datetime.fromisoformat(config['runtime']['stop_before_utc'].replace('Z','+00:00'))
    if not datetime.now(timezone.utc)<stop<expires: raise ValueError('fresh source retention window insufficient')
    for name in ('old_archive','source_archive'):
        spec=receipt[name]
        if any(sha256(spec[k])!=spec['sha256'] for k in ('path','durable_path')): raise ValueError('packed source archive changed')
    current=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',receipt['source'],current],check=True)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet',receipt['source'],current,'--',
        'server_ops/hgc','model_training/rigweave','model_training/jobs','model_training/tools'],check=True)
    return {'receipt':str(path),'sha256':sha256(path),'initial_source':receipt['source'],
        'preserved_old_directory':receipt['preserved_old_directory'],'fresh_source_window_verified':True,
        'old_directory_and_durable_archive_preserved':True,'old_files_restored':0}
