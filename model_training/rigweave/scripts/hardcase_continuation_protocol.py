"""Explicit330k-to400k continuation with exact weighted-prefix restoration."""
import copy
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from reference_flow_protocol import ROOT, load_backbone_config, scientific_sha, sha256

RUN = '/work/wangyy/virtual_query_matched_5k_20261025'
PARENT_CONFIG = {'relative_path':'model_training/experiments/hardcase3608_20261101_v1.json',
    'sha256':'d263a58cd2cc586f55fa8e9130adc92b2c920a2de6be67d0e03d4d0df4e36387'}
CHECKPOINT = {'path':RUN+'/outputs/train_v30/checkpoint_030000.pt',
    'sha256':'a62dd25991f948298889a68525ee2b0babb9cceb10cfd435e6cd0668422134f7'}
PARENT_RESULT = {'path':RUN+'/outputs/train_v30/result.json',
    'sha256':'a38d2fa9114c9d49e3334aefef4356323f70f27d4893be0d09243e4ef8b77e86'}
BOUNDARIES = [5000,10000,20000,30000]+list(range(40000,100001,10000))
MILESTONES = list(range(40000,100001,10000))
SOURCE_CHANGES = {'model_training/rigweave/scripts/'+name for name in (
    'anchor_full_protocol.py','anchor_full_runtime.py','hardcase_adaptation_protocol.py',
    'reference_flow_protocol.py')}
SOURCE_CHANGES.add('model_training/tools/hardcase_adaptation_ops.py')
NEW_SOURCE = 'model_training/rigweave/scripts/hardcase_continuation_protocol.py'


def parent_config():
    path = ROOT/PARENT_CONFIG['relative_path']
    if sha256(path) != PARENT_CONFIG['sha256']:
        raise ValueError('frozen30k parent configuration changed')
    return json.loads(path.read_text())


def validate_config(config):
    from hardcase_adaptation_protocol import validate_config as validate_parent
    parent = parent_config()
    validate_parent(parent)
    expected = copy.deepcopy(parent)
    expected['training'].update(additional_assets=100000, checkpoint_assets=BOUNDARIES,
        resume=CHECKPOINT)
    expected['training']['optimization'].update(start_lr=2e-6,peak_lr=2e-6,end_lr=2e-6,
        schedule_samples=100000,div_factor=1)
    for key in ('family','schema_version','backbone','architecture','source_recipe','evaluation','training'):
        if config.get(key) != expected[key]:
            raise ValueError('approved400k continuation changed: '+key)
    if config.get('continuation') != {'kind':'hardcase330k_to400k_v1','parent_config':PARENT_CONFIG,
            'checkpoint':CHECKPOINT,'start_samples':30000,'start_updates':628,
            'start_input_frame_visits':392704,'target_samples':100000,
            'retain_model_Adam_RNG_and_weighted_prefix':True,'no_quality_based_early_stop':True}:
        raise ValueError('unreviewed continuation source or target')
    sampling = copy.deepcopy(config['sampling'])
    for key in ('path','sha256','continuation_target_assets'):
        sampling.pop(key, None)
    original = copy.deepcopy(parent['sampling'])
    original.pop('path'); original.pop('sha256')
    if sampling != original or config['sampling'].get('continuation_target_assets') != 100000:
        raise ValueError('continuation changed3/2/1 or evaluation membership')
    runtime = copy.deepcopy(config['runtime'])
    if runtime.pop('required_h_rt_seconds') != 345600 or runtime.pop('stop_before_utc') != '2026-11-04T11:40:04Z':
        raise ValueError('continuation requires verified96h grant and save margin')
    old_runtime = copy.deepcopy(parent['runtime'])
    old_runtime.pop('required_h_rt_seconds'); old_runtime.pop('stop_before_utc')
    if runtime != old_runtime:
        raise ValueError('continuation changed native GPU/CPU resources or environment')
    retention = {'keep_last_full':7,'full_milestones':MILESTONES,'weight_milestones':MILESTONES,
        'keep_best_stage1':True,'keep_best_stage2':True}
    if config['artifacts']['retention'] != retention or config['data_cache'] != parent['data_cache']:
        raise ValueError('continuation retention or immutable SSD cache changed')
    base = load_backbone_config(config)
    base['optimization'] = copy.deepcopy(config['training']['optimization'])
    base['evaluation']['selection'] = copy.deepcopy(config['sampling']['selection'])
    base['evaluation']['valid_assets'] = 112
    base['hardcase_sampling'] = copy.deepcopy(config['sampling'])
    return base


@lru_cache(maxsize=4)
def sampling(path, expected_sha):
    from hardcase_adaptation_protocol import load_sampling, weighted_order
    if sha256(path) != expected_sha:
        raise ValueError('extended weighted plan changed')
    plan = json.loads(Path(path).read_text())
    parent = load_sampling(parent_config()['sampling'])
    for key in ('train_tiers','train_asset_ids','validation_strata','development_ids','locked_ids','profile_sha256'):
        if plan[key] != parent[key]:
            raise ValueError('extended plan changed immutable membership')
    order = plan['exposure_order']
    if (len(order) != 100000 or order[:30000] != parent['exposure_order']
            or order != weighted_order(plan['train_tiers'], budget=100000)):
        raise ValueError('extended weighted order does not preserve the exact30k prefix')
    return plan


def source_compatible(saved_sources, current_sources):
    if set(current_sources) != set(saved_sources)|{NEW_SOURCE}:
        return False
    return all(current_sources[name] == value for name,value in saved_sources.items() if name not in SOURCE_CHANGES)


def restore_stream(config, saved_state, rank, world, *, from_parent):
    from anchor_full_data import make_stream
    if not from_parent:
        return make_stream(validate_config(config),rank,world,saved_state)
    from hardcase_adaptation_protocol import validate_config as validate_parent
    old = make_stream(validate_parent(parent_config()),rank,world,saved_state)
    new = make_stream(validate_config(config),rank,world)
    if old.cursor != old.local_total or old.cursor*world != 30000:
        raise ValueError('only the fully consumed30k stream can extend')
    if (not np.array_equal(old.order,new.order[:30000])
            or not np.array_equal(old.update_sizes,new.update_sizes[:628])
            or not np.array_equal(old.frame_counts,new.frame_counts[:628])):
        raise ValueError('extension changed consumed assets, frames or update boundaries')
    new.cursor = old.cursor
    return new


def validate_resume(config, saved, names, sources, world):
    parent = parent_config()
    from_parent = saved['scientific_sha256'] == scientific_sha(parent)
    samples = saved['additional_asset_exposures']
    if (saved['optimizer_parameter_names'] != names or len(saved['rank_states']) != world
            or not saved['optimizer']['state'] or saved['global_backbone_asset_exposures'] != 300000+samples
            or saved['initialization'].get('project_optimizer_loaded') is not True):
        raise ValueError('incomplete model/Adam/rank state')
    if from_parent:
        if (samples != 30000 or saved['step'] != 628 or saved['input_frame_exposures'] != 392704
                or saved['configuration'] != dict(parent, _config_sha256=PARENT_CONFIG['sha256'])
                or not source_compatible(saved['source_sha256'],sources)):
            raise ValueError('source is not the exact compatible completed30k run')
    elif (saved['scientific_sha256'] != scientific_sha(config) or saved['source_sha256'] != sources
            or type(samples) is not int or not 30000 < samples <= 100000):
        raise ValueError('checkpoint is not part of this400k continuation')
    for rank,state in enumerate(saved['rank_states']):
        if state['rank'] != rank or state['stream']['cursor']*world != samples:
            raise ValueError('restored rank cursor differs')
        stream = restore_stream(config,state['stream'],rank,world,from_parent=from_parent)
        if (saved['step'] != int(np.searchsorted(stream.global_ends,samples))+1
                or saved['input_frame_exposures'] != int(np.dot(stream.update_sizes[:saved['step']].astype(np.int64),stream.frame_counts[:saved['step']]))):
            raise ValueError('checkpoint counters differ from exact extended frame plan')
    return from_parent


def load_model(config, device, rank, world, resume, *, cpu=False):
    import torch
    import torch.distributed as dist
    from anchor_full_runtime import initialize
    from anchor_full_protocol import source_hashes
    from fusion_evaluation import tree_fingerprint
    from virtual_pair_runtime import parameter_names, apply_rng
    from id_xyz_stages import random_state
    if resume is None:
        raise ValueError('continuation must restore a complete checkpoint')
    if rank == 0 and sha256(resume['path']) != resume['sha256']:
        raise ValueError('continuation checkpoint SHA256 differs')
    if not cpu:
        dist.barrier(device_ids=[device.index])
    model,optimizer,tokenizer,codec,base,_ = initialize(config,device)
    saved = torch.load(resume['path'],map_location='cpu',mmap=True,weights_only=False)
    from_parent = validate_resume(config,saved,parameter_names(model),source_hashes(),world)
    expected_model,expected_optimizer = tree_fingerprint(saved['model']),tree_fingerprint(saved['optimizer'])
    model.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    if tree_fingerprint(model.state_dict()) != expected_model or tree_fingerprint(optimizer.state_dict()) != expected_optimizer:
        raise ValueError('restoring continuation changed model or Adam tensors')
    stream = restore_stream(config,saved['rank_states'][rank]['stream'],rank,world,from_parent=from_parent)
    expected_rng = {k:v for k,v in saved['rank_states'][rank].items() if k not in ('rank','stream')}
    if not cpu:
        apply_rng(saved['rank_states'][rank])
        if tree_fingerprint(random_state()) != tree_fingerprint(expected_rng):
            raise ValueError('restored rank RNG differs from source')
    model.continuation_restore_report = {'complete':True,'model_sha256':expected_model,
        'model_fingerprint_method':'tree_fingerprint_state_dict',
        'restored_checkpoint':copy.deepcopy(resume),'global_model_asset_visits':300000+saved['additional_asset_exposures'],
        'optimizer_sha256':expected_optimizer,'optimizer_state_entries':len(optimizer.state),
        'rank_RNG_sha256':tree_fingerprint(expected_rng),'rank_RNG_restored':not cpu,
        'weighted_prefix_exact':True,'samples':saved['additional_asset_exposures'],
        'step':saved['step'],'sampler':stream.state_dict(),'from_parent':from_parent}
    best = saved.get('best',{})
    if from_parent:
        if sha256(PARENT_RESULT['path']) != PARENT_RESULT['sha256']:
            raise ValueError('completed parent result changed')
        parent_result = json.loads(Path(PARENT_RESULT['path']).read_text())
        if not parent_result['complete'] or not parent_result['passed'] or parent_result['samples'] != 30000:
            raise ValueError('parent did not finish the30k endpoint evaluation')
        best = parent_result['best']
    values = (model,optimizer,tokenizer,codec,base,saved['initialization'],stream,
        saved['additional_asset_exposures'],saved['input_frame_exposures'],saved['step'],best)
    return values


def inherited_capacity(config):
    from anchor_full_protocol import source_hashes
    parent = parent_config()
    spec = parent['execution']['capacity']
    if config['execution']['capacity'] != spec:
        raise ValueError('inherited capacity evidence changed')
    for key,expected in (('path','sha256'),('resources_path','resources_sha256')):
        if sha256(spec[key]) != spec[expected]:
            raise ValueError('original native capacity receipt changed')
    result = json.loads(Path(spec['path']).read_text())
    resource = json.loads(Path(spec['resources_path']).read_text())
    if (not result['complete'] or not result['passed'] or not resource['passed']
            or not resource['complete'] or resource['exit_code'] != 0 or not resource['payload_cleanup_complete']
            or not resource['kernel_scheduler_membership'] or resource['allocation'] != 131170166
            or not source_compatible(result['source_sha256'],source_hashes())
            or len(result['ranks']) != 2 or any(not r['resume_roundtrip_exact'] for r in result['ranks'])):
        raise ValueError('unchanged dual-GPU forward/capacity and recovery are not established')
    return result


def audit(config, output, source):
    import torch
    from anchor_full_data import audit_plan
    from anchor_full_protocol import source_hashes
    from anchor_full_runtime import release, write_json
    from hardcase_adaptation_protocol import audit_retained_cache
    base = validate_config(config)
    cache = audit_retained_cache(config,base)
    inherited_capacity(config)
    loaded = load_model(config,torch.device('cpu'),0,2,config['training']['resume'],cpu=True)
    model,optimizer,_,_,_,initial,stream,*_ = loaded
    report = copy.deepcopy(model.continuation_restore_report)
    del loaded,model,optimizer
    release()
    plan = audit_plan(base)
    if plan['assets'] != 100000 or plan['updates'] != 2091 or torch.cuda.is_initialized():
        raise ValueError('continuation update count or CPU-only audit differs')
    result = {'complete':True,'passed':True,'source':source,'source_sha256':source_hashes(),
        'scientific_sha256':scientific_sha(config),'initialization':initial,'cache':cache,
        'plan':dict(plan,new_asset_visits=70000,new_updates=1463),'validation_panel':112,
        'restoration':report,'capacity_inherited_from':config['execution']['capacity'],
        'cuda_initialized':False,'original_CPU_loader_and_capacity_not_rerun':True}
    write_json(output/'result.json',result)
    return result


def development_result(config, output):
    fields = ('asset_id','raw_frame_index','normalization_sha256','surface_sha256',
        'original_GT_sha256','target_mesh_diagonal','gt_joint_count')
    baseline = json.loads(Path(RUN+'/outputs/train_v30/evaluation_030000/stage1_rows.json').read_text())
    identities = sorted(tuple(r[k] for k in fields) for r in baseline)
    for n in MILESTONES:
        incoming = json.loads((Path(output)/f'evaluation_{n:06d}/stage1_rows.json').read_text())
        if sorted(tuple(r[k] for k in fields) for r in incoming) != identities:
            raise ValueError('continuation development inputs/GT differ from30k baseline')
    rows = [json.loads((Path(output)/f'evaluation_{n:06d}/summary.json').read_text()) for n in MILESTONES]
    if any(not r['complete'] for r in rows):
        raise ValueError('incomplete continuation development panel')
    return {'complete':True,'requested_endpoint':100000,'total_model_asset_visits':400000,
        'new_asset_visits':70000,'nodes':rows,'no_automatic_model_promotion':True,
        'matched_to_retained30k_development_inputs':True,
        'no_original_3pp_shortlist_early_stop':True,'official_case_comparison_pending':True}
