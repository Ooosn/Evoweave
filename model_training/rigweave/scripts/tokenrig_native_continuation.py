"""Resume the complete native Qwen 300k checkpoint without restarting Adam."""
import copy
import json
import math
from pathlib import Path

from anchor_full_protocol import scientific_sha, sha256
from reference_flow_protocol import ROOT, load_backbone_config
from tokenrig_comparison_matrix import IDENTITY

PHASE = 'tokenrig_native_cont400k_20261107'
PARENT_SOURCE = '2f50732ee08e98d609c38bc2f73ec3657e5c7022'
PARENT_CONFIG = ROOT/'model_training/experiments/tokenrig_native_full_20261105.json'
PARENT_CONFIG_SHA = 'e4fece4995c54dab87318bc13ab2280c47670fe1e920a371c7e63626fe159abb'
PARENT_ROOT = '/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_full'
PARENT_ASSETS, PARENT_STEPS, PARENT_FRAMES = 300000, 6270, 3919984
MILESTONES = list(range(310000, 400001, 10000))
PARENT_CHECKPOINT = {'path': PARENT_ROOT+'/checkpoint_300000.pt',
    'sha256': '743f55cc65f833c874c6c0e1ba87181f85fbe04d053d63f7be6047f15161e844'}
HARNESS_CHANGES = {
    'anchor_full_protocol.py', 'anchor_full_runtime.py', 'anchor_full_evaluation.py',
    'anchor_joint_runtime.py', 'reference_flow_protocol.py', 'virtual_pair_guard.py',
}


def checked_json(spec):
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('pinned continuation evidence changed: '+spec['path'])
    return json.loads(Path(spec['path']).read_text())


def parent_config():
    return checked_json({'path': str(PARENT_CONFIG), 'sha256': PARENT_CONFIG_SHA})


def validate_config(config):
    parent = parent_config()
    cont = config['native_continuation']
    if (cont['parent_checkpoint'] != PARENT_CHECKPOINT or cont['parent_source'] != PARENT_SOURCE
            or cont['learning_rate'] != 'hold_completed_onecycle_endpoint'
            or config['execution']['phase'] != PHASE
            or config['family'] != parent['family'] or config['schema_version'] != parent['schema_version']):
        raise ValueError('unreviewed native continuation origin or schedule')
    for key in ('backbone', 'architecture', 'tokenrig_decoder', 'data_cache', 'capacity'):
        if config[key] != parent[key]:
            raise ValueError('native continuation changed the parent model or dataset: '+key)
    expected_train = {**parent['training'], 'additional_assets': 400000,
        'checkpoint_assets': parent['training']['checkpoint_assets'] + MILESTONES,
        'optimizer': 'retain_parent_AdamW_all_groups', 'resume': PARENT_CHECKPOINT}
    if config['training'] != expected_train:
        raise ValueError('native continuation changed batch, supervision, sampler, or checkpoint origin')
    if config['evaluation'] != {**parent['evaluation'], 'ar_interval_assets': 10000}:
        raise ValueError('the same development panel and metrics are required at every new node')
    if config['artifacts']['retention'] != {'keep_last_full': 2, 'full_milestones': MILESTONES,
            'weight_milestones': [], 'keep_best_stage1': True, 'keep_best_stage2': True}:
        raise ValueError('all ten complete recoveries must be retained')
    runtime = config['runtime']
    for key, value in parent['runtime'].items():
        if key not in ('repo', 'output_root', 'allocation', 'stop_before_utc') and runtime[key] != value:
            raise ValueError('native continuation changed verified execution resources: '+key)
    if (runtime['allocation'] != 131942123 or runtime['stop_before_utc'] != '2026-11-08T18:20:00Z'
            or runtime['required_h_rt_seconds'] != 345600):
        raise ValueError('continuation needs the new native grant and conservative scratch save margin')
    return load_backbone_config(config)


def source_compatible(old, new):
    script = 'model_training/rigweave/scripts/'
    extension = script+'tokenrig_native_continuation.py'
    allowed = {script+name for name in HARNESS_CHANGES} | {'model_training/jobs/launch_virtual_pair.py'}
    return set(new) == set(old) | {extension} and all(new[k] == v for k, v in old.items() if k not in allowed)


def inherited_capacity(config):
    cont = config['native_continuation']
    review = checked_json(cont['parent_completion'])
    resources = checked_json(cont['parent_resources'])
    runtime = config['runtime']
    if (not review['training_and_prescribed_validation_complete'] or review['source'] != PARENT_SOURCE
            or review['config_sha256'] != PARENT_CONFIG_SHA
            or review['resource_receipt_sha256'] != cont['parent_resources']['sha256']
            or [(r['visits'], r['updates'], r['input_frame_visits']) for r in review['ranks']]
                != [(PARENT_ASSETS, PARENT_STEPS, PARENT_FRAMES)] * 2
            or not resources['complete'] or not resources['passed'] or resources['exit_code'] != 0
            or not resources['kernel_scheduler_membership'] or not resources['payload_cleanup_complete']
            or resources['source'] != PARENT_SOURCE or resources['physical_gpus'] != runtime['physical_gpus']
            or resources['cuda_visible_devices'] != ','.join(runtime['gpu_uuids'])
            or resources['cpu_affinity'] != runtime['cpu_affinity']):
        raise ValueError('unchanged native training capacity is not established')
    peaks = resources['peak_gpu_memory_mib']
    if set(peaks) != {str(i) for i in runtime['physical_gpus']} or any(not 0 < v <= 78*1024 for v in peaks.values()):
        raise ValueError('parent training exceeded the approved GPU ceiling')
    return {'complete': True, 'passed': True, 'inherited_parent_training_peaks_mib': peaks,
        'no_repeated_capacity_stress': True}


def completed_streams(config, saved):
    from anchor_full_data import make_stream
    from virtual_fusion_normal_runtime import MatchedFrameStream
    parent_base = load_backbone_config(parent_config())
    states = saved['rank_states']
    if [r['rank'] for r in states] != [0, 1]:
        raise ValueError('complete rank 0/1 sampler and RNG states are required')
    streams = []
    for rank, state in enumerate(states):
        parent_stream = make_stream(parent_base, rank, 2, state['stream'])
        if parent_stream.cursor * 2 != PARENT_ASSETS:
            raise ValueError('parent sampler did not complete the 300k budget')
        streams.append(MatchedFrameStream.extend_completed(range(parent_stream.contract['dataset_rows']),
            state['stream'], target_assets=400000, update_boundaries=config['training']['checkpoint_assets']))
    return streams


def restore_parent(config, device, rank, world, resume, *, cpu=False):
    import torch
    import torch.distributed as dist
    from anchor_full_protocol import source_hashes
    from anchor_full_runtime import initialize
    from fusion_evaluation import tree_fingerprint
    from id_xyz_stages import random_state
    from virtual_pair_runtime import apply_rng, parameter_names

    base = validate_config(config)
    if world != 2 or resume != PARENT_CHECKPOINT:
        raise ValueError('a complete, fixed 300k parent recovery is required')
    if rank == 0 and sha256(resume['path']) != resume['sha256']:
        raise ValueError('parent complete checkpoint SHA256 differs')
    if not cpu:
        dist.barrier(device_ids=[device.index])
    saved = torch.load(resume['path'], map_location='cpu', mmap=True, weights_only=False)
    parent = parent_config()
    if (saved['code_commit'] != PARENT_SOURCE or saved['scientific_sha256'] != scientific_sha(parent)
            or not source_compatible(saved['source_sha256'], source_hashes())
            or (saved['additional_asset_exposures'], saved['global_backbone_asset_exposures'],
                saved['input_frame_exposures'], saved['step']) != (PARENT_ASSETS,PARENT_ASSETS,PARENT_FRAMES,PARENT_STEPS)
            or not saved['optimizer']['state'] or len(saved['optimizer']['param_groups']) != 2):
        raise ValueError('parent model, optimizer or source provenance differs')
    streams = completed_streams(config, saved)
    model, optimizer, tokenizer, codec, _, _ = initialize(parent, device)
    if saved['optimizer_parameter_names'] != parameter_names(model):
        raise ValueError('Adam parameter ordering differs')
    model_hash, optimizer_hash = tree_fingerprint(saved['model']), tree_fingerprint(saved['optimizer'])
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    if tree_fingerprint(model.state_dict()) != model_hash or tree_fingerprint(optimizer.state_dict()) != optimizer_hash:
        raise ValueError('model or Adam changed during restore')
    rng = [{k:v for k,v in state.items() if k not in ('rank','stream')} for state in saved['rank_states']]
    if any(set(state) != {'python_rng','numpy_rng','torch_rng','cuda_rng'} for state in rng):
        raise ValueError('both complete rank RNG states are required')
    if not cpu:
        apply_rng(rng[rank])
        if tree_fingerprint(random_state()) != tree_fingerprint(rng[rank]):
            raise ValueError('rank RNG restore differs')
    initial = {'parent_checkpoint': resume, 'parent_source': PARENT_SOURCE,
        'parent_initialization': saved['initialization'], 'project_checkpoint_loaded': True,
        'project_optimizer_loaded': True, 'optimizer_state_entries': len(optimizer.state),
        'model_sha256': model_hash, 'optimizer_sha256': optimizer_hash,
        'parent_rank_rng_sha256': [tree_fingerprint(state) for state in rng],
        'completed_sampler_prefix_unchanged': True, 'extended_plan_sha256': streams[rank].plan_sha256}
    model.continuation_restore_report = {**initial, 'samples': PARENT_ASSETS, 'step': PARENT_STEPS,
        'rank': rank, 'rank_RNG_restored': not cpu, 'sampler': streams[rank].state_dict()}
    del saved
    base['tokenrig_decoder'] = config['tokenrig_decoder']
    return (model, optimizer, tokenizer, codec, base, initial, streams[rank],
        PARENT_ASSETS, PARENT_FRAMES, PARENT_STEPS, {})


def validate_saved(config, saved, names, sources, world):
    import numpy as np
    from anchor_full_data import make_stream
    samples = saved['additional_asset_exposures']
    initial = saved['initialization']
    if (world != 2 or saved['scientific_sha256'] != scientific_sha(config)
            or saved['source_sha256'] != sources or saved['optimizer_parameter_names'] != names
            or not initial['project_optimizer_loaded'] or initial['parent_checkpoint'] != PARENT_CHECKPOINT
            or not PARENT_ASSETS < samples <= 400000 or saved['global_backbone_asset_exposures'] != samples
            or [r['rank'] for r in saved['rank_states']] != [0,1] or not saved['optimizer']['state']):
        raise ValueError('new continuation recovery has a different contract')
    base = validate_config(config)
    for rank, state in enumerate(saved['rank_states']):
        stream = make_stream(base, rank, world, state['stream'])
        updates = int(np.searchsorted(stream.global_ends, samples, side='left')) + 1
        frames = int(np.dot(stream.update_sizes[:updates].astype(np.int64), stream.frame_counts[:updates]))
        if (stream.cursor*world != samples or int(stream.global_ends[updates-1]) != samples
                or saved['step'] != updates or saved['input_frame_exposures'] != frames):
            raise ValueError('new checkpoint sampler/update/frame counters differ')


def panel_identity(rows):
    keyed = {(row['asset_id'],row['frame_index']):tuple(row[k] for k in IDENTITY) for row in rows}
    assets = {asset for asset,_ in keyed}
    if len(rows) != 1024 or len(assets) != 128 or set(keyed) != {(a,f) for a in assets for f in range(8)}:
        raise ValueError('development panel must contain the same 128 assets and eight views each')
    return keyed


def verify_panel(config, output, samples):
    parent = checked_json(config['native_continuation']['parent_stage1_rows'])
    path = Path(output)/f'evaluation_{samples:06d}'/'stage1_rows.json'
    if panel_identity(json.loads(path.read_text())) != panel_identity(parent):
        raise ValueError('development raw frames, normalization, surfaces or original GT differ')


def verify_first_update(value, expected):
    for key in ('step','additional_asset_exposures','assets_in_update','frames_in_update'):
        if value[key] != expected[key]:
            raise ValueError('first continuation update count differs: '+key)
    fields = ('manifest_indices','asset_ids','sampling_seeds','frames')
    actual = [{k:batch[k] for k in fields} for batch in value['microbatches']]
    if actual != expected['microbatches']:
        raise ValueError('actual continuation sample indices, assets, frames or seeds differ')
    if (value['learning_rate_asset_position'] != PARENT_ASSETS
            or not all(math.isclose(value[k], v, rel_tol=0, abs_tol=1e-15)
                for k,v in (('lr',2e-6),('head_lr',6e-6),('beta1',.95)))
            or not all(math.isfinite(value[k]) for k in ('ar_loss','xyz_loss','bone_loss','gradient_norm'))
            or value['gradient_groups'] is None):
        raise ValueError('first continuation updates violate the finite loss/gradient or LR contract')


def audit(config, output, source):
    import numpy as np
    import torch
    from anchor_full_data import CachedFiles, make_stream, verify_ssd
    from anchor_full_evaluation import canonical_input
    from anchor_full_protocol import source_hashes
    from anchor_full_runtime import release
    from reference_flow_runtime import read_all
    from stage1_surface_repair_runtime import validate_canonical
    from virtual_pair_runtime import evaluation_rows, next_update_batches
    from virtual_query_xyz_runtime import fixed_dataset, write_json

    capacity = inherited_capacity(config)
    base, entries = read_all(config)
    resolver = CachedFiles(config,base)
    cont = config['native_continuation']
    if resolver.index_sha256 != cont['cache_index_sha256']:
        raise ValueError('retained training cache index differs')
    storage = verify_ssd(config['data_cache']['root'])
    old_rows = checked_json(cont['parent_stage1_rows'])
    panel_identity(old_rows)
    panel = evaluation_rows(base,entries)
    dataset = fixed_dataset(panel,base,8,mode='seen_eval',file_resolver=resolver)
    for index in range(len(panel)):
        record,_ = canonical_input(dataset,index,base,config,'seen_eval',torch.device('cpu'))
        validate_canonical(record,old_rows)
    archive = cont['panel_archive']
    if sha256(archive['durable_path']) != archive['sha256'] or archive['canonical_assets'] != 128:
        raise ValueError('retained canonical panel archive changed')
    del dataset, record
    loaded = restore_parent(config,torch.device('cpu'),0,2,PARENT_CHECKPOINT,cpu=True)
    initial, stream = loaded[5:7]
    report = copy.deepcopy(loaded[0].continuation_restore_report)
    first = {}
    for rank in (0,1):
        state = copy.deepcopy(stream.state_dict())
        state['contract']['rank'] = rank
        planner = make_stream(base,rank,2,state)
        samples, first[str(rank)] = PARENT_ASSETS, []
        for step in (6271,6272):
            goal,batches = next_update_batches(planner,samples,2)
            samples += goal
            first[str(rank)].append({'step':step,'additional_asset_exposures':samples,'assets_in_update':goal,
                'frames_in_update':goal*batches[0][0].frames,'microbatches':[{
                    'manifest_indices':[r.item_index for r in batch],
                    'asset_ids':[entries['train'][r.item_index]['asset_id'] for r in batch],
                    'sampling_seeds':[r.sampling_seed for r in batch],'frames':batch[0].frames} for batch in batches]})
    frames = int(np.dot(stream.update_sizes.astype(np.int64),stream.frame_counts))
    steps = len(stream.update_sizes)
    if steps != 8360 or stream.target_assets != 400000 or torch.cuda.is_initialized():
        raise ValueError('native continuation must add exactly 2090 updates without GPU audit work')
    del loaded
    release()
    result = {'complete':True,'passed':True,'source':source,'source_sha256':source_hashes(),
        'scientific_sha256':scientific_sha(config),'cuda_initialized':False,
        'initialization':initial,'restoration':report,'capacity':capacity,
        'cache':{'sha256':resolver.index_sha256,'files':len(resolver.files),'storage':storage},
        'plan':{'assets':400000,'updates':steps,'input_frame_visits':frames,
            'new_assets':100000,'new_updates':2090,'new_input_frame_visits':frames-PARENT_FRAMES,
            'first_updates':first},'exact_retained_development_panel':True}
    write_json(output/'result.json',result)
    return result


def verify_audit(config):
    from anchor_full_protocol import source_hashes
    spec = config['execution']['CPU_audit']
    result = checked_json(spec)
    resources = checked_json({'path':spec['resources_path'],'sha256':spec['resources_sha256']})
    if (not result['complete'] or not result['passed'] or result['cuda_initialized']
            or result['scientific_sha256'] != scientific_sha(config) or result['source_sha256'] != source_hashes()
            or not result['exact_retained_development_panel'] or result['source'] != resources['source']
            or not resources['complete'] or not resources['passed'] or resources['exit_code'] != 0
            or resources['allocation'] != config['runtime']['allocation'] or not resources['kernel_scheduler_membership']
            or not resources['payload_cleanup_complete']):
        raise ValueError('native continuation CPU restoration/admission is not accepted')
    return result
