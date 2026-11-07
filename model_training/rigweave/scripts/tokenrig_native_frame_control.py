"""The fixed native Qwen 5k/50k matched frame-interaction comparison."""
import copy

from tokenrig_native_continuation import checked_json, inherited_capacity, parent_config
from reference_flow_protocol import ROOT

PHASE = 'tokenrig_native_frame_control_20261108'
MANIFEST = {
    'path': '/hshare1/ZETTAI_path_WA_slash_home_KARA/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal5000_inputs_v1/train5000_manifest.jsonl',
    'rows': 5000,
    'sha256': 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88',
}
MILESTONES = list(range(10000, 50001, 10000))
RETENTION = {'keep_last_full': 2, 'full_milestones': MILESTONES,
    'weight_milestones': [], 'keep_best_stage1': True, 'keep_best_stage2': True}


def recipe(arm):
    if arm not in ('full', 'frame_local'):
        raise ValueError('unreviewed frame-control arm')
    parent = parent_config()
    return {**parent, 'training': {**parent['training'], 'additional_assets': 50000,
        'checkpoint_assets': MILESTONES, 'resume': None},
        'native_frame_control': {'arm': arm, 'train_manifest': MANIFEST,
            'interaction_scope': 'shared_mesh_backbone', 'frame_local_group_tokens': 16},
        'artifacts': {**parent['artifacts'], 'retention': RETENTION}}


def validate_config(config):
    from reference_flow_protocol import load_backbone_config
    exact = recipe(config['native_frame_control']['arm'])
    for key in ('schema_version', 'family', 'backbone', 'architecture', 'training',
                'evaluation', 'data_cache', 'tokenrig_decoder', 'capacity', 'native_frame_control'):
        if config[key] != exact[key]:
            raise ValueError('frame-control recipe differs: '+key)
    if (config['execution']['phase'] != PHASE or config['artifacts']['retention'] != RETENTION
            or config['runtime']['allocation'] != 131942123):
        raise ValueError('frame-control scope or native grant differs')
    for key, value in exact['runtime'].items():
        if key not in ('repo','output_root','allocation','stop_before_utc') and config['runtime'][key] != value:
            raise ValueError('frame-control runtime differs: '+key)
    return load_backbone_config(config)


def cache_base(base):
    from reference_flow_protocol import load_backbone_config
    original = load_backbone_config(parent_config())
    value = copy.copy(base)
    value['data'] = {**base['data'], 'manifests': original['data']['manifests']}
    return value


def retained_evidence():
    path = ROOT/'model_training/experiments/tokenrig_native_cont400k_20261107.json'
    return checked_json({'path':str(path),
        'sha256':'fe6fdd65e82af2ac138fa7ba1887089e88b30a8ecfb6d6e5d318a23889c0f1af'})['native_continuation']


def same_native_capacity(config):
    return inherited_capacity({**config, 'native_continuation':retained_evidence()})
