"""Full-data joint training from official UniRig, without project optimizer history."""
import copy
from pathlib import Path

from reference_flow_protocol import load_backbone_config, scientific_sha, sha256

SCOPE = 'anchor_full_20261030_v1'
BASE_SCOPE = 'anchor_full_backbone_20261030_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_anchor_full.py'
TRAIN_ROWS = 18002
VALID_ROWS = 967
TRAIN_SHA = '653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27'
VALID_SHA = 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a'
OFFICIAL_SHA = 'd8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4'


def validate_config(config):
    if config.get('family') == 'hardcase3608_20261101_v1':
        from hardcase_adaptation_protocol import validate_config as validate_adaptation
        return validate_adaptation(config)
    train, evaluation = config['training'], config['evaluation']
    if (config.get('family') != SCOPE or config.get('schema_version') != 1
            or config['backbone'].get('initialization') != 'official_UniRig_only'
            or config['backbone'].get('source_asset_exposures') != 0
            or set(config['backbone']) != {'config', 'initialization', 'source_asset_exposures'}):
        raise ValueError('full training must not inherit a project checkpoint')
    expected = {'additional_assets': 300000, 'effective_assets_per_update': 48,
        'stream_seed': 20261030, 'head_seed': 20261030, 'regime': 'joint_official',
        'ar_enabled': True, 'variant': 'reference_transport', 'coordinate_weight': 10,
        'bone_weight': .25, 'warmup_assets': 1000, 'head_lr_multiplier': 3,
        'gradient_clip': 'separate_backbone_and_head_norm1',
        'reference': 'uniform_GT_complete_tree_per_asset_visit',
        'optimizer': 'fresh_AdamW_all_groups', 'learning_rate_asset_offset': 0,
        'backbone_exposure_offset': 0}
    if any(train.get(k) != v or type(train.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('full-data training recipe changed')
    if train['checkpoint_assets'] != list(range(10000, 300001, 10000)):
        raise ValueError('unreviewed checkpoint/validation cadence')
    policy = {str(t): 24 if t <= 6 else 12 if t <= 12 else 6 for t in range(2, 25)}
    if train['frame_policy'] != policy:
        raise ValueError('physical batch policy changed')
    base = load_backbone_config(config)
    if base['family'] != BASE_SCOPE or base['initialization']['sha256'] != OFFICIAL_SHA:
        raise ValueError('full training requires the pinned official initialization')
    if [(base['data']['manifests'][s]['rows'], base['data']['manifests'][s]['sha256'])
            for s in ('train', 'valid')] != [(TRAIN_ROWS, TRAIN_SHA), (VALID_ROWS, VALID_SHA)]:
        raise ValueError('full immutable train/validation membership changed')
    data = base['data']
    if (data['frames_min'], data['frames_max'], data['heldout_frames'], data['normalization'],
            data['global_rt_removal'], data['coordinate_clipping']) != (2, 24, 8, 'input_sequence_mesh_bbox', True, False):
        raise ValueError('RT, normalization or frame partition contract changed')
    if base['optimization'] != {'schedule': 'onecycle', 'peak_lr': .0001, 'warmup_fraction': .1,
            'div_factor': 5, 'final_div_factor': 10, 'weight_decay': 0,
            'schedule_samples': 300000, 'clip_grad_norm': 1, 'auxiliary_weight': 0}:
        raise ValueError('full300k learning-rate schedule changed')
    if (base['initialization']['seed'] != 20261030 or base['pair']['target_selection'] != {
            'algorithm': 'sha256_python_random_uniform20_40_floor_min2_sample_v1', 'seed': 20261030}):
        raise ValueError('initialization/supervision seed changed')
    if config['architecture'] != {'input_dim': 1024, 'dim': 256, 'heads': 8,
            'reference_depth': 2, 'target_depth': 4, 'max_joints': 512,
            'max_hop': 8, 'gradient_checkpointing': True}:
        raise ValueError('stage2 architecture changed')
    if (evaluation['precision'] != 'fp32_noTF32' or evaluation['decoder_attention'] != 'eager'
            or evaluation['decoding'] != 'greedy_validation_only'
            or evaluation['selection_assets'] != 128 or evaluation['final_assets'] != VALID_ROWS
            or evaluation['input_frames'] != 8 or evaluation['reference_slots'] != [0, 4]
            or evaluation['ar_frame_indices'] != list(range(8)) or evaluation['max_new_tokens'] != 2048
            or evaluation['xyz_interval_assets'] != 10000 or evaluation['ar_interval_assets'] != 50000):
        raise ValueError('validation precision, coverage or generation policy changed')
    if (config['data_cache']['root'] != '/work/wangyy/anchor_full_20261030_data'
            or config['data_cache']['index'] != config['data_cache']['root'] + '/index.json'
            or config['data_cache']['originals_unchanged'] is not True):
        raise ValueError('data cache escaped the declared SSD root')
    if config['artifacts']['retention'] != {'keep_last_full': 2,
            'full_milestones': [100000, 200000, 300000],
            'weight_milestones': [50000, 100000, 150000, 200000, 250000, 300000],
            'keep_best_stage1': True, 'keep_best_stage2': True}:
        raise ValueError('checkpoint retention policy changed')
    return base


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = list((root/'model_training/rigweave/src').rglob('*.py'))
    names = ['anchor_full_protocol.py', 'anchor_full_data.py', 'anchor_full_runtime.py', 'anchor_full_cleanup.py',
        'anchor_full_evaluation.py', 'train_anchor_full.py', 'anchor_joint_runtime.py',
        'virtual_query_xyz_runtime.py', 'reference_flow_runtime.py', 'reference_flow_protocol.py',
        'virtual_pair_runtime.py', 'virtual_frame_runtime.py', 'virtual_pair_loading.py',
        'virtual_fusion_normal_runtime.py', 'train_virtual_query_xyz.py', 'stage1_feedback.py',
        'single_query_protocol.py', 'id_xyz_stages.py']
    names += ['hardcase_adaptation_protocol.py', 'prepare_hardcase_adaptation.py',
        'train_hardcase_adaptation.py', 'virtual_pair_guard.py']
    paths += [Path(__file__).with_name(name) for name in names]
    paths += [root/'model_training/tools/hardcase_adaptation_ops.py',
        root/'model_training/jobs/launch_virtual_pair.py']
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def validate_resume(config, saved, names, sources, world=2):
    if config.get('family') == 'hardcase3608_20261101_v1':
        from hardcase_adaptation_protocol import validate_resume as validate_adaptation_resume
        return validate_adaptation_resume(config, saved, names, sources, world)
    if (saved['scientific_sha256'] != scientific_sha(config)
            or saved['optimizer_parameter_names'] != names or saved['source_sha256'] != sources
            or saved['initialization'].get('project_checkpoint_loaded') is not False
            or saved['global_backbone_asset_exposures'] != saved['additional_asset_exposures']):
        raise ValueError('resume does not belong to this fresh full-data run')
    samples = saved['additional_asset_exposures']
    if type(samples) is not int or not 0 < samples <= config['training']['additional_assets']:
        raise ValueError('invalid resume exposure count')
    states = saved['rank_states']
    if len(states) != world or [r['rank'] for r in states] != list(range(world)):
        raise ValueError('resume rank states differ')
    if any(r['stream']['cursor'] * world != samples for r in states):
        raise ValueError('resume data cursor differs from completed optimizer updates')


def full_checkpoints_to_keep(steps, retention):
    steps = sorted(set(steps))
    return set(steps[-retention['keep_last_full']:]) | (set(steps) & set(retention['full_milestones']))


def cache_identity(base):
    return copy.deepcopy(base['data']['manifests'])
