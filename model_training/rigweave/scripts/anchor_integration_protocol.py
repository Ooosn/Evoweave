"""Controlled value of the pretrained/joint skeleton-generation framework."""
import copy

from reference_flow_protocol import DONOR_SHA, load_backbone_config, scientific_sha, sha256

SCOPE = 'anchor_integration_5k_20261028_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_anchor_integration.py'
REGIMES = ('joint_pretrained', 'pretrained_stage2', 'scratch_stage2')
PARENT_SOURCE = '9db222d0070122b08c6d5dde3329f3be69d46c16'
PARENT_SCIENCE = '0a3f889063f52c0e7687b10b47fca109d2987a1bcca1ad1729bd8b1c653430b9'
PARENT_SHA = 'cfd446c1f6808304aa6a691884c2ed77d19db0f1850512dfd97b83e7eb0a0016'


def validate_config(config):
    train = config['training']
    if (config.get('family') != SCOPE or config.get('schema_version') != 1
            or config['backbone']['checkpoint']['sha256'] != DONOR_SHA
            or config['backbone']['source_asset_exposures'] != 20000
            or train['regimes'] != list(REGIMES) or train.get('regime') is not None
            or train['variant'] != 'reference_transport'
            or train['additional_assets'] != 80000 or train['comparison_assets'] != 60000
            or train['minimum_assets'] != 60000
            or train['comparison_interval_assets'] != 10000
            or train['checkpoint_assets'] != list(range(5000, 80001, 5000))
            or train['effective_assets_per_update'] != 48
            or train['head_seed'] != 20261028 or train['stream_seed'] != 20261028
            or train['scratch_seed'] != 20261028
            or train['coordinate_weight'] != 10 or train['bone_weight'] != .25
            or train['warmup_assets'] != 1000 or train['head_lr_multiplier'] != 3
            or train['gradient_clip'] != 'separate_backbone_and_head_norm1'
            or train['learning_rate_asset_offset'] != 20000
            or train['optimizer'] != 'retain_pretrained_Adam_or_fresh_scratch_Adam'
            or train['loss'] != 'matched_coordinate_objective_with_regime_controlled_AR'
            or train['reference'] != 'uniform_GT_complete_tree_per_asset_visit'):
        raise ValueError('unreviewed integration comparison recipe')
    parent = train['parent_continuation']
    if (parent['sha256'] != PARENT_SHA or parent['code_commit'] != PARENT_SOURCE
            or parent['scientific_sha256'] != PARENT_SCIENCE
            or parent['additional_asset_exposures'] != 60000
            or parent['input_frame_exposures'] != 787504 or parent['step'] != 1260):
        raise ValueError('undeclared parent continuation')
    if (config['architecture'] != {'input_dim': 1024, 'dim': 256, 'heads': 8,
            'reference_depth': 2, 'target_depth': 4, 'max_joints': 512,
            'max_hop': 8, 'gradient_checkpointing': True}
            or config['evaluation']['reference_slots'] != [0, 4]
            or config['evaluation']['input_frames'] != 8
            or config['evaluation']['selection_assets'] != 128
            or config['evaluation']['final_assets'] != 967
            or config['evaluation']['train_diagnostic_assets'] != 32
            or config['evaluation']['reference_kind'] != 'GT_only'):
        raise ValueError('integration architecture/evaluation contract changed')
    expected = {str(n): 24 if n <= 6 else 12 if n <= 12 else 6 for n in range(2, 25)}
    base = load_backbone_config(config)
    if (train['frame_policy'] != expected or base['data']['manifests']['train']['rows'] != 5000
            or base['data']['manifests']['valid']['rows'] != 967):
        raise ValueError('full5k/data/physical-batch contract changed')
    phase = config['execution']['phase']
    if phase not in ('controls_to60k', 'common80k'):
        raise ValueError('unreviewed integration phase')
    if phase == 'common80k':
        resumes = config['execution'].get('control_resumes', {})
        if set(resumes) != set(REGIMES[1:]) or any(
                r.get('additional_asset_exposures') != 60000 or not r.get('both_copies_verified')
                or not r.get('complete') or len(r.get('sha256', '')) != 64 for r in resumes.values()):
            raise ValueError('common80k requires verified completed60k controls')
    return base


def regime_config(config, regime):
    if regime not in REGIMES:
        raise ValueError('unknown integration regime')
    result = copy.deepcopy(config)
    result['training'].update(regime=regime, ar_enabled=regime == 'joint_pretrained',
        backbone_initialization='random' if regime == 'scratch_stage2' else 'pretrained',
        backbone_exposure_offset=0 if regime == 'scratch_stage2' else 20000)
    result['artifacts']['durable_root'] += '/' + regime
    return result


def resume_kind(config, saved, resume):
    if saved['scientific_sha256'] == scientific_sha(config):
        return 'same_science'
    parent = config['training']['parent_continuation']
    expected = {'code_commit': PARENT_SOURCE, 'scientific_sha256': PARENT_SCIENCE,
        'additional_asset_exposures': 60000, 'input_frame_exposures': 787504, 'step': 1260}
    if (config['training']['regime'] != 'joint_pretrained'
            or resume['path'] != parent['path'] or resume['sha256'] != PARENT_SHA
            or any(saved.get(k) != value for k, value in expected.items())):
        raise ValueError('resume is neither same-science nor declared completed60k parent')
    old = saved['configuration']
    current = config
    for section in ('architecture',):
        if old[section] != current[section]:
            raise ValueError('parent model architecture changed')
    for key in ('variant', 'frame_policy', 'effective_assets_per_update', 'stream_seed', 'head_seed',
                'coordinate_weight', 'bone_weight', 'warmup_assets', 'head_lr_multiplier', 'gradient_clip'):
        if old['training'][key] != current['training'][key]:
            raise ValueError('parent training prefix changed: ' + key)
    if old['backbone'] != current['backbone']:
        raise ValueError('parent backbone source changed')
    return 'declared_parent'
