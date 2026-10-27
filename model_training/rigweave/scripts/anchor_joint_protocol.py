"""Matched full5k test of transporting reference joint-anchor associations."""
from reference_flow_protocol import load_backbone_config, sample_seed, scientific_sha, sha256, DONOR_SHA

SCOPE = 'anchor_joint_binding_5k_20261027_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_anchor_joint.py'
VARIANTS = ('reference_transport', 'target_content')


def validate_config(config):
    if config.get('family') != SCOPE or config.get('schema_version') != 1:
        raise ValueError('wrong anchor-binding family')
    train, evaluation = config['training'], config['evaluation']
    if (config['backbone']['checkpoint']['sha256'] != DONOR_SHA
            or train['additional_assets'] != 60000 or train['minimum_assets'] != 40000
            or train['checkpoint_assets'] != list(range(5000, 60001, 5000))
            or train['variants'] != list(VARIANTS) or train.get('variant') is not None
            or train['effective_assets_per_update'] != 48
            or train['loss'] != 'AR_plus_GT_XYZ_L1_plus_target_bone_vector_L1'
            or train['coordinate_weight'] != 10. or train['bone_weight'] != .25
            or train['warmup_assets'] != 1000 or train['head_lr_multiplier'] != 3.
            or train['reference'] != 'uniform_GT_complete_tree_per_asset_visit'
            or train['optimizer'] != 'retain_backbone_Adam_add_head_group'
            or evaluation['input_frames'] != 8 or evaluation['reference_slots'] != [0, 4]
            or evaluation['selection_assets'] != 128 or evaluation['final_assets'] != 967
            or evaluation['reference_kind'] != 'GT_only'):
        raise ValueError('unreviewed matched experiment recipe')
    base = load_backbone_config(config)
    if base['data']['manifests']['train']['rows'] != 5000 or base['data']['manifests']['valid']['rows'] != 967:
        raise ValueError('immutable full5000/967 required')
    if set(train['frame_policy']) != {str(i) for i in range(2, 25)}:
        raise ValueError('all2to24poses required')
    if any(type(v) is not int or v < 1 or 24 % v for v in train['frame_policy'].values()):
        raise ValueError('physical batch must divide per-rank effective batch')
    return base


def should_extend(summaries):
    """Common stopping decision uses only the predeclared128asset selection panel."""
    reasons = []
    for variant in VARIANTS:
        old, new = summaries[variant][30000], summaries[variant][40000]
        if new['normalized_mpjpe'] < old['normalized_mpjpe'] * .99:
            reasons.append(variant + ':MPJPE_improved_at_least1percent')
        for key in ('pck_005', 'edge_f1'):
            if new[key] > old[key] + .005:
                reasons.append(variant + ':' + key + '_improved_at_least0.005')
    return {'extend_both_to': 60000 if reasons else None, 'reasons': reasons,
        'selection_assets': 128, 'not_a_quality_threshold': True,
        'meaning': 'fixed common budget extension, not proof of convergence'}
