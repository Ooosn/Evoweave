"""Pinned 3/2/1 structural adaptation of the completed full300k joint model."""
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from reference_flow_protocol import ROOT, load_backbone_config, scientific_sha, sha256

SCOPE = 'hardcase3608_20261101_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_hardcase_adaptation.py'
PROFILE_SHA = '27666a73a6de586c23345b4d7a6d107eb8f29fb5fea3b30e9524a4e6ae4d6231'
SOURCE_SHA = 'a22824dd9134d1840ea60d75254589f1d95f248d495fe98381cc976dbc0fcaa7'
SOURCE_SCIENTIFIC_SHA = '00faf09254e850f5ec3032696e96b70473d7c5eeb1d09d1a19a183881f7b83f9'
COUNTS = {'core': 1860, 'extension': 1748, 'background': 14394}
WEIGHTS = {'core': 3, 'extension': 2, 'background': 1}
OPTIMIZATION = {'schedule': 'hardcase_warmup_cosine', 'peak_lr': 1e-5,
    'start_lr': 2e-6, 'end_lr': 2e-6, 'warmup_samples': 1000,
    'schedule_samples': 30000, 'div_factor': 5, 'weight_decay': 0,
    'clip_grad_norm': 1, 'auxiliary_weight': 0}


def classify(row):
    rare = row['train_family_assets']
    if rare <= 20 and (row['joints'] >= 64 or row['leaves'] >= 16 or row['max_depth'] >= 20):
        return 'core'
    if rare <= 50 and (row['joints'] >= 48 or row['leaves'] >= 12 or row['max_depth'] >= 16):
        return 'extension'
    return 'background'


def weighted_order(tiers, *, seed=20261101, budget=30000):
    import numpy as np
    pools = {name: np.asarray([i for i, t in enumerate(tiers) if t == name], dtype='<u4')
        for name in WEIGHTS}
    if any(len(pools[name]) != COUNTS[name] for name in pools):
        raise ValueError('candidate pools differ from the approved1860/1748/14394 partition')
    result, cycle = [], 0
    while sum(map(len, result)) < budget:
        labels = np.concatenate([np.full(len(pools[name])*WEIGHTS[name], number, dtype='u1')
            for number, name in enumerate(pools)])
        rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 9731]))
        rng.shuffle(labels)
        order = np.empty(len(labels), dtype='<u4')
        for number, (name, pool) in enumerate(pools.items()):
            # Each tier completes a permutation before repeating its assets.
            visits = np.concatenate([rng.permutation(pool) for _ in range(WEIGHTS[name])])
            order[labels == number] = visits
        result.append(order)
        cycle += 1
    return np.concatenate(result)[:budget].tolist()


def learning_rate(samples, optimization=OPTIMIZATION):
    if type(samples) is not int or not 0 <= samples <= optimization['schedule_samples']:
        raise ValueError('adaptation LR position is outside the approved exposure budget')
    warm = optimization['warmup_samples']
    if samples <= warm:
        return optimization['start_lr'] + (optimization['peak_lr']-optimization['start_lr'])*samples/warm
    progress = (samples-warm)/(optimization['schedule_samples']-warm)
    return optimization['end_lr'] + .5*(optimization['peak_lr']-optimization['end_lr'])*(1+math.cos(math.pi*progress))


@lru_cache(maxsize=8)
def _sampling(path, expected):
    import numpy as np
    if sha256(path) != expected:
        raise ValueError('frozen weighted sampling plan changed')
    plan = json.loads(Path(path).read_text())
    tiers, ids, order = plan['train_tiers'], plan['train_asset_ids'], plan['exposure_order']
    if (plan['profile_sha256'] != PROFILE_SHA or len(ids) != 18002 or len(set(ids)) != 18002
            or len(tiers) != 18002 or {n: tiers.count(n) for n in WEIGHTS} != COUNTS
            or len(order) != 30000 or any(type(i) is not int or not 0 <= i < 18002 for i in order)
            or set(order) != set(range(18002))):
        raise ValueError('sampling membership, counts or coverage changed')
    cycle = sum(COUNTS[n]*WEIGHTS[n] for n in WEIGHTS)
    counts = np.bincount(order[:cycle], minlength=18002)
    if counts.tolist() != [WEIGHTS[t] for t in tiers]:
        raise ValueError('the full weighted cycle does not implement exact3/2/1 weights')
    valid = set(plan['validation_strata'])
    dev, locked = map(set, (plan['development_ids'], plan['locked_ids']))
    if (len(valid) != 967 or len(dev) != 112 or len(locked) != 855 or dev & locked
            or dev | locked != valid or set(ids) & valid):
        raise ValueError('train/development/locked validation isolation changed')
    return plan


def load_sampling(spec):
    return _sampling(spec['path'], spec['sha256'])


def validate_config(config):
    from anchor_full_protocol import validate_config as validate_full
    if config.get('family') != SCOPE or config.get('schema_version') != 1:
        raise ValueError('unknown adaptation family')
    source_recipe = config['source_recipe']
    path = ROOT/source_recipe['relative_path']
    if sha256(path) != source_recipe['sha256']:
        raise ValueError('full300k source recipe changed')
    original = json.loads(path.read_text())
    validate_full(original)
    train = config['training']
    expected = {'additional_assets': 30000, 'effective_assets_per_update': 48,
        'stream_seed': 20261101, 'head_seed': 20261030, 'regime': 'joint_hardcase_adaptation',
        'ar_enabled': True, 'variant': 'reference_transport', 'coordinate_weight': 10,
        'bone_weight': .25, 'warmup_assets': 1, 'head_lr_multiplier': 3,
        'gradient_clip': 'separate_backbone_and_head_norm1',
        'reference': 'uniform_GT_complete_tree_per_asset_visit',
        'optimizer': 'retain_full300k_AdamW_all_groups', 'learning_rate_asset_offset': 0,
        'backbone_exposure_offset': 300000, 'checkpoint_assets': [5000, 10000, 20000, 30000],
        'optimization': OPTIMIZATION}
    if any(train.get(k) != v or type(train.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('approved adaptation training recipe changed')
    if (train['frame_policy'] != original['training']['frame_policy']
            or config['architecture'] != original['architecture']
            or config['backbone']['checkpoint']['sha256'] != SOURCE_SHA
            or config['backbone']['source_asset_exposures'] != 300000
            or config['sampling']['counts'] != COUNTS or config['sampling']['weights'] != WEIGHTS):
        raise ValueError('architecture, source checkpoint or3/2/1 partition changed')
    evaluation = config['evaluation']
    if (evaluation['selection_assets'] != 112 or evaluation['final_assets'] != 967
            or evaluation['ar_interval_assets'] != 5000 or evaluation['xyz_interval_assets'] != 5000
            or evaluation['precision'] != 'fp32_noTF32' or evaluation['decoder_attention'] != 'eager'
            or evaluation['decoding'] != 'greedy_validation_only'
            or evaluation['input_frames'] != 8 or evaluation['reference_slots'] != [0, 4]
            or evaluation['ar_frame_indices'] != list(range(8)) or evaluation['max_new_tokens'] != 2048):
        raise ValueError('fixed development evaluation contract changed')
    if (config['artifacts']['retention'] != {'keep_last_full': 4,
            'full_milestones': [5000, 10000, 20000, 30000],
            'weight_milestones': [5000, 10000, 20000, 30000],
            'keep_best_stage1': True, 'keep_best_stage2': True}
            or config['data_cache'] != original['data_cache']):
        raise ValueError('retention or immutable SSD cache contract changed')
    base = load_backbone_config(config)
    base['optimization'] = copy.deepcopy(OPTIMIZATION)
    base['evaluation']['selection'] = copy.deepcopy(config['sampling']['selection'])
    base['evaluation']['valid_assets'] = 112
    base['hardcase_sampling'] = copy.deepcopy(config['sampling'])
    return base


def initialize_from_source(config, model, optimizer):
    import torch
    from anchor_joint_runtime import head_digest
    from fusion_evaluation import tree_fingerprint
    from virtual_pair_runtime import parameter_names
    spec = config['backbone']['checkpoint']
    if sha256(spec['path']) != SOURCE_SHA:
        raise ValueError('full300k recovery checkpoint checksum differs')
    saved = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (saved['scientific_sha256'] != SOURCE_SCIENTIFIC_SHA
            or saved['additional_asset_exposures'] != 300000 or saved['step'] != 6270
            or saved['optimizer_parameter_names'] != parameter_names(model)
            or len(saved['optimizer']['param_groups']) != 2 or not saved['optimizer']['state']):
        raise ValueError('source is not the complete compatible full300k recovery checkpoint')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    expected_optimizer = tree_fingerprint(saved['optimizer'])
    if tree_fingerprint(optimizer.state_dict()) != expected_optimizer:
        raise RuntimeError('Adam groups or moments changed while importing full300k')
    initial = {'project_checkpoint_loaded': True, 'project_optimizer_loaded': True,
        'source_checkpoint_sha256': SOURCE_SHA, 'source_asset_exposures': 300000,
        'source_optimizer_updates': 6270, 'learning_rate_asset_offset': 0,
        'optimizer_initial_state_entries': len(optimizer.state),
        'optimizer_parameter_names': parameter_names(model),
        'initial_optimizer_sha256': expected_optimizer,
        'initial_model_sha256': head_digest(model), 'head_initial_sha256': head_digest(model.head),
        'AR_and_XYZ_active_from_start': True, 'coordinate_weight_from_first_update': 10,
        'new_data_order_and_LR_phase': True}
    del saved
    return initial


def validate_resume(config, saved, names, sources, world):
    samples = saved['additional_asset_exposures']
    if (saved['scientific_sha256'] != scientific_sha(config)
            or saved['optimizer_parameter_names'] != names or saved['source_sha256'] != sources
            or saved['initialization'].get('source_checkpoint_sha256') != SOURCE_SHA
            or saved['initialization'].get('project_optimizer_loaded') is not True
            or not saved['optimizer']['state'] or type(samples) is not int or not 0 < samples <= 30000
            or saved['global_backbone_asset_exposures'] != 300000+samples):
        raise ValueError('resume does not belong to this weighted adaptation run')
    states = saved['rank_states']
    if (len(states) != world or [r['rank'] for r in states] != list(range(world))
            or any(r['stream']['cursor']*world != samples for r in states)):
        raise ValueError('weighted adaptation resume rank state differs')


def audit_retained_cache(config, base):
    from anchor_full_data import CachedFiles, verify_ssd
    prior = config['execution']['retained_cache_audit']
    if sha256(prior['path']) != prior['sha256']:
        raise ValueError('original accepted cache audit changed')
    receipt = json.loads(Path(prior['path']).read_text())
    resolver = CachedFiles(config, base)
    cache = receipt['cache']
    if not receipt['passed'] or not receipt['complete'] or resolver.index_sha256 != cache['sha256']:
        raise ValueError('immutable SSD cache differs from the accepted full-data cache')
    storage = verify_ssd(config['data_cache']['root'])
    return {**cache, 'retained_accepted_cache': True, 'all_current_file_sizes_checked': True,
        'storage': storage, 'fresh_full_content_rehash': False}


def assess_development(baseline, candidate):
    def focus(value):
        groups = [value['stage1_strata'][name] for name in ('core', 'extension')]
        return sum(g['assets']*g['original_edge_f1'] for g in groups)/sum(g['assets'] for g in groups)
    def pose_error(value):
        return next(g['normalized_mpjpe'] for g in value['groups']
            if g['split'] == 'valid' and g['method'] == 'head')
    hard_gain = focus(candidate)-focus(baseline)
    background_delta = (candidate['stage1_strata']['background']['original_edge_f1']
        -baseline['stage1_strata']['background']['original_edge_f1'])
    failure_delta = candidate['stage1']['failures']-baseline['stage1']['failures']
    pose_ratio = pose_error(candidate)/max(pose_error(baseline), 1e-12)
    return {'samples': candidate['samples'], 'focus_edge_F1_gain': hard_gain,
        'background_edge_F1_delta': background_delta, 'parse_failure_delta': failure_delta,
        'stage2_pose_error_ratio': pose_ratio,
        'eligible_for_official_and_locked_verification': hard_gain >= .03 and background_delta >= -.005
            and failure_delta <= 0 and pose_ratio <= 1.02,
        'engineering_targets_not_proof_of_final_quality': True}


def select_development_checkpoint(output):
    output = Path(output)
    baseline_dir = output/'evaluation_000000'
    baseline = json.loads((baseline_dir/'summary.json').read_text())
    original = json.loads((baseline_dir/'stage1_rows.json').read_text())
    fields = ('asset_id', 'raw_frame_index', 'normalization_sha256', 'surface_sha256',
        'original_GT_sha256', 'target_mesh_diagonal', 'gt_joint_count')
    identities = sorted(tuple(r[k] for k in fields) for r in original)
    decisions = []
    for samples in (5000, 10000, 20000, 30000):
        directory = output/f'evaluation_{samples:06d}'
        candidate = json.loads((directory/'summary.json').read_text())
        rows = json.loads((directory/'stage1_rows.json').read_text())
        if identities != sorted(tuple(r[k] for k in fields) for r in rows):
            raise ValueError('development checkpoint inputs or GT differ from the full300k baseline')
        if not baseline['complete'] or not candidate['complete']:
            raise ValueError('incomplete development evaluation cannot select a model')
        decisions.append(assess_development(baseline, candidate))
    eligible = [d for d in decisions if d['eligible_for_official_and_locked_verification']]
    selected = max(eligible, key=lambda d: (d['focus_edge_F1_gain'], -d['samples'])) if eligible else None
    return {'complete': True, 'decisions': decisions,
        'selected_samples': selected['samples'] if selected else None,
        'keep_full300k_if_none_eligible': True, 'locked_or_official_acceptance_complete': False,
        'matched_development_inputs_verified': True,
        'next': 'Verify the fixed selected candidate versus full300k on locked validation and official beam inference; no retuning.'}
