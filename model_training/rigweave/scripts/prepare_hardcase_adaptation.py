#!/usr/bin/env python3
"""Freeze weighted train visits and independent structural validation strata."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from hardcase_adaptation_protocol import (COUNTS, OPTIMIZATION, PROFILE_SHA, SCOPE,
    SOURCE_SHA, WEIGHTS, classify, sha256, weighted_order)


def write_new(path, value):
    content = json.dumps(value, indent=2, sort_keys=True)+'\n'
    if path.exists():
        if path.read_bytes() != content.encode('utf-8'):
            raise RuntimeError('refuse to replace a frozen preparation artifact: '+str(path))
    else:
        path.write_text(content, encoding='utf-8', newline='\n')


def build_plan(rows):
    train = [r for r in rows if r['split'] == 'train']
    valid = [r for r in rows if r['split'] == 'valid']
    freq = Counter(r['unordered_rooted_tree_sha256'] for r in train)
    if (len(train) != 18002 or len(valid) != 967
            or any(r['train_family_assets'] != freq[r['unordered_rooted_tree_sha256']] for r in rows)):
        raise ValueError('profile split sizes or training-only family frequencies differ')
    tiers = [classify(r) for r in train]
    if dict(Counter(tiers)) != COUNTS:
        raise ValueError('approved candidate pool counts changed')
    order = weighted_order(tiers)
    strata = {r['asset_id']: classify(r) for r in valid}
    known = {valid[i]['asset_id'] for i in (5, 15, 23, 24, 30, 44, 54, 61, 70, 95, 105, 108)}
    development = []
    quotas = {'core': 48, 'extension': 16, 'background': 48}
    for name, goal in quotas.items():
        pool = [r for r in valid if strata[r['asset_id']] == name]
        chosen = [r for r in pool if r['asset_id'] in known]
        family_counts = Counter(r['unordered_rooted_tree_sha256'] for r in chosen)
        ranked = sorted(pool, key=lambda r: hashlib.sha256(
            ('20261101:development:'+r['source']+':'+r['asset_id']).encode()).digest())
        for limit in (1, 2, 967):
            for row in ranked:
                if len(chosen) >= goal:
                    break
                if row not in chosen and family_counts[row['unordered_rooted_tree_sha256']] < limit:
                    chosen.append(row)
                    family_counts[row['unordered_rooted_tree_sha256']] += 1
        if len(chosen) != goal:
            raise ValueError('not enough validation assets for the frozen structural panel')
        development.extend(r['asset_id'] for r in chosen)
    ids = [r['asset_id'] for r in train]
    if len(set(ids)) != len(ids) or set(ids) & set(strata):
        raise ValueError('asset-level split identity collision')
    actual = Counter(tiers[i] for i in order)
    mass = sum(COUNTS[n]*WEIGHTS[n] for n in WEIGHTS)
    return {'profile_sha256': PROFILE_SHA, 'seed': 20261101,
        'algorithm': 'exact3_2_1_weighted_cycles_with_within_tier_permutations_v1',
        'train_asset_ids': ids, 'train_tiers': tiers, 'exposure_order': order,
        'validation_strata': strata, 'development_ids': development,
        'locked_ids': [r['asset_id'] for r in valid if r['asset_id'] not in set(development)],
        'known_diagnostic_ids_in_development': sorted(known),
        'report': {'train_pool_counts': COUNTS, 'relative_asset_weights': WEIGHTS,
            'weighted_cycle_visits': mass, 'total_visits': len(order),
            'unique_train_assets_visited': len(set(order)), 'actual_tier_visits': dict(actual),
            'actual_tier_fractions': {n: actual[n]/len(order) for n in WEIGHTS},
            'expected_tier_fractions': {n: COUNTS[n]*WEIGHTS[n]/mass for n in WEIGHTS},
            'validation_stratum_counts': dict(Counter(strata.values())),
            'development_counts': quotas, 'locked_validation_assets': 855,
            'locked_means_not_used_for_this_selection_not_historically_unseen': True}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-manifest', type=Path)
    parser.add_argument('--valid-manifest', type=Path)
    parser.add_argument('--write-config', type=Path)
    parser.add_argument('--remote-preparation', type=Path)
    args = parser.parse_args()
    if sha256(args.profile) != PROFILE_SHA:
        raise ValueError('topology profile checksum changed')
    plan = build_plan(json.loads(args.profile.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'sampling_plan.json'
    write_new(path, plan)
    result = {'path': str(path), 'sha256': sha256(path), 'report': plan['report']}
    if args.train_manifest or args.valid_manifest:
        from anchor_full_protocol import TRAIN_SHA, VALID_SHA
        if (sha256(args.train_manifest) != TRAIN_SHA or sha256(args.valid_manifest) != VALID_SHA):
            raise ValueError('immutable train or validation manifest changed')
        train = [json.loads(line) for line in args.train_manifest.read_text().splitlines() if line]
        valid = [json.loads(line) for line in args.valid_manifest.read_text().splitlines() if line]
        if [r['asset_id'] for r in train] != plan['train_asset_ids'] or {r['asset_id'] for r in valid} != set(plan['validation_strata']):
            raise ValueError('topology profile order does not match the actual manifest')
        selected = {r['asset_id']: r for r in valid}
        selection = args.output/'development_selection.json'
        write_new(selection, {'valid': [selected[i] for i in plan['development_ids']]})
        result['selection'] = {'path': str(selection), 'sha256': sha256(selection)}
    write_new(args.output/'preparation.json', result)
    if args.write_config:
        if not args.remote_preparation:
            raise ValueError('the executable config needs the actual remote manifest preparation')
        remote = json.loads(args.remote_preparation.read_text())
        if remote['sha256'] != result['sha256']:
            raise ValueError('local and remote weighted plans differ')
        write_new(args.write_config, make_config(remote))
    print(json.dumps(result, indent=2))


def make_config(preparation):
    import copy
    from hardcase_adaptation_protocol import ROOT
    original_path = ROOT/'model_training/experiments/anchor_full_20261030_v1.json'
    original = json.loads(original_path.read_text())
    config = copy.deepcopy(original)
    config['family'] = SCOPE
    config['source_recipe'] = {'relative_path': str(original_path.relative_to(ROOT)).replace('\\', '/'),
        'sha256': sha256(original_path)}
    config['runtime'].update(allocation=131170166, keeper_pid=2268188,
        stop_before_utc='2026-11-01T11:40:04Z', required_h_rt_seconds=86400)
    config['backbone'].update(initialization='full300k_complete_checkpoint', source_asset_exposures=300000,
        checkpoint={'path': '/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v26/checkpoint_300000.pt',
            'sha256': SOURCE_SHA, 'bytes': 8278702930})
    config['training'].update(additional_assets=30000, checkpoint_assets=[5000, 10000, 20000, 30000],
        stream_seed=20261101, regime='joint_hardcase_adaptation', warmup_assets=1,
        optimizer='retain_full300k_AdamW_all_groups', backbone_exposure_offset=300000,
        optimization=OPTIMIZATION, resume=None)
    config['sampling'] = {'path': preparation['path'], 'sha256': preparation['sha256'],
        'selection': preparation['selection'], 'counts': COUNTS, 'weights': WEIGHTS}
    config['evaluation'].update(selection_assets=112, xyz_interval_assets=5000, ar_interval_assets=5000)
    config['artifacts'].update(durable_root='/home/wangyy/evorig/outputs/hardcase3608_20261101/v1',
        retention={'keep_last_full': 4, 'full_milestones': [5000, 10000, 20000, 30000],
            'weight_milestones': [5000, 10000, 20000, 30000],
            'keep_best_stage1': True, 'keep_best_stage2': True})
    config['execution'] = {'phase': 'hardcase3608', 'CPU_audit': None, 'capacity': None,
        'retained_cache_audit': original['execution']['CPU_audit']}
    return config


if __name__ == '__main__':
    main()
