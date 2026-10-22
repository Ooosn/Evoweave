"""Score the original B65 panel once and project the unchanged small panel."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


GROUP_SIZES = {'train_seen': 32, 'train_heldout': 32, 'valid': 128}
SMALL_GROUP_SIZES = {'train_seen': 8, 'train_heldout': 8, 'valid': 16}
LARGE_EVALUATION = {'train_assets': 32, 'valid_assets': 128, 'input_frames': 8,
                    'frame_indices': list(range(8)), 'seed': 20261022}
SMALL_EVALUATION = {'train_assets': 8, 'valid_assets': 16, 'input_frames': 8,
                    'frame_indices': [0, 2, 4, 6], 'seed': 20261022}


def _expanded_config(config):
    declared = config.get('large_evaluation')
    expected = {'enabled': True, **LARGE_EVALUATION,
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (not isinstance(declared, dict) or declared != expected
            or declared.get('enabled') is not True
            or any(type(declared.get(key)) is not int
                   for key in ('train_assets', 'valid_assets', 'input_frames', 'seed'))
            or any(type(frame) is not int for frame in declared.get('frame_indices', []))):
        raise ValueError('large_evaluation size, frames, seed or implementation_sha256 differs')
    if config.get('evaluation') != SMALL_EVALUATION:
        raise ValueError('the original small evaluation contract must remain unchanged')
    expanded = deepcopy(config)
    expanded['evaluation'] = deepcopy(LARGE_EVALUATION)
    return expanded


def _panel_ids(panel, counts):
    if set(panel) != set(counts):
        raise ValueError('reference panel groups differ')
    identities = {}
    for group, count in counts.items():
        rows = panel[group]
        ids = [row['asset_id'] for row in rows]
        if len(rows) != count or len(set(ids)) != count:
            raise ValueError('reference panel has missing or duplicate assets')
        identities[group] = set(ids)
    if (panel['train_seen'] != panel['train_heldout']
            or identities['train_seen'] & identities['valid']):
        raise ValueError('reference panel train/heldout/valid membership differs')
    return identities


def _validate_reference(panel, reference, counts, frames):
    identities = _panel_ids(panel, counts)
    expected = {(group, asset, frame) for group, assets in identities.items()
                for asset in assets for frame in frames}
    if set(reference) != expected:
        raise ValueError('reference has missing, duplicate or unexpected asset/frame keys')
    raw_frames = {}
    for key, row in reference.items():
        group, asset, frame = key
        if (row.get('asset_id') != asset or row.get('frame_index') != frame
                or type(row.get('frame_index')) is not int
                or row.get('group', group) != group
                or type(row.get('raw_frame_index')) is not int
                or not isinstance(row.get('target_tokens'), list) or not row['target_tokens']
                or any(type(token) is not int for token in row['target_tokens'])
                or not math.isfinite(float(row.get('original_edge_f1', float('nan'))))):
            raise ValueError('reference group, asset, frame or raw-GT identity differs')
        raw_frames.setdefault((group, asset), []).append(row['raw_frame_index'])
    if any(len(values) != len(set(values)) for values in raw_frames.values()):
        raise ValueError('reference repeats a raw frame within an asset/group')
    for asset in identities['train_seen']:
        if set(raw_frames['train_seen', asset]) & set(raw_frames['train_heldout', asset]):
            raise ValueError('reference crosses the declared heldout partition')


def load_large_reference_panel(config, entries):
    """Return (panel, reference, manifest), using the frozen loader on a copy."""
    from structure_xyz_fill_runtime import load_small_reference_panel

    expanded = _expanded_config(config)
    panel, reference, manifest = load_small_reference_panel(expanded, entries)
    _validate_reference(panel, reference, GROUP_SIZES, LARGE_EVALUATION['frame_indices'])
    selection = json.loads(Path(config['reference_evaluation']['files']['selection']['path']).read_text())
    for group, count in GROUP_SIZES.items():
        selected = selection[group]
        by_id = {row['asset_id']: row for row in selected}
        if (len(selected) != count or len(by_id) != count
                or set(by_id) != {row['asset_id'] for row in panel[group]}):
            raise ValueError('large panel differs from the original B65 selection')
        for row in panel[group]:
            if any(row[field] != by_id[row['asset_id']][field] for field in ('path', 'root_rt')):
                raise ValueError('large panel changed an original manifest/raw-GT descriptor')
    return panel, reference, manifest


def _empty_result():
    # Local projections can be empty; zero denominators must not become scores.
    summary = dict.fromkeys((
        'success', 'mean_original_edge_f1', 'mean_gt_structure_xyz_ce',
        'mean_gt_structure_xyz_accuracy', 'mean_gt_structure_xyz_l1',
        'mean_gt_structure_edge_f1', 'mean_j2j', 'mean_j2j_successful',
        'j2j_coverage', 'mean_structure_dfs_edge_f1', 'exact_count',
        'gt_structure_xyz_ce', 'gt_structure_xyz_accuracy', 'gt_structure_xyz_l1',
        'original_edge_f1', 'mean_reference_original_edge_f1'))
    summary.update(asset_count=0, score_frame_count=0, success_count=0, failure_count=0,
                   counts={'assets': 0, 'rows': 0, 'successes': 0, 'failures': 0})
    return {'rows': [], 'structure_rows': [], 'summary': summary}


def _validate_result(result, group, selected, frames, reference):
    assets = {row['asset_id'] for row in selected}
    expected = {(asset, frame) for asset in assets for frame in frames}
    rows, structures = result['rows'], result['structure_rows']
    actual = [(row['asset_id'], row['frame_index']) for row in rows]
    structure_ids = [row['asset_id'] for row in structures]
    if (len(actual) != len(expected) or set(actual) != expected
            or len(structure_ids) != len(assets) or set(structure_ids) != assets):
        raise RuntimeError('evaluation missed or duplicated an asset/view/structure')
    if any(row.get('group') != group for row in rows + structures):
        raise RuntimeError('evaluation row belongs to the wrong group')
    for row in rows:
        key = (group, row['asset_id'], row['frame_index'])
        original = reference[key]
        if row['raw_frame_index'] != original['raw_frame_index']:
            raise RuntimeError('evaluation raw frame differs from the pinned reference')
        # New target_tokens are structure IDs, not the reference's AR XYZ tokens.
        # The existing evaluator checks those AR tokens against freshly loaded GT.
        for field in ('original_target_joints', 'original_target_parents'):
            if field in original and row.get(field) != original[field]:
                raise RuntimeError('evaluation raw GT differs from the pinned reference')
        score = float(row['original_edge_f1'])
        if not math.isfinite(score) or (not row['success'] and score != 0):
            raise RuntimeError('evaluation changed finite/failed-generation F1 handling')


def _project_small_result(result, group, selected, reference):
    from structure_xyz_fill_evaluation import summarize_fill_rows

    assets = {row['asset_id'] for row in selected}
    frames = SMALL_EVALUATION['frame_indices']
    rows = [row for row in result['rows'] if row['asset_id'] in assets and row['frame_index'] in frames]
    structures = [row for row in result['structure_rows'] if row['asset_id'] in assets]
    projected = {'rows': rows, 'structure_rows': structures,
                 'summary': summarize_fill_rows(rows, structures) if rows else _empty_result()['summary']}
    _validate_result(projected, group, selected, frames, reference)
    return projected


def _summary(groups, samples, panel, panel_sha256, evaluation):
    return {'complete': True, 'asset_exposures': samples, 'groups': groups,
            'panel_sha256': panel_sha256, 'input_frames': evaluation['input_frames'],
            'scored_frame_indices': evaluation['frame_indices'],
            'evaluation_assets': {group: len(rows) for group, rows in panel.items()},
            'full_validation': False, 'failed_generations_scored_zero': True,
            'quality_acceptance': False}


def _aggregate(large_reports, small_reports, large_panel, panel, large_reference, reference,
               samples, large_hash, small_hash):
    from structure_xyz_fill_evaluation import summarize_fill_rows

    world = len(large_reports)
    if len(small_reports) != world:
        raise RuntimeError('large and small reports have different rank counts')
    for rank, (large, small) in enumerate(zip(large_reports, small_reports)):
        for report, digest in ((large, large_hash), (small, small_hash)):
            if (report['complete'] is not True or report['rank'] != rank
                    or report['asset_exposures'] != samples or report['panel_sha256'] != digest
                    or set(report['groups']) != set(GROUP_SIZES)):
                raise RuntimeError('rank did not complete this exact evaluation panel')
        for group in GROUP_SIZES:
            selected = large_panel[group][rank::world]
            small_ids = {row['asset_id'] for row in panel[group]}
            small_selected = [row for row in selected if row['asset_id'] in small_ids]
            _validate_result(large['groups'][group], group, selected,
                             LARGE_EVALUATION['frame_indices'], large_reference)
            projected = _project_small_result(large['groups'][group], group, small_selected, reference)
            if small['groups'][group] != projected:
                raise RuntimeError('saved small-panel projection changed large-panel values')
    large_groups, small_groups = {}, {}
    for reports, source_panel, refs, frames, groups in (
            (large_reports, large_panel, large_reference, LARGE_EVALUATION['frame_indices'], large_groups),
            (small_reports, panel, reference, SMALL_EVALUATION['frame_indices'], small_groups)):
        for group in GROUP_SIZES:
            rows = [row for report in reports for row in report['groups'][group]['rows']]
            structures = [row for report in reports for row in report['groups'][group]['structure_rows']]
            _validate_result({'rows': rows, 'structure_rows': structures}, group, source_panel[group], frames, refs)
            groups[group] = summarize_fill_rows(rows, structures)
    return (_summary(large_groups, samples, large_panel, large_hash, LARGE_EVALUATION),
            _summary(small_groups, samples, panel, small_hash, SMALL_EVALUATION))


def run_large_evaluation(model, tokenizer, codec, config, args, device, rank, world, samples,
                         panel, reference, panel_sha256):
    """Write both panels; return the legacy small summary on rank 0, else None."""
    import torch.distributed as dist
    from structure_xyz_fill_evaluation import evaluate_fill_items
    from structure_xyz_fill_runtime import (
        fixed_dataset, load_small_reference_panel, prepare_fill_batch, read_entries, write_json,
    )

    _expanded_config(config)
    if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('invalid evaluation rank/world')
    if type(samples) is not int or samples < 0:
        raise ValueError('evaluation asset exposure count must be a nonnegative integer')
    entries = read_entries(config)
    large_panel, large_reference, large_manifest = load_large_reference_panel(config, entries)
    expected_panel, expected_reference, small_manifest = load_small_reference_panel(config, entries)
    if (panel != expected_panel or reference != expected_reference
            or panel_sha256 != small_manifest['panel_sha256']):
        raise ValueError('supplied small panel/reference/hash differs from the frozen selection')
    _validate_reference(panel, reference, SMALL_GROUP_SIZES, SMALL_EVALUATION['frame_indices'])
    if any(key not in large_reference or row != large_reference[key] for key, row in reference.items()):
        raise ValueError('small raw-GT reference is not an exact subset of the large panel')
    large_hash = large_manifest['panel_sha256']
    large_root = Path(args.output) / f'evaluation_large_{samples:06d}'
    small_root = Path(args.output) / f'evaluation_{samples:06d}'
    large_local = {'rank': rank, 'asset_exposures': samples, 'groups': {}, 'complete': False,
                   'panel_sha256': large_hash}
    small_local = {'rank': rank, 'asset_exposures': samples, 'groups': {}, 'complete': False,
                   'panel_sha256': panel_sha256}
    was_training = bool(model.training)
    model.eval()
    try:
        for group, rows in large_panel.items():
            selected = rows[rank::world]
            if selected:
                dataset = fixed_dataset(selected, config, LARGE_EVALUATION['input_frames'],
                                        mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')

                def items():
                    for index in range(len(dataset)):
                        surface, targets, geometry, _ = prepare_fill_batch(dataset, [index], config, codec, device)
                        actual = set(geometry['selected_frames'][0].tolist())
                        reserved = set(geometry['heldout_raw_ids'][0])
                        if (len(actual) != 8 or (group == 'train_heldout' and actual != reserved)
                                or (group != 'train_heldout' and actual & reserved)):
                            raise ValueError('evaluation crossed the declared heldout partition')
                        yield surface, targets, geometry

                keys = {(group, row['asset_id'], frame) for row in selected for frame in range(8)}
                result = evaluate_fill_items(model, items(), codec, frame_indices=LARGE_EVALUATION['frame_indices'],
                    group=group, original_tokenizer=tokenizer,
                    reference_lookup={key: large_reference[key] for key in keys},
                    structure_weight=config['optimization']['structure_weight'], **config['quantization'])
            else:
                result = _empty_result()
            _validate_result(result, group, selected, LARGE_EVALUATION['frame_indices'], large_reference)
            small_ids = {row['asset_id'] for row in panel[group]}
            small_selected = [row for row in selected if row['asset_id'] in small_ids]
            large_local['groups'][group] = result
            small_local['groups'][group] = _project_small_result(result, group, small_selected, reference)
            write_json(large_root / f'rank_{rank}.json', large_local)
            write_json(small_root / f'rank_{rank}.json', small_local)
        large_local['complete'] = small_local['complete'] = True
        write_json(large_root / f'rank_{rank}.json', large_local)
        write_json(small_root / f'rank_{rank}.json', small_local)
        if world > 1:
            dist.barrier()
        summary = None
        if rank == 0:
            large_reports = [json.loads((large_root / f'rank_{number}.json').read_text()) for number in range(world)]
            small_reports = [json.loads((small_root / f'rank_{number}.json').read_text()) for number in range(world)]
            large_summary, summary = _aggregate(large_reports, small_reports, large_panel, panel,
                large_reference, reference, samples, large_hash, panel_sha256)
            write_json(large_root / 'summary.json', large_summary)
            write_json(small_root / 'summary.json', summary)
            print('EVALUATION_LARGE ' + json.dumps(large_summary, allow_nan=False), flush=True)
            print('EVALUATION ' + json.dumps(summary, allow_nan=False), flush=True)
        if world > 1:
            dist.barrier()
        return summary
    finally:
        model.train(was_training)
