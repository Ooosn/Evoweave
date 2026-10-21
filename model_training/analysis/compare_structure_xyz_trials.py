"""Read saved matched XYZ-fill evaluations without rerunning or changing models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(rows):
    assets = sorted({row['asset_id'] for row in rows})
    if not rows:
        return {'assets': 0, 'views': 0}
    result = {'assets': len(assets), 'views': len(rows)}
    for name in ('original_edge_f1', 'gt_structure_xyz_ce', 'gt_structure_xyz_accuracy',
                 'gt_structure_xyz_l1', 'gt_structure_edge_f1', 'success', 'exact_count'):
        result[name] = float(np.mean([row[name] for row in rows]))
    successful = [row for row in rows if row['success']]
    result['successful_mean_abs_joint_count_error'] = (float(np.mean([
        abs(len(row['predicted_parents']) - len(row['original_target_parents'])) for row in successful]))
        if successful else None)
    result['reference_B65_F1'] = float(np.mean([row['reference_original_edge_f1'] for row in rows]))
    return result


def load_evaluation(directory):
    summary = read_json(directory / 'summary.json')
    if not summary['complete']:
        raise ValueError('only complete evaluations may be compared')
    ranks = [read_json(path) for path in sorted(directory.glob('rank_*.json'))]
    if len(ranks) != 2 or {rank['rank'] for rank in ranks} != {0, 1}:
        raise ValueError('evaluation requires exactly both declared ranks')
    if any(not rank['complete'] or rank['asset_exposures'] != summary['asset_exposures'] for rank in ranks):
        raise ValueError('rank completion/counter does not match the summary')
    groups, identities = {}, {}
    for group, count in summary['evaluation_assets'].items():
        rows = [row for rank in ranks for row in rank['groups'][group]['rows']]
        keys = [(row['asset_id'], row['frame_index']) for row in rows]
        if len(keys) != len(set(keys)) or len(keys) != count * len(summary['scored_frame_indices']):
            raise ValueError('evaluation rows are missing or duplicated')
        identities[group] = dict(zip(keys, rows))
        groups[group] = {label: summarize([row for row in rows if predicate(row)]) for label, predicate in (
            ('all', lambda row: True),
            ('joints52', lambda row: len(row['original_target_parents']) == 52),
            ('non52', lambda row: len(row['original_target_parents']) != 52),
        )}
    return {'asset_exposures': summary['asset_exposures'], 'panel_sha256': summary['panel_sha256'],
            'groups': groups, 'source_summary_sha256': hashlib.sha256((directory / 'summary.json').read_bytes()).hexdigest()}, identities


def compare(left, right):
    output = {}
    if set(left) != set(right):
        raise ValueError('comparison groups differ')
    for group in left:
        if set(left[group]) != set(right[group]):
            raise ValueError('comparison assets or frame slots differ')
        per_asset = {}
        for key, row in left[group].items():
            other = right[group][key]
            for field in ('raw_frame_index', 'original_target_joints', 'original_target_parents', 'target_tokens'):
                if row[field] != other[field]:
                    raise ValueError(f'paired GT/frame/structure differs: {group}/{key}/{field}')
            per_asset.setdefault(row['asset_id'], []).append(row['original_edge_f1'] - other['original_edge_f1'])
        differences = np.asarray([np.mean(values) for values in per_asset.values()], dtype=np.float64)
        output[group] = {'assets': len(differences), 'mean_F1_left_minus_right': float(differences.mean()),
                         'median_F1_left_minus_right': float(np.median(differences)),
                         'left_better_by_over_001': int((differences > .01).sum()),
                         'right_better_by_over_001': int((differences < -.01).sum()),
                         'within_001': int((np.abs(differences) <= .01).sum())}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=Path, required=True, help='Completed evaluation directory')
    parser.add_argument('--right', type=Path, help='Matched evaluation directory at the same exposure')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    left, left_rows = load_evaluation(args.left)
    result = {'left': left, 'scope': 'Small fixed-panel observations, not a full-validation or significance claim.'}
    if args.right:
        right, right_rows = load_evaluation(args.right)
        if (left['asset_exposures'], left['panel_sha256']) != (right['asset_exposures'], right['panel_sha256']):
            raise ValueError('comparison exposure or panel differs')
        result.update(right=right, paired=compare(left_rows, right_rows))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError('preserve existing comparison artifacts')
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
