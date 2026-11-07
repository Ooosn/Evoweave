"""Reaggregate existing validations without mixing mismatched input panels."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


IDENTITY = ('raw_frame_index', 'normalization_sha256', 'surface_sha256',
            'original_GT_sha256', 'target_mesh_diagonal', 'gt_joint_count')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def balanced(rows, metric):
    assets = defaultdict(list)
    for row in rows:
        value = row[metric]
        if not math.isfinite(value):
            raise ValueError('nonfinite comparison metric')
        assets[row['asset_id']].append(value)
    if not assets:
        raise ValueError('empty comparison')
    return sum(sum(v) / len(v) for v in assets.values()) / len(assets)


def strict_assets(panels):
    keyed = {name: {(r['asset_id'], r['frame_index']): r for r in rows}
             for name, rows in panels.items()}
    if any(len(keyed[name]) != len(rows) for name, rows in panels.items()):
        raise ValueError('duplicate development view')
    shared = set.intersection(*({r['asset_id'] for r in rows} for rows in panels.values()))
    accepted, excluded = [], {}
    first = next(iter(keyed.values()))
    for asset in sorted(shared):
        reasons = set()
        for frame in range(8):
            key = asset, frame
            if any(key not in rows for rows in keyed.values()):
                reasons.add('missing_view')
                continue
            for field in IDENTITY:
                if any(rows[key][field] != first[key][field] for rows in keyed.values()):
                    reasons.add(field)
        if reasons:
            excluded[asset] = sorted(reasons)
        else:
            accepted.append(asset)
    return accepted, excluded


def matrix(config):
    panels, targets, sources, full = {}, {}, {}, {}
    for name, root in config['development_panels'].items():
        root = Path(root)
        paths = [root / f for f in ('summary.json', 'stage1_rows.json', 'rows.json')]
        summary, ar, rows = [read(p) for p in paths]
        if not summary['complete'] or summary['precision'] != 'fp32_noTF32':
            raise ValueError('incomplete or different-precision development validation')
        if any(not r['success'] and r['original_edge_f1'] != 0 for r in ar):
            raise ValueError('failed generation received quality')
        head = [r for r in rows if r['split'] == 'valid' and r['method'] == 'head']
        if any(r['reference_kind'] != 'GT' for r in head):
            raise ValueError('development reference kind differs')
        panels[name], targets[name] = ar, head
        sources[name] = {str(p): digest(p) for p in paths}
        full[name] = {'assets': len({r['asset_id'] for r in ar}), 'views': len(ar),
            'stage1_edge_F1': balanced(ar, 'original_edge_f1'),
            'GT_MPJPE': balanced(head, 'normalized_mpjpe'),
            'GT_edge_F1': balanced(head, 'edge_f1')}
    scopes = {}
    for scope, labels in config['matrix_scopes'].items():
        ids, excluded = strict_assets({n: panels[n] for n in labels})
        if not ids:
            raise ValueError('no strictly matched development assets')
        values, keys = {}, None
        for label in labels:
            ar = [r for r in panels[label] if r['asset_id'] in ids]
            head = [r for r in targets[label] if r['asset_id'] in ids]
            current = {(r['asset_id'], r['reference_slot'], r['target_slot']):
                (r['surface_sha256'], r['reference_raw_frame'], r['target_raw_frame'], r['joint_count'])
                for r in head}
            if len(head) != len(ids) * 14 or len(current) != len(head) or (keys is not None and current != keys):
                raise ValueError('strict GT target pairs differ')
            keys = current
            values[label] = {'stage1_edge_F1': balanced(ar, 'original_edge_f1'),
                'GT_MPJPE': balanced(head, 'normalized_mpjpe'),
                'GT_edge_F1': balanced(head, 'edge_f1')}
        scopes[scope] = {'assets': len(ids), 'stage1_views': len(ids) * 8,
            'GT_target_pairs': len(ids) * 14, 'asset_ids': ids,
            'excluded_input_mismatch': excluded, 'metrics': values}
    return {'complete': True, 'full_panel_means_not_necessarily_paired': full,
        'strictly_paired_scopes': scopes, 'sources_sha256': sources,
        'decoding': 'greedy_fp32_noTF32', 'stage2_reference': 'GT',
        'not_official_15beam_pipeline_scores': True}
