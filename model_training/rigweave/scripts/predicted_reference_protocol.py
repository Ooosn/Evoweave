"""Inference-only GT/predicted-reference comparison on the existing panel."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from stage1_feedback_analysis import interval
from reference_flow_protocol import sha256

SCOPE = 'predicted_reference_20261029_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/evaluate_predicted_reference.py'
ROOT = Path(__file__).resolve().parents[3]
METHODS = ('GT_head', 'GT_copy', 'predicted_head', 'predicted_copy')
PAIR_FIELDS = ('surface_sha256', 'normalization_sha256', 'raw_frame_index',
               'reference_raw_frame', 'original_GT_sha256', 'target_mesh_diagonal')


def scientific_sha(config):
    keys = ('family', 'parent_config', 'stage2_config', 'checkpoints', 'stage1_source',
            'canonical_source', 'evaluation', 'capacity', 'prior_source')
    values = {k: config[k] for k in keys}
    if 'official_smoke' in config:
        values['official_smoke'] = config['official_smoke']
    if 'full_model_inputs' in config:
        values['full_model_inputs'] = config['full_model_inputs']
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def bound_config(spec):
    path = ROOT / spec['relative_path']
    if sha256(path) != spec['sha256']:
        raise ValueError('bound parent config changed')
    return json.loads(path.read_text())


def validate_config(config, mode):
    full_model = 'full_model_inputs' in config
    expected = {'assets': 128, 'input_frames': 8, 'reference_slots': [0, 4],
        'methods': list(METHODS), 'precision': 'fp32', 'allow_tf32': False,
        'decoder_attention': 'eager', 'max_new_tokens': 2048, 'max_joints': 512,
        'optimizer_updates': 0, 'bootstrap_seed': 20261029, 'bootstrap_repeats': 10000,
        'stage1_cache_policy': 'reuse_only_when_all_input_identities_match_else_generate_same_surface',
        'fitting': False, 'GT_reference_repair': False, 'payload_limit_seconds': 5400}
    if 'official_smoke' in config:
        from stage1_official_inference import OFFICIAL_OPTIONS, POLICY
        smoke = config['official_smoke']
        expanded = smoke['case_indices'] == [5, 15, 23, 24, 30, 44, 54, 61, 70, 95, 105, 108]
        expected.update(assets=12 if expanded else 4, reference_slots=[0], payload_limit_seconds=1800,
                        stage1_cache_policy='regenerate_all_official_no_greedy_reuse')
        if ((not expanded and smoke['case_indices'] != [2, 3, 64, 127]) or smoke['policy'] != POLICY
                or smoke['generation_kwargs'] != OFFICIAL_OPTIONS or smoke['base_seed'] != 20261029
                or smoke['static_fallback'] or smoke['full_beam_capacity_first'] != (not expanded or full_model)):
            raise ValueError('unreviewed official inference recipe')
        if expanded and ((not full_model and not smoke.get('inherited_official_capacity'))
                         or smoke.get('visual_point_limit') != 4096):
            raise ValueError('expanded visual run needs accepted official capacity and fixed export size')
    if mode not in ('audit', 'capacity') or config.get('family') != SCOPE or config.get('evaluation') != expected:
        raise ValueError('unreviewed predicted-reference evaluation')
    profiles = 0 if 'official_smoke' in config else 3
    if set(config['checkpoints']) != {'stage1', 'stage2'} or config['capacity']['profile_assets_per_rank'] != profiles:
        raise ValueError('checkpoint or profile scope differs')
    if full_model:
        specs = config['checkpoints']
        expected_sha = 'bef030ac403cd26d94f493d868bf426f694f047ed93327e1df51b541fe52ba4d'
        if ('official_smoke' not in config or not expanded
                or specs['stage1'] != specs['stage2'] or specs['stage1']['label'] != 'full_joint_300000'
                or specs['stage1']['sha256'] != expected_sha or specs['stage1']['layout'] != 'anchor'
                or smoke.get('inherited_official_capacity') is not None
                or config['capacity']['full_official_beam_decode_per_rank'] != 1
                or config['parent_config'] != config['stage2_config']):
            raise ValueError('full300k pipeline checkpoint or capacity contract differs')
    elif config['checkpoints']['stage1']['label'] != 'joint_080000' or config['checkpoints']['stage2']['label'] != 'pretrained_stage2_080000':
        raise ValueError('primary pipeline model selection differs')
    bound_config(config['stage2_config'])
    return bound_config(config['parent_config'])


def reference_from_row(row, max_joints=512):
    if not row['success']:
        return None, 'generation_' + row['status']
    xyz = np.asarray(row['predicted_joints'], dtype=np.float32)
    parents = [(-1 if p is None else p) for p in row['predicted_parents']]
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not 1 <= len(xyz) <= max_joints or not np.isfinite(xyz).all():
        return None, 'invalid_reference_coordinates'
    if (len(parents) != len(xyz) or any(type(p) is not int for p in parents)
            or parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(parents[1:], 1))):
        return None, 'invalid_reference_tree'
    return (xyz, parents), 'valid'


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not row['success'] and (row['edge_f1'] != 0 or row['exact_count'] or row['normalized_j2j'] is not None):
            raise ValueError('failed prediction received quality')
        grouped[row['asset_id']].append(row)
    if not grouped:
        raise ValueError('empty evaluation')
    result = {k: float(np.mean([np.mean([r[k] for r in values]) for values in grouped.values()]))
              for k in ('success', 'edge_f1', 'exact_count')}
    geometry = [[r['normalized_j2j'] for r in values if r['success']] for values in grouped.values()]
    valid = [v for v in geometry if v]
    known = [[r['normalized_mpjpe'] for r in values if r.get('normalized_mpjpe') is not None] for values in grouped.values()]
    known = [v for v in known if v]
    result.update(assets=len(grouped), views=len(rows), failures=sum(not r['success'] for r in rows),
        normalized_j2j_successful=float(np.mean([np.mean(v) for v in valid])) if valid else None,
        geometry_assets=len(valid), geometry_views=sum(map(len, valid)),
        normalized_mpjpe_known_id=float(np.mean([np.mean(v) for v in known])) if known else None)
    return result


def compare(left, right):
    def keyed(rows):
        value = {(r['asset_id'], r['reference_slot'], r['target_slot']): r for r in rows}
        if len(value) != len(rows):
            raise ValueError('duplicate comparison row')
        return value
    a, b = keyed(left), keyed(right)
    if not a or a.keys() != b.keys():
        raise ValueError('paired coverage differs')
    differences = {key: defaultdict(list) for key in ('success', 'edge_f1', 'exact_count', 'normalized_j2j')}
    count = 0
    for key, x in a.items():
        y = b[key]
        if any(x[f] != y[f] for f in PAIR_FIELDS):
            raise ValueError('paired mesh/GT/frame/normalization differs')
        if x['reference_kind'] == y['reference_kind'] and x['reference_sha256'] != y['reference_sha256']:
            raise ValueError('same-kind references differ')
        for metric in ('success', 'edge_f1', 'exact_count'):
            differences[metric][key[0]].append(float(x[metric]) - float(y[metric]))
        if x['success'] and y['success']:
            differences['normalized_j2j'][key[0]].append(x['normalized_j2j'] - y['normalized_j2j'])
            count += 1
    return {'paired_views': len(a), 'geometry_jointly_successful_views': count,
        'metrics': {metric: interval([np.mean(v) for _, v in sorted(values.items())]) for metric, values in differences.items()},
        'scope': 'Asset-equal; geometry only identical views successful in both arms; 10000 asset bootstrap; one training seed.'}
