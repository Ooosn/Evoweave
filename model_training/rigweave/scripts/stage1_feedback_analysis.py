"""Identity-bound, paired-asset first-stage comparisons."""
from collections import defaultdict
import copy

import numpy as np


def pair_rows(left, right, left_identity, right_identity):
    identities = []
    for identity in (left_identity, right_identity):
        item = copy.deepcopy(identity)
        item.pop('checkpoint_sha256')
        item.pop('checkpoint_layout')
        identities.append(item)
    if identities[0] != identities[1]:
        raise ValueError('evaluation protocol identities differ')
    def keyed(rows):
        result = {(r['asset_id'], r['frame_index']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate comparison rows')
        return result
    a, b = keyed(left), keyed(right)
    if not a or a.keys() != b.keys():
        raise ValueError('comparison coverage differs')
    result = []
    for key in sorted(a):
        for field in ('raw_frame_index', 'surface_sha256', 'normalization_sha256',
                      'original_GT_sha256', 'target_mesh_diagonal', 'gt_joint_count'):
            if a[key][field] != b[key][field]:
                raise ValueError('paired inputs differ: '+field)
        result.append((a[key], b[key]))
    return result


def interval(values, *, seed=20261029, repeats=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'assets': 0, 'mean_left_minus_right': None, 'paired_asset_95CI': None}
    if not np.isfinite(values).all():
        raise ValueError('nonfinite paired differences')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(1)
    return {'assets': len(values), 'mean_left_minus_right': float(values.mean()),
            'paired_asset_95CI': np.quantile(samples, [.025, .975]).tolist()}


def compare(left, right, left_identity, right_identity):
    pairs = pair_rows(left, right, left_identity, right_identity)
    metrics = ('original_edge_f1', 'success', 'exact_count', 'original_normalized_j2j')
    differences = {metric: defaultdict(list) for metric in metrics}
    shared_successes = 0
    for a, b in pairs:
        asset = a['asset_id']
        for metric in metrics[:3]:
            differences[metric][asset].append(float(a[metric])-float(b[metric]))
        if a['success'] and b['success']:
            shared_successes += 1
            differences[metrics[-1]][asset].append(a[metrics[-1]]-b[metrics[-1]])
    result = {metric: interval([np.mean(v) for v in assets.values()])
              for metric, assets in differences.items()}
    result.update(paired_views=len(pairs), geometry_jointly_successful_views=shared_successes,
        geometry_scope='Only views successful in BOTH arms; average within asset, then equal-weight assets. '
            'Symmetric nearest-joint distance divided by target mesh bbox diagonal, not MPJPE.',
        bootstrap='10000 paired-asset resamples; percentile95%; one training seed, not seed uncertainty')
    return result
