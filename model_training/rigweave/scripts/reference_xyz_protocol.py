"""CPU-only, bounded contract for the first reference-skeleton XYZ pilot."""
import hashlib
import json
from pathlib import Path

from fit_cohort_selection import _joint_count

SCOPE = 'reference_skeleton_xyz_20261027_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_reference_xyz.py'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_backbone_config(config):
    spec = config['backbone']['config']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('retained backbone configuration changed')
    return json.loads(Path(spec['path']).read_text())


def validate_config(config):
    if config.get('family') != SCOPE:
        raise ValueError('not the reference-XYZ pilot')
    pilot = config['pilot']
    expected = {'train_assets': 64, 'valid_assets': 32, 'frames': 8, 'updates': 800,
                'batch_per_rank': 2, 'seed': 20261027, 'reference_policy': 'uniform_per_visit',
                'backbone_mode': 'frozen', 'loss': 'continuous_L1_nonreference_mean_per_asset',
                'evaluation_reference_slots': [0, 4], 'local_rigid_neighbors': 32,
                'coordinate_clipping': False}
    for key, value in expected.items():
        if type(pilot.get(key)) is not type(value) or pilot[key] != value:
            raise ValueError(f'pilot contract differs: {key}')
    if pilot.get('learning_rate') != .0003 or pilot.get('weight_decay') != .0001:
        raise ValueError('unreviewed head optimizer')
    if config['backbone']['checkpoint']['sha256'] != '9704b23a194ca65836c6dfacd6637ca27542dc4b0836b1d3611185f0339cc888':
        raise ValueError('pilot must use the retained random20-40 checkpoint')
    if config['backbone']['config']['sha256'] != '53c03ec4052abc2f811a00641b7ceaaf1a625250de7a89fbecde294580d98324':
        raise ValueError('pilot must use the exact retained backbone recipe')
    if config['architecture'] != {'input_dim': 1024, 'dim': 256, 'heads': 8,
                                 'reference_depth': 2, 'target_depth': 2, 'max_joints': 512}:
        raise ValueError('unreviewed joint readout architecture')
    if config['runtime']['world_size'] != 2 or config['runtime']['execution_mode'] != 'scheduler_job':
        raise ValueError('pilot requires the existing scheduler-owned dual-GPU grant')


def stratum(count):
    if count <= 0:
        raise ValueError('joint count metadata must be positive')
    return 'low' if count <= 20 else 'mid' if count < 52 else 'gt52' if count == 52 else 'high'


def select_cohort(rows, split, count, seed):
    if count % 4 or split not in ('train', 'valid'):
        raise ValueError('cohort requires four equal joint-count strata')
    pools = {name: [] for name in ('low', 'mid', 'gt52', 'high')}
    ids, paths = set(), set()
    for row in rows:
        asset, path = row['asset_id'], row['path']
        if asset in ids or path in paths:
            raise ValueError('cohort source has duplicate asset IDs or paths')
        ids.add(asset)
        paths.add(path)
        if any(row.get(key, split) != split for key in ('split', 'source_split')):
            raise ValueError('cohort row has the wrong split')
        joints, _ = _joint_count(row)
        group = stratum(joints)
        key = hashlib.sha256(f'{SCOPE}|{seed}|{split}|{asset}'.encode()).hexdigest()
        pools[group].append((key, row, joints))
    selected, records = [], []
    for group, pool in pools.items():
        if len(pool) < count // 4:
            raise ValueError(f'insufficient {split}/{group} assets')
        for _, row, joints in sorted(pool, key=lambda item: item[0])[:count // 4]:
            selected.append(row)
            records.append({'asset_id': row['asset_id'], 'path': row['path'],
                            'joint_count': joints, 'stratum': group})
    return selected, records
