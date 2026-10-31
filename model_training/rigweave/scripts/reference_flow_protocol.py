"""Exact full-5000 shared-backbone flow experiment contract."""
import copy
import hashlib
import json
from pathlib import Path

from reference_xyz_protocol import sha256

SCOPE = 'shared_reference_flow_5k_20261027_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_reference_flow.py'
ROOT = Path(__file__).resolve().parents[3]
DONOR_SHA = '9704b23a194ca65836c6dfacd6637ca27542dc4b0836b1d3611185f0339cc888'


def load_backbone_config(config):
    spec = config['backbone']['config']
    path = ROOT / spec['relative_path']
    if sha256(path) != spec['sha256']:
        raise ValueError('backbone configuration changed')
    base = json.loads(path.read_text())
    base['runtime'] = copy.deepcopy(config['runtime'])
    base['data']['stream_seed'] = config['training']['stream_seed']
    base['training']['target_assets'] = config['training']['additional_assets']
    base['training']['checkpoint_assets'] = config['training']['checkpoint_assets']
    base['training']['frame_policy'] = config['training']['frame_policy']
    for case in base['capacity']['cases']:
        case['batch_per_rank'] = base['training']['frame_policy'][str(case['frames'])]
    return base


def validate_config(config):
    if config.get('family') != SCOPE or config.get('schema_version') != 1:
        raise ValueError('unexpected shared flow family')
    if (config['backbone']['checkpoint']['sha256'] != DONOR_SHA
            or config['backbone']['source_asset_exposures'] != 20000
            or config['training']['additional_assets'] != 20000
            or config['training']['checkpoint_assets'] != [5000, 10000, 15000, 20000]
            or config['training']['effective_assets_per_update'] != 48
            or config['training']['loss'] != 'AR_plus_linear_flow_velocity_MSE'
            or config['training']['flow_weight'] != 1.
            or config['training']['flow_warmup_assets'] != 1000
            or config['training']['reference'] != 'uniform_GT_complete_tree_per_asset_visit'
            or config['training']['optimizer'] != 'retain_backbone_Adam_add_flow_group'
            or config['evaluation']['integration'] != 'Euler'
            or config['evaluation']['steps'] != 32
            or config['evaluation']['reference_slots'] != [0, 4]
            or config['evaluation']['input_frames'] != 8
            or config['evaluation']['final_GT_valid_assets'] != 967
            or config['evaluation']['natural_valid_assets'] != 128):
        raise ValueError('unreviewed experiment recipe')
    base = load_backbone_config(config)
    if base['data']['manifests']['train']['rows'] != 5000 or base['data']['manifests']['valid']['rows'] != 967:
        raise ValueError('full immutable train/validation splits required')
    if set(config['training']['frame_policy']) != {str(i) for i in range(2, 25)}:
        raise ValueError('all2to24input poses required')
    if any(type(v) is not int or v < 1 or 24 % v for v in config['training']['frame_policy'].values()):
        raise ValueError('physical batches must divide the per-rank effective batch')
    return base


def scientific_sha(config):
    fields = copy.deepcopy({k: config[k] for k in ('family', 'backbone', 'architecture', 'training', 'evaluation')})
    fields['training'].pop('resume', None)
    if config.get('family') == 'hardcase3608_20261101_v1':
        fields['sampling'] = copy.deepcopy(config['sampling'])
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def sample_seed(seed, identity, domain):
    return int.from_bytes(hashlib.sha256(f'{seed}:{identity}:{domain}'.encode()).digest()[:8], 'little') % (2**63 - 1)
