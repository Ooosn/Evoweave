"""Exact three-arm follow-up to the native single-pose baseline."""
import json
from pathlib import Path

from skintokens_static_protocol import check_preflights, validate_config as validate_static


SCOPE = 'skintokens_stage1_5k_20261109'
ENTRYPOINT = 'model_training/rigweave/scripts/train_skintokens_stage1.py'
ARMS = ('frame_local', 'temporal_multi', 'temporal_query')
ROOT = Path(__file__).resolve().parents[3]
STATIC_CONFIG = ROOT / 'model_training/experiments/skintokens_static_5k_20261109.json'
SCRATCH = '/work/wangyy/tokenrig_alignment_20261105'
DURABLE = '/home/wangyy/evorig/outputs/tokenrig_alignment_20261105'
REPO = f'{SCRATCH}/code_{SCOPE}'


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('unknown Stage 1 ablation arm')
    static = json.loads(STATIC_CONFIG.read_text())
    validate_static(static)
    spec = static['tokenrig_decoder']
    return {**static, 'family': SCOPE, 'execution': {'phase': SCOPE}, 'arm': arm,
        'runtime': {**static['runtime'], 'repo': REPO,
                    'output_root': f'{SCRATCH}/outputs_{SCOPE}/{arm}'},
        'training': {**static['training'],
                     'frame_policy': {str(t): 24 if t <= 12 else 12 for t in range(2, 25)}},
        'tokenrig_decoder': {**spec, 'conditioning': {**spec['conditioning'],
            'motion_initialization': 'identity_residuals_native_norm_v1'}},
        'artifacts': {'durable_root': f'{DURABLE}/{SCOPE}/{arm}'}}


def validate_config(config):
    if config != recipe(config['arm']):
        raise ValueError('Stage 1 recipe differs from the approved matched 5k/30k controls')


def verify_evidence(op, config, mode, config_sha):
    validate_config(config)
    if op.get('arm') != config['arm']:
        raise ValueError('wrong Stage 1 ablation arm in the live operation')
    check_preflights(op, config, mode, config_sha, scope=SCOPE)
