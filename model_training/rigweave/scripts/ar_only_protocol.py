"""Exact continuation of the donor AR branch, matched to joint+10/20/30/40k."""
from reference_flow_protocol import DONOR_SHA, load_backbone_config, scientific_sha, sha256

SCOPE = 'ar_only_feedback_5k_20261029_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/train_ar_only.py'
VARIANT = 'fraction20_40_query'
CHECKPOINTS = list(range(5000, 40001, 5000))
COMPARISONS = [10000, 20000, 30000, 40000]


def validate_config(config):
    train, evaluation = config['training'], config['evaluation']
    policy = {str(n): 24 if n <= 6 else 12 if n <= 12 else 6 for n in range(2, 25)}
    expected = {'additional_assets': 40000, 'checkpoint_assets': CHECKPOINTS,
        'comparison_assets': COMPARISONS, 'effective_assets_per_update': 48,
        'stream_seed': 20261028, 'frame_policy': policy, 'learning_rate_asset_offset': 20000,
        'loss': 'AR_only', 'optimizer': 'retain_complete_donor_Adam',
        'gradient_clip': 'backbone_norm1', 'precision': 'bf16', 'variant': VARIANT}
    if config.get('family') != SCOPE or config.get('schema_version') != 1:
        raise ValueError('wrong AR-only family')
    if any(train.get(k) != v for k, v in expected.items()):
        raise ValueError('unreviewed AR-only training recipe')
    if (config['backbone']['checkpoint']['sha256'] != DONOR_SHA
            or config['backbone']['source_asset_exposures'] != 20000
            or config['architecture'] != {'stage2_enabled': False, 'max_joints': 512}):
        raise ValueError('donor or architecture changed')
    if evaluation != {'input_frames': 8, 'frame_indices': list(range(8)),
            'selection_assets': 128, 'max_new_tokens': 2048, 'precision': 'fp32',
            'tf32': False, 'decoder_attention': 'eager', 'prompt': 'BOS_articulationxl', 'strategy': 'greedy',
            'reference_kind': 'none', 'scoring': 'original_GT'}:
        raise ValueError('unreviewed free-generation contract')
    base = load_backbone_config(config)
    if (base['optimization']['schedule_samples'] != 100000
            or base['data']['manifests']['train']['rows'] != 5000
            or base['data']['manifests']['valid']['rows'] != 967):
        raise ValueError('original LR horizon or dataset changed')
    if config['capacity'] != {'optimizer_steps_per_case': 2,
            'accumulation_microbatches': 2, 'full_decoder_tokens': 2048}:
        raise ValueError('capacity coverage changed')
    return base


def validate_donor(payload, names, world):
    if (payload['variant'] != VARIANT or payload['asset_exposures'] != 20000
            or payload['code_commit'] != 'a0bda62bb8fba70320310b063da8c7898ddaf441'
            or payload['optimizer_parameter_names'] != names
            or len(payload['rank_states']) != world
            or [r['rank'] for r in payload['rank_states']] != list(range(world))
            or len(payload['optimizer']['param_groups']) != 1
            or not payload['optimizer']['state']):
        raise ValueError('donor model/Adam/rank identity changed')


def validate_resume(config, saved, names, world):
    if (saved['scientific_sha256'] != scientific_sha(config)
            or saved['optimizer_parameter_names'] != names
            or len(saved['rank_states']) != world
            or [r['rank'] for r in saved['rank_states']] != list(range(world))
            or len(saved['optimizer']['param_groups']) != 1
            or saved['global_backbone_asset_exposures'] != 20000 + saved['additional_asset_exposures']
            or not 0 < saved['additional_asset_exposures'] <= 40000):
        raise ValueError('AR-only resume contract changed')
