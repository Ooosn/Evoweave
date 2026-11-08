"""Admission contract for the released single-pose skeleton baseline."""
import hashlib
import json
from pathlib import Path


SCOPE = 'skintokens_static_5k_20261109'
ENTRYPOINT = 'model_training/rigweave/scripts/train_skintokens_static.py'
MILESTONES = (10000, 20000, 30000)
MANIFEST_SHA = 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88'
RELEASE_SHA = 'f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692'
PANEL_SHA = '02c67d4828b8fe4972aadd947986741857107aebaf77e5db252efb24438697d8'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def validate_config(config):
    train, evaluate = config['training'], config['evaluation']
    if (config['schema_version'] != 1 or config['family'] != SCOPE
            or config['execution']['phase'] != SCOPE or config['stage2_enabled'] is not False
            or config['train_manifest']['rows'] != 5000 or config['train_manifest']['sha256'] != MANIFEST_SHA
            or config['tokenrig_decoder']['checkpoint_sha256'] != RELEASE_SHA
            or config['tokenrig_decoder']['conditioning']['prefix_tokens'] != 512
            or 'motion_initialization' in config['tokenrig_decoder']['conditioning']
            or train != {'additional_assets': 30000, 'effective_assets_per_update': 48,
                'frame_policy': {'1': 24}, 'checkpoint_assets': list(MILESTONES),
                'stream_seed': 20261030, 'initialization_seed': 20261030, 'resume': None}
            or evaluate['assets'] != 128 or evaluate['reference_slot'] != 0
            or evaluate['checkpoints'] != [0, *MILESTONES] or evaluate['max_new_tokens'] != 2048
            or evaluate['precision'] != 'fp32_noTF32' or evaluate['decoder_attention'] != 'eager'
            or evaluate['decoding'] != 'greedy' or evaluate['panel_archive']['sha256'] != PANEL_SHA):
        raise ValueError('unreviewed single-pose SkinTokens configuration')


def verify_evidence(op, config, mode, config_sha):
    validate_config(config)
    if op.get('experiment') != SCOPE or mode not in ('audit', 'capacity', 'train'):
        raise ValueError('unexpected native single-pose stage')
    needed = () if mode == 'audit' else ('audit',) if mode == 'capacity' else ('audit', 'capacity')
    for stage in needed:
        evidence = op['stage_evidence'][stage]
        root = Path(config['runtime']['output_root'])
        output = evidence.get('output_name', stage)
        if output not in (('audit',) if stage == 'audit' else ('capacity', 'capacity_recompute')):
            raise ValueError(f'unreviewed {stage} evidence path')
        paths = (root / output / 'result.json', root / f'{output}_resources.json')
        if any(sha256(path) != evidence[key] for path, key in
               zip(paths, ('result_sha256', 'resources_sha256'))):
            raise ValueError(f'{stage} evidence differs')
        result, resources = (json.loads(path.read_text()) for path in paths)
        if (result.get('config_sha256') != config_sha or result.get('source') != resources.get('source')
                or result.get('stage2_enabled') is not False
                or not all(value.get('complete') and value.get('passed') for value in (result, resources))
                or not resources.get('kernel_scheduler_membership')
                or not resources.get('payload_cleanup_complete') or resources.get('exit_code') != 0
                or resources.get('allocation') != config['runtime']['allocation']
                or (stage == 'audit' and result.get('cuda_initialized') is not False)
                or (stage == 'capacity' and result.get('optimizer_updates') != 0)):
            raise ValueError(f'{stage} did not pass and clean up under the approved grant')
