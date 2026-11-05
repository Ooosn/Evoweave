"""Verify the retained recovery point after an explicitly requested pause."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    request = json.loads((root/'outputs/user_pause_request_20261105.json').read_text())
    spec = request['resume_checkpoint']
    resources = json.loads((root/'outputs/train_resources.json').read_text())
    if not resources['complete'] or not resources['payload_cleanup_complete']:
        raise RuntimeError('the original payload has not finished cleanup')
    files = [Path(spec[key]) for key in ('path', 'durable_path')]
    for path in files:
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise RuntimeError('retained recovery point or mirror changed')
    saved = torch.load(files[0], map_location='cpu', mmap=True, weights_only=False)
    if saved['additional_asset_exposures'] != 250000 or saved['step'] != 5225:
        raise RuntimeError('unexpected retained recovery position')
    states = saved['optimizer']['state']
    names = saved['optimizer_parameter_names']
    if not states or len(saved['optimizer']['param_groups']) != 2:
        raise RuntimeError('Adam state or parameter groups are missing')
    for state in states.values():
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
            raise RuntimeError('an Adam parameter is incomplete')
        if not all(torch.isfinite(state[key]).all().item()
                   for key in ('step', 'exp_avg', 'exp_avg_sq')):
            raise RuntimeError('nonfinite Adam state')
    for rank, state in enumerate(saved['rank_states']):
        if state['rank'] != rank or state['stream']['cursor'] * 2 != 250000:
            raise RuntimeError('sampler resume position differs')
    if len(saved['rank_states']) != 2:
        raise RuntimeError('both rank RNG/sampler states are required')
    report = {
        'complete': True, 'paused_at_user_request': True,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': saved['code_commit'], 'resume_checkpoint': spec,
        'resume_samples': saved['additional_asset_exposures'], 'resume_updates': saved['step'],
        'live_samples_at_interrupt': [x['global_backbone_asset_exposures']
                                     for x in request['rank_progress']],
        'optimizer_entries': len(states), 'optimizer_parameter_groups': len(saved['optimizer']['param_groups']),
        'optimizer_parameter_names': len(names),
        'all_Adam_moments_finite': True, 'rank_state_keys': [sorted(s) for s in saved['rank_states']],
        'learning_rate_resume': 'Reconstructed from saved configuration and asset position, as in the original runtime.',
        'payload_cleanup_complete': True, 'allocation_released': False,
        'training_completed': False, 'model_quality_accepted': False,
        'original_supervisor_exit_code': resources.get('exit_code'),
    }
    for destination in (root/'outputs/user_pause_review_20261105.json',
                        files[1].parent/'user_pause_review_20261105.json'):
        destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
