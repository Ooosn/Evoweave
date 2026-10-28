"""Exact inference-only replay of two retained common80k cases."""

PROFILE = 'checkpoint_replay_20261029_v1'
CHECKPOINT_SHA = '3ea27d7537852a42aab043938fbd7f341aaf3127a2edba907b5648506950e701'
CASES = ['oxl_2aa3c8aa648c6cba21fb', 'oxl_24e8e5621802f809e815']
EXAMPLE_SHAS = ['0da4e3287e78bd60e943a1fc8620553e2e50f305bbdd8b7ec59d1cbb5c06f651',
                '0cc38d47252a8c6d06e91bae775aaf33bf2f8dc254d1c65cb40fac1bd600c8bf']


def validate_replay(config, mode=None):
    spec = config['capacity']
    if (set(spec) != {'profile', 'regime', 'checkpoint', 'case_ids', 'examples',
                     'optimizer_steps', 'model_loads', 'repeats_per_load', 'precisions', 'max_seconds'}
            or spec['profile'] != PROFILE or spec['regime'] != 'pretrained_stage2'
            or spec['case_ids'] != CASES or spec['optimizer_steps'] != 0
            or spec['model_loads'] != 2 or spec['repeats_per_load'] != 2
            or spec['precisions'] != ['bf16', 'fp32'] or spec['max_seconds'] != 900
            or config['execution']['phase'] != 'common80k'
            or (mode is not None and mode != 'capacity')):
        raise ValueError('replay is exactly two fixed cases, two loads, no optimizer updates, capacity mode only')
    checkpoint = spec['checkpoint']
    expected_path = '/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v19/pretrained_stage2/checkpoint_080000.pt'
    if (checkpoint['path'] != expected_path or checkpoint['sha256'] != CHECKPOINT_SHA
            or checkpoint['additional_asset_exposures'] != 80000
            or checkpoint['bytes'] != 8278609298 or not checkpoint['both_copies_verified']):
        raise ValueError('replay checkpoint differs')
    if len(spec['examples']) != 2:
        raise ValueError('both retained example ranks required')
    for rank, example in enumerate(spec['examples']):
        expected = f'/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v19/pretrained_stage2/evaluation_080000/examples_rank_{rank}.pt'
        if example['path'] != expected or example['sha256'] != EXAMPLE_SHAS[rank]:
            raise ValueError('retained fixed-case identity differs')
    return spec
