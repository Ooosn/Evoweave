"""Narrow official-only parameter loading for the independent XYZ-fill model."""
from pathlib import Path

import torch


COMMON_POLICY = {
    'kind': 'official_surface_fresh_motion_structure',
    'trained_experiment_weights_allowed': False,
    'optimizer_restored': False,
}


def validate_initialization_policy(config):
    common = config.get('common_initialization')
    if (not isinstance(common, dict) or set(common) != set(COMMON_POLICY)
            or common['kind'] != COMMON_POLICY['kind']
            or common['trained_experiment_weights_allowed'] is not False
            or common['optimizer_restored'] is not False):
        raise ValueError('only official surface weights with fresh motion/structure modules are allowed')
    initial = config['initialization']
    if Path(initial['checkpoint']).resolve() != Path(initial['official_checkpoint']).resolve():
        raise ValueError('initialization must identify the official checkpoint, not a trained experiment')
    if config['fill_initialization']['kind'] not in {'random', 'official_text_positions'}:
        raise ValueError('only fresh fill weights or explicitly selected official positions are allowed')
    if any(type(config[section].get('seed')) is not int for section in ('initialization', 'fill_initialization')):
        raise ValueError('initialization seeds must be explicit integers')


def load_official_surface_parameters(surface, projection, state):
    report = {}
    for prefix, module in (('mesh_encoder.', surface), ('output_proj.', projection)):
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not selected:
            raise ValueError(f'official checkpoint is missing {prefix}')
        if any(not isinstance(value, torch.Tensor) or
               (value.is_floating_point() and not bool(torch.isfinite(value).all()))
               for value in selected.values()):
            raise ValueError(f'official checkpoint contains invalid or nonfinite {prefix} tensors')
        module.load_state_dict(selected, strict=True)
        report[prefix] = {'strict': True, 'tensors': len(selected),
                          'elements': sum(value.numel() for value in selected.values())}
    return report


def official_text_positions(state, *, dim, query_tokens, position_capacity):
    key = 'transformer.model.decoder.embed_positions.weight'
    position = state.get(key)
    if (not isinstance(position, torch.Tensor) or not position.is_floating_point()
            or tuple(position.shape) != (position_capacity + 2, dim)
            or not bool(torch.isfinite(position).all())):
        raise ValueError('official OPT position table has incompatible shape or nonfinite values')
    if (type(query_tokens) is not int or type(position_capacity) is not int
            or not 0 <= query_tokens < position_capacity):
        raise ValueError('official text positions need a valid mesh-prefix length')
    # The original OPT table reserves two rows, then the old 1024-token mesh prefix.
    return position, query_tokens + 2
