from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_structure_xyz_continue160000 import FIXED_WITHOUT_ADAM, ROOT, make_stream, plan_handoff, validate_adam_identity


def inputs():
    base = json.loads((ROOT / 'model_training/experiments/structure_xyz_fill_random_80000_v1.json').read_text())
    future = deepcopy(base)
    boundaries = base['training']['evaluation_assets'] + list(range(90000, 160001, 10000))
    future['training'].update(target_assets=160000, evaluation_assets=boundaries, checkpoint_assets=boundaries)
    entries = [{'asset_id': str(index)} for index in range(5000)]
    states = []
    for rank in (0, 1):
        stream = make_stream(base, entries, rank=rank, world=2)
        stream.cursor = 40000
        states.append({'stream': stream.state_dict()})
    return base, future, entries, states


def test_plan_preserves_prefix_and_covers_both_rank_first_five_updates():
    result = plan_handoff(*inputs())
    assert result['plan_sha256'] == 'f12a885946f44b46a8fa84753812f4102e8c361d06563b3d426d23866ad726d3'
    assert result['total_updates'] == 3355 and result['new_updates'] == 1672
    assert result['supervised_frames'] == 2068624
    for report in result['ranks']:
        assert [row['step'] for row in report['first_five_updates']] == list(range(1684, 1689))
        assert [row['asset_exposures'] for row in report['first_five_updates']] == list(range(80048, 80241, 48))
        assert [row['microbatches'][0]['frames'] for row in report['first_five_updates']] == [14, 4, 14, 12, 23]
        assert sum(len(m['requests']) for row in report['first_five_updates'] for m in row['microbatches']) == 120
    assert result['milestone_counters'][-1] == {
        'asset_exposures': 160000, 'step': 3355, 'supervised_frame_exposures': 2068624}


def test_wrong_saved_cursor_is_rejected():
    base, future, entries, states = inputs()
    states[0]['stream']['cursor'] -= 24
    with pytest.raises(ValueError):
        plan_handoff(base, future, entries, states)


def adam_fixture():
    names = [f'parameter{index}' for index in range(626)] + sorted(FIXED_WITHOUT_ADAM)
    groups, named = [], []
    for index in range(4):
        identifiers = list(range(index, len(names), 4))
        groups.append({'name': f'group{index}', 'params': identifiers})
        named.append({'group': f'group{index}', 'parameters': [names[i] for i in identifiers]})
    return {'optimizer_parameter_names': named, 'model': {name: torch.zeros(2) for name in names},
            'optimizer': {'param_groups': groups, 'state': {index: {'step': torch.tensor(1683),
                          'exp_avg': torch.zeros(2), 'exp_avg_sq': torch.zeros(2)} for index in range(626)}}}


def test_exact_fixed_parameters_may_lack_adam_but_every_trained_parameter_is_required():
    saved = adam_fixture()
    count, _, fixed = validate_adam_identity(saved)
    assert count == 626 and set(fixed) == FIXED_WITHOUT_ADAM
    del saved['optimizer']['state'][5]
    with pytest.raises(AssertionError):
        validate_adam_identity(saved)


def test_adam_step_and_shape_are_checked():
    saved = adam_fixture()
    saved['optimizer']['state'][5]['step'] = torch.tensor(1682)
    with pytest.raises(AssertionError):
        validate_adam_identity(saved)
    saved = adam_fixture()
    saved['optimizer']['state'][5]['exp_avg'] = torch.zeros(3)
    with pytest.raises(AssertionError):
        validate_adam_identity(saved)
