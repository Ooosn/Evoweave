from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_structure_xyz_loss_balance import gradient_statistics, summarize, validate_resource_runtime


def test_resource_handoff_requires_published_grant_and_same_environment():
    original = {'allocation': 1, 'host': 'old', 'physical_gpus': [5, 6],
                'world_size': 2, 'repo': '/same/code', 'python': '/same/python'}
    runtime = {**original, 'allocation': 2, 'host': 'new', 'physical_gpus': [0, 1]}
    run = {key: runtime[key] for key in ('allocation', 'host', 'physical_gpus')}
    validate_resource_runtime({'runtime': original}, runtime, run)
    with pytest.raises(ValueError):
        validate_resource_runtime({'runtime': original}, {**runtime, 'python': '/other/python'}, run)
    with pytest.raises(ValueError):
        validate_resource_runtime({'runtime': original}, runtime, {**run, 'allocation': 3})
    with pytest.raises(ValueError):
        validate_resource_runtime({'runtime': original}, {**runtime, 'world_size': 4}, run)


def test_norm_ratio_and_cosine_include_disconnected_parameters():
    result = gradient_statistics([torch.tensor([3., 4.]), None],
                                 [torch.tensor([6., 8.]), torch.tensor([2.])])
    assert result['structure_norm'] == 5.
    assert result['xyz_norm'] == pytest.approx(104 ** .5)
    assert result['structure_over_xyz'] == pytest.approx(5 / 104 ** .5)
    assert result['cosine'] == pytest.approx(50 / (25 * 104) ** .5)


def test_zero_and_nonfinite_gradient_handling():
    result = gradient_statistics([None], [torch.tensor([1.])])
    assert result['structure_over_xyz'] == 0.
    assert result['cosine'] is None
    with pytest.raises(FloatingPointError):
        gradient_statistics([torch.tensor([float('nan')])], [None])
    with pytest.raises(ValueError):
        gradient_statistics([], [None])


def test_separate_backward_does_not_update_or_accumulate():
    shared = torch.nn.Parameter(torch.tensor([1., 2.]))
    structure = torch.nn.Parameter(torch.tensor([2., 1.]))
    xyz = torch.nn.Parameter(torch.tensor([3., 4.]))
    params = [shared, structure, xyz]
    before = [p.detach().clone() for p in params]
    left = (shared * structure).sum()
    right = (shared * xyz).sum()
    a = torch.autograd.grad(left, params, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(right, params, allow_unused=True)
    assert gradient_statistics(a[:1], b[:1])['structure_over_xyz'] == pytest.approx(5 ** .5 / 5)
    assert b[1] is None and a[2] is None
    assert all(p.grad is None and torch.equal(p, old) for p, old in zip(params, before))


def test_summary_handles_empty_subsets_and_equal_asset_weight():
    def row(joints, loss):
        metric = {'tokens': 2, 'raw_ce': loss, 'legal_ce': loss,
                  'raw_accuracy': .8, 'legal_accuracy': .9}
        return {'target_joint_count': joints, 'structure_loss': loss, 'xyz_loss': 2.,
                'structure_accuracy': .8, 'xyz_accuracy': .5,
                'gradients': {name: gradient_statistics([torch.tensor([loss])], [torch.tensor([2.])])
                              for name in ('surface_tokenizer', 'motion_encoder', 'shared', 'structure_decoder', 'fill_decoder')},
                'structure_metrics': {'all': metric, 'choice': metric,
                                      'by_category': {name: metric for name in
                                                      ('new_joint_id', 'branch', 'parent_reference', 'eos')}}}
    result = summarize([row(52, 1.), row(20, 3.)])
    assert result['all']['structure_loss'] == 2.
    assert result['non52']['structure_loss'] == 3.
    assert result['all']['structure_categories']['choice']['tokens'] == 4
    assert summarize([row(52, 1.)])['non52']['structure_loss'] is None
