from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rigweave/src'))

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.structure_xyz_fill import allowed_structure_tokens
from structure_condition_metrics import (
    build_prefix_layout, choose_partner_indices, contrast_logits, summarize_logits,
)


@pytest.fixture
def codec():
    return AuxSkeletonTokens(max_joints=8, max_sequence_tokens=32)


@pytest.fixture
def branched(codec):
    return build_prefix_layout([1, 4, 5, 6, 3, 4, 7, 2], codec)


def test_layout_uses_actual_prefix_grammar_and_previous_branch(codec, branched):
    tokens = [1, 4, 5, 6, 3, 4, 7, 2]
    assert set(branched) == {'labels', 'categories', 'allowed', 'choice'}
    assert branched['labels'].dtype == np.int64
    assert branched['allowed'].dtype == branched['choice'].dtype == np.bool_
    assert branched['labels'].tolist() == tokens[1:]
    assert branched['categories'] == ['new_joint_id'] * 3 + ['branch', 'parent_reference', 'new_joint_id', 'eos']
    assert branched['choice'].tolist() == [False, True, True, True, True, False, True]
    for index in range(len(tokens) - 1):
        assert np.flatnonzero(branched['allowed'][index]).tolist() == sorted(
            allowed_structure_tokens(tokens[:index + 1], codec))


@pytest.mark.parametrize('parents', [[-1], [-1, 0, 1], [-1, 0, 0, 2], [-1, 0, 1, 2, 1, 0, 5, 5]])
def test_layout_accepts_actual_codec_encodings(codec, parents):
    tokens = codec.encode(parents).tokens
    layout = build_prefix_layout(tokens, codec)
    np.testing.assert_array_equal(layout['labels'], tokens[1:])
    assert layout['categories'].count('new_joint_id') == len(parents)
    assert layout['categories'].count('eos') == 1


@pytest.mark.parametrize('tokens', [
    [], [1], [1, 2], [4, 2], [1, 4, 5], [1, 4, 2, 2], [1, 4, 2, 0],
    [1, 4, 6, 2], [1, 4, 3, 4, 5, 2], [1, 4, 5, 3, 5, 6, 2],
    [1, 4, 5, 3, 4, 2], [1, 4, 5, 3, 9, 6, 2], [1, 4, -1, 2],
    [1, 4, 12, 2], [[1, 4, 2]], [1., 4., 2.], [True, False, True],
])
def test_invalid_or_incomplete_prefix_rejected(codec, tokens):
    with pytest.raises(ValueError):
        build_prefix_layout(tokens, codec)


def test_sequence_and_joint_capacity_rejected():
    with pytest.raises(ValueError, match='capacity'):
        build_prefix_layout([1, 4, 5, 2], AuxSkeletonTokens(max_joints=8, max_sequence_tokens=3))
    with pytest.raises(ValueError):
        build_prefix_layout([1, 4, 5, 6, 2], AuxSkeletonTokens(max_joints=2))


def test_uniform_ce_probabilities_categories_and_forced_exclusion(codec, branched):
    result = summarize_logits(np.zeros(branched['allowed'].shape), branched)
    assert set(result) == {'all', 'choice', 'by_category', 'positions'}
    assert result['all']['tokens'] == 7
    assert result['choice']['tokens'] == 5
    assert result['all']['raw_ce'] == pytest.approx(np.log(codec.vocab_size))
    counts = branched['allowed'].sum(axis=1)
    assert result['all']['legal_ce'] == pytest.approx(np.log(counts).mean())
    assert result['choice']['legal_ce'] == pytest.approx(np.log(counts[branched['choice']]).mean())
    assert result['by_category']['parent_reference']['tokens'] == 1
    assert result['by_category']['new_joint_id']['tokens'] == 4
    for index in (0, 5):
        assert result['positions'][index]['legal_nll'] == 0
        assert result['positions'][index]['legal_probability'] == 1
        assert result['positions'][index]['choice'] is False
    assert result['positions'][0]['raw_probability'] == pytest.approx(1 / codec.vocab_size)
    json.dumps(result, allow_nan=False)


def two_token_layout():
    return {'labels': np.array([0], dtype=np.int64), 'categories': ['eos'],
            'allowed': np.array([[True, True]]), 'choice': np.array([True])}


def test_known_ce_accuracy_and_positions():
    result = summarize_logits(np.log([[.8, .2]]), two_token_layout())
    assert result['all'] == pytest.approx({'tokens': 1, 'raw_ce': -np.log(.8), 'raw_accuracy': 1.,
                                         'legal_ce': -np.log(.8), 'legal_accuracy': 1.})
    assert set(result['positions'][0]) == {'label', 'category', 'choice', 'raw_nll', 'legal_nll',
                                         'raw_probability', 'legal_probability', 'raw_argmax', 'legal_argmax'}
    assert result['positions'][0]['raw_probability'] == pytest.approx(.8)


def test_illegal_argmax_is_removed_and_large_logits_are_stable(codec):
    layout = build_prefix_layout([1, 4, 2], codec)
    logits = np.full(layout['allowed'].shape, -10000.)
    logits[:, codec.pad] = 10000.
    result = summarize_logits(logits, layout)
    assert result['all']['raw_accuracy'] == 0
    assert result['all']['legal_accuracy'] == 1
    assert result['all']['raw_ce'] == pytest.approx(20000.)
    assert result['all']['legal_ce'] == pytest.approx(np.log(2) / 2)
    assert result['positions'][0]['raw_probability'] == 0
    assert result['positions'][0]['legal_argmax'] == 4
    json.dumps(result, allow_nan=False)


def test_empty_choice_and_categories_return_none():
    layout = build_prefix_layout([1, 4, 2], AuxSkeletonTokens(max_joints=1))
    values = np.zeros(layout['allowed'].shape)
    result = summarize_logits(values, layout)
    for group in (result['choice'], result['by_category']['branch'], result['by_category']['parent_reference']):
        assert group == {'tokens': 0, 'raw_ce': None, 'raw_accuracy': None,
                         'legal_ce': None, 'legal_accuracy': None}
    assert contrast_logits(values, values, layout)['choice'] == {
        'tokens': 0, 'raw_js': None, 'legal_js': None,
        'raw_argmax_change_fraction': None, 'legal_argmax_change_fraction': None}


def test_empty_scored_layout_has_no_nan():
    layout = {'labels': np.empty(0, dtype=np.int64), 'categories': [],
              'allowed': np.empty((0, 3), dtype=bool), 'choice': np.empty(0, dtype=bool)}
    for result in (summarize_logits(np.empty((0, 3)), layout),
                   contrast_logits(np.empty((0, 3)), np.empty((0, 3)), layout)):
        assert result['all']['tokens'] == 0
        assert result['positions'] == []
        assert all(value is None for key, value in result['all'].items() if key != 'tokens')
        json.dumps(result, allow_nan=False)


def test_known_js_in_nats_and_argmax_changes():
    left, right = np.log([[.9, .1]]), np.log([[.1, .9]])
    result = contrast_logits(left, right, two_token_layout())
    expected = .9 * np.log(1.8) + .1 * np.log(.2)
    assert set(result) == {'all', 'choice', 'positions'}
    assert result['all'] == pytest.approx({'tokens': 1, 'raw_js': expected, 'legal_js': expected,
                                         'raw_argmax_change_fraction': 1., 'legal_argmax_change_fraction': 1.})
    assert result['all'] == result['choice']
    assert result['positions'] == [{'raw_js': pytest.approx(expected), 'legal_js': pytest.approx(expected),
                                   'raw_argmax_changed': True, 'legal_argmax_changed': True}]


def test_js_zero_probability_terms_and_disjoint_support():
    result = contrast_logits([[10000., -10000.]], [[-10000., 10000.]], two_token_layout())
    assert result['all']['raw_js'] == pytest.approx(np.log(2))
    assert result['all']['legal_js'] == pytest.approx(np.log(2))
    json.dumps(result, allow_nan=False)


def test_contrast_forced_legal_distribution_stays_identical():
    layout = build_prefix_layout([1, 4, 2], AuxSkeletonTokens(max_joints=1))
    correct = np.zeros(layout['allowed'].shape)
    counter = correct.copy()
    counter[:, 0] = 20
    result = contrast_logits(correct, counter, layout)
    assert result['all']['raw_js'] > 0
    assert result['all']['legal_js'] == 0
    assert result['all']['legal_argmax_change_fraction'] == 0
    assert result['choice']['tokens'] == 0


def test_identical_and_per_row_shift_invariance(branched):
    values = np.arange(branched['allowed'].size, dtype=np.float64).reshape(branched['allowed'].shape) / 8
    shifted = values + np.arange(len(values))[:, None] * 1024 + 4096
    for counter in (values.copy(), shifted):
        result = contrast_logits(values, counter, branched)
        for name in ('all', 'choice'):
            assert result[name]['raw_js'] == 0
            assert result[name]['legal_js'] == 0
            assert result[name]['raw_argmax_change_fraction'] == 0
            assert result[name]['legal_argmax_change_fraction'] == 0
        assert summarize_logits(values, branched) == summarize_logits(counter, branched)


def test_js_symmetry_and_bounds(branched):
    rng = np.random.default_rng(17)
    a, b = [rng.normal(size=branched['allowed'].shape) * 30 for _ in range(2)]
    forward, backward = contrast_logits(a, b, branched), contrast_logits(b, a, branched)
    assert forward == backward
    for position in forward['positions']:
        assert 0 <= position['raw_js'] <= np.log(2) + 1e-14
        assert 0 <= position['legal_js'] <= np.log(2) + 1e-14


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_logits_rejected_even_in_illegal_columns(branched, bad):
    values = np.zeros(branched['allowed'].shape)
    invalid = values.copy()
    invalid[0, 0] = bad
    for call in (lambda: summarize_logits(invalid, branched),
                 lambda: contrast_logits(invalid, values, branched),
                 lambda: contrast_logits(values, invalid, branched)):
        with pytest.raises(ValueError, match='finite'):
            call()


@pytest.mark.parametrize('bad', [np.zeros(7), np.zeros((7, 11)), np.zeros((6, 12)),
                               np.zeros((1, 7, 12)), np.zeros((7, 12), dtype=complex)])
def test_wrong_logits_shape_or_type_rejected(branched, bad):
    with pytest.raises(ValueError):
        summarize_logits(bad, branched)
    with pytest.raises(ValueError):
        contrast_logits(np.zeros(branched['allowed'].shape), bad, branched)


@pytest.mark.parametrize('bad_field', ['zero_allowed', 'illegal_label', 'choice', 'categories', 'shape', 'dtype'])
def test_malformed_layout_rejected_without_nan(branched, bad_field):
    invalid = deepcopy(branched)
    if bad_field == 'zero_allowed':
        invalid['allowed'][0] = False
    elif bad_field == 'illegal_label':
        invalid['labels'][0] = 0
    elif bad_field == 'choice':
        invalid['choice'][0] = True
    elif bad_field == 'categories':
        invalid['categories'][0] = 'unknown'
    elif bad_field == 'shape':
        invalid['labels'] = invalid['labels'][1:]
    else:
        invalid['allowed'] = invalid['allowed'].astype(int)
    with pytest.raises(ValueError):
        summarize_logits(np.zeros(branched['allowed'].shape), invalid)
    with pytest.raises(ValueError):
        contrast_logits(np.zeros(branched['allowed'].shape), np.zeros(branched['allowed'].shape), invalid)


def test_functions_do_not_mutate_inputs(branched):
    saved = deepcopy(branched)
    values = np.zeros(branched['allowed'].shape)
    summarize_logits(values, branched)
    contrast_logits(values, values, branched)
    for key in branched:
        np.testing.assert_array_equal(branched[key], saved[key])
    assert not values.any()


def test_partner_selection_first_cyclic_different_tuple_not_permutation():
    parents = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 0], [-1, 0, 1]]
    saved = deepcopy(parents)
    assert choose_partner_indices(parents) == [2, 2, 3, 2]
    assert parents == saved
    assert choose_partner_indices([[-1], [-1, 0]]) == [1, 0]


@pytest.mark.parametrize('parents', [[], [[-1]], [[-1, 0], [-1, 0]], [[-1.], [-1, 0]], [[], [-1]]])
def test_partner_selection_requires_different_integer_parent_rows(parents):
    with pytest.raises(ValueError):
        choose_partner_indices(parents)
