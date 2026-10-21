from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_structure_xyz_condition import contrast_segments, mismatch_position, summarize_cases, verify_operation


def case(group, joints, correct_ce, other_ce):
    def metrics(ce, tokens=2):
        return {'tokens': tokens, 'raw_ce': ce, 'raw_accuracy': .5,
                'legal_ce': ce, 'legal_accuracy': .5}

    modes = {}
    for mode, ce in (('correct', correct_ce), ('different_asset', other_ce), ('zero', 3.)):
        modes[mode] = {'all': metrics(ce), 'choice': metrics(ce),
                       'by_category': {category: metrics(ce) for category in
                                       ('new_joint_id', 'branch', 'parent_reference', 'eos')}}
        modes[mode]['by_category']['branch'] = {key: (0 if key == 'tokens' else None)
                                               for key in metrics(ce)}
    contrasts = {mode: {'choice': {'tokens': 2, 'raw_js': .1, 'legal_js': .2,
                                   'raw_argmax_change_fraction': .25,
                                   'legal_argmax_change_fraction': .5}}
                 for mode in ('different_asset', 'zero')}
    prefix = {'modes': modes, 'contrasts': contrasts}
    return {'group': group, 'target_joint_count': joints, 'teacher': prefix, 'self': deepcopy(prefix)}


def test_mismatch_is_next_token_position():
    assert mismatch_position([1, 4, 5, 2], [1, 4, 2]) == 1
    assert mismatch_position([1, 4, 2], [1, 4, 2]) is None
    assert mismatch_position([1, 4, 5], [1, 4, 5, 2]) == 2


def test_wrong_diagnostic_purpose_fails_before_runtime_access():
    with pytest.raises(ValueError, match='declared read-only'):
        verify_operation(None, {'purpose': 'train'}, {})


def test_mismatch_segments_are_nonoverlapping_and_bounded():
    contrast = {'positions': [{'raw_js': float(i), 'legal_js': float(i) / 2} for i in range(50)]}
    result = contrast_segments(contrast, 3)
    assert [part['positions'] for part in result.values()] == [3, 8, 24, 15]
    assert result['mismatch_through_next7']['raw_js'] == pytest.approx(6.5)
    exact = contrast_segments(contrast, None)
    assert exact['before_first_mismatch']['positions'] == 50
    assert exact['after32']['positions'] == 0
    assert exact['after32']['legal_js'] is None


def test_summary_uses_equal_case_weight_and_keeps_paired_deltas():
    rows = [case('valid', 52, 1., 1.4), case('valid', 21, 2., 1.8)]
    rows[1]['teacher']['modes']['correct']['all']['tokens'] = 100
    result = summarize_cases(rows)['valid']['all']['prefixes']['teacher']
    assert result['modes']['correct']['all']['raw_ce'] == pytest.approx(1.5)
    assert result['modes']['correct']['all']['total_tokens'] == 102
    contrast = result['contrasts']['different_asset']
    assert contrast['choice_legal_ce_counter_minus_correct'] == pytest.approx(.1)
    assert contrast['correct_lower_ce_cases'] == 1
    assert contrast['counter_lower_ce_cases'] == 1
    assert result['modes']['correct']['branch']['contributing_cases'] == 0
    assert result['modes']['correct']['branch']['raw_ce'] is None


def test_summary_separates_groups_counts_and_empty_subsets():
    result = summarize_cases([case('train_seen', 21, 1., 2.), case('valid', 52, 2., 3.)])
    assert result['train_seen']['non52']['cases'] == 1
    assert result['valid']['joints52']['cases'] == 1
    assert result['train_heldout']['all']['cases'] == 0
    assert result['train_seen']['joints52']['prefixes']['teacher']['modes']['correct']['all']['raw_ce'] is None
