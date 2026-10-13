"""CPU-only acceptance checks; these receipts never authorize a GPU launch."""
from __future__ import annotations

import math
from collections import Counter


def view_key(row):
    return row['asset_id'], row['raw_frame_index']


def main_view_passes(row):
    distance = row.get('quantized_j2j')
    exact = bool(row.get('target_tokens')) and row.get('generated_tokens') == row['target_tokens']
    return bool(row['success'] and row['status'] == 'eos' and row['exact_count']
                and distance is not None and math.isfinite(distance) and distance <= 4 / 256
                and (row['quantized_edge_f1'] >= .95 or exact))


def assess_fit(report):
    rows, auxiliary = report['rows'], report['auxiliary_rows']
    counts = Counter(row['asset_id'] for row in rows)
    if (len(rows) != 32 or len({view_key(row) for row in rows}) != 32 or len(counts) != 8
            or set(counts.values()) != {4} or len(auxiliary) != 8
            or {row['asset_id'] for row in auxiliary} != set(counts)):
        raise ValueError('fitting gate requires the complete fixed eight assets and four distinct poses each')
    if any(not math.isfinite(row['main_accuracy']) for row in rows):
        raise ValueError('nonfinite teacher accuracy')
    accuracy = sum(row['main_accuracy'] for row in rows) / 32
    main_eos = sum(bool(row['success'] and row['status'] == 'eos') for row in rows)
    passing = sum(main_view_passes(row) for row in rows)
    aux_eos = sum(bool(row['success'] and row['natural_eos'] and row['status'] == 'eos') for row in auxiliary)
    aux_exact = sum(bool(row['success'] and row['natural_eos'] and row['status'] == 'eos'
                         and row.get('predicted_parents') and row['predicted_parents'] == row['target_parents'])
                    for row in auxiliary)
    main_passed = main_eos == 32 and accuracy >= .99 and passing >= 29
    auxiliary_passed = aux_eos == 8 and aux_exact == 8
    return {'step': report['step'], 'accepted': main_passed and auxiliary_passed,
            'main_passed': main_passed, 'auxiliary_passed': auxiliary_passed,
            'main_valid_eos': main_eos, 'main_teacher_accuracy': accuracy,
            'passing_main_views': passing, 'auxiliary_valid_eos': aux_eos,
            'auxiliary_exact_parent_trees': aux_exact,
            'unique_main_predictions': len({tuple(row['generated_tokens']) for row in rows}),
            'unique_auxiliary_predictions': len({tuple(row['generated_tokens']) for row in auxiliary})}


def review_fit_pair(left_result, right_result, left_evaluations, right_evaluations, left_ready, right_ready):
    expected_steps = [0, 32, 64, 128, 256, 512, 1024]
    for result, evaluations, ready in ((left_result, left_evaluations, left_ready),
                                        (right_result, right_evaluations, right_ready)):
        if (not result.get('execution_complete') or result['completed_updates'] != 1024
                or result['asset_exposures'] != 8192 or not ready.get('passed')):
            raise ValueError('both trials must complete the matched budget and pre-update pair verification')
        if [report['step'] for report in evaluations] != expected_steps:
            raise ValueError('missing or reordered scheduled evaluations')
    for key in ('full_initial_state_sha256', 'inputs_sha256', 'comparison_contract_sha256', 'code_commit', 'assets'):
        if left_ready[key] != right_ready[key]:
            raise ValueError(f'paired pre-update evidence differs: {key}')
    if left_result['code_commit'] != right_result['code_commit'] or left_result['code_commit'] != left_ready['code_commit']:
        raise ValueError('paired source identity differs')
    reference = {view_key(row): row['target_tokens'] for row in left_evaluations[0]['rows']}
    for report in [*left_evaluations, *right_evaluations]:
        if {view_key(row): row['target_tokens'] for row in report['rows']} != reference:
            raise ValueError('fixed targets or poses changed across evaluation checkpoints')
    arms = {}
    for name, result, evaluations in (('a', left_result, left_evaluations), ('b', right_result, right_evaluations)):
        table = [assess_fit(report) for report in evaluations]
        passing = [row['step'] for row in table if row['step'] > 0 and row['accepted']]
        first = min(passing) if passing else None
        if result['first_accepted_step'] != first or bool(result['accepted']) != bool(passing):
            raise ValueError('trainer acceptance differs from independent complete-generation gate')
        arms[name] = {'first_accepted_step': first, 'final_accepted': table[-1]['accepted'], 'evaluations': table}
    selected = next((name for name in ('a', 'b') if arms[name]['first_accepted_step'] is not None), None)
    return {'stage': 'paired_fusion_G2_review', 'accepted': selected is not None, 'selected_arm': selected,
            'scope': 'fixed8/T4 fitting only; not heldout/generalization/capacity acceptance',
            'arms': arms, 'formal_training_allowed': False}


def compare_validation(baseline, candidate, *, group='valid'):
    def indexed(report):
        rows = [row for row in report['rows'] if row['group'] == group]
        lookup = {view_key(row): row for row in rows}
        if not rows or len(lookup) != len(rows):
            raise ValueError('comparison group is empty or contains duplicate poses')
        return lookup

    left, right = indexed(baseline), indexed(candidate)
    if set(left) != set(right):
        raise ValueError('validation comparison requires exactly the same assets and poses')
    if any(left[key]['target_tokens'] != right[key]['target_tokens'] for key in left):
        raise ValueError('validation target tokens differ')

    def summary(rows):
        values = list(rows.values())
        f1 = [float(row['original_edge_f1']) if row['success'] else 0. for row in values]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in f1):
            raise ValueError('invalid successful-generation F1')
        return {'success_rate': sum(bool(row['success']) for row in values) / len(values),
                'original_edge_f1_failed_as_zero': sum(f1) / len(values)}

    a, b = summary(left), summary(right)
    common = [key for key in left if left[key]['success'] and right[key]['success']]
    for key in common:
        if any(row['quantized_j2j'] is None or not math.isfinite(row['quantized_j2j'])
               for row in (left[key], right[key])):
            raise ValueError('successful common view lacks a finite J2J')
    delta = sum(right[key]['quantized_j2j'] - left[key]['quantized_j2j'] for key in common) / len(common) if common else None
    gates = {'generation_success_not_lower': b['success_rate'] >= a['success_rate'],
             'failure_as_zero_F1_not_lower': b['original_edge_f1_failed_as_zero'] >= a['original_edge_f1_failed_as_zero'],
             'paired_J2J_within_one_bin': delta is not None and delta <= 1 / 256}
    return {'group': group, 'views': len(left), 'baseline': a, 'candidate': b,
            'common_success_views': len(common), 'paired_J2J_delta': delta, 'J2J_tolerance': 1 / 256,
            'gates': gates, 'accepted': all(gates.values()), 'f1_reference': 'continuous_original_GT',
            'J2J_reference': 'quantized_GT_on_identical_common_success_views',
            'statistical_superiority_claim': False}
