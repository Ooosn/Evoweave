"""Independent generation-quality and matched-validation acceptance tests."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fusion_acceptance import assess_fit, compare_validation, review_fit_pair


def fixture(step=32):
    rows, auxiliary = [], []
    for asset in range(8):
        for frame in range(4):
            tokens = [1, asset+3, frame+15, 2]
            rows.append({'asset_id': str(asset), 'raw_frame_index': frame+10,
                         'success': True, 'status': 'eos', 'exact_count': True,
                         'main_accuracy': 1., 'quantized_j2j': 0., 'quantized_edge_f1': 1.,
                         'original_edge_f1': 1., 'target_tokens': tokens,
                         'generated_tokens': tokens.copy(), 'group': 'valid'})
        parents = [-1] + list(range(asset))
        auxiliary.append({'asset_id': str(asset), 'success': True, 'natural_eos': True, 'status': 'eos',
                          'target_parents': parents, 'predicted_parents': parents.copy(),
                          'generated_tokens': list(range(asset+3))})
    return {'step': step, 'rows': rows, 'auxiliary_rows': auxiliary}


class AcceptanceTests(unittest.TestCase):
    def test_main_and_auxiliary_must_pass_same_checkpoint(self):
        first, second = fixture(32), fixture(64)
        first['auxiliary_rows'][0]['predicted_parents'] = [-1, 0]
        for row in second['rows'][:4]:
            row.update(generated_tokens=[1, 99, 2], quantized_edge_f1=.5)
        a, b = assess_fit(first), assess_fit(second)
        self.assertTrue(a['main_passed'])
        self.assertFalse(a['auxiliary_passed'])
        self.assertFalse(b['main_passed'])
        self.assertTrue(b['auxiliary_passed'])
        self.assertFalse(a['accepted'] or b['accepted'])

    def test_exact_tokens_override_nonreflexive_F1_but_not_distance(self):
        report = fixture()
        for row in report['rows']:
            row['quantized_edge_f1'] = .6
        self.assertTrue(assess_fit(report)['accepted'])
        for row in report['rows'][:4]:
            row['quantized_j2j'] = 4 / 256 + 1e-6
        self.assertFalse(assess_fit(report)['accepted'])

    def test_complete_natural_eos_and_cohort_are_mandatory(self):
        report = fixture()
        report['rows'][0]['status'] = 'max_tokens'
        self.assertFalse(assess_fit(report)['accepted'])
        report = fixture()
        report['auxiliary_rows'][0]['natural_eos'] = False
        self.assertFalse(assess_fit(report)['accepted'])
        report = fixture()
        report['rows'][0] = deepcopy(report['rows'][1])
        with self.assertRaises(ValueError):
            assess_fit(report)

    def test_fit_pair_same_budget_first_pass_and_regression_are_explicit(self):
        result = {'execution_complete': True, 'completed_updates': 1024, 'asset_exposures': 8192,
                  'first_accepted_step': 32, 'accepted': True, 'code_commit': 'source'}
        ready = {'passed': True, 'full_initial_state_sha256': 'initial', 'inputs_sha256': 'inputs',
                 'comparison_contract_sha256': 'recipe', 'code_commit': 'source', 'assets': list(range(8))}
        left = [fixture(step) for step in (0, 32, 64, 128, 256, 512, 1024)]
        right = deepcopy(left)
        for row in left[-1]['rows']:
            row['main_accuracy'] = .98
        report = review_fit_pair(result, result, left, right, ready, ready)
        self.assertEqual(report['selected_arm'], 'a')
        self.assertFalse(report['arms']['a']['final_accepted'])
        self.assertTrue(report['arms']['b']['final_accepted'])
        with self.assertRaises(ValueError):
            review_fit_pair({**result, 'completed_updates': 128}, result, left, right, ready, ready)
        with self.assertRaises(ValueError):
            review_fit_pair(result, result, left, right, ready, {**ready, 'inputs_sha256': 'wrong'})
        right[-1]['rows'][0]['target_tokens'] = [1, 3, 999, 2]
        with self.assertRaises(ValueError):
            review_fit_pair(result, result, left, right, ready, ready)

    def test_validation_scores_failures_zero_not_success_only(self):
        baseline, candidate = fixture(), fixture()
        for row in baseline['rows']:
            row['original_edge_f1'] = .8
        for row in candidate['rows'][:16]:
            row['success'] = False
        result = compare_validation(baseline, candidate)
        self.assertEqual(result['candidate']['original_edge_f1_failed_as_zero'], .5)
        self.assertFalse(result['accepted'])

    def test_paired_J2J_uses_same_common_success_poses_and_exact_boundary(self):
        baseline, candidate = fixture(), fixture()
        baseline['rows'][0].update(success=False, quantized_j2j=None)
        candidate['rows'][0]['quantized_j2j'] = 99.
        for row in candidate['rows'][1:]:
            row['quantized_j2j'] = 1 / 256
        result = compare_validation(baseline, candidate)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['common_success_views'], 31)
        self.assertEqual(result['paired_J2J_delta'], 1 / 256)
        candidate['rows'][1]['quantized_j2j'] += .001
        self.assertFalse(compare_validation(baseline, candidate)['accepted'])

    def test_no_common_success_is_unknown_not_an_automatic_pass(self):
        baseline, candidate = fixture(), fixture()
        for report in (baseline, candidate):
            for row in report['rows']:
                row.update(success=False, quantized_j2j=None)
        result = compare_validation(baseline, candidate)
        self.assertIsNone(result['paired_J2J_delta'])
        self.assertFalse(result['accepted'])

    def test_validation_rejects_changed_targets_or_missing_views(self):
        baseline, candidate = fixture(), fixture()
        candidate['rows'][0]['target_tokens'][1] += 1
        with self.assertRaises(ValueError):
            compare_validation(baseline, candidate)
        candidate = fixture()
        candidate['rows'].pop()
        with self.assertRaises(ValueError):
            compare_validation(baseline, candidate)


if __name__ == '__main__':
    unittest.main()
