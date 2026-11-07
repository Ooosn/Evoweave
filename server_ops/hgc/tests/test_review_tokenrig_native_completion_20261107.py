import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'review_tokenrig_native_completion_20261107.py'
SPEC = importlib.util.spec_from_file_location('native_completion', PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class CompletionAggregationTests(unittest.TestCase):
    def test_equal_asset_weight_not_equal_row_weight(self):
        rows = [{'asset_id': 'a', 'score': 0.}, {'asset_id': 'a', 'score': 0.},
            {'asset_id': 'b', 'score': 1.}]
        self.assertEqual(review.balanced(rows, 'score'), .5)

    def test_rejects_nonfinite_metrics(self):
        with self.assertRaises(RuntimeError):
            review.balanced([{'asset_id': 'a', 'score': float('nan')}], 'score')

    def test_groups_do_not_mix_head_and_copy(self):
        rows = [{'asset_id': 'a', 'split': 'valid', 'method': method, 'reference_kind': 'GT',
            **{key: value for key in ('normalized_mpjpe', 'edge_f1', 'normalized_j2j', 'pck_005')}}
            for method, value in [('head', .2), ('copy', .8)]]
        groups = [{'split': 'valid', 'method': method, 'assets': 1, 'rows': 1,
            **{key: value for key in ('normalized_mpjpe', 'edge_f1', 'normalized_j2j', 'pck_005')}}
            for method, value in [('head', .2), ('copy', .8)]]
        review.verify_groups(rows, {'groups': groups})
        groups[0]['normalized_mpjpe'] = .8
        with self.assertRaises(RuntimeError):
            review.verify_groups(rows, {'groups': groups})


if __name__ == '__main__':
    unittest.main()
