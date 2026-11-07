from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tokenrig_comparison_matrix import balanced, strict_assets


class ComparisonMatrixTests(unittest.TestCase):
    def test_any_identity_mismatch_excludes_whole_asset(self):
        import copy
        rows = [{'asset_id': a, 'frame_index': f, 'raw_frame_index': f,
            'normalization_sha256': 'n', 'surface_sha256': 's', 'original_GT_sha256': 'g',
            'target_mesh_diagonal': 1., 'gt_joint_count': 5} for a in ('a', 'b') for f in range(8)]
        other = copy.deepcopy(rows)
        other[2]['surface_sha256'] = 'different'
        ids, excluded = strict_assets({'old': rows, 'new': other})
        self.assertEqual(ids, ['b'])
        self.assertEqual(excluded, {'a': ['surface_sha256']})

    def test_equal_asset_weight_and_nonfinite_rejection(self):
        self.assertEqual(balanced([{'asset_id': 'a', 's': 0.}, {'asset_id': 'a', 's': 0.},
            {'asset_id': 'b', 's': 1.}], 's'), .5)
        with self.assertRaises(ValueError):
            balanced([{'asset_id': 'a', 's': float('nan')}], 's')


if __name__ == '__main__':
    unittest.main()
