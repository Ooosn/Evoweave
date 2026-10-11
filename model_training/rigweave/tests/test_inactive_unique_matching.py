from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from report_inactive_joint_capability import unique_matches


class UniqueInactiveMatchingTest(unittest.TestCase):
    def test_one_prediction_cannot_cover_two_target_joints(self):
        result = unique_matches([[0, 0, 0], [0.01, 0, 0]], [-1, 0], [[0.005, 0, 0]], [-1])
        self.assertEqual(sum(result["covered"]), 1)
        self.assertEqual(sum(result["located_edge_found"]), 0)

    def test_cardinality_precedes_distance(self):
        result = unique_matches([[0, 0, 0], [0.04, 0, 0]], [-1, 0],
                                [[0, 0, 0], [-0.04, 0, 0]], [-1, 0])
        self.assertEqual(result["assignment"], [1, 0])
        self.assertEqual(result["covered"], [True, True])

    def test_connection_requires_correct_direction_and_locations(self):
        target = [[0, 0, 0], [0, 0.2, 0], [0, 0.4, 0]]
        result = unique_matches(target, [-1, 0, 1], target, [-1, 0, 0])
        self.assertEqual(result["located_edge_found"], [False, True, False])
        failed = unique_matches(target, [-1, 0, 1], None, [])
        self.assertEqual(failed["covered"], [False, False, False])


if __name__ == "__main__":
    unittest.main()
