import unittest

import numpy as np

from rigweave.dynamic_rig.skeleton_topology_metrics import _edge_overlap, _edge_set, _topology_metrics


class OriginalTopologyMetricTests(unittest.TestCase):
    def test_invalid_parent_entries_are_not_edges(self):
        self.assertEqual(_edge_set([None, 0, 2, 99, -1]), {(0, 1)})

    def test_joint_reordering_uses_original_nearest_mapping(self):
        truth = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0]], dtype=np.float32)
        order = [2, 0, 1]
        prediction = truth[order]
        result = _topology_metrics(prediction, [2, -1, 1], truth, [-1, 0, 1])
        self.assertEqual(result['edge_f1'], 1.)
        self.assertEqual(result['pred_matched_edges'], 2)

    def test_partial_tree_not_exact_equality(self):
        truth = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]], dtype=np.float32)
        result = _topology_metrics(truth, [-1, 0, 0], truth, [-1, 0, 1])
        self.assertEqual(result['edge_f1'], .5)

    def test_coincident_joints_keep_original_tie_behavior(self):
        joints = np.zeros((2, 3), dtype=np.float32)
        result = _topology_metrics(joints, [-1, 0], joints, [-1, 0])
        self.assertEqual(result['edge_f1'], 0.)

    def test_root_only_and_empty_target(self):
        root = np.zeros((1, 3), dtype=np.float32)
        self.assertEqual(_topology_metrics(root, [-1], root, [-1])['edge_f1'], 1.)
        self.assertEqual(_edge_overlap(np.zeros((2, 3)), [-1, 0], np.empty((0, 3)), [])['score'], 0.)


if __name__ == '__main__':
    unittest.main()
