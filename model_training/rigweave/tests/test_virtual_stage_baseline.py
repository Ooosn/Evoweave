from types import SimpleNamespace
import unittest

import numpy as np
import torch

from eval_virtual_stage_baseline import joint_coordinate_offsets, summarize, verify_saved_input


class VirtualStageBaselineTests(unittest.TestCase):
    def test_branch_coordinate_positions_preserve_joint_identity(self):
        tokenizer = SimpleNamespace(token_id_branch=8, eos=9, num_discrete=8)
        ids = [10, 11, 0, 1, 2, 3, 4, 5, 8, 0, 1, 2, 6, 7, 0, 9]
        np.testing.assert_array_equal(joint_coordinate_offsets(ids, [-1, 0, 0], tokenizer),
                                      [[2, 3, 4], [5, 6, 7], [12, 13, 14]])
        with self.assertRaises(ValueError):
            joint_coordinate_offsets(ids, [-1, 0, 1], tokenizer)

    def test_saved_inputs_fail_on_frame_target_or_scale_changes(self):
        batch = {"frame_target_joints": torch.zeros(1, 2, 1, 3), "target_parents": torch.tensor([[-1]]),
                 "selected_frames": torch.tensor([[2, 3]]), "query_center": torch.zeros(1, 3),
                 "query_scale": torch.ones(1), "path": ["asset.npz"]}
        saved = {"target_joints_common": np.zeros((2, 1, 3)), "target_parents": [-1],
                 "selected_frames": [[2, 3]], "query_center": [[0., 0., 0.]], "query_scale": [1.], "path": ["asset.npz"]}
        verify_saved_input(batch, saved)
        for field, changed in (("selected_frames", [[2, 4]]), ("query_scale", [1.1]),
                               ("target_joints_common", np.ones((2, 1, 3)))):
            with self.assertRaises(ValueError):
                verify_saved_input(batch, {**saved, field: changed})

    def test_empty_split_does_not_imply_zero_error(self):
        result = summarize([])
        self.assertIsNone(result["valid"]["edge_f1"])
        self.assertEqual(result["valid"]["assets"], 0)


if __name__ == "__main__":
    unittest.main()
