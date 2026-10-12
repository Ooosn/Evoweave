import unittest

import torch

from diagnose_virtual_fitting import fitting_passed, input_digest, single_asset_inputs


class SingleFitTests(unittest.TestCase):
    def test_static_input_changes_all_frame_fields_without_mutating_source(self):
        surface = {k: torch.randn(1, 8, 4, 3) for k in
                   ("dense_points", "dense_normals", "query_points", "query_normals")}
        surface["query_indices"] = torch.arange(4)[None]
        targets = {"asset_id": ["one"], "frame_target_joints": torch.randn(1, 8, 5, 3),
                   "selected_frames": torch.arange(8)[None], "frame_mesh_bbox_side": torch.randn(1, 8),
                   "target_parents": torch.tensor([[-1, 0, 1, 2, 3]])}
        before = input_digest(surface, targets)
        static, labels = single_asset_inputs(surface, targets, "single_static")
        self.assertEqual(before, input_digest(surface, targets))
        for k in ("dense_points", "dense_normals", "query_points", "query_normals"):
            self.assertTrue(torch.equal(static[k], surface[k][:, :1].expand_as(surface[k])))
        self.assertTrue(torch.equal(labels["frame_target_joints"], targets["frame_target_joints"][:, :1].expand_as(targets["frame_target_joints"])))
        self.assertTrue(torch.equal(labels["target_parents"], targets["target_parents"]))
        self.assertNotEqual(before, input_digest(static, labels))

    def test_dynamic_control_retains_values_and_hash(self):
        surface = {"query_points": torch.randn(1, 8, 4, 3)}
        targets = {"asset_id": ["one"], "frame_target_joints": torch.randn(1, 8, 5, 3)}
        same, labels = single_asset_inputs(surface, targets, "single_dynamic")
        self.assertEqual(input_digest(surface, targets), input_digest(same, labels))
        with self.assertRaises(ValueError):
            single_asset_inputs(surface, {"asset_id": ["one", "two"]}, "single_dynamic")

    def test_fit_needs_coordinates_structure_and_generation(self):
        good = {"xyz_l1": .004, "query_xyz_l1": .003, "structure_ce": .002,
                "all_generated_parents_exact": True, "generation_F1": .98, "oracle_query_F1": 1.}
        self.assertTrue(fitting_passed(good, .01))
        for field, value in (("xyz_l1", .05), ("query_xyz_l1", .05), ("structure_ce", .1),
                             ("all_generated_parents_exact", False), ("generation_F1", .5)):
            self.assertFalse(fitting_passed({**good, field: value}, .01))


if __name__ == "__main__":
    unittest.main()
