import unittest

import torch

from rigweave.dynamic_rig.reference_skeleton_xyz import (
    ReferenceSkeletonXYZ, ReferenceXYZConfig, joint_relations, localization_loss,
    local_rigid_transfer, validate_reference,
)


class ReferenceXYZTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261027)
        self.config = ReferenceXYZConfig(input_dim=8, dim=32, heads=4, reference_depth=1, target_depth=1)
        self.model = ReferenceSkeletonXYZ(self.config)
        self.features = torch.randn(2, 3, 8, 8)
        self.points = torch.randn(2, 3, 8, 3) * .3
        self.normals = torch.nn.functional.normalize(torch.randn_like(self.points), dim=-1)
        self.xyz = torch.randn(2, 4, 3) * .2
        self.parents = torch.tensor([[-1, 0, 0, 1], [-1, 0, -2, -2]])
        self.mask = self.parents != -2
        self.ref = torch.tensor([0, 1])
        self.xyz[~self.mask] = 0

    def call(self, **overrides):
        values = dict(frame_features=self.features, anchor_points=self.points, anchor_normals=self.normals,
                      reference_xyz=self.xyz, parents=self.parents, joint_mask=self.mask, reference_frame=self.ref)
        values.update(overrides)
        return self.model(**values)

    def enable_output(self):
        torch.nn.init.normal_(self.model.output[-1].weight, std=.04)

    def test_shape_copy_initialization_and_nonreference_loss(self):
        prediction = self.call()
        expected = self.xyz[:, None].expand(-1, 3, -1, -1)
        torch.testing.assert_close(prediction, expected)
        targets = prediction.detach() + .2
        targets[torch.arange(2), self.ref] = 999
        loss = localization_loss(prediction, targets, self.mask, self.ref)
        torch.testing.assert_close(loss, torch.full((2,), .2))
        self.enable_output()
        prediction = self.call()
        torch.testing.assert_close(prediction[torch.arange(2), self.ref], self.xyz)
        self.assertTrue((prediction[~self.mask[:, None].expand(-1, 3, -1)] == 0).all())

    def test_structure_validation(self):
        for bad in (torch.tensor([[-1, 0, 2, 1], [-1, 0, -2, -2]]),
                    torch.tensor([[-1, -1, 0, 1], [-1, 0, -2, -2]])):
            with self.assertRaises(ValueError):
                self.call(parents=bad)
        with self.assertRaises(ValueError):
            self.call(reference_frame=torch.tensor([3, 0]))
        with self.assertRaises(ValueError):
            localization_loss(torch.zeros(2, 1, 4, 3), torch.zeros(2, 1, 4, 3), self.mask, torch.zeros(2, dtype=torch.long))
        relation = joint_relations(self.parents)
        self.assertEqual(relation[0, 1, 0], 2)
        self.assertEqual(relation[0, 0, 1], 3)
        self.assertEqual(relation[0, 1, 2], 4)
        self.assertEqual(relation[0, 3, 2], 0)

    def test_frame_and_anchor_permutation(self):
        self.model.eval()
        self.enable_output()
        expected = self.call()
        order = torch.tensor([2, 0, 1])
        actual = self.call(frame_features=self.features[:, order], anchor_points=self.points[:, order],
                           anchor_normals=self.normals[:, order], reference_frame=torch.argsort(order)[self.ref])
        torch.testing.assert_close(actual, expected[:, order], atol=2.e-6, rtol=2.e-5)
        order = torch.tensor([7, 1, 5, 2, 3, 4, 0, 6])
        actual = self.call(frame_features=self.features[:, :, order], anchor_points=self.points[:, :, order],
                           anchor_normals=self.normals[:, :, order])
        torch.testing.assert_close(actual, expected, atol=2.e-6, rtol=2.e-5)

    def test_joint_permutation_and_padding(self):
        self.model.eval()
        self.enable_output()
        arguments = dict(frame_features=self.features[:1], anchor_points=self.points[:1],
                         anchor_normals=self.normals[:1], reference_xyz=self.xyz[:1],
                         parents=self.parents[:1], joint_mask=self.mask[:1], reference_frame=self.ref[:1])
        expected = self.model(**arguments)
        order = torch.tensor([0, 2, 1, 3])
        arguments.update(reference_xyz=self.xyz[:1, order], parents=torch.tensor([[-1, 0, 0, 2]]))
        actual = self.model(**arguments)
        torch.testing.assert_close(actual, expected[:, :, order], atol=2.e-6, rtol=2.e-5)
        single = self.model(self.features[1:], self.points[1:], self.normals[1:], self.xyz[1:, :2],
                            self.parents[1:, :2], self.mask[1:, :2], self.ref[1:])
        torch.testing.assert_close(single, self.call()[1:, :, :2], atol=2.e-6, rtol=2.e-5)

    def test_target_geometry_reference_and_feature_gradients(self):
        self.enable_output()
        features = self.features.clone().requires_grad_()
        points = self.points.clone().requires_grad_()
        xyz = self.xyz.clone().requires_grad_()
        prediction = self.call(frame_features=features, anchor_points=points, reference_xyz=xyz)
        target = torch.randn_like(prediction) * .2
        localization_loss(prediction, target, self.mask, self.ref).mean().backward()
        for name, value in (('features', features), ('geometry', points), ('reference', xyz)):
            self.assertTrue(torch.isfinite(value.grad).all(), name)
            self.assertGreater(float(value.grad.abs().sum()), 0, name)
        for blocks in (self.model.reference_blocks, self.model.target_blocks):
            self.assertGreater(float(blocks[0].cross.in_proj_weight.grad.abs().sum()), 0)
            self.assertGreater(float(blocks[0].graph_bias.weight.grad.abs().sum()), 0)

    def test_rigid_transfer_rotation_translation(self):
        reference = torch.randn(2, 16, 3)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = torch.tensor([.2, -.3, .4])
        targets = reference @ rotation + translation
        points = torch.stack((reference, targets), 1)
        joints = torch.randn(2, 4, 3)
        prediction, degenerate = local_rigid_transfer(joints, points, torch.zeros(2, dtype=torch.long))
        self.assertFalse(bool(degenerate.any()))
        torch.testing.assert_close(prediction[:, 0], joints, atol=2.e-5, rtol=2.e-5)
        torch.testing.assert_close(prediction[:, 1], joints @ rotation + translation, atol=2.e-5, rtol=2.e-5)

    def test_rigid_transfer_degenerate_translation(self):
        reference = torch.zeros(1, 8, 3)
        target = reference + torch.tensor([.2, .3, -.4])
        points = torch.stack((reference, target), 1)
        joints = torch.randn(1, 4, 3)
        prediction, degenerate = local_rigid_transfer(joints, points, torch.zeros(1, dtype=torch.long))
        self.assertTrue(bool(degenerate.all()))
        torch.testing.assert_close(prediction[:, 1], joints + torch.tensor([.2, .3, -.4]))

    def test_synthetic_translation_can_fit(self):
        points = self.points[:, :1].expand(-1, 3, -1, -1).clone()
        translations = torch.tensor([[[0., 0., 0.], [.2, -.1, .1], [-.1, .2, -.2]],
                                     [[0., 0., 0.], [-.2, .1, .2], [.1, -.2, -.1]]])
        points += translations[:, :, None]
        targets = self.xyz[:, None] + translations[:, :, None]
        ref = torch.zeros(2, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.003)
        initial = float(localization_loss(self.call(anchor_points=points, reference_frame=ref),
                                         targets, self.mask, ref).mean().detach())
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.call(anchor_points=points, reference_frame=ref)
            loss = localization_loss(prediction, targets, self.mask, ref).mean()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial * .15)


if __name__ == '__main__':
    unittest.main()
