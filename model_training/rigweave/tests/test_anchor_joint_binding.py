import copy
import unittest

import torch

from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig, coordinate_objective
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots


class AnchorJointBindingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.config = AnchorJointConfig(input_dim=12, dim=16, heads=4, reference_depth=1, target_depth=1, max_joints=8)
        self.head = AnchorJointBinding(self.config)
        self.features = torch.randn(2, 3, 7, 12, requires_grad=True)
        self.points = torch.randn(2, 3, 7, 3)
        self.normals = torch.randn_like(self.points)
        self.target = torch.randn(2, 3, 4, 3) * .2
        self.parents = torch.tensor([[-1, 0, 0, 1], [-1, 0, -2, -2]])
        self.mask = self.parents != -2
        self.reference = torch.tensor([0, 0])

    def args(self):
        return [self.features, self.points, self.normals, reference_slots(self.target, self.reference),
            self.parents, self.mask, self.reference]

    def loss(self, prediction, target=None):
        return coordinate_objective(prediction, self.target if target is None else target,
            self.parents, self.mask, self.reference)['loss_per_asset'].mean()

    def test_both_arms_have_identical_parameters_and_initial_values(self):
        other = AnchorJointBinding(self.config, 'target_content')
        other.load_state_dict(self.head.state_dict(), strict=True)
        self.assertEqual(list(other.state_dict()), list(self.head.state_dict()))
        self.assertEqual(sum(p.numel() for p in other.parameters()), sum(p.numel() for p in self.head.parameters()))

    def test_reference_exact_and_padding_zero(self):
        output = self.head(*self.args())
        expected = self.target[:, 0].masked_fill(~self.mask[..., None], 0.)
        torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(output[1, :, 2:], torch.zeros(3, 2, 3)))

    def test_reference_association_reused_at_every_pose(self):
        _, weights = self.head(*self.args(), return_association=True)
        torch.testing.assert_close(weights[:, 0], weights[:, 1], atol=0, rtol=0)
        sums = weights.sum(-1)
        torch.testing.assert_close(sums, self.mask[:, None, None].expand_as(sums).float())

    def test_pose_permutation(self):
        self.head.eval()
        expected = self.head(*self.args())
        order = torch.tensor([2, 0, 1])
        args = self.args()
        args[:3] = [x[:, order] for x in args[:3]]
        args[-1] = torch.argsort(order)[self.reference]
        torch.testing.assert_close(self.head(*args), expected[:, order], atol=2e-6, rtol=2e-6)

    def test_common_anchor_permutation(self):
        self.head.eval()
        args = self.args()
        order = torch.tensor([3, 6, 0, 5, 1, 2, 4])
        args[:3] = [x[:, :, order] for x in args[:3]]
        torch.testing.assert_close(self.head(*args), self.head(*self.args()), atol=2e-6, rtol=2e-6)

    def test_legal_joint_relabeling(self):
        self.head.eval()
        args = [a[:1] for a in self.args()]
        expected = self.head(*args)
        order = torch.tensor([0, 2, 1, 3])
        args[3] = args[3][:, order]
        args[4] = torch.tensor([[-1, 0, 0, 2]])
        torch.testing.assert_close(self.head(*args), expected[:, :, order], atol=2e-6, rtol=2e-6)

    def test_target_content_is_invariant_to_target_anchor_order_but_transport_is_not(self):
        args = self.args()
        args[:3] = [x.clone() for x in args[:3]]
        order = torch.tensor([3, 6, 0, 5, 1, 2, 4])
        for x in args[:3]:
            x[:, 1:] = x[:, 1:, order]
        self.head.eval()
        before, after = self.head(*self.args()), self.head(*args)
        self.assertGreater(float((before - after).detach().abs().max()), 1e-6)
        self.head.mode = 'target_content'
        torch.testing.assert_close(self.head(*self.args()), self.head(*args), atol=2e-6, rtol=2e-6)

    def test_feature_gradients_reach_every_pose_and_all_head_parameters(self):
        self.loss(self.head(*self.args())).backward()
        for t in range(3):
            self.assertGreater(float(self.features.grad[:, t].norm()), 0.)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.head.parameters()))

    def test_loss_excludes_reference_padding_and_matches_definition(self):
        prediction = self.head(*self.args())
        changed = self.target.clone()
        changed[:, 0] = 1000
        changed[1, :, 2:] = float('nan')
        torch.testing.assert_close(self.loss(prediction, changed), self.loss(prediction))
        exact = coordinate_objective(self.target.masked_fill(~self.mask[:, None, :, None], 0.),
            self.target, self.parents, self.mask, self.reference)
        self.assertTrue(torch.equal(exact['loss_per_asset'], torch.zeros(2)))

    def test_no_target_argument_or_rng_consumption(self):
        self.head.eval()
        rng = torch.get_rng_state().clone()
        torch.testing.assert_close(self.head(*self.args()), self.head(*self.args()), atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_checkpoint_gradients_match(self):
        other = copy.deepcopy(self.head)
        other.config = AnchorJointConfig(**{**self.config.__dict__, 'gradient_checkpointing': False})
        self.loss(self.head(*self.args())).backward()
        self.loss(other(*self.args())).backward()
        for a, b in zip(self.head.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)

    def test_synthetic_coordinates_can_be_learned(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=.01)
        before = float(self.loss(self.head(*self.args())).detach())
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.head(*self.args()))
            loss.backward()
            optimizer.step()
        self.assertLess(float(self.loss(self.head(*self.args())).detach()), before * .35)

    def test_invalid_reference_rejected(self):
        args = self.args()
        args[4] = self.parents.clone()
        args[4][0, 1] = 2
        with self.assertRaises(ValueError):
            self.head(*args)


if __name__ == '__main__':
    unittest.main()
