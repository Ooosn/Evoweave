import unittest

import torch

from rigweave.dynamic_rig.reference_flow import ReferenceFlow, ReferenceFlowConfig, graph_hops
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots


class ReferenceFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.head = ReferenceFlow(ReferenceFlowConfig(input_dim=12, dim=16, heads=4,
            reference_depth=1, flow_depth=2, max_joints=8))
        self.features = torch.randn(2, 3, 7, 12, requires_grad=True)
        self.points = torch.randn(2, 3, 7, 3)
        self.normals = torch.randn_like(self.points)
        self.targets = torch.randn(2, 3, 4, 3) * .2
        self.parents = torch.tensor([[-1, 0, 0, 1], [-1, 0, -2, -2]])
        self.mask = self.parents != -2
        self.reference = torch.tensor([0, 2])
        self.noise = torch.randn_like(self.targets)
        self.tau = torch.tensor([.2, .7])

    def inputs(self):
        return (self.features, self.points, self.normals, reference_slots(self.targets, self.reference),
                self.parents, self.mask, self.reference)

    def loss(self, **kwargs):
        return self.head(*self.inputs(), targets=kwargs.get('targets', self.targets),
            noise=self.noise, tau=self.tau)['loss_per_asset']

    def test_flow_alone_reaches_mesh_features(self):
        self.loss().mean().backward()
        self.assertGreater(float(self.features.grad.norm()), 0.)
        for frame in range(3):
            self.assertGreater(float(self.features.grad[:, frame].norm()), 0.)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in self.head.parameters()))

    def test_exact_reference_and_padding_during_all_steps(self):
        self.head.eval()
        condition = self.head.condition(*self.inputs())
        x = self.head.clamp_reference(self.noise, condition)
        for k in range(4):
            v = self.head.velocity(x, torch.full((2,), k / 4), condition)
            self.assertTrue(torch.equal(reference_slots(v, self.reference), torch.zeros(2, 4, 3)))
            x = self.head.clamp_reference(x + v / 4, condition)
            expected = reference_slots(self.targets, self.reference).masked_fill(~self.mask[:, :, None], 0.)
            self.assertTrue(torch.equal(reference_slots(x, self.reference), expected))
            self.assertTrue(torch.equal(x[1, :, 2:], torch.zeros(3, 2, 3)))
        actual = self.head.generate(*self.inputs(), noise=self.noise, steps=4)
        torch.testing.assert_close(actual, x)

    def test_padding_excluded_from_loss(self):
        expected = self.loss()
        changed = self.targets.clone()
        changed[1, :, 2:] = float('nan')
        torch.testing.assert_close(self.loss(targets=changed), expected)

    def test_pose_permutation_equivariance(self):
        self.head.eval()
        expected = self.head.generate(*self.inputs(), noise=self.noise, steps=2)
        order = torch.tensor([2, 0, 1])
        ref = torch.argsort(order)[self.reference]
        actual = self.head.generate(self.features[:, order], self.points[:, order], self.normals[:, order],
            reference_slots(self.targets, self.reference), self.parents, self.mask, ref,
            noise=self.noise[:, order], steps=2)
        torch.testing.assert_close(actual, expected[:, order], atol=3e-6, rtol=3e-6)

    def test_generate_has_no_targets_and_does_not_consume_rng(self):
        self.head.eval()
        before = torch.get_rng_state().clone()
        a = self.head.generate(*self.inputs(), noise=self.noise, steps=2)
        b = self.head.generate(*self.inputs(), noise=self.noise, steps=2)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_velocity_loss_matches_linear_path(self):
        torch.nn.init.zeros_(self.head.output[-1].weight)
        torch.nn.init.zeros_(self.head.output[-1].bias)
        actual = self.loss()
        valid = self.mask[:, None] & (torch.arange(3)[None] != self.reference[:, None])[:, :, None]
        expected = (self.targets - self.noise).square().mean(-1).masked_fill(~valid, 0.).sum((1, 2)) / valid.sum((1, 2))
        torch.testing.assert_close(actual, expected)

    def test_graph_hops(self):
        hops = graph_hops(self.parents, self.mask, 8)
        self.assertEqual(hops[0].tolist(), [[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 3], [2, 1, 3, 0]])

    def test_invalid_tree_and_time_fail_closed(self):
        self.parents[0, 2] = 3
        with self.assertRaises(ValueError):
            self.loss()
        self.parents[0, 2] = 0
        self.tau[0] = 1.1
        with self.assertRaises(ValueError):
            self.loss()

    def test_one_pose_is_not_a_training_case(self):
        args = self.inputs()
        with self.assertRaises(ValueError):
            self.head.condition(*(x[:, :1] for x in args[:3]), *args[3:])

    def test_checkpointed_and_plain_gradients_match(self):
        import dataclasses
        plain = ReferenceFlow(dataclasses.replace(self.head.config, gradient_checkpointing=False))
        plain.load_state_dict(self.head.state_dict())
        self.loss().sum().backward()
        plain(*self.inputs(), targets=self.targets, noise=self.noise, tau=self.tau)['loss_per_asset'].sum().backward()
        for a, b in zip(self.head.parameters(), plain.parameters()):
            torch.testing.assert_close(a.grad, b.grad)


if __name__ == '__main__':
    unittest.main()
