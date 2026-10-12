from dataclasses import replace
import unittest

import torch

from rigweave.dynamic_rig.virtual_frame_rig import AnchorCoordinateReadout, ContinuousJointLocalizer
from test_virtual_frame_rig import settings


class CoordinateReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(231)

    def test_attachment_preserves_original_prediction_and_existing_parameters(self):
        config = settings()
        original = ContinuousJointLocalizer(config).eval()
        modified = ContinuousJointLocalizer(config).eval()
        modified.load_state_dict(original.state_dict(), strict=True)
        modified.coordinate_readout = AnchorCoordinateReadout(config.dim, 8)
        modified.config = replace(config, position_coordinate_readout=True, position_coordinate_dim=8)
        args = (torch.randn(2, 4, 16), torch.tensor([[True]*4, [True, True, False, False]]),
                torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 3))
        torch.testing.assert_close(original(*args), modified(*args), rtol=0, atol=2e-7)
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, modified.state_dict()[name], rtol=0, atol=0)
        modified(*args).sum().backward()
        self.assertGreater(modified.coordinate_readout.key.weight.grad.abs().sum().item(), 0)

    def test_learned_readout_depends_on_pairing_and_is_permutation_invariant(self):
        head = AnchorCoordinateReadout(16, 8)
        torch.nn.init.normal_(head.key.weight, std=.3)
        joints, memory, points = torch.randn(2, 4, 16), torch.randn(2, 7, 16), torch.randn(2, 7, 3)
        order = torch.tensor([4, 1, 6, 0, 2, 5, 3])
        expected = head(joints, memory, points)
        torch.testing.assert_close(expected, head(joints, memory[:, order], points[:, order]))
        self.assertGreater((expected - head(joints, memory[:, order], points)).abs().mean().item(), .01)
        torch.testing.assert_close(expected, head(joints, memory, points + 12), rtol=1e-5, atol=2e-6)

    def test_coordinate_values_have_direct_finite_gradients(self):
        head = AnchorCoordinateReadout(16, 8)
        torch.nn.init.normal_(head.key.weight, std=.3)
        joints = torch.randn(2, 4, 16, requires_grad=True)
        memory = torch.randn(2, 7, 16, requires_grad=True)
        points = torch.randn(2, 7, 3, requires_grad=True)
        head(joints, memory, points).square().sum().backward()
        for value in (joints, memory, points, *head.parameters()):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
