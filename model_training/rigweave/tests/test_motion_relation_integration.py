from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingBlock, AnchorWiseAlternatingMotionEncoder
from rigweave.dynamic_rig.motion_evidence import MotionEvidence
from rigweave.dynamic_rig.motion_relation_residual import MotionRelationResidual


def evidence():
    states = torch.rand(1, 5, 5, 3).softmax(-1)
    return MotionEvidence(states, torch.rand(1, 5, 5), torch.ones(1, 5, dtype=torch.bool))


class RelationIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.encoder = AnchorWiseAlternatingMotionEncoder(16, depth=2, heads=2, register_tokens=2,
            motion_evidence_fusion="bias", motion_evidence_heads=2, gradient_checkpointing=True)
        self.encoder.requires_grad_(False)
        self.encoder.train()

    def test_default_checkpoint_and_zero_init_parity(self):
        state = self.encoder.state_dict()
        self.assertFalse(any("relation_residual" in key for key in state))
        self.encoder.load_state_dict(state, strict=True)
        x = torch.randn(1, 3, 5, 16, requires_grad=True)
        e = evidence()
        expected = self.encoder(x, motion_evidence=e)
        self.encoder.enable_relation_residual(4)
        actual = self.encoder(x, motion_evidence=e)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        for block in self.encoder.blocks:
            self.assertIsNotNone(block.relation_residual.out_proj.weight.grad)
            self.assertGreater(float(block.relation_residual.out_proj.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for n, p in self.encoder.named_parameters() if "relation_residual" not in n))
        with self.assertRaises(RuntimeError):
            self.encoder.enable_relation_residual(4)

    def test_residual_does_not_touch_prefix(self):
        block = AnchorWiseAlternatingBlock(4, 2)
        block.pose_inner = nn.Identity()
        block.anchor_temporal = nn.Identity()
        block.relation_residual = MotionRelationResidual(4, 2)
        nn.init.normal_(block.relation_residual.out_proj.weight, std=0.1)
        x = torch.randn(1, 2, 8, 4)
        actual = block(x, relation_states=evidence().states)
        torch.testing.assert_close(actual[:, :, :3], x[:, :, :3], atol=0, rtol=0)
        self.assertFalse(torch.equal(actual[:, :, 3:], x[:, :, 3:]))

    def test_override_only_changes_relation_branch(self):
        x = torch.randn(1, 3, 5, 16, requires_grad=True)
        e = evidence()
        with self.assertRaises(ValueError):
            self.encoder(x, motion_evidence=e, relation_states=e.states)
        self.encoder.enable_relation_residual(4)
        unknown = torch.zeros_like(e.states)
        unknown[..., 0] = 1
        snapshots = []
        handle = self.encoder.blocks[0].motion_injection.register_forward_pre_hook(
            lambda module, args: snapshots.append(args[2].states.clone()))
        self.encoder(x, motion_evidence=e, relation_states=unknown)
        handle.remove()
        torch.testing.assert_close(snapshots[0], e.states, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.encoder(x, motion_evidence=e, relation_states=unknown[:, :4])


if __name__ == "__main__":
    unittest.main()
