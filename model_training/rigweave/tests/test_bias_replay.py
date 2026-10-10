from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bias_replay import MotionCapture, MotionInput, captured_forward, rng_snapshot, rng_digest, restore_rng, tensor_delta


@dataclass
class Evidence:
    states: torch.Tensor
    anchor_features: torch.Tensor
    valid_groups: torch.Tensor


class Encoder(torch.nn.Module):
    def forward(self, tokens, *, query_points, motion_evidence):
        return tokens + torch.rand_like(tokens) + motion_evidence.states + query_points


class Model(torch.nn.Module):
    condition_fusion = "dynamic"
    branch_prior = None

    def __init__(self):
        super().__init__()
        self.conditioner = torch.nn.Module()
        self.conditioner.motion_encoder = Encoder()

    def build_condition(self, batch, refs):
        return self.conditioner.motion_encoder(batch["tokens"], query_points=batch["points"], motion_evidence=batch["evidence"])

    def _ar_losses(self, condition, batch):
        return {"ce_loss": condition.square().mean()}


class ReplayTest(unittest.TestCase):
    def test_capture_replay_and_independent_storage(self):
        model = Model()
        batch = {"tokens": torch.ones(2), "points": torch.zeros(2),
                 "evidence": Evidence(torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.bool))}
        condition, _, frozen, _, _ = captured_forward(model, batch, None)
        self.assertTrue(torch.equal(condition, frozen.forward(model.conditioner.motion_encoder)))
        batch["tokens"].zero_()
        batch["evidence"].states.zero_()
        self.assertTrue(torch.equal(frozen.tokens, torch.ones(2)))
        self.assertTrue(torch.equal(frozen.evidence.states, torch.ones(2)))
        self.assertFalse(model.conditioner.motion_encoder._forward_pre_hooks)

    def test_rng_roundtrip(self):
        state = rng_snapshot()
        values = torch.rand(4)
        restore_rng(state)
        self.assertTrue(torch.equal(values, torch.rand(4)))
        self.assertEqual(rng_digest(state), rng_digest(state))

    def test_contract_rejects_other_fusions(self):
        model = Model()
        model.condition_fusion = "static_blend"
        with self.assertRaises(ValueError):
            MotionCapture(model)
        with self.assertRaises(ValueError):
            MotionInput.capture((torch.ones(2),), {})

    def test_delta_includes_boolean_validity(self):
        self.assertEqual(tensor_delta(torch.tensor([True]), torch.tensor([False]))["max_abs"], 1)
        with self.assertRaises(ValueError):
            tensor_delta(torch.ones(2), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
