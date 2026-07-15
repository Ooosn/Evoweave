from __future__ import annotations

import unittest

import torch

from rigweave.dynamic_rig.puppeteer_dynamic import (
    _sequence_cross_entropy,
    _termination_decision_metrics,
)


class PuppeteerLossTest(unittest.TestCase):
    def test_sequence_mean_gives_short_and_long_sequences_equal_mass(self) -> None:
        logits = torch.zeros((2, 4, 5), dtype=torch.float32)
        labels = torch.tensor(
            [
                [1, -100, -100, -100],
                [2, 2, 2, -100],
            ],
            dtype=torch.long,
        )
        logits[1, :3, 2] = 8.0

        token_mean = _sequence_cross_entropy(logits, labels, reduction="token_mean")
        sequence_mean = _sequence_cross_entropy(logits, labels, reduction="sequence_mean")

        self.assertGreater(float(sequence_mean), float(token_mean))
        short_loss = torch.nn.functional.cross_entropy(logits[0, :1], labels[0, :1])
        long_loss = torch.nn.functional.cross_entropy(logits[1, :3], labels[1, :3])
        self.assertTrue(torch.allclose(sequence_mean, 0.5 * (short_loss + long_loss)))

    def test_termination_decision_reuses_coordinate_and_eos_logits(self) -> None:
        logits = torch.full((1, 4, 7), -8.0, dtype=torch.float32)
        labels = torch.tensor([[3, 3, 4, 1]], dtype=torch.long)
        roles = torch.tensor([[3, 6, 3, 1]], dtype=torch.long)

        # Position 0 is joint 0 and cannot terminate. Position 2 continues;
        # position 3 terminates.
        logits[0, 2, 3:7] = 4.0
        logits[0, 2, 1] = -4.0
        logits[0, 3, 3:7] = -4.0
        logits[0, 3, 1] = 4.0
        correct = _termination_decision_metrics(
            logits,
            labels,
            roles,
            regular_start=3,
            regular_end=7,
            eos_token_id=1,
        )

        wrong_logits = logits.clone()
        wrong_logits[0, 3, 1] = -8.0
        wrong_logits[0, 3, 3:7] = 4.0
        wrong = _termination_decision_metrics(
            wrong_logits,
            labels,
            roles,
            regular_start=3,
            regular_end=7,
            eos_token_id=1,
        )

        self.assertEqual(float(correct["stop_acc"]), 1.0)
        self.assertEqual(float(correct["continue_acc"]), 1.0)
        self.assertLess(float(correct["loss"]), float(wrong["loss"]))


if __name__ == "__main__":
    unittest.main()
