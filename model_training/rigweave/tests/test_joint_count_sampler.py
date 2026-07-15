from __future__ import annotations

import unittest

from rigweave.dynamic_rig.joint_count_sampler import (
    JointCountMixtureSampler,
    parse_joint_count_bin_uppers,
)


class JointCountMixtureSamplerTest(unittest.TestCase):
    def test_probability_mass_interpolates_between_natural_and_uniform_bins(self) -> None:
        counts = [5] * 10 + [15] * 30 + [25] * 60
        sampler = JointCountMixtureSampler(
            counts,
            bin_upper_bounds=(10, 20, 30),
            mixture_alpha=0.5,
            seed=7,
        )

        self.assertEqual(sampler.bucket_counts, (10, 30, 60))
        expected = (
            0.5 * 0.10 + 0.5 / 3,
            0.5 * 0.30 + 0.5 / 3,
            0.5 * 0.60 + 0.5 / 3,
        )
        for actual, wanted in zip(sampler.expected_bucket_probabilities, expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertAlmostEqual(float(sampler.weights.sum()), 1.0)

    def test_distributed_draw_is_deterministic_and_epoch_dependent(self) -> None:
        counts = [5] * 10 + [15] * 30 + [25] * 60
        rank0 = JointCountMixtureSampler(
            counts,
            bin_upper_bounds=(10, 20, 30),
            mixture_alpha=0.5,
            num_replicas=2,
            rank=0,
            seed=11,
        )
        rank1 = JointCountMixtureSampler(
            counts,
            bin_upper_bounds=(10, 20, 30),
            mixture_alpha=0.5,
            num_replicas=2,
            rank=1,
            seed=11,
        )

        first0 = list(rank0)
        first1 = list(rank1)
        self.assertEqual(len(first0), 50)
        self.assertEqual(len(first1), 50)
        self.assertEqual(first0, list(rank0))
        rank0.set_epoch(1)
        self.assertNotEqual(first0, list(rank0))

    def test_manifest_counts_and_bins_are_strict(self) -> None:
        self.assertEqual(parse_joint_count_bin_uppers("10,20,40"), (10, 20, 40))
        with self.assertRaisesRegex(ValueError, "target_joint_count"):
            JointCountMixtureSampler(
                [5, None, 15],
                bin_upper_bounds=(10, 20),
                mixture_alpha=0.5,
            )
        with self.assertRaisesRegex(ValueError, "empty bins"):
            JointCountMixtureSampler(
                [5, 6, 7],
                bin_upper_bounds=(10, 20),
                mixture_alpha=0.5,
            )


if __name__ == "__main__":
    unittest.main()
