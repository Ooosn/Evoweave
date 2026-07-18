from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from rigweave.dynamic_rig.topology_sampler import (
    TopologyFamilyMixtureSampler,
    load_parent_topology_signatures,
)


def chain(length: int) -> tuple[int, ...]:
    return tuple([-1] + list(range(length - 1)))


class TopologyFamilyMixtureSamplerTest(unittest.TestCase):
    def test_probability_mass_interpolates_natural_and_uniform_families(self) -> None:
        signatures = [chain(2)] * 10 + [chain(3)] * 30 + [chain(4)] * 60
        sampler = TopologyFamilyMixtureSampler(
            signatures,
            mixture_alpha=0.5,
            seed=7,
        )

        self.assertEqual(sampler.family_count, 3)
        self.assertEqual(sampler.family_histogram, {10: 1, 30: 1, 60: 1})
        self.assertAlmostEqual(float(sampler.weights.sum()), 1.0)
        expected_family_masses = (
            0.5 * 0.10 + 0.5 / 3,
            0.5 * 0.30 + 0.5 / 3,
            0.5 * 0.60 + 0.5 / 3,
        )
        offsets = (0, 10, 40, 100)
        for start, end, expected in zip(offsets, offsets[1:], expected_family_masses):
            self.assertAlmostEqual(float(sampler.weights[start:end].sum()), expected)

    def test_distributed_draw_is_deterministic_and_epoch_dependent(self) -> None:
        signatures = [chain(2)] * 10 + [chain(3)] * 30 + [chain(4)] * 60
        rank0 = TopologyFamilyMixtureSampler(
            signatures,
            mixture_alpha=0.75,
            num_replicas=2,
            rank=0,
            seed=11,
        )
        rank1 = TopologyFamilyMixtureSampler(
            signatures,
            mixture_alpha=0.75,
            num_replicas=2,
            rank=1,
            seed=11,
        )

        first0 = list(rank0)
        first1 = list(rank1)
        self.assertEqual(len(first0), 50)
        self.assertEqual(len(first1), 50)
        self.assertEqual(first0, list(rank0))
        self.assertEqual(len(first0) + len(first1), rank0.total_size)
        rank0.set_epoch(1)
        self.assertNotEqual(first0, list(rank0))

    def test_npz_scan_preserves_order_and_rejects_invalid_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npz"
            second = root / "second.npz"
            invalid = root / "invalid.npz"
            np.savez(first, target_parents=np.asarray(chain(3), dtype=np.int64))
            np.savez(second, target_parents=np.asarray((-1, 0, 0), dtype=np.int64))
            np.savez(invalid, target_parents=np.asarray((-1, -1), dtype=np.int64))

            signatures = load_parent_topology_signatures(
                [first, second],
                num_workers=2,
            )
            self.assertEqual(signatures, [chain(3), (-1, 0, 0)])
            with self.assertRaisesRegex(ValueError, "exactly one root"):
                load_parent_topology_signatures([invalid], num_workers=0)


if __name__ == "__main__":
    unittest.main()
