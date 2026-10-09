from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.data import _select_query_sequence
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest, VariableFrameBatchSampler


class FrameBatchSamplerTest(unittest.TestCase):
    def test_batch_counts_are_replayable_and_ddp_consistent(self):
        dataset = range(120)
        left_base = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)
        right_base = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True)
        left = VariableFrameBatchSampler(left_base, 3, min_frames=2, max_frames=24, seed=39)
        right = VariableFrameBatchSampler(right_base, 3, min_frames=2, max_frames=24, seed=39)
        a, b = list(left), list(right)
        self.assertEqual(a, list(left))
        self.assertEqual([row[0].frames for row in a], [row[0].frames for row in b])
        for row in a + b:
            self.assertEqual(len({request.frames for request in row}), 1)
            self.assertTrue(all(isinstance(request, FrameSampleRequest) for request in row))
            self.assertTrue(2 <= row[0].frames <= 24)
        self.assertEqual(len(a), len(left))
        left_base.set_epoch(1)
        left.set_epoch(1)
        self.assertNotEqual(a, list(left))

    def test_bounds_and_partial_batch(self):
        sampler = VariableFrameBatchSampler(SequentialSampler(range(5)), 3, min_frames=2, max_frames=2, seed=1)
        rows = list(sampler)
        self.assertEqual([len(row) for row in rows], [3, 2])
        self.assertEqual({request.frames for row in rows for request in row}, {2})
        with self.assertRaises(ValueError):
            VariableFrameBatchSampler(SequentialSampler(range(5)), 3, min_frames=1, max_frames=24, seed=1)


class EvidenceFrameQuotaTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(93)
        self.vertices = rng.normal(size=(40, 12, 3)).astype(np.float32)
        self.joints = rng.normal(size=(40, 3, 3)).astype(np.float32)

    def select(self, frames, *, minimum=1, seed=3, joints=None):
        return _select_query_sequence(
            self.vertices, self.joints if joints is None else joints, None,
            frame_count=frames, path=Path("test_asset.npz"), index=0,
            random_query=True, seed=91, sampling_seed=seed,
            motion_fps_ratio=0.75, motion_vertex_samples=12, minimum_random_frames=minimum,
        )

    def test_random_minimum_takes_priority_over_fps(self):
        for total, random_count in [(2, 1), (3, 1), (5, 1), (9, 2), (24, 6)]:
            with self.subTest(total=total):
                def fps(features, candidates, count):
                    return candidates[:count].tolist()
                with patch("rigweave.dynamic_rig.data._farthest_frames", side_effect=fps) as mocked:
                    _, targets, _, chosen = self.select(total)
                self.assertEqual(mocked.call_args.args[2], total - 1 - random_count)
                self.assertEqual(len(chosen), total)
                self.assertEqual(len(set(chosen.tolist())), total)
                np.testing.assert_array_equal(targets, self.joints[chosen[0]])
        with patch("rigweave.dynamic_rig.data._farthest_frames", side_effect=lambda f, c, n: c[:n].tolist()) as mocked:
            self.select(2, minimum=0)
        self.assertEqual(mocked.call_args.args[2], 1)

    def test_query_and_sampling_do_not_use_skeleton_labels(self):
        _, _, _, selected = self.select(24)
        frames, _, _, modified = self.select(24, joints=self.joints + 1000)
        np.testing.assert_array_equal(selected, modified)
        np.testing.assert_array_equal(frames, self.vertices[selected])
        _, _, _, fewer = self.select(2)
        self.assertEqual(selected[0], fewer[0])
        self.assertTrue(np.any(selected != self.select(24, seed=4)[3]))


if __name__ == "__main__":
    unittest.main()
