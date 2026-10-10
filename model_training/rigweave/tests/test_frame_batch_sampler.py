from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.data import _select_query_sequence
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest, VariableFrameBatchSampler


def _legacy_reference(sampler, batch_size, *, min_frames, max_frames, seed, epoch, drop_last):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 99011]))
    rows = []
    for indices in BatchSampler(sampler, batch_size, drop_last):
        frames = int(rng.integers(min_frames, max_frames + 1))
        requests = []
        for index in indices:
            sampling_seed = int(np.random.SeedSequence(
                [seed, epoch, int(index), 129]).generate_state(1, dtype=np.uint32)[0])
            requests.append(FrameSampleRequest(int(index), frames, sampling_seed))
        rows.append(requests)
    return rows


class FrameBatchSamplerTest(unittest.TestCase):
    def test_legacy_requests_and_length_match_original_implementation(self):
        for count in (0, 1, 5, 17, 120):
            for batch_size in (1, 3, 7):
                for min_frames, max_frames in ((2, 2), (2, 24)):
                    for epoch in (0, 1, 4):
                        for drop_last in (False, True):
                            with self.subTest(
                                count=count, batch_size=batch_size, bounds=(min_frames, max_frames),
                                epoch=epoch, drop_last=drop_last,
                            ):
                                base = SequentialSampler(range(count))
                                kwargs = dict(
                                    min_frames=min_frames, max_frames=max_frames, seed=39,
                                    drop_last=drop_last,
                                )
                                sampler = VariableFrameBatchSampler(base, batch_size, **kwargs)
                                sampler.set_epoch(epoch)
                                expected = _legacy_reference(base, batch_size, epoch=epoch, **kwargs)
                                self.assertEqual(list(sampler), expected)
                                self.assertEqual(list(sampler), expected)
                                self.assertEqual(len(sampler), len(expected))

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


class FrameBudgetSamplerTest(unittest.TestCase):
    def make_sampler(self, count, *, frame_budget=72, max_batch_size=8, **kwargs):
        return VariableFrameBatchSampler(
            SequentialSampler(range(count)), 3, min_frames=2, max_frames=24, seed=39,
            frame_budget=frame_budget, max_batch_size=max_batch_size, **kwargs,
        )

    def test_budget_cap_and_uniform_per_microbatch_frame_stream(self):
        for budget, cap in ((24, 7), (72, 8), (100, 3), (24, 1), (72, 100)):
            with self.subTest(budget=budget, cap=cap):
                sampler = self.make_sampler(2048, frame_budget=budget, max_batch_size=cap)
                rows = list(sampler)
                rng = np.random.default_rng(np.random.SeedSequence([39, 0, 99011]))
                expected_frames = rng.integers(2, 25, size=len(rows)).tolist()
                self.assertEqual([row[0].frames for row in rows], expected_frames)
                self.assertEqual(set(expected_frames), set(range(2, 25)))
                self.assertEqual(len(sampler), len(rows))
                remaining = 2048
                for row, frames in zip(rows, expected_frames):
                    self.assertEqual(len(row), min(remaining, cap, budget // frames))
                    self.assertEqual({request.frames for request in row}, {frames})
                    self.assertTrue(all(isinstance(request, FrameSampleRequest) for request in row))
                    self.assertLessEqual(len(row) * frames, budget)
                    self.assertLessEqual(len(row), cap)
                    remaining -= len(row)
                self.assertEqual(remaining, 0)
                self.assertEqual([request.item_index for row in rows for request in row], list(range(2048)))

    def test_tail_drop_last_and_empty_lengths(self):
        for count in (0, 1, 2, 3, 5, 8, 9, 23, 24, 47, 120, 121):
            for epoch in (0, 1, 7):
                for drop_last in (False, True):
                    with self.subTest(count=count, epoch=epoch, drop_last=drop_last):
                        sampler = self.make_sampler(count, drop_last=drop_last)
                        sampler.set_epoch(epoch)
                        rng = np.random.default_rng(np.random.SeedSequence([39, epoch, 99011]))
                        expected_sizes = []
                        remaining = count
                        while remaining:
                            frames = int(rng.integers(2, 25))
                            size = min(8, 72 // frames)
                            if drop_last and remaining < size:
                                break
                            expected_sizes.append(min(size, remaining))
                            remaining -= expected_sizes[-1]
                        rows = list(sampler)
                        self.assertEqual([len(row) for row in rows], expected_sizes)
                        self.assertEqual(len(sampler), len(rows))
                        self.assertEqual(
                            [request.item_index for row in rows for request in row],
                            list(range(sum(expected_sizes))),
                        )
        for drop_last, expected in ((False, [3, 3, 2]), (True, [3, 3])):
            sampler = VariableFrameBatchSampler(
                SequentialSampler(range(8)), 99, min_frames=24, max_frames=24,
                seed=39, frame_budget=72, max_batch_size=8, drop_last=drop_last,
            )
            self.assertEqual([len(row) for row in sampler], expected)
            self.assertEqual(len(sampler), len(expected))

    def test_reiteration_epoch_replay_and_unchanged_per_asset_seeds(self):
        sampler = self.make_sampler(120)
        initial = list(sampler)
        self.assertEqual(initial, list(sampler))
        sampler.set_epoch(1)
        changed = list(sampler)
        self.assertNotEqual(initial, changed)
        self.assertEqual(changed, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(initial, list(sampler))
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            legacy = VariableFrameBatchSampler(
                SequentialSampler(range(120)), 3, min_frames=2, max_frames=24, seed=39,
            )
            legacy.set_epoch(epoch)
            expected = {request.item_index: request.sampling_seed for row in legacy for request in row}
            actual = {request.item_index: request.sampling_seed for row in sampler for request in row}
            self.assertEqual(actual, expected)

    def test_ddp_schedules_match_for_distinct_indices_and_padding(self):
        for base_drop_last in (False, True):
            for drop_last in (False, True):
                for epoch in (0, 1, 7):
                    with self.subTest(base_drop_last=base_drop_last, drop_last=drop_last, epoch=epoch):
                        rank_rows = []
                        for rank in range(3):
                            base = DistributedSampler(
                                range(121), num_replicas=3, rank=rank, shuffle=True,
                                seed=93, drop_last=base_drop_last,
                            )
                            base.set_epoch(epoch)
                            sampler = VariableFrameBatchSampler(
                                base, 3, min_frames=2, max_frames=24, seed=39,
                                frame_budget=72, max_batch_size=8, drop_last=drop_last,
                            )
                            sampler.set_epoch(epoch)
                            rows = list(sampler)
                            self.assertEqual(len(sampler), len(rows))
                            indices = [request.item_index for row in rows for request in row]
                            self.assertEqual(indices, list(base)[:len(indices)])
                            rank_rows.append(rows)
                        self.assertNotEqual(rank_rows[0], rank_rows[1])
                        schedules = [[(row[0].frames, len(row)) for row in rows] for rows in rank_rows]
                        self.assertEqual(schedules[0], schedules[1])
                        self.assertEqual(schedules[0], schedules[2])

    def test_length_does_not_consume_underlying_sampler_or_random_state(self):
        class TrackingSampler(Sampler[int]):
            def __init__(self):
                self.generator = torch.Generator().manual_seed(105)
                self.sampler = RandomSampler(range(120), generator=self.generator)
                self.iteration_count = 0

            def __len__(self):
                return len(self.sampler)

            def __iter__(self):
                self.iteration_count += 1
                return iter(self.sampler)

        base = TrackingSampler()
        sampler = VariableFrameBatchSampler(
            base, 3, min_frames=2, max_frames=24, seed=39, frame_budget=72, max_batch_size=8,
        )
        generator_state = base.generator.get_state().clone()
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        for epoch in (0, 1, 7, 0):
            sampler.set_epoch(epoch)
            self.assertEqual(len(sampler), len(sampler))
        self.assertEqual(base.iteration_count, 0)
        self.assertTrue(torch.equal(base.generator.get_state(), generator_state))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        rows = list(sampler)
        self.assertEqual(len(rows), len(sampler))
        self.assertEqual(base.iteration_count, 1)

        sequential = self.make_sampler(120)
        expected = list(sequential)
        iterator = iter(sequential)
        self.assertEqual(next(iterator), expected[0])
        self.assertEqual(len(sequential), len(expected))
        self.assertEqual(list(iterator), expected[1:])

    def test_zero_budget_preserves_legacy_even_with_explicit_cap(self):
        sampler = self.make_sampler(120, frame_budget=0, max_batch_size=1)
        expected = _legacy_reference(
            SequentialSampler(range(120)), 3, min_frames=2, max_frames=24,
            seed=39, epoch=0, drop_last=False,
        )
        self.assertEqual(list(sampler), expected)
        self.assertEqual(len(sampler), len(expected))

    def test_invalid_budgets_and_caps_fail_without_silent_clamping(self):
        for budget in (-1, 1, 23, 1.5, True, None, "72"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.make_sampler(120, frame_budget=budget)
        for cap in (None, 0, -1, 1.5, True, "8"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.make_sampler(120, max_batch_size=cap)
        with self.assertRaisesRegex(ValueError, "explicit max_batch_size"):
            VariableFrameBatchSampler(
                SequentialSampler(range(120)), 3, min_frames=2, max_frames=24,
                seed=39, frame_budget=72,
            )
        for kwargs in ({"batch_size": 0}, {"batch_size": True}, {"batch_size": 3, "drop_last": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VariableFrameBatchSampler(
                    SequentialSampler(range(120)), min_frames=2, max_frames=24,
                    seed=39, frame_budget=72, max_batch_size=8, **kwargs,
                )


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
