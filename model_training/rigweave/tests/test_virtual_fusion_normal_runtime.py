"""CPU proofs for batch-independent online pose/exposure schedules."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from virtual_fusion_normal_runtime import MatchedFrameStream


def stream(rank=0, world=2, batch=7, state=None, **changes):
    args = dict(rank=rank, world_size=world, seed=20261013, target_assets=20000, effective_assets=48,
                update_boundaries=[5000, 10000, 15000, 20000],
                frame_policy={str(t): batch for t in range(2, 25)}, manifest_sha256='0' * 64, state=state)
    args.update(changes)
    return MatchedFrameStream(range(5000), **args)


def drain(value):
    result = []
    while value.cursor < value.local_total:
        result.extend(value.take())
    return result


class MatchedScheduleTests(unittest.TestCase):
    def test_physical_batch_policy_cannot_change_exposures_frames_or_seeds(self):
        for rank in (0, 1):
            one, large = stream(rank=rank, batch=1), stream(rank=rank, batch=24)
            self.assertEqual(one.plan_sha256, large.plan_sha256)
            self.assertEqual(drain(one), drain(large))

    def test_two_ranks_cover_exact_global_order_without_padding_or_dropping(self):
        left, right = stream(rank=0), stream(rank=1)
        a, b = drain(left), drain(right)
        global_requests, start = [], 0
        for size in left.update_sizes:
            local_size = int(size) // 2
            global_requests.extend(a[start:start+local_size] + b[start:start+local_size])
            start += local_size
        single = stream(rank=0, world=1, batch=24)
        self.assertEqual(left.plan_sha256, single.plan_sha256)
        self.assertEqual(global_requests, drain(single))
        self.assertEqual(len(global_requests), 20000)
        counts = Counter(request.item_index for request in global_requests)
        self.assertEqual(set(counts), set(range(5000)))
        self.assertEqual(Counter(counts.values()), {4: 5000})
        self.assertEqual(len({request.sampling_seed for request in global_requests}), 20000)
        self.assertEqual({request.frames for request in global_requests}, set(range(2, 25)))
        start = 0
        for size in left.update_sizes:
            self.assertEqual(len({request.frames for request in global_requests[start:start+int(size)]}), 1)
            start += int(size)
        self.assertEqual(Counter(left.update_sizes), {48: 416, 8: 4})
        self.assertEqual(Counter(row.item_index for row in global_requests[:5000]), {i: 1 for i in range(5000)})
        self.assertEqual(Counter(row.item_index for row in global_requests[:10000]), {i: 2 for i in range(5000)})
        self.assertEqual(left.global_ends[104::105].tolist(), [5000, 10000, 15000, 20000])
        self.assertEqual(left.global_ends[-1], 20000)

    def test_different_visits_to_an_asset_use_different_sampling_seeds(self):
        requests = drain(stream())
        grouped = {}
        for request in requests:
            grouped.setdefault(request.item_index, []).append(request)
        self.assertTrue(any(len({row.frames for row in rows}) > 1 for rows in grouped.values()))
        for rows in grouped.values():
            self.assertEqual(len(rows), len({row.sampling_seed for row in rows}))

    def test_saved_cursor_replays_exact_remaining_requests(self):
        original = stream(batch=7)
        original.take(5)
        original.take(2)
        state = original.state_dict()
        restored = stream(batch=7, state=state)
        self.assertEqual(drain(original), drain(restored))
        changed = deepcopy(state)
        changed['cursor'] = -1
        with self.assertRaises(ValueError):
            stream(batch=7, state=changed)
        with self.assertRaises(ValueError):
            stream(batch=8, state=state)

    def test_request_truncation_does_not_drop_or_resample_pending_assets(self):
        original, split = stream(batch=24), stream(batch=24)
        expected = original.take()
        self.assertEqual(split.take(3) + split.take(4) + split.take(17), expected)

    def test_validation_and_end_of_budget(self):
        for overrides in ({'effective_assets': 47}, {'target_assets': 80015}, {'seed': True},
                          {'manifest_sha256': 'unbound'}, {'min_frames': 1}, {'max_frames': 25},
                          {'frame_policy': {'2': 1}}, {'rank': 2}, {'world_size': 0},
                          {'update_boundaries': [10000]}, {'update_boundaries': [10001, 20000]},
                          {'update_boundaries': [20000, 10000]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                stream(**overrides)
        value = stream(target_assets=48, update_boundaries=[48])
        drain(value)
        with self.assertRaises(StopIteration):
            value.take()
        with self.assertRaises(ValueError):
            stream().take(False)


if __name__ == '__main__':
    unittest.main()
