"""CPU checks for deterministic selection over the unchanged matched input plan."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))

import single_query_protocol as protocol
from single_query_protocol import SELECTOR, SINGLE_SCOPE, SINGLE_VARIANT, select_target_frames, supervised_count
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from virtual_fusion_normal_runtime import MatchedFrameStream


CONFIG = json.loads((ROOT / 'model_training/experiments/virtual_query_matched_5k_20261025_v2.json').read_text())
SELECTION_SEED = 20261026


def configured_stream(rank, *, world_size=2, state=None, frame_policy=None):
    data, training = CONFIG['data'], CONFIG['training']
    manifest = data['manifests']['train']
    return MatchedFrameStream(
        range(manifest['rows']), rank=rank, world_size=world_size, seed=data['stream_seed'],
        target_assets=training['target_assets'], effective_assets=training['effective_assets_per_update'],
        update_boundaries=training['checkpoint_assets'],
        frame_policy=training['frame_policy'] if frame_policy is None else frame_policy,
        manifest_sha256=manifest['sha256'], min_frames=data['frames_min'], max_frames=data['frames_max'],
        state=state)


def collect(stream, limit, *, cap=None):
    requests, selected = [], []
    while stream.cursor < limit:
        batch = stream.take(min(cap or stream.local_update, limit - stream.cursor))
        requests.extend(batch)
        selected.extend(select_target_frames(batch, SELECTION_SEED))
    return requests, selected


class SingleQueryProtocolTests(unittest.TestCase):
    def test_constants_and_exact_encoding_vectors(self):
        self.assertEqual(SINGLE_SCOPE, 'real_query_single_target_5k_20261026_v1')
        self.assertEqual(SINGLE_VARIANT, 'single_query')
        self.assertEqual(SELECTOR, 'sha256_python_random_randrange_v1')
        vectors = (
            (0, FrameSampleRequest(0, 2, 0),
             'c62afcf461de3ec26d9015608fc53e833685d5ef8a5ad3741fade964dcc0cf3b', 1),
            (20261026, FrameSampleRequest(17, 24, 2**64 - 1),
             'd993943d791d86671c125dec570b4e6b31dfa7af83fd83cbc6796083a2254207', 14),
            (2**64 - 1, FrameSampleRequest(5000, 17, 2**32 + 7),
             '119a705d478fd6826f17e0b257d50128e180d0d7bbceb799861573b98d599ce8', 5),
        )
        for seed, request, digest, expected in vectors:
            with self.subTest(seed=seed, request=request), \
                    patch.object(protocol.random, 'Random', wraps=random.Random) as rng:
                self.assertEqual(select_target_frames([request], seed), [expected])
                rng.assert_called_once_with(int(digest, 16))

    def test_numpy_integrals_preserve_full_values_and_requests(self):
        requests = [FrameSampleRequest(np.int64(17), np.int64(24), np.uint64(2**64 - 1)),
                    FrameSampleRequest(np.int64(5000), np.int32(17), np.int64(2**32 + 7))]
        before = deepcopy(requests)
        expected = select_target_frames([FrameSampleRequest(*(int(value) for value in row))
                                         for row in requests], 2**64 - 1)
        actual = select_target_frames(requests, np.uint64(2**64 - 1))
        self.assertEqual(actual, expected)
        self.assertTrue(all(type(value) is int for value in actual))
        self.assertEqual(requests, before)
        self.assertEqual(select_target_frames(iter(requests), np.uint64(2**64 - 1)), expected)
        self.assertEqual(select_target_frames([], np.int64(0)), [])

    def test_invalid_integer_fields_and_frame_bounds_are_rejected(self):
        request = FrameSampleRequest(0, 2, 0)
        invalid = (True, False, np.bool_(True), -1, np.int64(-1), 2.0, np.float64(2),
                   float('nan'), float('inf'), '2', None)
        for value in invalid:
            with self.subTest(field='seed', value=value), self.assertRaises(ValueError):
                select_target_frames([], value)
            for field in ('item_index', 'frames', 'sampling_seed'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    select_target_frames([request._replace(**{field: value})], 0)
        for frames in (0, 1, 25, 2**64):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, '2..24'):
                select_target_frames([request._replace(frames=frames)], 0)

    def test_selection_preserves_global_python_numpy_and_torch_rngs(self):
        requests = [FrameSampleRequest(index, frames, 2**48 + index)
                    for index, frames in enumerate(range(2, 25))]
        before = deepcopy(requests)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        first = select_target_frames(requests, 2**64 - 1)
        self.assertEqual(select_target_frames(requests, 2**64 - 1), first)
        with self.assertRaises(ValueError):
            select_target_frames(requests + [FrameSampleRequest(1, 25, 1)], 0)
        self.assertEqual(random.getstate(), python_state)
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
        self.assertEqual(actual_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(requests, before)

    def test_reordering_repetition_and_batch_positions_do_not_change_selection(self):
        requests = [FrameSampleRequest(index, 2 + index % 23, 2**48 + index)
                    for index in range(80)]
        expected = select_target_frames(requests, SELECTION_SEED)
        self.assertEqual(select_target_frames(requests[::-1], SELECTION_SEED), expected[::-1])
        for cap in (1, 7, 24):
            actual = [choice for start in range(0, len(requests), cap)
                      for choice in select_target_frames(requests[start:start + cap], SELECTION_SEED)]
            self.assertEqual(actual, expected)
        self.assertEqual(select_target_frames([requests[31]] * 5, SELECTION_SEED), [expected[31]] * 5)

    def test_full_20k_two_rank_plan_state_and_input_stream_are_unchanged(self):
        original_config = deepcopy(CONFIG)
        combined = []
        for rank in (0, 1):
            actual, reference = configured_stream(rank), configured_stream(rank)
            plan = deepcopy(actual.plan_report())
            arrays = {name: getattr(actual, name).copy()
                      for name in ('order', 'frame_counts', 'update_sizes', 'global_ends', 'local_ends')}
            while actual.cursor < 10000:
                batch = actual.take()
                self.assertEqual(batch, reference.take())
                state, original_batch = deepcopy(actual.state_dict()), deepcopy(batch)
                selected = select_target_frames(batch, SELECTION_SEED)
                self.assertEqual(actual.state_dict(), state)
                self.assertEqual(actual.state_dict(), reference.state_dict())
                self.assertEqual(batch, original_batch)
                self.assertEqual(len(selected), len(batch))
                self.assertTrue(all(type(choice) is int and 0 <= choice < request.frames
                                    for request, choice in zip(batch, selected)))
                combined.extend(zip(batch, selected))
            self.assertEqual(actual.cursor, 10000)
            self.assertEqual(actual.plan_report(), plan)
            for name, expected in arrays.items():
                np.testing.assert_array_equal(getattr(actual, name), expected)
        self.assertEqual(CONFIG, original_config)
        combined.sort(key=lambda pair: pair[0].sampling_seed)
        self.assertEqual(len(combined), 20000)
        self.assertEqual(Counter(request.item_index for request, _ in combined), {i: 4 for i in range(5000)})
        self.assertEqual(len({request.sampling_seed for request, _ in combined}), 20000)
        self.assertTrue(all(request.sampling_seed > 2**32 for request, _ in combined))
        input_frames = sum(request.frames for request, _ in combined)
        self.assertEqual(input_frames, 263608)
        single_rank = configured_stream(0, world_size=1)
        requests, selected = collect(single_rank, 20000)
        self.assertEqual(combined, list(zip(requests, selected)))
        print(json.dumps({
            'diagnostic': 'single_query_20k_selection_balance', 'assets': len(combined),
            'input_frames': input_frames,
            'mean_normalized_selected_index': sum(choice / (request.frames - 1)
                                                   for request, choice in combined) / len(combined),
            'uniform_expected_mean': 0.5, 'statistical_acceptance_gate': False,
        }, sort_keys=True))

    def test_full_20k_replay_and_rebatching_preserve_requests_and_selections(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                expected_stream = configured_stream(rank)
                expected = collect(expected_stream, 10000)
                interrupted = configured_stream(rank)
                prefix_requests, prefix_selected = collect(interrupted, 3731, cap=7)
                saved = deepcopy(interrupted.state_dict())
                restored = configured_stream(rank, state=saved)
                suffix_requests, suffix_selected = collect(restored, 10000, cap=1)
                self.assertEqual((prefix_requests + suffix_requests, prefix_selected + suffix_selected), expected)
                self.assertEqual(interrupted.state_dict(), saved)
                self.assertEqual(restored.state_dict(), expected_stream.state_dict())
                rebatched = configured_stream(rank, frame_policy={str(t): 24 for t in range(2, 25)})
                self.assertEqual(rebatched.plan_sha256, expected_stream.plan_sha256)
                self.assertEqual(collect(rebatched, 10000), expected)

    def test_supervised_counts_match_family_and_accept_numpy_integers(self):
        for config in ({'family': SINGLE_SCOPE}, {'family': CONFIG['family']},
                       {'family': SINGLE_VARIANT}, {}):
            original = deepcopy(config)
            for assets, frames in ((0, 0), (0, 24), (1, 1), (48, 624),
                                   (np.int64(20000), np.int64(263608))):
                with self.subTest(config=config, assets=assets, frames=frames):
                    expected = int(assets if config.get('family') == SINGLE_SCOPE else frames)
                    actual = supervised_count(config, assets, frames)
                    self.assertIs(type(actual), int)
                    self.assertEqual(actual, expected)
            self.assertEqual(config, original)

    def test_supervised_counts_reject_invalid_types_and_inconsistent_counts(self):
        invalid = (True, False, np.bool_(False), -1, np.int64(-1), 1.0, np.float64(1),
                   float('nan'), float('inf'), '1', None)
        for config in ({'family': SINGLE_SCOPE}, {}):
            for value in invalid:
                with self.subTest(config=config, value=value):
                    with self.assertRaises(ValueError):
                        supervised_count(config, value, 24)
                    with self.assertRaises(ValueError):
                        supervised_count(config, 0, value)
            for assets, frames in ((1, 0), (3, 2)):
                with self.assertRaisesRegex(ValueError, 'exceed'):
                    supervised_count(config, assets, frames)


if __name__ == '__main__':
    unittest.main()
