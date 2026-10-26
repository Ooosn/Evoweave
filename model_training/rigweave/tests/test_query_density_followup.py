from copy import deepcopy
import hashlib
import json
import random
import unittest

import numpy as np
import torch

import virtual_pair_runtime as runtime
from single_query_protocol import (FOUR_SELECTOR, FRACTION2_SELECTOR, select_density_target_frames)
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from test_two_query_runtime import two_configuration


def configuration(algorithm):
    config = two_configuration()
    config['family'] = runtime.DENSITY_SCOPE
    config['pair']['target_selection']['algorithm'] = algorithm
    config['pair']['variants'] = [runtime.FOUR_VARIANT if algorithm == FOUR_SELECTOR else runtime.FRACTION2_VARIANT]
    return config


class DensityFollowupTests(unittest.TestCase):
    def test_new_rules_independent_recipe_global_rng_and_rebatching(self):
        requests = [FrameSampleRequest(i, t, (1 << 63) + i) for i, t in enumerate(list(range(2, 25)) * 20)]
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        for algorithm in (FOUR_SELECTOR, FRACTION2_SELECTOR):
            actual = select_density_target_frames(requests, 20261026, algorithm)
            expected = []
            for r in requests:
                text = f'{algorithm}|20261026|{r.item_index}|{r.sampling_seed}|{r.frames}'
                generator = random.Random(int.from_bytes(hashlib.sha256(text.encode('ascii')).digest(), 'big'))
                k = min(4, r.frames) if algorithm == FOUR_SELECTOR else max(2, int(r.frames * (.2 + .2 * generator.random())))
                expected.append(sorted(generator.sample(list(range(r.frames)), k)))
            self.assertEqual(actual, expected)
            self.assertEqual(select_density_target_frames(requests[::-1], 20261026, algorithm), actual[::-1])
            self.assertEqual(select_density_target_frames(requests[:17], 20261026, algorithm)
                             + select_density_target_frames(requests[17:], 20261026, algorithm), actual)
            self.assertTrue(all(2 <= len(v) <= r.frames and len(set(v)) == len(v) for r, v in zip(requests, actual)))
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_equal(before[1], np.random.get_state())
        torch.testing.assert_close(before[2], torch.get_rng_state(), atol=0, rtol=0)

    def test_new_runtime_policy_and_full20k_accounting(self):
        for algorithm in (FOUR_SELECTOR, FRACTION2_SELECTOR):
            config = configuration(algorithm)
            runtime.validate_config(config)
            report = runtime.audit_target_selection(config)
            inputs = sum(row['input_frames'] for row in report['ranks'])
            targets = sum(row['supervised_frames'] for row in report['ranks'])
            self.assertEqual(inputs, 263608)
            self.assertEqual(runtime.supervised_exposures(config, 20000, inputs), targets)
            self.assertGreater(targets, 40000)
            if algorithm == FOUR_SELECTOR:
                self.assertEqual(targets, 77920)
            wrong = deepcopy(config)
            wrong['pair']['variants'] = ['three_query']
            with self.assertRaises(ValueError):
                runtime.validate_config(wrong)


if __name__ == '__main__':
    unittest.main()
