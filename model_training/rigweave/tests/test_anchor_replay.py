import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from anchor_integration_protocol import validate_config, scientific_sha
from anchor_replay_protocol import validate_replay
from anchor_replay_runtime import difference, tensor_digest


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'model_training/experiments/anchor_checkpoint_replay_20261029_v1.json').read_text())

    def test_exact_two_case_inference_contract_preserves_science(self):
        validate_config(self.config)
        spec = validate_replay(self.config, 'capacity')
        self.assertEqual(spec['optimizer_steps'], 0)
        self.assertEqual(scientific_sha(self.config), '8e0467089d886766aa94b1ae6b6002f7ea03f1d2bff27ce161acef1c1b752da1')

    def test_diagnostic_config_cannot_run_as_training_or_audit(self):
        for mode in ('train', 'audit'):
            with self.assertRaises(ValueError):
                validate_replay(self.config, mode)

    def test_no_update_more_cases_repeats_or_unbounded_duration(self):
        for key, value in (('optimizer_steps', 1), ('model_loads', 3), ('repeats_per_load', 3),
                           ('case_ids', ['unknown']), ('precisions', ['bf16']), ('max_seconds', 10000)):
            config = copy.deepcopy(self.config)
            config['capacity'][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_wrong_checkpoint_or_examples_rejected(self):
        for field in ('checkpoint', 'examples'):
            config = copy.deepcopy(self.config)
            value = config['capacity'][field]
            (value[0] if isinstance(value, list) else value)['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                validate_replay(config)

    def test_tensor_comparison_distinguishes_identity_and_numeric_difference(self):
        a = torch.tensor([[1., 2.], [3., 4.]])
        self.assertTrue(difference(a, a.clone())['bitwise_equal'])
        b = a.clone()
        b[0, 0] += .25
        result = difference(a, b)
        self.assertFalse(result['bitwise_equal'])
        self.assertEqual(result['max_abs'], .25)
        self.assertEqual(result['mean_abs'], .0625)
        with self.assertRaises(ValueError):
            difference(a, a.double())
        self.assertNotEqual(tensor_digest(a), tensor_digest(a.reshape(4)))


if __name__ == '__main__':
    unittest.main()
