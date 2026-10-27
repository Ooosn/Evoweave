from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from reference_flow_protocol import SCOPE, ENTRYPOINT, scientific_sha, validate_config
from reference_flow_runtime import flow_inputs, ar_indices, summarize
from virtual_pair_runtime import make_stream, next_update_batches
from virtual_pair_guard import operation, validate_stage
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest


class FlowRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'model_training/experiments/shared_reference_flow_5k_20261027_v1.json'
        self.config = json.loads(self.path.read_text())
        self.base = validate_config(self.config)

    def test_full5000_each_pass_and_exact_resume(self):
        streams = [make_stream(self.base, rank, 2) for rank in range(2)]
        samples, indices = 0, []
        while samples < 20000:
            for rank in range(2):
                if samples == 10000:
                    streams[rank] = make_stream(self.base, rank, 2, streams[rank].state_dict())
                goal, batches = next_update_batches(streams[rank], samples, 2)
                indices.extend(r.item_index for batch in batches for r in batch)
                for batch in batches:
                    selected, _, mask = ar_indices(self.base, batch, 'cpu')
                    self.assertEqual(int(mask.sum()), sum(map(len, selected)))
                    self.assertTrue(all(len(x) >= 2 for x in selected))
            samples += goal
            if samples % 5000 == 0:
                self.assertEqual(sorted(indices), list(range(5000)))
                indices = []

    def test_noise_independent_of_padding_and_global_rng(self):
        request = FrameSampleRequest(1, 3, 154)
        small = {'frame_target_joints': torch.zeros(1, 3, 2, 3), 'joint_mask': torch.ones(1, 2, dtype=torch.bool)}
        large = {'frame_target_joints': torch.zeros(2, 3, 5, 3),
                 'joint_mask': torch.tensor([[True, True, False, False, False], [True] * 5])}
        before = torch.get_rng_state().clone()
        a = flow_inputs(small, [request], self.config, 'cpu')
        b = flow_inputs(large, [request, FrameSampleRequest(2, 3, 155)], self.config, 'cpu')
        self.assertEqual(a[0][0], b[0][0])
        self.assertEqual(a[2][0], b[2][0])
        torch.testing.assert_close(a[1][0], b[1][0, :, :2])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_protocol_rejects_small_dataset_and_hidden_loss_changes(self):
        for field, value in [('additional_assets', 64), ('loss', 'rigid'), ('flow_weight', 0.)]:
            bad = deepcopy(self.config)
            bad['training'][field] = value
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_resume_location_not_scientific_change(self):
        changed = deepcopy(self.config)
        changed['training']['resume'] = {'path': 'x', 'sha256': 'a'}
        self.assertEqual(scientific_sha(changed), scientific_sha(self.config))
        changed['architecture']['dim'] = 512
        self.assertNotEqual(scientific_sha(changed), scientific_sha(self.config))

    def test_entrypoint_and_training_admission_fail_closed(self):
        op = {'label': SCOPE, 'explicit_human_authorized': True, 'required_entrypoint': ENTRYPOINT}
        self.assertEqual(operation({'active_operation': op}), op)
        wrong = {**op, 'required_entrypoint': 'model_training/rigweave/scripts/train_reference_xyz.py'}
        with self.assertRaises(RuntimeError):
            operation({'active_operation': wrong})
        from reference_flow_protocol import sha256
        op.update(complete=False, gpu_work_allowed=True, formal_training_allowed=True, CPU_audit_accepted=True,
            capacity_accepted=False, runs=[{'config': str(self.path), 'output': str(ROOT / 'output'),
                'mode': 'train', 'approved': True, 'config_sha256': sha256(self.path)}])
        args = SimpleNamespace(config=self.path, output=ROOT / 'output', mode='train')
        with self.assertRaises(RuntimeError):
            validate_stage({'active_operation': op}, self.config, args)
        op['capacity_accepted'] = True
        self.assertEqual(validate_stage({'active_operation': op}, self.config, args), op)

    def test_macro_scoring_counts_failed_references(self):
        rows = [{'split': 'valid', 'reference_kind': 'predicted', 'method': 'flow', 'asset_id': asset,
                 'reference_success': success, 'edge_f1': score, 'normalized_j2j': distance}
                for asset, success, score, distance in [('a', True, .8, .1), ('a', True, .6, .2), ('b', False, 0., None)]]
        result = summarize(rows)[0]
        self.assertAlmostEqual(result['edge_f1'], .35)
        self.assertAlmostEqual(result['normalized_j2j'], .15)
        self.assertEqual(result['normalized_j2j_scored_assets'], 1)
        self.assertEqual(result['successful_rows'], 2)


if __name__ == '__main__':
    unittest.main()
