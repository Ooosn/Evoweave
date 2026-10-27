import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from anchor_joint_protocol import VARIANTS, SCOPE, ENTRYPOINT, sha256, scientific_sha, should_extend, validate_config


class AnchorJointProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'model_training/experiments/anchor_joint_binding_5k_20261027_v1.json').read_text())

    def test_full5k_contract(self):
        base = validate_config(self.config)
        self.assertEqual(base['data']['manifests']['train']['rows'], 5000)
        self.assertEqual(base['training']['target_assets'], 60000)

    def test_small_or_single_arm_trial_rejected(self):
        for field, value in (('minimum_assets', 5000), ('variants', ['reference_transport']), ('additional_assets', 20000)):
            changed = copy.deepcopy(self.config)
            changed['training'][field] = value
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_variant_part_of_scientific_identity(self):
        changed = copy.deepcopy(self.config)
        changed['training']['variant'] = 'reference_transport'
        self.assertNotEqual(scientific_sha(changed), scientific_sha(self.config))

    def curves(self):
        return {v: {n: {'normalized_mpjpe': .1, 'pck_005': .5, 'edge_f1': .5} for n in (30000, 40000)} for v in VARIANTS}

    def test_flat_pair_stops_at_common40k(self):
        self.assertIsNone(should_extend(self.curves())['extend_both_to'])

    def test_either_arm_improving_extends_both(self):
        for metric, value in (('normalized_mpjpe', .098), ('pck_005', .51), ('edge_f1', .51)):
            curves = self.curves()
            curves['target_content'][40000][metric] = value
            self.assertEqual(should_extend(curves)['extend_both_to'], 60000)

    def test_full5000_twelve_passes_and_resume_with_identical_references(self):
        from anchor_joint_runtime import reference_indices
        from virtual_pair_runtime import make_stream, next_update_batches
        base = validate_config(self.config)
        streams = [make_stream(base, r, 2) for r in range(2)]
        samples, indices, passes = 0, [], 0
        while samples < 60000:
            for rank in range(2):
                if samples == 40000:
                    streams[rank] = make_stream(base, rank, 2, streams[rank].state_dict())
                goal, batches = next_update_batches(streams[rank], samples, 2)
                indices.extend(r.item_index for batch in batches for r in batch)
                for batch in batches:
                    a = reference_indices(batch, self.config, 'cpu')
                    b = reference_indices(batch, self.config, 'cpu')
                    self.assertEqual(a.tolist(), b.tolist())
                    self.assertTrue(all(0 <= n < r.frames for n, r in zip(a.tolist(), batch)))
            samples += goal
            if samples % 5000 == 0:
                self.assertEqual(sorted(indices), list(range(5000)))
                indices = []
                passes += 1
        self.assertEqual(passes, 12)

    def test_guard_requires_capacity_acceptance(self):
        from virtual_pair_guard import operation, validate_stage
        path = ROOT / 'model_training/experiments/anchor_joint_binding_5k_20261027_v1.json'
        op = {'label': SCOPE, 'explicit_human_authorized': True, 'required_entrypoint': ENTRYPOINT,
            'complete': False, 'gpu_work_allowed': True, 'formal_training_allowed': True,
            'CPU_audit_accepted': True, 'capacity_accepted': False,
            'runs': [{'config': str(path), 'output': str(ROOT / 'output'), 'mode': 'train',
                'approved': True, 'config_sha256': sha256(path)}]}
        self.assertEqual(operation({'active_operation': op}), op)
        args = SimpleNamespace(config=path, output=ROOT / 'output', mode='train')
        with self.assertRaises(RuntimeError):
            validate_stage({'active_operation': op}, self.config, args)
        op['capacity_accepted'] = True
        self.assertEqual(validate_stage({'active_operation': op}, self.config, args), op)


if __name__ == '__main__':
    unittest.main()
