from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
import virtual_pair_runtime as runtime
import virtual_pair_guard as guard


def configuration():
    return json.loads((ROOT / 'model_training/experiments/virtual_query_matched_5k_20261025_v1.json').read_text())


class PairRuntimeTests(unittest.TestCase):
    def test_stage_requires_exact_approval_and_training_acceptance(self):
        path = ROOT / 'model_training/experiments/virtual_query_matched_5k_20261025_v1.json'
        config = configuration()
        args = SimpleNamespace(config=path, output=ROOT / 'unused_test_output', mode='audit')
        run = {'mode': 'audit', 'config': str(path), 'output': str(args.output),
               'approved': True, 'config_sha256': runtime.sha256(path)}
        op = {'label': guard.SCOPE, 'explicit_human_authorized': True, 'complete': False,
              'runs': [run], 'gpu_work_allowed': False}
        state = {'active_operation': op}
        guard.validate_stage(state, config, args)
        args.mode = run['mode'] = 'train'
        with self.assertRaises(RuntimeError):
            guard.validate_stage(state, config, args)
        op.update(gpu_work_allowed=True, formal_training_allowed=True,
                  CPU_audit_accepted=True, capacity_accepted=True)
        guard.validate_stage(state, config, args)
        run['config_sha256'] = '0' * 64
        with self.assertRaises(RuntimeError):
            guard.validate_stage(state, config, args)

    def test_fixed_5k_and_budget(self):
        config = configuration()
        runtime.validate_config(config)
        for key, value in [('target_assets', 5000), ('effective_assets_per_update', 24)]:
            wrong = deepcopy(config)
            wrong['training'][key] = value
            with self.assertRaises(ValueError):
                runtime.validate_config(wrong)
        config['data']['manifests']['train']['rows'] = 8
        with self.assertRaises(ValueError):
            runtime.validate_config(config)

    def test_no_loss_confound(self):
        for section, key in [('optimization', 'auxiliary_weight'), ('pair', 'consistency_weight')]:
            config = configuration()
            config[section][key] = .1
            with self.assertRaises(ValueError):
                runtime.validate_config(config)

    def test_pair_sampling_plan_and_first_full_epoch(self):
        config = configuration()
        streams = [runtime.make_stream(config, rank, 2) for rank in range(2)]
        self.assertEqual(streams[0].plan_sha256, streams[1].plan_sha256)
        self.assertEqual(len(set(streams[0].order[:5000].tolist())), 5000)
        self.assertEqual(int(streams[0].global_ends[-1]), 100000)
        self.assertEqual(len(streams[0].global_ends), 2100)
        for stream in streams:
            stream.take()
            replay = runtime.make_stream(config, stream.rank, 2, stream.state_dict())
            self.assertEqual(stream.take(), replay.take())

    def test_contract_excludes_only_operational_resume_root(self):
        config = configuration()
        changed = deepcopy(config)
        changed['runtime']['repo'] = '/new/allocation/code'
        changed['training']['resume_root'] = '/old/run'
        self.assertEqual(runtime.contract_sha(config), runtime.contract_sha(changed))
        changed['optimization']['peak_lr'] *= .5
        self.assertNotEqual(runtime.contract_sha(config), runtime.contract_sha(changed))

    def test_learning_rate_uses_same_full_horizon_after_segment(self):
        config = configuration()
        model = torch.nn.Linear(2, 1)
        optimizer = runtime.optimizer_for(model, config)
        first = runtime.set_lr(optimizer, 5000, config)
        second = runtime.set_lr(optimizer, 5000, config)
        self.assertEqual(first, second)
        self.assertNotEqual(first, runtime.set_lr(optimizer, 0, config))

    def test_exact_model_adam_sampler_restore_and_two_copies(self):
        config = configuration()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'local'
            output.mkdir()
            config['artifacts']['durable_root'] = str(Path(directory) / 'durable')
            model = torch.nn.Linear(2, 1)
            optimizer = runtime.optimizer_for(model, config)
            model(torch.ones(2, 2)).sum().backward()
            optimizer.step()
            stream = runtime.make_stream(config, 0, 1)
            rng = {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.tensor([], dtype=torch.uint8),
                   'numpy_rng': runtime.np.random.get_state(), 'python_rng': runtime.random.getstate()}
            with patch.object(runtime, 'random_state', return_value=rng):
                spec = runtime.save_checkpoint(model, optimizer, stream, config, output, 'virtual_on',
                    {}, 'test', 0, 0, 0, 0, 1, torch.device('cpu'))
            self.assertTrue(spec['both_copies_verified'])
            self.assertEqual(runtime.sha256(spec['path']), runtime.sha256(spec['durable_path']))
            target = torch.nn.Linear(2, 1)
            target_optimizer = runtime.optimizer_for(target, config)
            restored = runtime.restore_checkpoint(target, target_optimizer, config, spec, 'virtual_on', 0, 1)
            self.assertEqual(runtime.parameter_fingerprint(model), runtime.parameter_fingerprint(target))
            self.assertEqual(runtime.tree_fingerprint(optimizer.state_dict()), runtime.tree_fingerprint(target_optimizer.state_dict()))
            self.assertEqual(restored[0].state_dict(), stream.state_dict())
            self.assertEqual(runtime.latest_checkpoint(output, config, 'virtual_on'), spec)
            with self.assertRaises(RuntimeError):
                runtime.restore_checkpoint(target, target_optimizer, config, spec, 'virtual_off', 0, 1)
            with patch.object(runtime, 'random_state', return_value=rng), self.assertRaises(RuntimeError):
                runtime.save_checkpoint(model, optimizer, stream, config, output, 'virtual_on',
                    {}, 'test', 0, 0, 0, 0, 1, torch.device('cpu'))

    def test_deadline_is_explicit(self):
        config = configuration()
        config['runtime']['stop_before_utc'] = '2000-01-01T00:00:00Z'
        self.assertTrue(guard.deadline_due(config))
        config['runtime']['stop_before_utc'] = '2099-01-01T00:00:00Z'
        self.assertFalse(guard.deadline_due(config))
        config['runtime']['stop_before_utc'] = '2099-01-01T00:00:00'
        with self.assertRaises(ValueError):
            guard.deadline_due(config)

    def test_final_budget_without_evaluation_is_incomplete(self):
        config = configuration()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            progress = dict.fromkeys(runtime.VARIANTS, 100000)
            self.assertEqual(runtime.pending_variants(progress, output, config), list(runtime.VARIANTS))
            for variant in runtime.VARIANTS:
                runtime.write_json(output / variant / 'evaluation_100000/summary.json', {
                    'complete': True, 'samples': 100000, 'variant': variant,
                    'contract_sha256': runtime.contract_sha(config)})
            self.assertEqual(runtime.pending_variants(progress, output, config), [])


if __name__ == '__main__':
    unittest.main()
