"""CPU checks for the matched legacy reference, with the actual common loop."""
import ast
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import torch._dynamo
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'model_training/rigweave/scripts'
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(SCRIPTS))
import probe_virtual_fusion_reference as reference
from probe_virtual_query_reference import ReferenceConditionRig

BASELINE_PATH = ROOT / 'model_training/experiments/fusion_a_medium_v1.json'
DRAFT_PATH = ROOT / 'model_training/experiments/fusion_legacy_medium_v1.json'


def extract(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError('missing actual helper under test')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)


def pinned_baseline(payload=None):
    original = Path.read_bytes
    baseline = original(BASELINE_PATH) if payload is None else payload

    def read(path):
        return baseline if path.as_posix() == reference.BASELINE_CONFIG['path'] else original(path)

    return patch.object(Path, 'read_bytes', read)


class ConfigTests(unittest.TestCase):
    def test_primary_draft_matches_pinned_a_without_mutation_or_peer(self):
        config = json.loads(DRAFT_PATH.read_text())
        before = deepcopy(config)
        with pinned_baseline():
            matched = reference.validate_config(config)
        self.assertEqual(config, before)
        self.assertEqual(matched, config['reference_match'])
        self.assertNotIn('paired_output', config['probe'])
        self.assertFalse(config['probe']['evaluate_auxiliary'])
        self.assertEqual(config['comparison']['gradient_measurement_updates'], [])
        self.assertEqual((config['comparison']['max_updates'], config['comparison']['assets_per_update'],
                          config['optimization']['schedule_samples']), (512, 48, 24576))
        self.assertEqual(config['comparison']['generation_frame_indices'], [0, 1, 2, 3])
        self.assertEqual(config['comparison']['evaluation_updates'], [128, 256, 512])

    def test_pinned_a_byte_change_is_rejected(self):
        config = json.loads(DRAFT_PATH.read_text())
        with pinned_baseline(BASELINE_PATH.read_bytes() + b'\n'), self.assertRaisesRegex(ValueError, 'bytes changed'):
            reference.validate_config(config)

    def test_semantic_and_undeclared_drift_is_rejected(self):
        mutations = [
            ('data', 'normalization', 'different'), ('data', 'global_rt_removal', False),
            ('data', 'frame_seed', 20261014), ('surface_sampling', 'query_tokens', 512),
            ('initialization', 'seed', 20261014), ('initialization', 'checkpoint', '/old/checkpoint.pt'),
            ('architecture', 'fusion_mode', 'real_queries_virtual'),
            ('architecture', 'gradient_checkpointing', 1),
            ('runtime', 'physical_gpu', 7), ('runtime', 'host', 'pcg01i'),
            ('comparison', 'frames', 2), ('comparison', 'assets_per_update', 47),
            ('comparison', 'max_updates', 511), ('comparison', 'order_seed', 20261014),
            ('comparison', 'initial_evaluation', False), ('comparison', 'generation_frame_indices', [0]),
            ('comparison', 'evaluation_updates', [512]), ('comparison', 'gradient_measurement_updates', [1]),
            ('optimization', 'schedule_samples', 2048), ('optimization', 'peak_lr', .001),
            ('probe', 'auxiliary_weight', .1), ('probe', 'auxiliary_weight', False),
            ('probe', 'evaluate_auxiliary', True), ('probe', 'evaluate_auxiliary', 0),
            ('probe', 'paired_output', '/already/finished/a'), ('probe', 'early_stop', True),
        ]
        with pinned_baseline():
            for section, key, value in mutations:
                config = json.loads(DRAFT_PATH.read_text())
                config[section][key] = value
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    reference.validate_config(config)
            config = json.loads(DRAFT_PATH.read_text())
            config['unreviewed_option'] = True
            with self.assertRaises(ValueError):
                reference.validate_config(config)

    def test_route_and_match_contract_cannot_repin_the_baseline(self):
        for key in ('full_base_initial_state_sha256', 'inputs_sha256', 'training_order_sha256', 'baseline_config'):
            config = json.loads(DRAFT_PATH.read_text())
            config['reference_match'][key] = 'wrong'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'reference_match'):
                reference.validate_config(config)
        config = json.loads(DRAFT_PATH.read_text())
        config['reference_route'] = 'virtual_final_real'
        with self.assertRaises(ValueError):
            reference.validate_config(config)


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(config=reference.CODE_ROOT / 'model_training/experiments/fusion_legacy_medium_v1.json',
                                    output=reference.OUTPUT_ROOT / 'fusion_legacy_medium_v1')
        self.operation = {'mode': 'reference_compare', 'gpu_work_allowed': True, 'formal_training_allowed': False,
            'complete': False, 'required_entrypoint': reference.ENTRYPOINT, 'code_commit': 'published-source',
            'config': str(self.args.config), 'config_sha256': 'config-hash', 'output': str(self.args.output), 'physical_gpu': 5}
        self.state = {'architecture_contract': {'id': 'virtual_query_xyz_aux_ids_v1'}, 'active_operation': self.operation}

    def test_exact_reference_authority_and_isolated_paths(self):
        self.assertIs(reference.declared_operation(self.state, self.args, 'config-hash'), self.operation)
        for key, value in [('mode', 'compare'), ('mode', 'reference_fit'), ('complete', True),
                           ('gpu_work_allowed', False), ('formal_training_allowed', True),
                           ('required_entrypoint', 'model_training/rigweave/scripts/probe_virtual_fusion.py'),
                           ('config_sha256', 'wrong'), ('physical_gpu', 7)]:
            state = deepcopy(self.state)
            state['active_operation'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reference.declared_operation(state, self.args, 'config-hash')
        for args in (SimpleNamespace(config=Path('/old/config.json'), output=self.args.output),
                     SimpleNamespace(config=self.args.config, output=Path('/old/run'))):
            with self.assertRaises(ValueError):
                reference.declared_operation(self.state, args, 'config-hash')

    def test_device_gate_is_single_gpu_and_has_memory_safeguard(self):
        with patch.object(reference.socket, 'gethostname', return_value='pcg02'), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5'}), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'set_device') as set_device, \
                patch.object(torch.cuda, 'set_per_process_memory_fraction') as memory:
            self.assertEqual(str(reference.configure_device()), 'cuda:0')
            set_device.assert_called_once_with(0)
            memory.assert_called_once_with(.95, 0)
            with patch.object(torch.cuda, 'device_count', return_value=2), self.assertRaises(ValueError):
                reference.configure_device()
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7'}), self.assertRaises(ValueError):
                reference.configure_device()
            with patch.object(reference.socket, 'gethostname', return_value='pcg01'), self.assertRaises(ValueError):
                reference.configure_device()

    def test_failure_json_and_exclusive_output_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'config.json'
            config_path.write_text('{}')
            args = SimpleNamespace(config=config_path, output=root / 'run')
            with patch.object(reference.argparse.ArgumentParser, 'parse_args', return_value=args), \
                    patch.object(reference, 'validate_config'), patch.object(reference, 'verify_operation', return_value='source'), \
                    patch.object(reference, 'configure_device', return_value='cpu'), \
                    patch.object(reference, 'run_reference', side_effect=ValueError('fingerprint failure')):
                with self.assertRaisesRegex(ValueError, 'fingerprint failure'):
                    reference.main()
                result = json.loads((args.output / 'result.json').read_text())
                self.assertFalse(result['execution_complete'])
                self.assertFalse(result['accepted'])
                self.assertFalse(result['formal_training_allowed'])
                before = (args.output / 'result.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    reference.main()
                self.assertEqual((args.output / 'result.json').read_bytes(), before)


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.embedding, self.head = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        self.embedding.weight.grad = torch.ones_like(self.embedding.weight)
        self.head.weight.grad = torch.ones_like(self.head.weight)
        self.model = SimpleNamespace(xyz_decoder=SimpleNamespace(get_input_embeddings=lambda: self.embedding,
            get_output_embeddings=lambda: self.head))
        self.norms = {'surface_tokenizer': 1., 'motion_encoder': 2., 'xyz_decoder': 3.,
                      'frame_fusion': 0., 'auxiliary': 0., 'virtual_slots': 0.}

    def test_unused_virtual_fusion_auxiliary_branches_are_not_required(self):
        reference.require_reference_gradients(self.model, self.norms, 0.)

    def test_required_legacy_gradients_must_be_finite_positive(self):
        for key in ('surface_tokenizer', 'motion_encoder', 'xyz_decoder'):
            for value in (0., -1., float('nan'), float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    reference.require_reference_gradients(self.model, {**self.norms, key: value}, 0.)
        with self.assertRaises(ValueError):
            reference.require_reference_gradients(self.model, self.norms, .1)

    def test_embedding_and_head_gradients_cannot_be_missing_zero_or_nonfinite(self):
        for module in (self.embedding, self.head):
            for value in (None, 0., float('nan'), float('inf')):
                module.weight.grad = None if value is None else torch.full_like(module.weight, value)
                with self.subTest(module=module, value=value), self.assertRaises(RuntimeError):
                    reference.require_reference_gradients(self.model, self.norms, 0.)
            module.weight.grad = torch.ones_like(module.weight)

    def test_existing_focused_proof_uses_the_actual_legacy_encoder_and_xyz_loss(self):
        from test_virtual_query_xyz_legacy_parity import LegacyParityTests
        self.assertIs(reference.ReferenceConditionRig, ReferenceConditionRig)
        proof = LegacyParityTests('test_reference_controls_call_actual_legacy_encoder_and_preserve_target_frame')
        proof.setUp()
        proof.test_reference_controls_call_actual_legacy_encoder_and_preserve_target_frame()
        self.assertIsNone(proof.new.motion_encoder.virtual_slots.grad)
        self.assertTrue(all(parameter.grad is None for parameter in proof.new.frame_fusion.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in proof.new.auxiliary.parameters()))
        self.assertGreater(float(proof.new.xyz_decoder.get_input_embeddings().weight.grad.norm()), 0.)
        self.assertGreater(float(proof.new.xyz_decoder.get_output_embeddings().weight.grad.norm()), 0.)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding, self.head = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.surface_tokenizer, self.motion_encoder = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        self.motion_encoder.virtual_slots = nn.Parameter(torch.ones(1))
        self.xyz_decoder = TinyDecoder()
        self.frame_fusion, self.condition_norm, self.auxiliary = nn.Linear(1, 1), nn.Identity(), nn.Linear(1, 1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(.5)


class TinyReference(ReferenceConditionRig):
    def forward(self, surface, targets, *, auxiliary_weight):
        if auxiliary_weight != 0:
            raise AssertionError('auxiliary forward must not run')
        value = self.xyz_decoder.head(self.xyz_decoder.embedding(self.motion_encoder(self.surface_tokenizer(surface['value']))))
        loss = (value - targets['target']).square().mean()
        zero = loss.detach().new_zeros(())
        return {'loss': loss, 'main_loss': loss, 'auxiliary_loss': zero,
                'main_accuracy': zero, 'auxiliary_accuracy': zero}


def fake_common_module():
    module = ModuleType('probe_virtual_query_lambda')
    module.__dict__.update(Path=Path, hashlib=hashlib, json=json, np=np, torch=torch, time=time)
    extract(SCRIPTS / 'probe_virtual_query_lambda.py',
            ['transfer', 'comparison_plan', 'training_order', 'parameter_fingerprint', 'run_comparison'], module.__dict__)
    extract(SCRIPTS / 'probe_virtual_query_xyz.py', ['grad_norms'], module.__dict__)
    extract(SCRIPTS / 'virtual_query_xyz_runtime.py', ['sha256', 'write_json'], module.__dict__)
    module.require_gradients = lambda *args: (_ for _ in ()).throw(AssertionError('original gradient gate must be patched'))
    module.shared_gradient_measurement = lambda *args: (_ for _ in ()).throw(AssertionError('virtual/auxiliary gradients must not be measured'))
    module.set_exposure_lr = lambda optimizer, seen, cfg: (optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['betas'][0])
    return module


class InjectedCommonFlowTests(unittest.TestCase):
    def run_case(self, root, wrong_fingerprint=None):
        module = fake_common_module()
        train = [({'value': torch.tensor([[.5 + i / 100]])}, {'target': torch.tensor([[.4]])},
                  {'asset_id': ['train_%d' % i]}) for i in range(64)]
        groups = {'train': train, 'train_heldout': train[:8],
                  'valid': [({'value': torch.ones(1, 1)}, {'target': torch.zeros(1, 1)}, {'asset_id': ['valid_%d' % i]}) for i in range(32)]}
        base = TinyBase()
        baseline = json.loads(BASELINE_PATH.read_text())
        baseline['comparison']['max_updates'] = 2
        baseline['comparison']['evaluation_updates'] = [1, 2]
        baseline['optimization']['schedule_samples'] = 96
        baseline_path = root / 'baseline.json'
        baseline_path.write_text(json.dumps(baseline))
        baseline_spec = {'path': baseline_path.as_posix(), 'sha256': module.sha256(baseline_path)}
        order = module.training_order(64, 96, baseline['comparison']['order_seed'])
        expected = {'full_base_initial_state_sha256': module.parameter_fingerprint(base),
                    'inputs_sha256': reference.tree_fingerprint(groups['train'] + groups['train_heldout'] + groups['valid']),
                    'training_order_sha256': hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()}
        if wrong_fingerprint:
            expected[wrong_fingerprint] = '0' * 64
        config = deepcopy(baseline)
        del config['probe']['paired_output']
        config['probe'].update(auxiliary_weight=0, evaluate_auxiliary=False)
        config['comparison']['gradient_measurement_updates'] = []
        config.update(reference_route=reference.ROUTE, reference_match={'baseline_config': baseline_spec, **expected})
        config_path = root / 'config.json'
        config_path.write_text(json.dumps(config))
        output = root / 'output'
        output.mkdir()
        calls, events = {'build': 0, 'load': 0}, []

        def load(cfg):
            calls['load'] += 1
            return groups, {'selection_groups': {}}

        def build(cfg, device):
            calls['build'] += 1
            return base, object(), None, {'checkpoint_sha256': cfg['initialization']['sha256'],
                'old_method_trained_weights_loaded': False,
                'load_report': {key: {'strict': True} for key in ('transformer.', 'mesh_encoder.', 'output_proj.')}}

        def evaluate(model, tokenizer, data, index, cfg, directory, step, device):
            self.assertFalse(cfg['probe']['evaluate_auxiliary'])
            self.assertEqual(cfg['comparison']['generation_frame_indices'], [0, 1, 2, 3])
            events.append(('evaluation', step))
            return {'groups': {'valid': {'main_accuracy': 1.}}, 'rows': [], 'accepted': True}

        original_adam = torch.optim.AdamW

        def optimizer(*args, **kwargs):
            self.assertTrue((output / 'matched_reference.json').is_file())
            events.append(('optimizer_created',))
            actual = original_adam(*args, **kwargs)
            step = actual.step

            def update(*args, **kwargs):
                events.append(('optimizer_update',))
                return step(*args, **kwargs)

            actual.step = update
            return actual

        module.load_cache, module.build_model, module.evaluate_groups = load, build, evaluate
        import fusion_evaluation
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'probe_virtual_query_lambda': module}))
            stack.enter_context(patch.object(reference, 'BASELINE_CONFIG', baseline_spec))
            stack.enter_context(patch.object(reference, 'EXPECTED_FINGERPRINTS', expected))
            stack.enter_context(patch.object(reference, 'ReferenceConditionRig', TinyReference))
            stack.enter_context(patch.object(fusion_evaluation, 'pair_ready', side_effect=AssertionError('must not wait on finished peers')))
            stack.enter_context(patch.object(base.auxiliary, 'forward', side_effect=AssertionError('auxiliary must be unused')))
            stack.enter_context(patch.object(torch, 'autocast', lambda *args, **kwargs: nullcontext()))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(patch.object(torch.optim, 'AdamW', optimizer))
            if wrong_fingerprint:
                with self.assertRaisesRegex(ValueError, 'before optimizer updates'):
                    reference.run_reference(config, output, 'cpu', 'test-source', config_path=config_path)
                self.assertEqual(events, [])
                self.assertFalse((output / 'matched_reference.json').exists())
                self.assertFalse((output / 'trace.jsonl').exists())
                result = None
            else:
                result = reference.run_reference(config, output, 'cpu', 'test-source', config_path=config_path)
        self.assertEqual(calls, {'build': 1, 'load': 1})
        self.assertIs(module.load_cache, load)
        self.assertIs(module.build_model, build)
        return result, output, events, expected, base

    def test_actual_common_runner_injection_matches_before_updates_and_finishes_full_fixture_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, events, expected, base = self.run_case(Path(directory))
            matched = json.loads((output / 'matched_reference.json').read_text())
            initial = json.loads((output / 'initialization.json').read_text())
            trace = [json.loads(line) for line in (output / 'trace.jsonl').read_text().splitlines()]
            self.assertEqual(matched['optimizer_updates'], 0)
            self.assertEqual(matched['actual'], expected)
            self.assertFalse(matched['peer_wait_required'])
            self.assertEqual(initial['full_base_initial_state_sha256'], expected['full_base_initial_state_sha256'])
            self.assertNotEqual(initial['full_initial_state_sha256'], initial['full_base_initial_state_sha256'])
            self.assertEqual(result['completed_updates'], 2)
            self.assertEqual(result['asset_exposures'], 96)
            self.assertEqual(result['frame_exposures'], 384)
            self.assertEqual(result['training_order_sha256'], expected['training_order_sha256'])
            self.assertEqual([row['step'] for row in trace], [1, 2])
            self.assertEqual([row['asset_exposures'] for row in trace], [48, 96])
            self.assertEqual([event for event in events if event[0] == 'evaluation'],
                             [('evaluation', 0), ('evaluation', 1), ('evaluation', 2)])
            self.assertEqual(events.count(('optimizer_update',)), 2)
            self.assertFalse(result['accepted'])
            self.assertFalse(result['formal_training_allowed'])
            self.assertFalse(result['auxiliary_applicable'])
            self.assertIsNone(base.motion_encoder.virtual_slots.grad)
            self.assertTrue(all(parameter.grad is None for parameter in base.frame_fusion.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in base.auxiliary.parameters()))

    def test_each_bad_reference_fingerprint_prevents_any_optimizer_creation_or_update(self):
        for key in reference.EXPECTED_FINGERPRINTS:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                self.run_case(Path(directory), wrong_fingerprint=key)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
