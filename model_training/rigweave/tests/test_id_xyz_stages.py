"""CPU-only checks at the audit, receipt, sampler and checkpoint boundaries."""
from contextlib import ExitStack
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from transformers import OPTConfig

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT.parent / 'analysis')]
import id_xyz_runtime as runtime
import id_xyz_stages as stages
import id_xyz_launch_guard as guard
from rigweave.dynamic_rig.id_xyz_tokens import IdXYZTokens


def config(arm='consistency'):
    return json.loads((ROOT.parent / 'experiments' / f'id_xyz_{arm}_20000_v1.json').read_text())


def receipt(cfg):
    return {'passed': True, 'mode': 'audit', 'code_commit': 'a' * 40,
            'config_sha256': 'b' * 64, 'source_sha256': runtime.source_hashes(),
            'paired_contract_sha256': runtime.paired_contract_hash(cfg)}


class StopAtBoundary(Exception):
    pass


class TinyGroups(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('surface_tokenizer', 'motion_encoder', 'xyz_decoder'):
            setattr(self, name, nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(.1)


class ReferenceTokenizer:
    bos, eos, pad, token_id_branch, vocab_size = 257, 258, 259, 256, 271

    def cls_name_to_token(self, name):
        if name != 'articulationxl':
            raise ValueError(name)
        return 270

    def tokenize(self, value):
        def quantize(xyz):
            return np.rint((xyz + np.float32(1.)) * np.float32(128.)).clip(0, 255).astype(np.int64).tolist()
        result = [self.bos, self.cls_name_to_token(value.cls)]
        for index, bone in enumerate(value.bones):
            if value.branch[index]:
                result.extend([self.token_id_branch, *quantize(bone[:3])])
            result.extend(quantize(bone[3:]))
        return np.asarray([*result, self.eos], dtype=np.int64)

    def detokenize(self, values):
        xyz, index = [], 2
        while values[index] != self.eos:
            if values[index] == self.token_id_branch:
                index += 4
            xyz.append(values[index:index + 3])
            index += 3
        joints = (np.asarray(xyz, dtype=np.float32) + np.float32(.5)) / np.float32(128.) - np.float32(1.)
        return SimpleNamespace(joints=joints)


class IdXYZStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_config_pair_is_identical_except_arm_and_weight(self):
        left, right = config(), config('control')
        runtime.validate_config(left)
        runtime.validate_config(right)
        self.assertEqual((left.pop('arm'), right.pop('arm')), ('consistency', 'control'))
        self.assertEqual((left['consistency'].pop('weight'), right['consistency'].pop('weight')), (1, 0))
        self.assertEqual(left, right)
        self.assertEqual(runtime.paired_contract_hash(config()), runtime.paired_contract_hash(config('control')))

    def test_receipt_file_hash_pass_flag_and_producer_binding_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            payload = receipt(config())
            path.write_text(json.dumps(payload), encoding='utf-8')
            spec = {key: payload[key] for key in ('code_commit', 'config_sha256')}
            spec.update(path=str(path), sha256=runtime.sha256(path))
            self.assertEqual(stages.read_reviewed(spec), payload)
            for key in ('sha256', 'code_commit', 'config_sha256'):
                changed = dict(spec, **{key: 'changed'})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    stages.read_reviewed(changed)
            for passed in (False, 1, 'true', None):
                payload['passed'] = passed
                path.write_text(json.dumps(payload), encoding='utf-8')
                spec['sha256'] = runtime.sha256(path)
                with self.subTest(passed=passed), self.assertRaises(ValueError):
                    stages.read_reviewed(spec)

    def test_receipt_rejects_changed_source_or_training_contract(self):
        cfg = config()
        original = receipt(cfg)
        stages.validate_receipt(cfg, original)
        for key in ('source_sha256', 'paired_contract_sha256'):
            changed = dict(original, **{key: {}})
            with self.subTest(key=key), self.assertRaises(ValueError):
                stages.validate_receipt(cfg, changed)
        for section, field, value in (
                ('data', 'stream_seed', 7), ('initialization', 'id_seed', 9),
                ('training', 'effective_assets_per_update', 24),
                ('evaluation', 'seed', 44), ('final_evaluation', 'input_frames', 4)):
            changed = deepcopy(cfg)
            changed[section][field] = value
            with self.subTest(section=section), self.assertRaises(ValueError):
                stages.validate_receipt(changed, original)

    def test_receipt_rejects_changed_tokenizer_contract(self):
        cfg = config()
        original = receipt(cfg)
        changed = deepcopy(cfg)
        changed['runtime']['tokenizer_config_sha256'] = 'e' * 64
        with self.assertRaises(ValueError):
            stages.validate_receipt(changed, original)

    def test_learning_rate_implementation_is_bound_to_receipt(self):
        cfg = config()
        original = receipt(cfg)
        target = (ROOT / 'scripts/virtual_frame_runtime.py').resolve()
        real_open = Path.open

        def changed_open(path, *args, **kwargs):
            if path.resolve() == target:
                return io.BytesIO(b'changed exposure_learning_rate implementation')
            return real_open(path, *args, **kwargs)

        with patch.object(Path, 'open', changed_open), self.assertRaises(ValueError):
            stages.validate_receipt(cfg, original)

    def test_capacity_covers_every_configured_tier_at_its_upper_bound(self):
        cfg = config()
        stress = {'longest_tokens': [2, 3, 4], **{f'geometry_t{t}': [t + 10] for t in (6, 12, 24)}}
        cases = stages.capacity_cases(cfg, {'stress_groups': stress})
        self.assertEqual(len(cases), 9)
        self.assertEqual({(c['frames'], c['batch_per_rank']) for c in cases}, {(6, 12), (12, 6), (24, 3)})
        for frame in range(2, 25):
            ceiling = 6 if frame <= 6 else 12 if frame <= 12 else 24
            self.assertEqual(cfg['training']['frame_policy'][str(frame)], cfg['training']['frame_policy'][str(ceiling)])
        for frame in (5, 11, 23):
            changed = deepcopy(cfg)
            changed['training']['frame_policy'][str(frame)] = 24
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                runtime.validate_config(changed)
        for case in cases:
            requests = [stages.requests_for(cfg, case['indices'], frames=case['frames'],
                        batch=case['batch_per_rank'], rank=rank, number=3) for rank in (0, 1)]
            seeds = [request.sampling_seed for rank in requests for request in rank]
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertTrue(all(request.frames == case['frames'] for rank in requests for request in rank))

    def test_capacity_rejects_an_audit_receipt_in_place_of_preflight(self):
        cfg = config()
        cfg['safety']['primary_accepted'] = True
        wrong_stage = receipt(cfg)
        with patch.object(stages, 'read_reviewed', return_value=wrong_stage), \
                patch.object(stages, 'fixed_dataset') as dataset, \
                patch.object(stages, 'optimizer_for', side_effect=StopAtBoundary):
            try:
                stages.capacity(None, None, None, {}, {'train': []}, cfg,
                                SimpleNamespace(), torch.device('cpu'), 0, 2, {})
            except ValueError:
                pass
            except StopAtBoundary:
                pass
        self.assertFalse(dataset.called, 'an audit receipt was accepted as the required preflight receipt')

    def test_cpu_model_initialization_does_not_request_flash_attention(self):
        cfg = config()
        cfg['initialization']['model_config'] = str(ROOT.parent / 'experiments/shared_virtual_upstream_20261012.yaml')
        upstream = ModuleType('src.model.michelangelo.get_model')
        upstream.get_encoder_simplified = Mock(side_effect=AssertionError('surface model must not be constructed'))
        captured = []

        def stop_decoder(value):
            captured.append(value)
            raise StopAtBoundary

        def digest(path):
            key = 'sha256' if str(path) == cfg['initialization']['checkpoint'] else 'model_config_sha256'
            return cfg['initialization'][key]

        transformers_stub = ModuleType('transformers')
        transformers_stub.OPTConfig = SimpleNamespace(from_pretrained=lambda _path, **kw: OPTConfig(**kw))
        transformers_stub.OPTForCausalLM = stop_decoder
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {upstream.__name__: upstream, 'transformers': transformers_stub}))
            stack.enter_context(patch.object(runtime.subprocess, 'check_output', side_effect=[cfg['initialization']['upstream_source_commit'], '']))
            stack.enter_context(patch.object(runtime, 'sha256', side_effect=digest))
            stack.enter_context(patch.object(runtime, 'reset_random'))
            stack.enter_context(patch.object(runtime, 'load_tokenizer', return_value=(SimpleNamespace(vocab_size=271), None)))
            with self.assertRaises(StopAtBoundary):
                runtime.build_model(cfg, torch.device('cpu'))
        self.assertNotIn('flash', captured[0]._attn_implementation or '',
                         'CPU audit hides CUDA, so its decoder constructor cannot request FlashAttention')

    def test_cpu_target_audit_uses_existing_adapter_and_source_order_api(self):
        codec = IdXYZTokens(271, tuple(range(256)), bos=257, eos=258, pad=259,
                           branch=256, class_token=270)
        joints = torch.linspace(-.7, .7, 2 * 4 * 3).reshape(1, 2, 4, 3)
        geometry = {'joint_count': torch.tensor([4]), 'target_parents': torch.tensor([[-1, 0, 0, 1]]),
                    'frame_target_joints': joints, 'joint_mask': torch.ones(1, 4, dtype=torch.bool),
                    'frame_mask': torch.ones(1, 2, dtype=torch.bool),
                    'selected_frames': torch.tensor([[2, 7]]), 'heldout_raw_ids': [[30, 31]],
                    'asset_id': ['synthetic_nonidentity_dfs']}
        targets = runtime.build_targets(geometry, codec, 'cpu')
        self.assertEqual(targets['dfs_orders'], [[0, 1, 3, 2]])
        spec = ModuleType('src.tokenizer.spec')
        spec.TokenizeInput = SimpleNamespace
        with patch.dict(sys.modules, {spec.__name__: spec}):
            rows = stages.check_targets(geometry, targets, ReferenceTokenizer(), codec)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['parent_identity_exact'])
        self.assertTrue(rows[0]['official_xyz_bins_exact'])
        self.assertFalse(bool((targets['coordinate_mask'] & targets['structure_mask']).any()))

    def test_sampler_restores_next_request_and_exact_checkpoint_boundaries(self):
        cfg = config()
        streams = [runtime.make_stream(cfg, rank, 2) for rank in (0, 1)]
        self.assertEqual(streams[0].plan_sha256, streams[1].plan_sha256)
        self.assertTrue({5000, 10000, 15000, 20000}.issubset(set(streams[0].global_ends.tolist())))
        self.assertEqual(int(streams[0].global_ends[-1]), 20000)
        for rank, stream in enumerate(streams):
            stream.take()
            restored = runtime.make_stream(cfg, rank, 2, state=stream.state_dict())
            self.assertEqual(stream.take(), restored.take())
            control = runtime.make_stream(config('control'), rank, 2)
            self.assertEqual(stream.plan_sha256, control.plan_sha256)

    def test_checkpoint_preserves_model_optimizer_names_rank_rng_and_sampler(self):
        cfg = config()
        model = TinyGroups()
        optimizer = runtime.optimizer_for(model, cfg)
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()
        streams = [runtime.make_stream(cfg, rank, 2) for rank in (0, 1)]
        for stream in streams:
            stream.take()
        cuda_rng = torch.tensor([1, 9, 7], dtype=torch.uint8)
        rank_one = {'rank': 1, 'stream': streams[1].state_dict(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': cuda_rng.clone(), 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.cuda, 'get_rng_state', return_value=cuda_rng), \
                patch.object(stages, 'gather', side_effect=lambda value, world: [value, rank_one]), \
                patch.object(stages.dist, 'barrier'):
            args = SimpleNamespace(output=Path(directory))
            stages.save_checkpoint(model, optimizer, streams[0], cfg, args, {'initial_model_sha256': 'initial'},
                                   'commit', 48, 96, 1, 0, 2)
            path = args.output / 'checkpoint_000048.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            sidecar = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(sidecar['sha256'], runtime.sha256(path))
            self.assertFalse(path.with_suffix('.pt.tmp').exists())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(saved['model'][key], value))
        self.assertEqual(saved['optimizer']['param_groups'], optimizer.state_dict()['param_groups'])
        self.assertEqual({row['group'] for row in saved['optimizer_parameter_names']},
                         {'surface_tokenizer', 'motion_encoder', 'xyz_decoder'})
        self.assertEqual({name for row in saved['optimizer_parameter_names'] for name in row['parameters']},
                         set(dict(model.named_parameters())))
        for rank, state in enumerate(saved['rank_states']):
            self.assertEqual(state['rank'], rank)
            self.assertTrue(torch.equal(state['cuda_rng'], cuda_rng))
            self.assertTrue({'torch_rng', 'numpy_rng', 'python_rng'}.issubset(state))
            restored = runtime.make_stream(cfg, rank, 2, state=state['stream'])
            self.assertEqual(streams[rank].take(), restored.take())

    def test_preserve_random_restores_all_domains_after_exception(self):
        cpu = torch.get_rng_state().clone()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        cuda = torch.tensor([3, 5, 8], dtype=torch.uint8)
        with patch.object(torch.cuda, 'get_rng_state', return_value=cuda), \
                patch.object(torch.cuda, 'set_rng_state') as restore:
            with self.assertRaises(StopAtBoundary), stages.preserve_random():
                torch.rand(3)
                np.random.rand(3)
                random.random()
                raise StopAtBoundary
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(torch.equal(restore.call_args.args[0], cuda))

    def test_ssh_exception_is_bound_to_exact_scope_and_resources(self):
        cfg = config()
        exception = {key: cfg['runtime'][key] for key in ('allocation', 'host', 'physical_gpus',
                     'cpu_slots', 'rss_limit_bytes', 'gpu_memory_fraction_limit', 'keeper_pid')}
        exception.update(scope=guard.SCOPE, explicit_human_authorized=True,
                         kernel_scheduler_membership=False, authorization='explicit test authorization')
        state = {'active_operation': {'label': guard.SCOPE, 'ssh_exception': exception}}
        self.assertEqual(guard.validate_ssh_exception(state, cfg), exception)
        for field, changed in (('scope', 'other'), ('cpu_slots', True), ('physical_gpus', [4, 5]),
                               ('authorization', ''), ('explicit_human_authorized', 1)):
            bad = deepcopy(state)
            bad['active_operation']['ssh_exception'][field] = changed
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                guard.validate_ssh_exception(bad, cfg)


if __name__ == '__main__':
    unittest.main()
