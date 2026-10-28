import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from anchor_integration_protocol import REGIMES, regime_config, resume_kind, scientific_sha, validate_config
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig, SharedAnchorJointRig, coordinate_objective
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots
from virtual_fusion_normal_runtime import MatchedFrameStream


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.surface_tokenizer = nn.Linear(3, 12)
        self.motion_encoder = nn.Linear(12, 12)
        self.xyz_decoder = nn.Linear(12, 1)
        self.ar_calls = 0

    def encode(self, surface):
        return self.motion_encoder(self.surface_tokenizer(surface['query_points'])), None

    def forward_encoded(self, features, virtual, targets, **kwargs):
        self.ar_calls += 1
        value = self.xyz_decoder(features).square().mean((1, 2, 3))
        return {'loss_per_asset': value, 'main_accuracy': value.new_zeros(())}


class IntegrationObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.head = AnchorJointBinding(AnchorJointConfig(input_dim=12, dim=16, heads=4,
            reference_depth=1, target_depth=1, max_joints=8, gradient_checkpointing=False))
        self.backbone = TinyBackbone()
        self.surface = {'query_points': torch.randn(2, 3, 7, 3), 'query_normals': torch.randn(2, 3, 7, 3)}
        self.geometry = {'frame_target_joints': torch.randn(2, 3, 3, 3)*.2,
            'target_parents': torch.tensor([[-1, 0, 1], [-1, 0, 1]]),
            'joint_mask': torch.ones(2, 3, dtype=torch.bool)}
        self.args = (self.surface, None, self.geometry, None, None, torch.tensor([0, 1]), 10., .25)

    def test_default_joint_path_matches_legacy_formula_bitwise(self):
        a = SharedAnchorJointRig(self.backbone, self.head)
        b = SharedAnchorJointRig(copy.deepcopy(self.backbone), copy.deepcopy(self.head), ar_enabled=True)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        left = a(*self.args)
        features, virtual = b.backbone.encode(self.surface)
        ar = b.backbone.forward_encoded(features, virtual, None, auxiliary_weight=0.,
            target_frame_indices=None, target_frame_mask=None)
        reference = self.args[5]
        prediction = b.head(features, self.surface['query_points'], self.surface['query_normals'],
            reference_slots(self.geometry['frame_target_joints'], reference), self.geometry['target_parents'],
            self.geometry['joint_mask'], reference)
        losses = coordinate_objective(prediction, self.geometry['frame_target_joints'],
            self.geometry['target_parents'], self.geometry['joint_mask'], reference, .25)
        right = {**losses, 'loss_per_asset': ar['loss_per_asset']+10*losses['loss_per_asset'],
            'ar_loss': ar['loss_per_asset'], 'main_accuracy': ar['main_accuracy']}
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        left['loss_per_asset'].sum().backward()
        right['loss_per_asset'].sum().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_stage2_does_not_execute_or_differentiate_AR_branch(self):
        model = SharedAnchorJointRig(self.backbone, self.head, ar_enabled=False)
        with patch.object(self.backbone, 'forward_encoded', side_effect=AssertionError('AR executed')):
            result = model(*self.args)
        result['loss_per_asset'].sum().backward()
        self.assertTrue(all(p.grad is None for p in self.backbone.xyz_decoder.parameters()))
        for module in (self.backbone.surface_tokenizer, self.backbone.motion_encoder, self.head):
            self.assertGreater(sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None), 0)
        torch.testing.assert_close(result['loss_per_asset'], 10*(result['xyz_loss']+.25*result['bone_loss']))
        self.assertTrue(torch.equal(result['ar_loss'], torch.zeros(2)))

    def test_AR_switch_rejects_numeric_alias(self):
        with self.assertRaises(ValueError):
            SharedAnchorJointRig(self.backbone, self.head, ar_enabled=0)


class IntegrationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'model_training/experiments/anchor_integration_5k_20261028_v1.json').read_text())

    def test_declared_full_data_and_three_factors(self):
        base = validate_config(self.config)
        self.assertEqual(base['training']['target_assets'], 80000)
        self.assertEqual(base['data']['manifests']['train']['rows'], 5000)
        values = [regime_config(self.config, name) for name in REGIMES]
        self.assertEqual([v['training']['ar_enabled'] for v in values], [True, False, False])
        self.assertEqual([v['training']['backbone_initialization'] for v in values], ['pretrained', 'pretrained', 'random'])
        self.assertEqual([v['training']['backbone_exposure_offset'] for v in values], [20000, 20000, 0])
        self.assertEqual(len({scientific_sha(v) for v in values}), 3)

    def test_unreviewed_factor_changes_rejected(self):
        for key, value in [('additional_assets', 100000), ('loss', 'AR_only'), ('scratch_seed', 99), ('head_lr_multiplier', 1)]:
            changed = copy.deepcopy(self.config)
            changed['training'][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)
        with self.assertRaises(ValueError):
            regime_config(self.config, 'unknown')

    def test80k_requires_completed_control_admission(self):
        self.config['execution']['phase'] = 'common80k'
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def parent_payload(self):
        old = json.loads((ROOT/'model_training/experiments/anchor_joint_binding_5k_20261027_v1.json').read_text())
        old['training']['variant'] = 'reference_transport'
        return {**self.config['training']['parent_continuation'], 'configuration': old}

    def test_declared_parent_migration_is_narrow(self):
        run = regime_config(self.config, 'joint_pretrained')
        saved = self.parent_payload()
        resume = self.config['training']['parent_continuation']
        self.assertEqual(resume_kind(run, saved, resume), 'declared_parent')
        for name in ('pretrained_stage2', 'scratch_stage2'):
            with self.assertRaises(ValueError):
                resume_kind(regime_config(self.config, name), saved, resume)
        for key, value in [('step', 1259), ('code_commit', 'wrong'), ('input_frame_exposures', 1)]:
            with self.assertRaises(ValueError):
                resume_kind(run, {**saved, key: value}, resume)
        changed = copy.deepcopy(run)
        changed['training']['head_lr_multiplier'] = 1
        with self.assertRaises(ValueError):
            resume_kind(changed, saved, resume)

    def test_phase_is_not_a_new_learning_recipe(self):
        before = scientific_sha(self.config)
        self.config['execution']['phase'] = 'common80k'
        self.assertEqual(before, scientific_sha(self.config))

    def test_guard_binds_the_new_family_to_its_entrypoint(self):
        from anchor_integration_protocol import ENTRYPOINT, SCOPE
        from virtual_pair_guard import operation, scheduler_runtime
        active = {'label': SCOPE, 'explicit_human_authorized': True, 'required_entrypoint': ENTRYPOINT}
        self.assertEqual(operation({'active_operation': active}), active)
        self.assertEqual(scheduler_runtime(self.config)['cpu_slots'], 16)
        with self.assertRaises(RuntimeError):
            operation({'active_operation': {**active, 'required_entrypoint': 'wrong.py'}})

    def test_completed_stream_extension_preserves_every_prefix_request(self):
        policy = self.config['training']['frame_policy']
        for rank in range(2):
            args = dict(rank=rank, world_size=2, seed=20261028, effective_assets=48,
                frame_policy=policy, manifest_sha256='a'*64)
            old = MatchedFrameStream(range(5000), target_assets=60000,
                update_boundaries=list(range(5000, 60001, 5000)), **args)
            new = MatchedFrameStream(range(5000), target_assets=80000,
                update_boundaries=list(range(5000, 80001, 5000)), **args)
            while old.cursor < old.local_total:
                self.assertEqual(old.take(), new.take())
            extended = MatchedFrameStream.extend_completed(range(5000), old.state_dict(),
                target_assets=80000, update_boundaries=list(range(5000, 80001, 5000)))
            self.assertEqual(extended.state_dict(), new.state_dict())


@dataclass
class FakeConfig:
    query_tokens: int = 1024
    motion_depth: int = 12


class FakeLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = nn.Module()
        self.model.decoder.embed_tokens = nn.Embedding(5, 8)
        self.lm_head = nn.Linear(8, 5, bias=False)

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, **kwargs):
        pass


class FakeSurface(nn.Linear):
    width = 4

    def __init__(self):
        super().__init__(3, 4)


class FakeRig(nn.Module):
    def __init__(self, surface, main, config, **kwargs):
        super().__init__()
        self.surface_tokenizer, self.xyz_decoder = surface, main
        self.motion_encoder, self.frame_fusion, self.auxiliary = (nn.Linear(4, 4) for _ in range(3))


class RandomInitializationTests(unittest.TestCase):
    def test_constructor_random_path_never_reads_checkpoint(self):
        import virtual_query_xyz_runtime as runtime
        config = {'initialization': {'seed': 1, 'upstream_source_commit': 'revision', 'checkpoint': '/weights',
            'official_checkpoint': '/weights', 'sha256': 'weight-hash', 'model_config': '/model',
            'model_config_sha256': 'model-hash'},
            'runtime': {'unirig_root': '/upstream', 'tokenizer_config': '/tokenizer',
                'tokenizer_config_sha256': 'tokenizer-hash', 'opt_config_root': '/opt'},
            'architecture': {}, 'auxiliary_codec': {}}
        modules = {'transformers': SimpleNamespace(OPTConfig=SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace(hidden_size=8)), OPTForCausalLM=FakeLM),
            'src.model.michelangelo.get_model': SimpleNamespace(get_encoder_simplified=lambda **k: FakeSurface()),
            'train_dynamic_rig': SimpleNamespace(build_tokenizer=lambda path: SimpleNamespace(vocab_size=5))}
        architecture = json.dumps({'llm': {'pretrained_model_name_or_path': 'unused'},
            'mesh_encoder': {'__target__': 'michelangelo_encoder'}})
        with patch.dict(sys.modules, modules), \
                patch.object(runtime.subprocess, 'check_output', side_effect=lambda cmd, **k: 'revision' if 'rev-parse' in cmd else ''), \
                patch.object(runtime, 'sha256', side_effect=lambda path: {'/model': 'model-hash', '/tokenizer': 'tokenizer-hash'}[path]), \
                patch.object(Path, 'read_text', return_value=architecture), \
                patch.object(runtime, 'VirtualQueryXYZConfig', FakeConfig), \
                patch.object(runtime, 'VirtualQueryXYZRig', FakeRig), \
                patch.object(runtime, 'FixedQuerySurfaceTokenizer', side_effect=lambda a, b: nn.Sequential(a, b)), \
                patch.object(runtime, 'AuxSkeletonTokens', return_value=SimpleNamespace(vocab_size=5, pad=0)), \
                patch.object(torch, 'load', side_effect=AssertionError('pretrained checkpoint read')):
            model, _, _, record = runtime.build_model(config, 'cpu', load_official_weights=False, initialization_seed=41)
            self.assertIsNone(record['checkpoint_sha256'])
            self.assertEqual(record['load_report']['checkpoint_tensors_loaded'], 0)
            self.assertEqual(record['seed'], 41)
            self.assertTrue(any(bool(p.detach().abs().sum()) for p in model.parameters()))


class IntegrationDriverTests(unittest.TestCase):
    def setUp(self):
        import anchor_integration_runtime
        self.runtime = anchor_integration_runtime
        self.config = json.loads((ROOT/'model_training/experiments/anchor_integration_5k_20261028_v1.json').read_text())

    def test_controls_stop_at_matched60k_without_retraining_joint_arm(self):
        calls = []

        def segment(config, output, source, device, rank, world, end, resume):
            name = config['training']['regime']
            calls.append((name, end, resume))
            return {'complete': True, 'samples': end, 'checkpoints': [{'path': str(output/str(end))}],
                'initialization': {'head_initial_sha256': 'same'}}

        with patch.object(self.runtime, 'train_segment', side_effect=segment), \
                patch.object(self.runtime, 'write_json'), patch.object(self.runtime, 'evaluate') as evaluate:
            result = self.runtime.train(self.config, Path('/output'), 'source', None, 0, 2)
        self.assertEqual(len(calls), 12)
        self.assertEqual(calls[0][:2], ('pretrained_stage2', 10000))
        self.assertEqual(calls[-1][:2], ('scratch_stage2', 60000))
        self.assertTrue(all(name != 'joint_pretrained' for name, _, _ in calls))
        self.assertTrue(result['phase_complete'])
        self.assertFalse(result['operation_complete'])
        evaluate.assert_not_called()

    def test_partial_stage_stops_without_claiming_completion(self):
        with patch.object(self.runtime, 'train_segment', return_value={'complete': False, 'samples': 48, 'checkpoints': []}):
            result = self.runtime.train(self.config, Path('/output'), 'source', None, 0, 2)
        self.assertFalse(result['complete'])
        self.assertFalse(result['operation_complete'])

    def test80k_uses_three_declared_resumes_and_final_evaluations(self):
        self.config['execution'].update(phase='common80k', control_resumes={
            r: {'path': r+'60k'} for r in REGIMES[1:]})
        calls = []

        def segment(config, output, source, device, rank, world, end, resume):
            calls.append((config['training']['regime'], end, resume))
            return {'complete': True, 'samples': end, 'checkpoints': [{'path': str(end)}],
                'initialization': {'head_initial_sha256': 'same'}}

        with patch.object(self.runtime, 'train_segment', side_effect=segment), \
                patch.object(self.runtime, 'write_json'), \
                patch.object(self.runtime, 'load_model', return_value=(object(), object(), None, None, {}, None)), \
                patch.object(self.runtime, 'read_all', return_value=({}, {})), \
                patch.object(self.runtime, 'evaluate', return_value={'complete': True}) as evaluate, \
                patch.object(self.runtime, 'release'), patch.object(self.runtime.dist, 'barrier'):
            result = self.runtime.train(self.config, Path('/output'), 'source', SimpleNamespace(index=0), 0, 2)
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[0][2], self.config['training']['parent_continuation'])
        self.assertEqual([end for _, end, _ in calls], [70000]*3+[80000]*3)
        self.assertEqual(evaluate.call_count, 3)
        self.assertTrue(result['operation_complete'])


if __name__ == '__main__':
    unittest.main()
