"""CPU proofs for constant-LR continuation with preserved optimizer, RNG and exposure state."""
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_virtual_query_training import ROOT, namespace
from test_virtual_fusion_normal_training import namespace as normal_namespace
from virtual_fusion_continuation import (
    BASE_SHA256, BOUNDARIES, CHECKPOINT, OLD_PLAN_SHA256, SOURCE_CONFIG,
    CHAINED_CHECKPOINT, CHAINED_SOURCE_CONFIG, CHAINED_PLAN_SHA256,
    EXTENDED_CHECKPOINT, EXTENDED_SOURCE_CONFIG, EXTENDED_PLAN_SHA256, PURPOSE, continuation_contract,
    B_CHECKPOINT, B_SOURCE_CONFIG,
    A120_CHECKPOINT, A120_SOURCE_CONFIG, A120_PLAN_SHA256,
    extend_checkpoint_stream, restore_training_state, set_continuation_lr,
    validate_checkpoint_payload, validate_continuation_config,
)
from virtual_fusion_normal_runtime import MatchedFrameStream


def configs():
    folder = ROOT / 'model_training/experiments'
    return (json.loads((folder / 'fusion_normal_a_v1.json').read_text()),
            json.loads((folder / 'fusion_normal_a_extend40000_v1.json').read_text()))


def rng_state():
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}


def payload_fixture():
    original, _ = configs()
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9499999512612207, .999), weight_decay=0)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    for state in optimizer.state.values():
        state['step'].fill_(420)
    rank_states = []
    for rank in (0, 1):
        stream = normal_namespace()['make_stream'](original, rank, 2)
        stream.cursor = stream.local_total
        rank_states.append({'stream': stream.state_dict(), **rng_state()})
    initial = {'route': 'a', 'source': CHECKPOINT['source'], 'config_sha256': SOURCE_CONFIG['sha256'],
               'full_base_initial_state_sha256': BASE_SHA256, 'shared_initial_state_sha256': BASE_SHA256,
               'exposure_plan_sha256': OLD_PLAN_SHA256}
    return {'schema': 'virtual_query_xyz_aux_ids_v1', 'purpose': 'normal5000_trend',
            'source': CHECKPOINT['source'], 'config': original, 'architecture': {},
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'asset_exposures': 20000, 'step': 420, 'supervised_frame_exposures': 263608,
            'rank_states': rank_states, 'initialization': initial}


def chained_payload_fixture():
    payload = payload_fixture()
    original, source = configs()
    payload.update(purpose=PURPOSE, source=CHAINED_CHECKPOINT['source'], config=source,
                   asset_exposures=40000, step=840, supervised_frame_exposures=528296)
    for value in payload['optimizer']['state'].values():
        value['step'].fill_(840)
    payload['optimizer']['param_groups'][0]['lr'] = 1e-4
    for rank in (0, 1):
        stream = extend_checkpoint_stream(payload, rank)
        stream.cursor = stream.local_total
        payload['rank_states'][rank]['stream'] = stream.state_dict()
    payload['initialization']['training_continuation'] = {
        'checkpoint': deepcopy(CHECKPOINT), 'source_config': deepcopy(SOURCE_CONFIG), 'route': 'a',
        'source': CHAINED_CHECKPOINT['source'], 'config_sha256': CHAINED_SOURCE_CONFIG['sha256'],
        'start_asset_exposures': 20000, 'target_asset_exposures': 40000,
        'start_updates': 420, 'target_updates': 840,
        'start_supervised_frame_exposures': 263608, 'target_supervised_frame_exposures': 528296,
        'original_exposure_plan_sha256': OLD_PLAN_SHA256, 'extended_exposure_plan_sha256': CHAINED_PLAN_SHA256,
        'constant_learning_rate': 1e-4, 'optimizer_betas': 'restore_checkpoint',
        'optimizer_state_restore_required': True, 'fresh_official_initialization': False,
        'resumed_trained_weights_loaded': True, 'original_20000_prefix_verified': True,
    }
    return payload


def completed_a120_payload_fixture():
    payload = chained_payload_fixture()
    folder = ROOT / 'model_training/experiments'
    for target, checkpoint, source_ref in ((80000, EXTENDED_CHECKPOINT, EXTENDED_SOURCE_CONFIG),
                                            (120000, A120_CHECKPOINT, A120_SOURCE_CONFIG)):
        source = json.loads((folder / f'fusion_normal_a_extend{target}_v1.json').read_text())
        contract = continuation_contract(source)
        for rank in (0, 1):
            stream = extend_checkpoint_stream(payload, rank, target_assets=target)
            stream.cursor = stream.local_total
            payload['rank_states'][rank]['stream'] = stream.state_dict()
        updates = len(stream.update_sizes)
        frames = stream.plan_report()['supervised_frame_exposures']
        initial = payload['initialization']
        initial.setdefault('training_continuation_history', []).append(deepcopy(initial['training_continuation']))
        initial['training_continuation'].update(
            checkpoint=deepcopy(contract['checkpoint']), source_config=deepcopy(contract['source_config']),
            source=checkpoint['source'], config_sha256=source_ref['sha256'],
            start_asset_exposures=contract['start_assets'], target_asset_exposures=target,
            start_updates=contract['start_updates'], target_updates=updates,
            start_supervised_frame_exposures=contract['start_frames'], target_supervised_frame_exposures=frames,
            original_exposure_plan_sha256=contract['old_plan_sha256'],
            extended_exposure_plan_sha256=stream.plan_sha256,
            saved_prefix_asset_exposures_verified=contract['start_assets'])
        payload.update(source=checkpoint['source'], config=source, asset_exposures=target,
                       step=updates, supervised_frame_exposures=frames)
        for value in payload['optimizer']['state'].values():
            value['step'].fill_(updates)
    return payload


class ContinuationTests(unittest.TestCase):
    def test_a130k_uses_final_a120_and_only_lowers_continuation_rate(self):
        original, _ = configs()
        folder = ROOT / 'model_training/experiments'
        source = json.loads((folder / 'fusion_normal_a_extend120000_v1.json').read_text())
        config = json.loads((folder / 'fusion_normal_a_lr3e5_130000_v1.json').read_text())
        self.assertEqual(validate_continuation_config(config, original, 2), [125000, 130000])
        self.assertEqual(continuation_contract(config)['learning_rate'], 3e-5)
        self.assertEqual(config['continuation']['checkpoint'], A120_CHECKPOINT)
        payload = completed_a120_payload_fixture()
        validate_checkpoint_payload(payload, original, 2, config=config, source_config=source)
        for rank in (0, 1):
            previous = payload['rank_states'][rank]['stream']
            self.assertEqual(previous['plan_sha256'], A120_PLAN_SHA256)
            old = MatchedFrameStream(range(5000), **{
                key: value for key, value in previous['contract'].items() if key != 'dataset_rows'})
            stream = extend_checkpoint_stream(payload, rank, target_assets=130000)
            self.assertEqual(stream.cursor, 60000)
            self.assertEqual(len(stream.update_sizes), 2730)
            self.assertEqual(stream.plan_report()['supervised_frame_exposures'], 1685592)
            self.assertEqual(stream.plan_sha256, '29b3527f644f76d996ccfb3e7286d6eac43b29895d4c777ebb48d89a98a4dc41')
            np.testing.assert_array_equal(stream.order[:120000], old.order)
            np.testing.assert_array_equal(stream.frame_counts[:2520], old.frame_counts)
            np.testing.assert_array_equal(stream.update_sizes[:2520], old.update_sizes)
            self.assertEqual(stream.global_ends[2520], 120048)
            self.assertEqual(stream.frame_counts[2520:2528].tolist(), [20, 21, 4, 23, 10, 24, 6, 12])
        for section, key, value in [('continuation', 'learning_rate', 1e-4),
                                    ('continuation', 'checkpoint', EXTENDED_CHECKPOINT),
                                    ('optimization', 'schedule_samples', 130000),
                                    ('runtime', 'allocation', 129839822),
                                    ('evaluation', 'initial', True)]:
            changed = deepcopy(config)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_continuation_config(changed, original, 2)
        for mutate in (
            lambda data: data['initialization']['training_continuation_history'].pop(),
            lambda data: data['initialization']['training_continuation_history'][1].__setitem__('checkpoint', CHECKPOINT),
            lambda data: data['initialization']['training_continuation_history'][0].__setitem__('source', 'wrong'),
            lambda data: data['initialization']['training_continuation'].__setitem__('constant_learning_rate', 3e-5),
            lambda data: data['optimizer']['param_groups'][0].__setitem__('lr', 3e-5),
            lambda data: data['optimizer']['state'][0]['step'].fill_(2521),
            lambda data: data['rank_states'][1]['stream'].__setitem__('cursor', 40000),
        ):
            changed = deepcopy(payload)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_checkpoint_payload(changed, original, 2, config=config, source_config=source)

    def test_low_rate_restore_preserves_full_optimizer_and_rank_rng(self):
        payload = completed_a120_payload_fixture()
        for rank in (0, 1):
            stream = extend_checkpoint_stream(payload, rank, target_assets=130000)
            model = nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, betas=(.5, .8))
            state = dict(asset_exposures=120000, supervised_frame_exposures=1555504, step=2520,
                         optimizer_state_dict=payload['optimizer'], rank_rng_state=payload['rank_states'][rank],
                         learning_rate=3e-5)
            counters = restore_training_state(optimizer, stream, state, device='cpu', rank=rank,
                                              world=2, target_assets=130000)
            self.assertEqual(counters, (120000, 1555504, 2520))
            self.assertTrue(torch.equal(torch.get_rng_state(), payload['rank_states'][rank]['torch_rng']))
            self.assertEqual(random.getstate(), payload['rank_states'][rank]['python_rng'])
            np.testing.assert_array_equal(np.random.get_state()[1], payload['rank_states'][rank]['numpy_rng'][1])
            self.assertEqual(set_continuation_lr(optimizer, 3e-5), (3e-5, .9499999512612207))
            self.assertEqual(optimizer.param_groups[0]['lr'], 3e-5)
            self.assertEqual(optimizer.param_groups[0]['betas'], (.9499999512612207, .999))
            restored = optimizer.state_dict()['state']
            for key, value in payload['optimizer']['state'].items():
                for name in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertTrue(torch.equal(restored[key][name], value[name]))

    def test_b20k_to40k_preserves_b_route_and_exact_shared_stream(self):
        folder = ROOT / 'model_training/experiments'
        original = json.loads((folder / 'fusion_normal_b_v1.json').read_text())
        extended = json.loads((folder / 'fusion_normal_b_extend40000_v1.json').read_text())
        self.assertEqual(validate_continuation_config(extended, original, 2), [25000, 30000, 35000, 40000])
        payload = payload_fixture()
        payload.update(source=B_CHECKPOINT['source'], config=original)
        payload['initialization'].update(route='b', source=B_CHECKPOINT['source'], config_sha256=B_SOURCE_CONFIG['sha256'])
        validate_checkpoint_payload(payload, original, 2, config=extended, source_config=original)
        for rank in (0, 1):
            stream = extend_checkpoint_stream(payload, rank, target_assets=40000, route='b')
            self.assertEqual(stream.cursor, 10000)
            self.assertEqual(stream.plan_sha256, CHAINED_PLAN_SHA256)
            self.assertEqual(len(stream.update_sizes), 840)
            self.assertEqual(stream.plan_report()['supervised_frame_exposures'], 528296)
            self.assertEqual(stream.take()[0].sampling_seed, (original['data']['stream_seed'] << 32) + 20000 + rank * 24)
        for section, key, value in [('runtime', 'allocation', 129839822),
                                    ('architecture', 'fusion_mode', 'virtual_queries_real'),
                                    ('continuation', 'checkpoint', CHECKPOINT),
                                    ('continuation', 'learning_rate', 3e-5),
                                    ('training', 'target_assets', 80000)]:
            changed = deepcopy(extended)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_continuation_config(changed, original, 2)
        for mutate in (
            lambda data: data['initialization'].__setitem__('route', 'a'),
            lambda data: data['initialization'].__setitem__('source', CHECKPOINT['source']),
            lambda data: data.__setitem__('source', CHECKPOINT['source']),
            lambda data: data['rank_states'][1]['stream'].__setitem__('cursor', 0),
        ):
            changed = deepcopy(payload)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_checkpoint_payload(changed, original, 2, config=extended, source_config=original)
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(payload_fixture(), original, 2, config=extended, source_config=original)

    def test_a120k_preserves_a80k_prefix_history_optimizer_and_replacement_grant(self):
        original, _ = configs()
        folder = ROOT / 'model_training/experiments'
        source = json.loads((folder / 'fusion_normal_a_extend80000_v1.json').read_text())
        extended = json.loads((folder / 'fusion_normal_a_extend120000_v1.json').read_text())
        self.assertEqual(validate_continuation_config(extended, original, 2), list(range(85000, 120001, 5000)))
        payload = chained_payload_fixture()
        payload.update(source=EXTENDED_CHECKPOINT['source'], config=source,
                       asset_exposures=80000, step=1680, supervised_frame_exposures=1041832)
        for value in payload['optimizer']['state'].values():
            value['step'].fill_(1680)
        initial = payload['initialization']
        initial['training_continuation_history'] = [deepcopy(initial['training_continuation'])]
        initial['training_continuation'].update(
            checkpoint=deepcopy(CHAINED_CHECKPOINT), source_config=deepcopy(CHAINED_SOURCE_CONFIG),
            source=EXTENDED_CHECKPOINT['source'], config_sha256=EXTENDED_SOURCE_CONFIG['sha256'],
            start_asset_exposures=40000, target_asset_exposures=80000, start_updates=840, target_updates=1680,
            start_supervised_frame_exposures=528296, target_supervised_frame_exposures=1041832,
            original_exposure_plan_sha256=CHAINED_PLAN_SHA256,
            extended_exposure_plan_sha256=EXTENDED_PLAN_SHA256, saved_prefix_asset_exposures_verified=40000)
        for rank in (0, 1):
            old = extend_checkpoint_stream(payload, rank, target_assets=80000)
            old.cursor = old.local_total
            payload['rank_states'][rank]['stream'] = old.state_dict()
            stream = extend_checkpoint_stream(payload, rank, target_assets=120000)
            self.assertEqual(stream.cursor, 40000)
            self.assertEqual(len(stream.update_sizes), 2520)
            self.assertEqual(stream.plan_report()['supervised_frame_exposures'], 1555504)
            self.assertEqual(stream.plan_sha256, '26ff999633011ec96e21b0015e671e18c654a92c9e0b5a7bfe1cc7cc18df4a70')
            self.assertEqual(stream.global_ends[1680], 80048)
            np.testing.assert_array_equal(stream.order[:80000], old.order)
            np.testing.assert_array_equal(stream.frame_counts[:1680], old.frame_counts)
            np.testing.assert_array_equal(stream.update_sizes[:1680], old.update_sizes)
            self.assertEqual(stream.frame_counts[1680:1688].tolist(), [23, 11, 4, 14, 4, 14, 12, 23])
        validate_checkpoint_payload(payload, original, 2, config=extended, source_config=source)
        for section, key, value in [('runtime', 'allocation', 129839822), ('runtime', 'host', 'pcg01i'),
                                    ('runtime', 'physical_gpus', [0, 1]), ('optimization', 'peak_lr', 2e-4),
                                    ('continuation', 'checkpoint', CHAINED_CHECKPOINT)]:
            changed = deepcopy(extended)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_continuation_config(changed, original, 2)
        for mutate in (
            lambda data: data['initialization'].pop('training_continuation_history'),
            lambda data: data['initialization']['training_continuation_history'][0].__setitem__('source', 'wrong'),
            lambda data: data['initialization']['training_continuation'].__setitem__('checkpoint', CHECKPOINT),
            lambda data: data['optimizer']['state'][0]['step'].fill_(1681),
            lambda data: data['optimizer']['param_groups'][0].__setitem__('lr', 2e-6),
            lambda data: data['rank_states'][1]['stream'].__setitem__('cursor', 20000),
            lambda data: data['rank_states'][0].pop('cuda_rng'),
        ):
            changed = deepcopy(payload)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_checkpoint_payload(changed, original, 2, config=extended, source_config=source)

    def test_chained_config_preserves_original_policy_and_adds_only_new_milestones(self):
        original, source = configs()
        extended = json.loads((ROOT / 'model_training/experiments/fusion_normal_a_extend80000_v1.json').read_text())
        self.assertEqual(validate_continuation_config(extended, original, 2), list(range(45000, 80001, 5000)))
        contract = continuation_contract(extended)
        self.assertEqual(contract['checkpoint'], CHAINED_CHECKPOINT)
        self.assertEqual(contract['source_config'], CHAINED_SOURCE_CONFIG)
        for section, key, value in [('continuation', 'checkpoint', CHECKPOINT),
                                    ('training', 'target_assets', 80016),
                                    ('training', 'evaluation_assets', [40000, 80000]),
                                    ('optimization', 'schedule_samples', 80000),
                                    ('evaluation', 'initial', True)]:
            changed = deepcopy(extended)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_continuation_config(changed, original, 2)

    def test_chained_payload_extends_both_completed40k_streams_without_replay(self):
        original, source = configs()
        extended = json.loads((ROOT / 'model_training/experiments/fusion_normal_a_extend80000_v1.json').read_text())
        payload = chained_payload_fixture()
        validate_checkpoint_payload(payload, original, 2, config=extended, source_config=source)
        for rank in (0, 1):
            old = MatchedFrameStream(range(5000), **{
                key: value for key, value in payload['rank_states'][rank]['stream']['contract'].items()
                if key != 'dataset_rows'})
            stream = extend_checkpoint_stream(payload, rank, target_assets=80000)
            self.assertEqual(stream.cursor, 20000)
            self.assertEqual(len(stream.update_sizes), 1680)
            self.assertEqual(stream.plan_report()['supervised_frame_exposures'], 1041832)
            self.assertEqual(stream.plan_sha256, 'f561a7d8432792920292cc1cf06f74858872b06173187d9ac3f132efdf72b7f2')
            self.assertEqual(stream.global_ends[839], 40000)
            self.assertEqual(stream.global_ends[840], 40048)
            np.testing.assert_array_equal(stream.order[:40000], old.order)
            np.testing.assert_array_equal(stream.frame_counts[:840], old.frame_counts)
            np.testing.assert_array_equal(stream.update_sizes[:840], old.update_sizes)
            requests = stream.take()
            self.assertEqual(requests[0].sampling_seed, (original['data']['stream_seed'] << 32) + 40000 + rank * 24)
        for mutate in (
            lambda data: data.__setitem__('step', 420),
            lambda data: data.__setitem__('config', original),
            lambda data: data['initialization'].pop('training_continuation'),
            lambda data: data['initialization']['training_continuation'].__setitem__('checkpoint', CHAINED_CHECKPOINT),
            lambda data: data['rank_states'][0]['stream'].__setitem__('cursor', 10000),
            lambda data: data['optimizer']['param_groups'][0].__setitem__('lr', 2e-6),
            lambda data: data['optimizer']['param_groups'][0].__setitem__('betas', (.9, .999)),
        ):
            changed = deepcopy(payload)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_checkpoint_payload(changed, original, 2, config=extended, source_config=source)
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(payload_fixture(), original, 2, config=extended, source_config=source)

    def test_config_only_changes_explicit_budget_evaluation_and_resume_policy(self):
        original, extended = configs()
        normal_namespace()['validate_config'](original, 'train', 2)
        self.assertEqual(validate_continuation_config(extended, original, 2), [25000, 30000, 35000, 40000])
        for section, key, value in [('architecture', 'fusion_depth', 2), ('data', 'stream_seed', 1),
                                    ('optimization', 'schedule_samples', 40000),
                                    ('continuation', 'learning_rate', 2e-6),
                                    ('evaluation', 'initial', True), ('training', 'target_assets', 60000)]:
            changed = deepcopy(extended)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_continuation_config(changed, original, 2)
        with self.assertRaises(ValueError):
            validate_continuation_config(extended, original, 1)

    def test_checkpoint_is_full_original_a20k_and_both_rank_streams_extend(self):
        original, _ = configs()
        payload = payload_fixture()
        validate_checkpoint_payload(payload, original, 2)
        for rank in (0, 1):
            stream = extend_checkpoint_stream(payload, rank)
            self.assertEqual(stream.cursor, 10000)
            self.assertEqual(len(stream.update_sizes), 840)
            self.assertEqual(stream.plan_report()['supervised_frame_exposures'], 528296)
            self.assertEqual(stream.plan_sha256, '3321e85cb3608f51620b45ea9c54359b3b0043074ef9f4d645643d1e10a6ba09')
        for key, value in [('source', 'wrong'), ('asset_exposures', 10000), ('step', 0),
                           ('supervised_frame_exposures', 1), ('rank_states', payload['rank_states'][:1]),
                           ('optimizer', {'param_groups': [], 'state': {}})]:
            changed = deepcopy(payload)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint_payload(changed, original, 2)
        changed = deepcopy(payload)
        changed['rank_states'][1]['stream']['cursor'] = 0
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(changed, original, 2)
        changed = deepcopy(payload)
        for state in changed['optimizer']['state'].values():
            state['step'].fill_(1)
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(changed, original, 2)

    def test_constant_rate_preserves_betas_and_moments(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, betas=(.9499999512612207, .999))
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        old = deepcopy(optimizer.state_dict())
        self.assertEqual(set_continuation_lr(optimizer, 1e-4), (1e-4, .9499999512612207))
        for before, after in zip(old['state'].values(), optimizer.state_dict()['state'].values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(before[key], after[key]))
        self.assertEqual(optimizer.param_groups[0]['betas'], old['param_groups'][0]['betas'])
        self.assertEqual(set_continuation_lr(optimizer, 3e-5), (3e-5, .9499999512612207))
        for before, after in zip(old['state'].values(), optimizer.state_dict()['state'].values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(before[key], after[key]))
        for bad in (0, 2e-6, float('nan')):
            with self.assertRaises(ValueError):
                set_continuation_lr(optimizer, bad)

    def test_real_training_loop_resume_matches_uninterrupted_optimizer_and_rng(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))

            def forward(self, surface, target, auxiliary_weight):
                loss = (self.weight * surface).square()
                return {'loss_per_asset': loss, 'loss': loss.mean(), 'main_loss': loss.mean(),
                        'auxiliary_loss': loss.new_tensor(0.), 'main_accuracy': loss.new_tensor(1.)}

        def make_stream(total):
            return MatchedFrameStream(range(5), rank=0, world_size=1, seed=7, target_assets=total,
                effective_assets=8, update_boundaries=list(range(10, total + 1, 10)),
                frame_policy={str(t): 3 for t in range(2, 25)}, manifest_sha256='0' * 64)

        def run(folder, model, stream, resume=None):
            n, snapshots, gradient_checks = namespace(), {}, []

            def prepare(dataset, requests, *args):
                values = torch.tensor([r.item_index + 1. for r in requests])
                values += .01 * (torch.rand(len(requests)) + float(np.random.rand()) + random.random())
                return values, None, requests

            def schedule(optimizer, samples, config):
                lr = .001 / (1 + samples) if samples < 20 else 1e-4
                for group in optimizer.param_groups:
                    group['lr'] = lr
                return lr, optimizer.param_groups[0]['betas'][0]

            def save(model, optimizer, stream, config, args, samples, frames, step, *unused):
                snapshots[samples] = deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'stream': stream.state_dict(), 'asset_exposures': samples,
                    'supervised_frame_exposures': frames, 'step': step, **rng_state()})

            n.update(fixed_dataset=lambda *a, **k: [],
                optimizer_for=lambda model, config: torch.optim.AdamW(model.parameters(),
                    lr=.01, betas=(.5, .8) if resume else (.91, .999), weight_decay=0),
                wrap_model=lambda model, *a: model, prepare=prepare, set_exposure_lr=schedule,
                grad_norms=lambda model: {'weight': float(model.weight.grad.abs())}, save_checkpoint=save)
            plan = {'training': {'target_assets': stream.target_assets, 'effective_assets_per_update': 8,
                                 'evaluation_assets': list(range(10, stream.target_assets + 1, 10))},
                    'optimization': {'auxiliary_weight': 0., 'clip_grad_norm': 1e9}}
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(torch, 'autocast', lambda *a, **k: nullcontext()))
                for name in ('reset_peak_memory_stats', 'synchronize', 'max_memory_allocated', 'max_memory_reserved'):
                    stack.enter_context(patch.object(torch.cuda, name, return_value=0))
                n['run_training'](model, None, None, {}, {'train': []}, plan,
                    SimpleNamespace(output=folder), 'cpu', 0, 1, 'cpu_test', stream=stream,
                    milestones=list(range(10, stream.target_assets + 1, 10)), evaluation_callback=lambda *a: None,
                    gradient_check=lambda *a: gradient_checks.append(True), resume_state=resume)
            rows = [json.loads(line) for line in (folder / 'trace_rank_0.jsonl').read_text().splitlines()]
            return snapshots, rows, gradient_checks, json.loads((folder / 'rank_0.json').read_text())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('whole', 'first', 'resumed', 'resumed_again'):
                (root / name).mkdir()
            torch.manual_seed(9)
            np.random.seed(9)
            random.seed(9)
            whole_model = Model()
            whole, _, _, _ = run(root / 'whole', whole_model, make_stream(80))
            torch.manual_seed(9)
            np.random.seed(9)
            random.seed(9)
            first, _, _, _ = run(root / 'first', Model(), make_stream(20))
            checkpoint_path = root / 'checkpoint.pt'
            torch.save(first[20], checkpoint_path)
            saved = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
            extended = MatchedFrameStream.extend_completed(range(5), saved['stream'],
                target_assets=40, update_boundaries=[10, 20, 30, 40])
            restored_model = Model()
            restored_model.load_state_dict(saved['model'], strict=True)
            torch.rand(11)
            np.random.rand(7)
            random.random()
            resume = {key: saved[key] for key in ('asset_exposures', 'supervised_frame_exposures', 'step')}
            resume.update(optimizer_state_dict=saved['optimizer'], rank_rng_state=saved,
                          learning_rate=1e-4, provenance={'kind': 'cpu_test_continuation'})
            tail, rows, gradient_checks, report = run(root / 'resumed', restored_model, extended, resume)
            torch.save(tail[40], root / 'checkpoint40.pt')
            saved_again = torch.load(root / 'checkpoint40.pt', map_location='cpu', mmap=True, weights_only=False)
            second_stream = MatchedFrameStream.extend_completed(range(5), saved_again['stream'],
                target_assets=80, update_boundaries=list(range(10, 81, 10)))
            second_model = Model()
            second_model.load_state_dict(saved_again['model'], strict=True)
            torch.rand(13)
            np.random.rand(9)
            random.random()
            second_resume = {key: saved_again[key] for key in ('asset_exposures', 'supervised_frame_exposures', 'step')}
            second_resume.update(optimizer_state_dict=saved_again['optimizer'], rank_rng_state=saved_again,
                                 learning_rate=1e-4, provenance={'kind': 'cpu_test_chained_continuation'})
            second_tail, second_rows, second_checks, second_report = run(
                root / 'resumed_again', second_model, second_stream, second_resume)
        self.assertTrue(torch.equal(whole[40]['model']['weight'], tail[40]['model']['weight']))
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertTrue(torch.equal(whole[40]['optimizer']['state'][0][key], tail[40]['optimizer']['state'][0][key]))
        self.assertTrue(torch.equal(whole[40]['torch_rng'], tail[40]['torch_rng']))
        np.testing.assert_array_equal(whole[40]['numpy_rng'][1], tail[40]['numpy_rng'][1])
        self.assertEqual(whole[40]['python_rng'], tail[40]['python_rng'])
        self.assertEqual([r['step'] for r in rows], [5, 6, 7, 8])
        self.assertEqual([r['asset_exposures'] for r in rows], [28, 30, 38, 40])
        self.assertTrue(all(r['lr'] == 1e-4 and r['beta1'] == .91 for r in rows))
        self.assertEqual(gradient_checks, [True])
        self.assertFalse(report['fresh_official_initialization'])
        self.assertTrue(report['optimizer_state_restored'])
        self.assertEqual(report['restored_optimizer']['param_groups'][0]['betas'], [.91, .999])
        self.assertTrue(torch.equal(whole[80]['model']['weight'], second_tail[80]['model']['weight']))
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertTrue(torch.equal(whole[80]['optimizer']['state'][0][key],
                                        second_tail[80]['optimizer']['state'][0][key]))
        self.assertTrue(torch.equal(whole[80]['torch_rng'], second_tail[80]['torch_rng']))
        np.testing.assert_array_equal(whole[80]['numpy_rng'][1], second_tail[80]['numpy_rng'][1])
        self.assertEqual(whole[80]['python_rng'], second_tail[80]['python_rng'])
        self.assertEqual([row['step'] for row in second_rows], list(range(9, 17)))
        self.assertTrue(all(row['lr'] == 1e-4 and row['beta1'] == .91 for row in second_rows))
        self.assertEqual(second_checks, [True])
        self.assertEqual(second_report['restored_optimizer']['param_groups'][0]['lr'], 1e-4)


if __name__ == '__main__':
    unittest.main()
