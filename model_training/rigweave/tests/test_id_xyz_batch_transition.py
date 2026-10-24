"""CPU-only provenance and replay checks for physical microbatch changes."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_id_xyz_resume as fixtures

runtime, resume, stages = fixtures.runtime, fixtures.resume, fixtures.stages


def write_spec(path, value, *, reviewed=False):
    path.write_text(json.dumps(value), encoding='utf-8')
    result = {'path': str(path), 'sha256': runtime.sha256(path)}
    if reviewed:
        result.update({key: value[key] for key in ('code_commit', 'config_sha256')})
    return result


def consume(stream, assets=24):
    chunks, requests = [], []
    while len(requests) < assets:
        chunk = stream.take(max_assets=assets - len(requests))
        chunks.append(len(chunk))
        requests.extend(chunk)
    return chunks, requests


class IdXYZBatchTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.no_cuda = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU test may not initialize CUDA'))
        cls.no_cuda.start()
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.parent = fixtures.recipe(20000)
        cls.source = fixtures.recipe(100000)
        cls.source['continuation']['parent_configuration'] = write_spec(root / 'parent.json', cls.parent)
        cls.current_sources = runtime.source_hashes()
        cls.source_hashes = dict(cls.current_sources)
        for name in resume.CONTINUATION_SOURCE_CHANGES:
            cls.source_hashes[name] = 'a' * 64
        cls.audit = {'passed': True, 'mode': 'resume_audit',
            'code_commit': '5092b1676e03f8a2dd51aaf82f23c4f6f162d72c',
            'config_sha256': '0f6339c2cebbc3dd0d243b47fad339a8032025617218cc19106d9c6bd794bf95',
            'source_sha256': cls.source_hashes,
            'paired_contract_sha256': runtime.paired_contract_hash(cls.source)}
        cls.streams = [runtime.make_stream(cls.source, rank, 2) for rank in (0, 1)]
        for stream in cls.streams:
            stream.cursor = 10000
        cls.frames = int(np.dot(cls.streams[0].update_sizes[:420].astype(np.int64), cls.streams[0].frame_counts[:420]))
        cls.model = fixtures.TinyGroups()
        cls.optimizer = runtime.optimizer_for(cls.model, cls.source)
        for samples in [0, *cls.streams[0].global_ends[:419].tolist()]:
            fixtures.update(cls.model, cls.optimizer, samples, cls.source)
        cls.saved = deepcopy({'model': cls.model.state_dict(), 'optimizer': cls.optimizer.state_dict(),
            'optimizer_parameter_names': resume.optimizer_names(cls.model, cls.optimizer),
            'training_contract': runtime.training_contract(cls.source), 'configuration': cls.source,
            'code_commit': '382e14c4e110d952cc99cadee029f934d8a93273', 'source_sha256': cls.source_hashes,
            'asset_exposures': 20000, 'step': 420, 'input_frame_exposures': cls.frames,
            'supervised_frame_exposures': cls.frames, 'initialization': {'initial_model_sha256': 'tiny-initial'},
            'rank_states': [{'rank': rank, 'stream': stream.state_dict(),
                'torch_rng': torch.Generator(device='cpu').manual_seed(701 + rank).get_state(),
                'cuda_rng': torch.zeros(16, dtype=torch.uint8),
                'numpy_rng': np.random.RandomState(801 + rank).get_state(),
                'python_rng': random.Random(901 + rank).getstate()} for rank, stream in enumerate(cls.streams)]})

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        cls.no_cuda.stop()
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = deepcopy(self.source)
        self.config['runtime']['repo'] = '/unused/batch_code'
        self.config['training']['frame_policy'] = {key: value * 2 for key, value in self.source['training']['frame_policy'].items()}
        self.config['training']['resume_checkpoint'] = {
            'path': str(self.root / 'tiny20k.pt'), 'sha256': '0' * 64, 'bytes': 1,
            'producer_head': self.saved['code_commit'], 'asset_exposures': 20000, 'step': 420}
        self.config['batch_transition'] = {'kind': 'physical_batch_only_v1',
            'source_configuration': write_spec(self.root / 'source.json', self.source),
            'source_audit': write_spec(self.root / 'audit.json', self.audit, reviewed=True)}

    def replace_source(self, value, config=None):
        config = self.config if config is None else config
        config['batch_transition']['source_configuration'] = write_spec(self.root / 'changed-source.json', value)

    def replace_audit(self, value, config=None):
        config = self.config if config is None else config
        config['batch_transition']['source_audit'] = write_spec(self.root / 'changed-audit.json', value, reviewed=True)

    def restore(self, saved=None, config=None, rank=0):
        config = self.config if config is None else config
        model = fixtures.TinyGroups()
        optimizer = runtime.optimizer_for(model, config)
        with patch.object(torch, 'load', side_effect=AssertionError('only the tiny in-memory fixture is allowed')), \
                patch.object(runtime, 'source_hashes', return_value=self.current_sources):
            result = resume.restore_checkpoint(model, optimizer, config, rank=rank, world=2,
                device=torch.device('cpu'), saved=deepcopy(self.saved if saved is None else saved))
        return model, optimizer, result

    def restore_stream(self, state, config=None, *, step=420, samples=20000, frames=None, rank=0):
        return resume.restore_stream(self.config if config is None else config, state, samples=samples,
            frames=self.frames if frames is None else frames, step=step, rank=rank, world=2)

    def temporal_v2(self):
        config, audit, saved = deepcopy(self.config), deepcopy(self.audit), deepcopy(self.saved)
        execution = {'kind': 'independent_temporal_sequence_chunks_v1', 'max_sequences': 13452,
                     'source_before_sha256': 'b' * 64,
                     'source_after_sha256': self.current_sources[resume.TEMPORAL_SOURCE]}
        config['batch_transition'].update(kind='physical_batch_temporal_chunks_v2', temporal_execution=execution)
        audit['source_sha256'][resume.TEMPORAL_SOURCE] = execution['source_before_sha256']
        saved['source_sha256'] = deepcopy(audit['source_sha256'])
        self.replace_audit(audit, config)
        return config, audit, saved

    def prior_receipt(self, hashes):
        return {'mode': 'preflight', 'paired_contract_sha256': runtime.paired_contract_hash(self.parent),
                'source_sha256': {key: value for key, value in hashes.items()
                                  if key != 'scripts/id_xyz_resume.py'}}

    def test_draft_pins_real_source_audit_cpu1_global48_and_target_policy(self):
        path = fixtures.ROOT.parent / 'experiments/id_xyz_consistency_100000_batch_v1.json'
        draft = json.loads(path.read_text())
        transition = draft['batch_transition']
        self.assertEqual(transition['kind'], 'physical_batch_temporal_chunks_v2')
        self.assertEqual(resume.temporal_execution(draft), {
            'kind': 'independent_temporal_sequence_chunks_v1', 'max_sequences': 13452,
            'source_before_sha256': '999b3be8672e9b12a69aed32b3174ec7c24bfa2415c56235b7010d933dc640e6',
            'source_after_sha256': self.current_sources[resume.TEMPORAL_SOURCE]})
        self.assertEqual(transition['source_configuration']['sha256'],
                         '27d62327284dc9f7aac65a41987d132ea2980ecab045a032e61cfbaa034b196c')
        self.assertEqual(transition['source_audit']['sha256'],
                         '0607d7e31a3be76ab009dfad12646b819817252510354c3c2b4550c4807ac74c')
        self.assertEqual(transition['source_audit']['code_commit'], self.audit['code_commit'])
        self.assertEqual(transition['source_audit']['config_sha256'], self.audit['config_sha256'])
        self.assertEqual(draft['runtime']['cpu_slots'], 1)
        self.assertEqual(draft['training']['effective_assets_per_update'], 48)
        self.assertEqual(draft['training']['frame_policy'], self.config['training']['frame_policy'])
        self.assertEqual(draft['continuation'], fixtures.recipe(100000)['continuation'])

    def test_temporal_v1_cannot_authorize_a_core_source_change(self):
        self.assertIsNone(resume.temporal_execution(self.config))
        self.assertIsNone(resume.temporal_execution(self.source))
        config, audit, _ = self.temporal_v2()
        config['batch_transition']['kind'] = 'physical_batch_only_v1'
        config['batch_transition'].pop('temporal_execution')
        self.assertEqual(resume.approved_source_changes(config, audit['source_sha256'], self.current_sources),
                         resume.CONTINUATION_SOURCE_CHANGES)
        with self.assertRaises(ValueError):
            resume.batch_source_configuration(config)
        with self.assertRaises(ValueError):
            resume.validate_prior_receipt(config, self.prior_receipt(audit['source_sha256']))

    def test_temporal_v2_accepts_only_the_exact_extra_source_and_prior_receipts(self):
        config, audit, _ = self.temporal_v2()
        self.assertEqual(resume.temporal_execution(config), config['batch_transition']['temporal_execution'])
        self.assertEqual(resume.approved_source_changes(config, audit['source_sha256'], self.current_sources),
                         resume.CONTINUATION_SOURCE_CHANGES | {resume.TEMPORAL_SOURCE})
        self.assertEqual(resume.batch_source_configuration(config), self.source)
        self.assertEqual(resume.parent_configuration(config), self.parent)
        runtime.validate_config(config)
        resume.validate_prior_receipt(config, self.prior_receipt(audit['source_sha256']))

    def test_temporal_v2_rejects_invalid_schema_types_cap_and_digest_format(self):
        config, _, _ = self.temporal_v2()
        transition = config['batch_transition']
        spec = transition['temporal_execution']
        variants = [None, {}, [], dict(spec, unknown=True),
                    {key: value for key, value in spec.items() if key != 'kind'}]
        for value in (-1, 0, 1, 13451, 13453, True, False, 13452., '13452', None):
            variants.append(dict(spec, max_sequences=value))
        variants.append(dict(spec, kind='frame_chunks'))
        for key in ('source_before_sha256', 'source_after_sha256'):
            for value in ('A' * 64, 'g' * 64, 'a' * 63, 'a' * 65, '', None, 123):
                variants.append(dict(spec, **{key: value}))
        variants.append(dict(spec, source_before_sha256=spec['source_after_sha256']))
        for index, value in enumerate(variants):
            changed = deepcopy(config)
            changed['batch_transition']['temporal_execution'] = value
            with self.subTest(variant=index), self.assertRaises(ValueError):
                resume.temporal_execution(changed)
            with self.subTest(batch_validation=index), self.assertRaises(ValueError):
                resume.batch_source_configuration(changed)
        for value in (dict(transition, unknown=True), dict(transition, kind='physical_batch_only_v1'),
                      {key: item for key, item in transition.items() if key != 'temporal_execution'}):
            changed = deepcopy(config)
            changed['batch_transition'] = value
            with self.subTest(transition=value), self.assertRaises(ValueError):
                resume.temporal_execution(changed)

    def test_temporal_v2_rejects_mismatched_or_missing_before_and_after_source_hashes(self):
        config, audit, _ = self.temporal_v2()
        for side in ('previous', 'current'):
            for missing in (False, True):
                previous, current = deepcopy(audit['source_sha256']), deepcopy(self.current_sources)
                selected = previous if side == 'previous' else current
                if missing:
                    selected.pop(resume.TEMPORAL_SOURCE)
                else:
                    selected[resume.TEMPORAL_SOURCE] = 'c' * 64
                with self.subTest(side=side, missing=missing), self.assertRaises(ValueError):
                    resume.approved_source_changes(config, previous, current)
        for field in ('source_before_sha256', 'source_after_sha256'):
            changed = deepcopy(config)
            changed['batch_transition']['temporal_execution'][field] = 'c' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.batch_source_configuration(changed)

    def test_temporal_v2_does_not_permit_any_other_core_data_or_sampling_source_change(self):
        config, audit, _ = self.temporal_v2()
        allowed = resume.CONTINUATION_SOURCE_CHANGES | {resume.TEMPORAL_SOURCE}
        protected = sorted(set(self.current_sources) - allowed)
        self.assertTrue(any('data' in key for key in protected))
        self.assertTrue(any('id_xyz_consistency' in key for key in protected))
        for name in protected:
            changed = dict(self.current_sources, **{name: 'd' * 64})
            with self.subTest(source=name), patch.object(runtime, 'source_hashes', return_value=changed):
                with self.assertRaises(ValueError):
                    resume.batch_source_configuration(config)
                with self.assertRaises(ValueError):
                    resume.validate_prior_receipt(config, self.prior_receipt(audit['source_sha256']))

    def test_temporal_v2_prior_receipt_independently_checks_exact_before_hash(self):
        config, audit, _ = self.temporal_v2()
        receipt = self.prior_receipt(audit['source_sha256'])
        receipt['source_sha256'][resume.TEMPORAL_SOURCE] = 'c' * 64
        self.assertEqual(resume.batch_source_configuration(config), self.source)
        with self.assertRaisesRegex(ValueError, 'before/after'):
            resume.validate_prior_receipt(config, receipt)

    def test_temporal_v2_training_contract_and_paired_digest_bind_execution(self):
        config, _, _ = self.temporal_v2()
        spec = config['batch_transition']['temporal_execution']
        contract = runtime.training_contract(config)
        self.assertEqual(contract['temporal_execution'], spec)
        self.assertNotIn('temporal_execution', runtime.training_contract(self.config))
        self.assertEqual({key: value for key, value in contract.items() if key != 'temporal_execution'},
                         runtime.training_contract(self.config))
        self.assertNotEqual(runtime.paired_contract_hash(config), runtime.paired_contract_hash(self.config))
        for key in ('source_before_sha256', 'source_after_sha256'):
            changed = deepcopy(config)
            changed['batch_transition']['temporal_execution'][key] = 'c' * 64
            self.assertNotEqual(runtime.training_contract(changed), contract)
            self.assertNotEqual(runtime.paired_contract_hash(changed), runtime.paired_contract_hash(config))

    def test_temporal_v2_requires_fresh_resume_audit_and_accepts_exact_current_receipt(self):
        config, _, _ = self.temporal_v2()
        receipt = {'mode': 'resume_audit', 'source_sha256': self.current_sources,
                   'paired_contract_sha256': runtime.paired_contract_hash(config)}
        with patch.object(stages, 'source_hashes', return_value=self.current_sources):
            stages.validate_receipt(config, receipt, mode='resume_audit')
            for old_contract in (self.source, self.config):
                stale = dict(receipt, paired_contract_sha256=runtime.paired_contract_hash(old_contract))
                with self.subTest(old_contract=old_contract.get('batch_transition')), self.assertRaises(ValueError):
                    stages.validate_receipt(config, stale, mode='resume_audit')
            stale = dict(receipt, source_sha256=dict(self.current_sources, **{resume.TEMPORAL_SOURCE: 'b' * 64}))
            with self.assertRaises(ValueError):
                stages.validate_receipt(config, stale, mode='resume_audit')

    def test_temporal_v2_tiny20k_full_state_and_next48_requests_restore_exactly(self):
        config, _, saved = self.temporal_v2()
        requests = []
        for rank in (0, 1):
            model, optimizer, (stream, rng, initial, report) = self.restore(saved, config, rank)
            self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(self.model.state_dict()))
            self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(self.model, self.optimizer))
            self.assertEqual(resume.rng_fingerprint(rng), resume.rng_fingerprint(saved['rank_states'][rank]))
            self.assertEqual(initial, saved['initialization'])
            self.assertEqual((report['asset_exposures'], report['step'], report['input_frame_exposures'],
                              report['supervised_frame_exposures']), (20000, 420, self.frames, self.frames))
            self.assertEqual(report['physical_batch_transition']['temporal_execution'],
                             config['batch_transition']['temporal_execution'])
            self.assertEqual((stream.cursor, stream.plan_sha256), (10000, self.streams[rank].plan_sha256))
            for name in ('order', 'frame_counts', 'update_sizes', 'global_ends'):
                np.testing.assert_array_equal(getattr(stream, name), getattr(self.streams[rank], name))
            _, actual = consume(stream)
            self.assertEqual(actual, consume(deepcopy(self.streams[rank]))[1])
            requests.extend(actual)
            baseline = deepcopy(self.model)
            baseline_optimizer = runtime.optimizer_for(baseline, self.source)
            baseline_optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
            fixtures.update(model, optimizer, 20000, config)
            fixtures.update(baseline, baseline_optimizer, 20000, self.source)
            self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(baseline.state_dict()))
            self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(baseline, baseline_optimizer))
        self.assertEqual([item.sampling_seed for item in requests],
                         list(range((20261013 << 32) + 20000, (20261013 << 32) + 20048)))

    def test_temporal_v2_future_checkpoint_resumes_and_rejects_execution_contract_tampering(self):
        config, _, saved = self.temporal_v2()
        model, optimizer, _ = self.restore(saved, config)
        fixtures.update(model, optimizer, 20000, config)
        future = deepcopy(saved)
        future.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
            training_contract=deepcopy(runtime.training_contract(config)), configuration=deepcopy(config),
            source_sha256=deepcopy(self.current_sources), code_commit='f' * 40, asset_exposures=20048, step=421)
        frames = self.frames + 48 * int(self.streams[0].frame_counts[420])
        future.update(input_frame_exposures=frames, supervised_frame_exposures=frames)
        for rank in (0, 1):
            stream = runtime.make_stream(config, rank, 2)
            stream.cursor = 10024
            future['rank_states'][rank]['stream'] = stream.state_dict()
        config['training']['resume_checkpoint'].update(producer_head='f' * 40, asset_exposures=20048, step=421)
        for rank in (0, 1):
            actual_model, actual_optimizer, (stream, rng, _, report) = self.restore(future, config, rank)
            self.assertEqual((report['step'], report['asset_exposures'], stream.cursor), (421, 20048, 10024))
            self.assertEqual(resume.state_fingerprint(actual_model.state_dict()), resume.state_fingerprint(model.state_dict()))
            self.assertEqual(resume.optimizer_fingerprint(actual_model, actual_optimizer), resume.optimizer_fingerprint(model, optimizer))
            self.assertEqual(resume.rng_fingerprint(rng), resume.rng_fingerprint(future['rank_states'][rank]))
        for change in ('removed', 'cap', 'before', 'after'):
            changed = deepcopy(future)
            contract = changed['training_contract']
            if change == 'removed':
                contract.pop('temporal_execution')
            elif change == 'cap':
                contract['temporal_execution']['max_sequences'] = 26904
            else:
                contract['temporal_execution']['source_' + change + '_sha256'] = 'c' * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.restore(changed, config)

    def test_all_seven_unchanged_or_doubled_tier_combinations_are_accepted(self):
        for increased in itertools.product((False, True), repeat=3):
            if not any(increased):
                continue
            config = deepcopy(self.config)
            for frame in range(2, 25):
                tier = 0 if frame <= 6 else 1 if frame <= 12 else 2
                config['training']['frame_policy'][str(frame)] = self.source['training']['frame_policy'][str(frame)] * (2 if increased[tier] else 1)
            with self.subTest(increased=increased):
                self.assertEqual(resume.batch_source_configuration(config), self.source)
                self.assertEqual(resume.parent_configuration(config), self.parent)
                runtime.validate_config(config)

    def test_policy_rejects_no_increase_missing_T_noninteger_or_inconsistent_tiers(self):
        policies = [deepcopy(self.source['training']['frame_policy'])]
        for frame, value in [('2', True), ('2', 0), ('2', 18), ('2', 36), ('7', 9), ('24', 12), ('5', 12)]:
            changed = deepcopy(self.config['training']['frame_policy'])
            changed[frame] = value
            policies.append(changed)
        missing = deepcopy(self.config['training']['frame_policy'])
        missing.pop('3')
        policies.extend([missing, dict(self.config['training']['frame_policy'], **{'25': 6})])
        for policy in policies:
            changed = deepcopy(self.config)
            changed['training']['frame_policy'] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                resume.batch_source_configuration(changed)

    def test_source_configuration_file_and_declared_digest_are_immutable(self):
        changed = deepcopy(self.config)
        changed['batch_transition']['source_configuration']['sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            resume.batch_source_configuration(changed)
        path = Path(self.config['batch_transition']['source_configuration']['path'])
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(ValueError):
            resume.batch_source_configuration(self.config)

    def test_source_must_be_original_100k_without_nested_batch_transition(self):
        for field, value in [('target', 20000), ('nested', {'kind': 'physical_batch_only_v1'})]:
            source = deepcopy(self.source)
            if field == 'target':
                source['training']['target_assets'] = value
            else:
                source['batch_transition'] = value
            self.replace_source(source)
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.batch_source_configuration(self.config)

    def test_model_loss_data_normalization_precision_cpu_lr_and_global_batch_cannot_change(self):
        changes = [('architecture', 'gradient_checkpointing', False), ('model_contract', 'all_real_frames_supervised', False),
            ('data', 'normalization', 'per_frame'), ('data', 'heldout_frames', 4), ('data', 'stream_seed', 17),
            ('surface_sampling', 'surface_samples', 8192), ('codec', 'max_sequence_tokens', 1024),
            ('consistency', 'weight', 0), ('optimization', 'peak_lr', 3e-5), ('runtime', 'cpu_slots', 8),
            ('runtime', 'precision', 'float16'), ('runtime', 'gpu_memory_fraction_limit', .9),
            ('training', 'effective_assets_per_update', 96), ('training', 'target_assets', 200000),
            ('evaluation', 'input_frames', 4), ('continuation', 'handoff_assets', 20000)]
        for section, field, value in changes:
            changed = deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                resume.batch_source_configuration(changed)

    def test_missing_bad_or_wrong_producer_source_audit_is_rejected(self):
        for replacement in (None, {}):
            changed = deepcopy(self.config)
            changed['batch_transition']['source_audit'] = replacement
            with self.subTest(replacement=replacement), self.assertRaises((ValueError, KeyError)):
                resume.batch_source_configuration(changed)
        for key in ('sha256', 'code_commit', 'config_sha256'):
            changed = deepcopy(self.config)
            changed['batch_transition']['source_audit'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.batch_source_configuration(changed)
        changed = deepcopy(self.config)
        changed['batch_transition'].pop('source_audit')
        with self.assertRaises(ValueError):
            resume.batch_source_configuration(changed)

    def test_audit_pass_mode_recipe_and_full_source_coverage_are_bound(self):
        for field, value in [('passed', False), ('mode', 'capacity'), ('paired_contract_sha256', '0' * 64)]:
            audit = deepcopy(self.audit)
            audit[field] = value
            self.replace_audit(audit)
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.batch_source_configuration(self.config)
        for change in ('missing', 'extra', 'core_hash'):
            audit = deepcopy(self.audit)
            hashes = audit['source_sha256']
            key = 'scripts/virtual_fusion_normal_runtime.py'
            if change == 'missing':
                hashes.pop(key)
            elif change == 'extra':
                hashes['unreviewed.py'] = 'f' * 64
            else:
                hashes[key] = 'f' * 64
            self.replace_audit(audit)
            with self.subTest(change=change), self.assertRaises(ValueError):
                resume.batch_source_configuration(self.config)

    def test_old20k_data_and_preflight_evidence_still_validate(self):
        hashes = {key: value for key, value in self.source_hashes.items() if key != 'scripts/id_xyz_resume.py'}
        for mode in ('audit', 'preflight', 'capacity'):
            receipt = {'mode': mode, 'paired_contract_sha256': runtime.paired_contract_hash(self.parent), 'source_sha256': hashes}
            with self.subTest(mode=mode):
                stages.validate_receipt(self.config, receipt, mode=mode)

    def test_new_capacity_receipt_accepts_exact_current_contract_before_old_parent_fallback(self):
        receipt = {'mode': 'capacity', 'paired_contract_sha256': runtime.paired_contract_hash(self.config),
                   'source_sha256': self.current_sources}
        with patch.object(runtime, 'source_hashes', return_value=self.current_sources), \
                patch.object(stages, 'source_hashes', return_value=self.current_sources), \
                patch.object(resume, 'validate_prior_receipt', side_effect=AssertionError('must accept current evidence first')):
            stages.validate_receipt(self.config, receipt, mode='capacity')
        receipt['source_sha256'] = {}
        with self.assertRaises(ValueError):
            stages.validate_receipt(self.config, receipt, mode='capacity')

    def test_full_plan_and_next48_global_requests_are_identical_despite_different_chunks(self):
        global_old, global_new = [], []
        for rank, old in enumerate(self.streams):
            restored = self.restore_stream(old.state_dict(), rank=rank)
            self.assertEqual(restored.cursor, old.cursor)
            self.assertEqual(restored.plan_sha256, old.plan_sha256)
            self.assertEqual(restored.contract['update_boundaries'], old.contract['update_boundaries'])
            for attribute in ('order', 'frame_counts', 'update_sizes', 'global_ends'):
                np.testing.assert_array_equal(getattr(restored, attribute), getattr(old, attribute))
            old_chunks, old_requests = consume(deepcopy(old))
            new_chunks, new_requests = consume(restored)
            self.assertEqual(old_requests, new_requests)
            self.assertEqual(len(old_chunks), 2 * len(new_chunks))
            global_old.extend(old_requests)
            global_new.extend(new_requests)
        self.assertEqual(global_old, global_new)
        self.assertEqual(len(global_new), 48)
        self.assertEqual([request.sampling_seed for request in global_new],
                         list(range((20261013 << 32) + 20000, (20261013 << 32) + 20048)))
        self.assertTrue(all(type(request.sampling_seed) is int for request in global_new))

    def test_chunk_only_replay_equivalence_holds_for_all_three_T_tiers(self):
        plan = self.streams[0]
        for low, high in ((2, 6), (7, 12), (13, 24)):
            update = next(index for index in range(420, len(plan.frame_counts)) if low <= plan.frame_counts[index] <= high)
            samples = int(plan.global_ends[update - 1])
            frames = int(np.dot(plan.update_sizes[:update].astype(np.int64), plan.frame_counts[:update]))
            for rank in (0, 1):
                old = runtime.make_stream(self.source, rank, 2)
                old.cursor = samples // 2
                restored = self.restore_stream(old.state_dict(), step=update, samples=samples, frames=frames, rank=rank)
                old_chunks, old_requests = consume(old)
                new_chunks, new_requests = consume(restored)
                with self.subTest(tier=(low, high), rank=rank):
                    self.assertEqual(old_requests, new_requests)
                    self.assertEqual(len(old_chunks), 2 * len(new_chunks))

    def test_policy_change_without_declaration_or_with_wrong_old_policy_is_rejected(self):
        changed = deepcopy(self.config)
        changed.pop('batch_transition')
        with self.assertRaises(ValueError):
            self.restore_stream(self.streams[0].state_dict(), changed)
        wrong = self.streams[0].state_dict()
        wrong['contract']['frame_policy'] = {key: 1 for key in wrong['contract']['frame_policy']}
        with self.assertRaises(ValueError):
            self.restore_stream(wrong)

    def test_changed_global_order_or_T_plan_is_rejected(self):
        real_make = runtime.make_stream
        for attribute in ('order', 'frame_counts', 'update_sizes'):
            def changed_make(config, rank, world):
                stream = real_make(config, rank, world)
                values = getattr(stream, attribute)
                values[-1] = values[-1] + 1
                return stream
            with self.subTest(attribute=attribute), patch.object(runtime, 'make_stream', side_effect=changed_make), self.assertRaises(ValueError):
                self.restore_stream(self.streams[0].state_dict())

    def test_tiny420_step_model_adam_rng_and_next_update_restore_exactly(self):
        model, optimizer, (stream, rng, initial, report) = self.restore()
        self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(self.model.state_dict()))
        self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(self.model, self.optimizer))
        self.assertEqual(resume.rng_fingerprint(rng), resume.rng_fingerprint(self.saved['rank_states'][0]))
        self.assertEqual(stream.contract['frame_policy'], self.config['training']['frame_policy'])
        self.assertEqual((report['asset_exposures'], report['step'], report['input_frame_exposures']), (20000, 420, self.frames))
        self.assertEqual(initial, self.saved['initialization'])
        self.assertEqual(report, json.loads(json.dumps(report)))
        self.assertEqual(report, self.restore(rank=1)[-1][-1])
        baseline = deepcopy(self.model)
        baseline_optimizer = runtime.optimizer_for(baseline, self.source)
        baseline_optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
        fixtures.update(baseline, baseline_optimizer, 20000, self.source)
        fixtures.update(model, optimizer, 20000, self.config)
        self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(baseline.state_dict()))
        self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(baseline, baseline_optimizer))

    def test_same_new_recipe_future_checkpoint_resumes_at_step421(self):
        model, optimizer, (_, _, _, report) = self.restore()
        fixtures.update(model, optimizer, 20000, self.config)
        future = deepcopy(self.saved)
        future.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                      training_contract=deepcopy(runtime.training_contract(self.config)),
                      configuration=deepcopy(self.config), source_sha256=self.current_sources,
                      code_commit='f' * 40, asset_exposures=20048, step=421)
        frames = self.frames + 48 * int(self.streams[0].frame_counts[420])
        future.update(input_frame_exposures=frames, supervised_frame_exposures=frames)
        for rank in (0, 1):
            stream = runtime.make_stream(self.config, rank, 2)
            stream.cursor = 10024
            future['rank_states'][rank]['stream'] = stream.state_dict()
        config = deepcopy(self.config)
        config['training']['resume_checkpoint'].update(producer_head='f' * 40, asset_exposures=20048, step=421)
        restored_model, restored_optimizer, result = self.restore(future, config)
        self.assertEqual(result[-1]['step'], 421)
        self.assertEqual(result[0].cursor, 10024)
        self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(restored_model.state_dict()))
        self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(restored_model, restored_optimizer))

    def test_source_checkpoint_cannot_claim_new_policy_inside_old_training_contract(self):
        bad = deepcopy(self.saved)
        bad['rank_states'][0]['stream']['contract']['frame_policy'] = deepcopy(self.config['training']['frame_policy'])
        with self.assertRaises(ValueError):
            self.restore(bad)

    def test_source100k_checkpoint_cannot_embed_an_old20k_sampling_plan(self):
        bad = deepcopy(self.saved)
        for rank in (0, 1):
            stream = runtime.make_stream(self.parent, rank, 2)
            stream.cursor = 10000
            bad['rank_states'][rank]['stream'] = stream.state_dict()
        with self.assertRaises(ValueError):
            self.restore(bad)

    def test_checkpoint_producer_source_hash_contract_and_parameter_order_are_bound(self):
        for change in ('producer', 'source_hashes', 'loss_contract', 'parameter_order'):
            bad = deepcopy(self.saved)
            if change == 'producer':
                bad['code_commit'] = '0' * 40
            elif change == 'source_hashes':
                bad['source_sha256']['scripts/id_xyz_resume.py'] = '0' * 64
            elif change == 'loss_contract':
                bad['training_contract']['consistency']['weight'] = 0
            else:
                bad['optimizer_parameter_names'][0]['parameters'].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.restore(bad)

    def test_tampered_serialized_model_checkpoint_hash_fails_before_loading(self):
        path = self.root / 'tiny20k.pt'
        torch.save(self.saved, path)
        spec = self.config['training']['resume_checkpoint']
        spec.update(bytes=path.stat().st_size, sha256=runtime.sha256(path))
        self.assertLess(spec['bytes'], 100000)
        bad = deepcopy(self.saved)
        next(iter(bad['model'].values())).flatten()[0] += .1
        torch.save(bad, path)
        model = fixtures.TinyGroups()
        with patch.object(torch, 'load') as load, self.assertRaises(ValueError):
            resume.restore_checkpoint(model, runtime.optimizer_for(model, self.config), self.config,
                rank=0, world=2, device=torch.device('cpu'))
        load.assert_not_called()

    def test_extended_LR_beta_and_precision_are_unchanged_by_batch_transition(self):
        for samples in (0, 4992, 5000, 20000, 20048, 75000, 100000):
            self.assertEqual(resume.extended_learning_rate(samples, self.config),
                             resume.extended_learning_rate(samples, self.source))
        fixtures.IdXYZResumeTests.test_precision_remains_float32_parameters_with_bfloat16_training_autocast(self)


if __name__ == '__main__':
    unittest.main()
