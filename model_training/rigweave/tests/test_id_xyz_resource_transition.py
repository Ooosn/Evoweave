"""Tiny CPU-only checks for the reviewed 35k CPU-budget continuation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_id_xyz_batch_transition as batch_fixtures

fixtures = batch_fixtures.fixtures
runtime, resume, stages = fixtures.runtime, fixtures.resume, fixtures.stages
CPUS = [58, 63, 64, 65]
SOURCE_HEAD = '7' * 40


def document(documents, name, value, reviewed=False):
    path = str(Path('__cpu_transition_memory__') / name)
    text = json.dumps(value)
    documents[path] = text
    spec = {'path': path, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    if reviewed:
        spec.update({key: value[key] for key in ('code_commit', 'config_sha256')})
    return spec


class IdXYZResourceTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        no_cuda = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU-only test'))
        no_cuda.start()
        cls.addClassCleanup(no_cuda.stop)
        cls.base_documents = {}
        cls.parent = fixtures.recipe(20000)
        cls.original = fixtures.recipe(100000)
        cls.original['continuation']['parent_configuration'] = document(
            cls.base_documents, 'parent.json', cls.parent)
        cls.current_sources = runtime.source_hashes()
        before_batch = dict(cls.current_sources)
        for key in resume.CONTINUATION_SOURCE_CHANGES:
            before_batch[key] = 'a' * 64
        before_batch[resume.TEMPORAL_SOURCE] = 'b' * 64
        original_spec = document(cls.base_documents, 'original.json', cls.original)
        batch_audit = {'passed': True, 'mode': 'resume_audit', 'code_commit': '5' * 40,
                       'config_sha256': original_spec['sha256'], 'source_sha256': before_batch,
                       'paired_contract_sha256': runtime.paired_contract_hash(cls.original)}
        cls.source = deepcopy(cls.original)
        cls.source['runtime']['repo'] = '/unused/frozen_batch_code'
        cls.source['training']['frame_policy'] = {
            key: value * 2 for key, value in cls.original['training']['frame_policy'].items()}
        cls.source['batch_transition'] = {
            'kind': 'physical_batch_temporal_chunks_v2', 'source_configuration': original_spec,
            'source_audit': document(cls.base_documents, 'batch-audit.json', batch_audit, reviewed=True),
            'temporal_execution': {'kind': 'independent_temporal_sequence_chunks_v1',
                'max_sequences': 13452, 'source_before_sha256': before_batch[resume.TEMPORAL_SOURCE],
                'source_after_sha256': cls.current_sources[resume.TEMPORAL_SOURCE]}}
        cls.old_sources = dict(cls.current_sources)
        for name in resume.RESOURCE_SOURCE_CHANGES:
            cls.old_sources[name] = 'c' * 64
        source_spec = document(cls.base_documents, 'source.json', cls.source)
        cls.audit = {'passed': True, 'mode': 'resume_audit', 'code_commit': SOURCE_HEAD,
                     'config_sha256': source_spec['sha256'], 'source_sha256': cls.old_sources,
                     'paired_contract_sha256': runtime.paired_contract_hash(cls.source)}
        cls.config = deepcopy(cls.source)
        cls.config['runtime'].update(repo='/unused/cpu4_code', execution_cpu_budget=4, cpu_affinity=CPUS.copy())
        cls.config['training']['resume_checkpoint'] = {
            'path': '__never_read_production_checkpoint__', 'sha256': '0' * 64, 'bytes': 1,
            'producer_head': SOURCE_HEAD, 'asset_exposures': 35000, 'step': 735}
        cls.config['resource_transition'] = {
            'kind': 'cpu_budget_only_v1', 'source_configuration': source_spec,
            'source_audit': document(cls.base_documents, 'source-audit.json', cls.audit, reviewed=True),
            'source_head': SOURCE_HEAD, 'handoff_assets': 35000}
        cls.streams = [runtime.make_stream(cls.source, rank, 2) for rank in (0, 1)]
        for stream in cls.streams:
            stream.cursor = 17500
        cls.frames = int(np.dot(cls.streams[0].update_sizes[:735].astype(np.int64),
                                cls.streams[0].frame_counts[:735]))
        cls.model = fixtures.TinyGroups()
        cls.optimizer = runtime.optimizer_for(cls.model, cls.source)
        for samples in [0, *cls.streams[0].global_ends[:734].tolist()]:
            fixtures.update(cls.model, cls.optimizer, samples, cls.source)
        cls.saved = deepcopy({
            'model': cls.model.state_dict(), 'optimizer': cls.optimizer.state_dict(),
            'optimizer_parameter_names': resume.optimizer_names(cls.model, cls.optimizer),
            'training_contract': runtime.training_contract(cls.source), 'configuration': cls.source,
            'code_commit': SOURCE_HEAD, 'source_sha256': cls.old_sources,
            'asset_exposures': 35000, 'step': 735, 'input_frame_exposures': cls.frames,
            'supervised_frame_exposures': cls.frames, 'initialization': {'initial_model_sha256': 'tiny35k'},
            'rank_states': [{'rank': rank, 'stream': stream.state_dict(),
                'torch_rng': torch.Generator(device='cpu').manual_seed(701 + rank).get_state(),
                'cuda_rng': torch.zeros(16, dtype=torch.uint8),
                'numpy_rng': np.random.RandomState(801 + rank).get_state(),
                'python_rng': random.Random(901 + rank).getstate()} for rank, stream in enumerate(cls.streams)]})

    def setUp(self):
        self.documents = deepcopy(self.base_documents)
        self.config = deepcopy(type(self).config)
        original_read = Path.read_text
        original_sha = runtime.sha256
        def read(path, *args, **kwargs):
            return self.documents[str(path)] if str(path) in self.documents else original_read(path, *args, **kwargs)
        def sha(path):
            return (hashlib.sha256(self.documents[str(path)].encode()).hexdigest()
                    if str(path) in self.documents else original_sha(path))
        for mocked in (patch.object(Path, 'read_text', read), patch.object(runtime, 'sha256', sha),
                       patch.object(stages, 'sha256', sha),
                       patch.object(runtime, 'source_hashes', return_value=self.current_sources),
                       patch.object(stages, 'source_hashes', return_value=self.current_sources),
                       patch.object(torch, 'load', side_effect=AssertionError('only in-memory tiny checkpoints'))):
            mocked.start()
            self.addCleanup(mocked.stop)

    def replace_source(self, source):
        self.config['resource_transition']['source_configuration'] = document(
            self.documents, 'changed-source.json', source)

    def replace_audit(self, audit):
        self.config['resource_transition']['source_audit'] = document(
            self.documents, 'changed-audit.json', audit, reviewed=True)

    def restore(self, saved=None, config=None, rank=0):
        config = self.config if config is None else config
        model = fixtures.TinyGroups()
        optimizer = runtime.optimizer_for(model, config)
        result = resume.restore_checkpoint(model, optimizer, config, rank=rank, world=2,
            device=torch.device('cpu'), saved=deepcopy(self.saved if saved is None else saved))
        return model, optimizer, result

    def test_frozen_batch_recipe_and_contract_are_preserved(self):
        before = deepcopy(self.config)
        self.assertEqual(resume.resource_source_configuration(self.config), self.source)
        self.assertEqual(resume.parent_configuration(self.config), self.parent)
        self.assertEqual(resume.batch_source_configuration(self.config), self.original)
        self.assertEqual(runtime.training_contract(self.config), runtime.training_contract(self.source))
        self.assertEqual(runtime.paired_contract_hash(self.config), runtime.paired_contract_hash(self.source))
        self.assertEqual(self.config, before)
        self.assertEqual(self.frames, 456552)

    def test_no_transition_returns_no_resource_receipt_override(self):
        self.assertIsNone(resume.resource_source_configuration(self.source))
        self.assertIs(resume.validate_resource_receipt(self.source, {}), False)

    def test_kind_shape_handoff_and_source_head_are_bound(self):
        transition = self.config['resource_transition']
        variants = [{}, [], dict(transition, kind='other'), dict(transition, extra=True)]
        variants += [{k: v for k, v in transition.items() if k != key} for key in transition]
        variants += [dict(transition, handoff_assets=x) for x in (True, 35000., '35000', 20000, 100000, 35001)]
        variants += [dict(transition, source_head=x) for x in ('a'*39, 'A'*40, 'g'*40, None)]
        for value in variants:
            config = deepcopy(self.config)
            config['resource_transition'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                resume.resource_source_configuration(config)

    def test_source_and_reviewed_audit_digests_are_required(self):
        for field in ('source_configuration', 'source_audit'):
            for spec in (None, {}, {'path': 'bad', 'sha256': 'bad'}):
                changed = deepcopy(self.config)
                changed['resource_transition'][field] = spec
                with self.subTest(field=field, spec=spec), self.assertRaises((ValueError, TypeError, KeyError, FileNotFoundError)):
                    resume.resource_source_configuration(changed)
            changed = deepcopy(self.config)
            changed['resource_transition'][field]['sha256'] = 'f' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.resource_source_configuration(changed)

    def test_source_must_be_unmodified_one_cpu_batch_recipe(self):
        for change in ('missing_batch', 'nested_resource', 'execution_budget', 'old_affinity', 'bool_slots'):
            source = deepcopy(self.source)
            if change == 'missing_batch':
                source.pop('batch_transition')
            elif change == 'nested_resource':
                source['resource_transition'] = {'kind': 'cpu_budget_only_v1'}
            elif change == 'execution_budget':
                source['runtime']['execution_cpu_budget'] = 1
            elif change == 'old_affinity':
                source['runtime']['cpu_affinity'] = [58]
            else:
                source['runtime']['cpu_slots'] = True
            self.replace_source(source)
            with self.subTest(change=change), self.assertRaises(ValueError):
                resume.resource_source_configuration(self.config)

    def test_only_cpu_binding_repo_resume_and_safety_may_change(self):
        permitted = deepcopy(self.config)
        permitted['runtime']['repo'] = '/unused/another_cpu4_code'
        permitted['training']['resume_checkpoint']['path'] = '__tiny_not_read__'
        permitted['safety'] = {'primary_accepted': False}
        self.assertEqual(resume.resource_source_configuration(permitted), self.source)
        changes = [('architecture', 'dropout', .1), ('data', 'normalization', 'per_frame'),
                   ('data', 'stream_seed', 1), ('model_contract', 'all_real_frames_supervised', False),
                   ('consistency', 'weight', 0), ('optimization', 'peak_lr', 3e-5),
                   ('training', 'effective_assets_per_update', 96), ('training', 'target_assets', 200000),
                   ('training', 'frame_policy', {str(t): 1 for t in range(2, 25)}),
                   ('runtime', 'cpu_slots', 4), ('runtime', 'cpu_slots', True),
                   ('runtime', 'rss_limit_bytes', 1), ('runtime', 'gpu_memory_fraction_limit', .9),
                   ('runtime', 'physical_gpus', [5, 7]), ('runtime', 'python', '/other/python'),
                   ('runtime', 'require_scheduler_membership', True), ('evaluation', 'input_frames', 4)]
        for section, key, value in changes:
            changed = deepcopy(self.config)
            changed[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                resume.resource_source_configuration(changed)

    def test_cpu_budget_and_exact_affinity_types_are_strict(self):
        for key, values in [('execution_cpu_budget', (1, 3, 8, True, 4.0, '4', None)),
                            ('cpu_affinity', ([58], [58, 58, 64, 65], [65, 64, 63, 58],
                                              [-1, 63, 64, 65], [True, 63, 64, 65], tuple(CPUS)))]:
            for value in values:
                changed = deepcopy(self.config)
                changed['runtime'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resume.resource_source_configuration(changed)

    def test_audit_must_be_reviewed_passed_and_cover_frozen_contract(self):
        for key, value in [('passed', False), ('mode', 'capacity'), ('paired_contract_sha256', '0'*64)]:
            audit = deepcopy(self.audit)
            audit[key] = value
            self.replace_audit(audit)
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.resource_source_configuration(self.config)
        for key in ('code_commit', 'config_sha256'):
            changed = deepcopy(type(self).config)
            changed['resource_transition']['source_audit'][key] = 'f'*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.resource_source_configuration(changed)

    def test_core_source_changes_are_limited_to_two_integration_files(self):
        allowed = {'scripts/id_xyz_resume.py', 'scripts/id_xyz_stages.py'}
        self.assertEqual(resume.RESOURCE_SOURCE_CHANGES, allowed)
        self.assertEqual({k for k in self.old_sources if self.old_sources[k] != self.current_sources[k]}, allowed)
        self.assertEqual(resume.resource_source_configuration(self.config), self.source)
        for name in sorted(set(self.current_sources)-allowed):
            current = dict(self.current_sources, **{name: 'd'*64})
            with self.subTest(name=name), patch.object(runtime, 'source_hashes', return_value=current):
                with self.assertRaises(ValueError):
                    resume.resource_source_configuration(self.config)
        for current in (dict(self.current_sources, unreviewed='e'*64),
                        {k: v for k, v in self.current_sources.items() if k != 'scripts/id_xyz_runtime.py'}):
            with patch.object(runtime, 'source_hashes', return_value=current), self.assertRaises(ValueError):
                resume.resource_source_configuration(self.config)

    def test_exact_old_capacity_is_reused_but_unreviewed_hashes_rejected(self):
        receipt = {'mode': 'capacity', 'paired_contract_sha256': runtime.paired_contract_hash(self.source),
                   'source_sha256': deepcopy(self.old_sources)}
        self.assertIs(resume.validate_resource_receipt(self.config, receipt), True)
        stages.validate_receipt(self.config, receipt, mode='capacity')
        for change in ('producer', 'missing', 'extra'):
            bad = deepcopy(receipt)
            if change == 'producer':
                bad['source_sha256']['scripts/id_xyz_stages.py'] = 'f'*64
            elif change == 'missing':
                bad['source_sha256'].pop('scripts/id_xyz_stages.py')
            else:
                bad['source_sha256']['unreviewed'] = 'f'*64
            with self.subTest(change=change), self.assertRaises(ValueError):
                stages.validate_receipt(self.config, bad, mode='capacity')
        with self.assertRaises(ValueError):
            stages.validate_receipt(self.config, dict(receipt, mode='preflight'), mode='capacity')

    def test_tiny735_step_model_adam_rng_and_sampler_restore(self):
        requests = []
        for rank in (0, 1):
            model, optimizer, (stream, rng, initial, report) = self.restore(rank=rank)
            self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(self.model.state_dict()))
            self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(self.model, self.optimizer))
            self.assertEqual(resume.rng_fingerprint(rng), resume.rng_fingerprint(self.saved['rank_states'][rank]))
            self.assertEqual(initial, self.saved['initialization'])
            self.assertEqual((report['asset_exposures'], report['step'], report['input_frame_exposures']), (35000, 735, 456552))
            self.assertEqual(report['resource_transition']['cpu_affinity'], CPUS)
            self.assertTrue(report['resource_transition']['checkpoint_from_before_transition'])
            self.assertEqual((stream.cursor, stream.plan_sha256), (17500, self.streams[rank].plan_sha256))
            for name in ('order', 'frame_counts', 'update_sizes', 'global_ends'):
                np.testing.assert_array_equal(getattr(stream, name), getattr(self.streams[rank], name))
            actual = batch_fixtures.consume(stream)[1]
            self.assertEqual(actual, batch_fixtures.consume(deepcopy(self.streams[rank]))[1])
            requests.extend(actual)
            baseline = deepcopy(self.model)
            baseline_optimizer = runtime.optimizer_for(baseline, self.source)
            baseline_optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
            fixtures.update(model, optimizer, 35000, self.config)
            fixtures.update(baseline, baseline_optimizer, 35000, self.source)
            self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(baseline.state_dict()))
            self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(baseline, baseline_optimizer))
        self.assertEqual([r.sampling_seed for r in requests], list(range((20261013 << 32)+35000, (20261013 << 32)+35048)))

    def test_wrong_checkpoint_source_counters_or_sampler_are_rejected(self):
        for change in ('head', 'hash', 'configuration', 'assets', 'step', 'frames', 'cursor', 'adam'):
            saved = deepcopy(self.saved)
            if change == 'head':
                saved['code_commit'] = 'f'*40
            elif change == 'hash':
                saved['source_sha256']['scripts/id_xyz_resume.py'] = 'f'*64
            elif change == 'configuration':
                saved['configuration']['runtime']['repo'] = '/different/source'
            elif change == 'assets':
                saved['asset_exposures'] -= 48
            elif change == 'step':
                saved['step'] -= 1
            elif change == 'frames':
                saved['input_frame_exposures'] += 1
            elif change == 'cursor':
                saved['rank_states'][0]['stream']['cursor'] -= 24
            else:
                next(iter(saved['optimizer']['state'].values()))['step'] -= 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.restore(saved)


if __name__ == '__main__':
    unittest.main()
