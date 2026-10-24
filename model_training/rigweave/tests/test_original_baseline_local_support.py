from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
import eval_dynamic_rig_ce as evaluator
import train_dynamic_rig as trainer
import original_baseline_eval_guard as guard


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


work_guard = module('baseline_work_guard_test', ROOT / 'model_training/tools/agent_work_guard.py')
launcher = module('baseline_local_launcher_test', ROOT / 'model_training/jobs/launch_original_baseline_eval.py')


class TinyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class StrictCheckpointTests(unittest.TestCase):
    def build(self, state, strict=True, official=False):
        import rigweave.dynamic_rig as dynamic
        import rigweave.dynamic_rig.model as models
        import rigweave.dynamic_rig.unirig_wrapper as wrappers
        args = SimpleNamespace(**{**evaluator.CHECKPOINT_DEFAULTS,
            'checkpoint': 'unused.pt', 'model_config': 'model.yaml', 'unirig_checkpoint': 'official.pt'})
        unirig = SimpleNamespace(hidden_size=4, mesh_encoder=nn.Identity(), output_proj=nn.Identity())
        with patch.object(evaluator, 'load_unirig', return_value=unirig) as load, \
             patch.object(evaluator, '_move_dynamic_model_to_device'), \
             patch.object(dynamic, 'FixedQuerySurfaceTokenizer', TinyModule), \
             patch.object(dynamic, 'LegacyTemporalMotionEncoder', TinyModule), \
             patch.object(dynamic, 'FrameTypeAnchorWiseAlternatingMotionEncoder', TinyModule), \
             patch.object(dynamic, 'TemporalMotionEncoder', TinyModule), \
             patch.object(models, 'DynamicRigConditioner', TinyModule), \
             patch.object(wrappers, 'DynamicRigUniRigAR', TinyModule):
            result = evaluator._build_dynamic_model(args, object(), torch.device('cpu'),
                strict_checkpoint=strict, initialize_from_official=official,
                checkpoint_payload={'model': state})
            return result, load.call_args.args[-1]

    def test_strict_full_restore_does_not_need_official_weights(self):
        model, initial = self.build({'weight': torch.tensor([3.0])})
        self.assertIsNone(initial)
        self.assertEqual(model.weight.item(), 3.)

    def test_skipping_official_requires_strict_restore(self):
        with self.assertRaises(ValueError):
            self.build({'weight': torch.tensor([3.0])}, strict=False)

    def test_missing_extra_or_wrong_shape_fails(self):
        for state in ({}, {'weight': torch.zeros(2)}, {'weight': torch.zeros(1), 'extra': torch.zeros(1)}):
            with self.subTest(keys=list(state)), self.assertRaises(RuntimeError):
                self.build(state)

    def test_existing_official_initialization_remains_default(self):
        model, initial = self.build({'weight': torch.tensor([2.0])}, strict=False, official=True)
        self.assertEqual(initial, 'official.pt')
        self.assertEqual(model.weight.item(), 2.)

    def test_windows_memory_logging_is_explicit(self):
        with patch.object(trainer, 'resource', None):
            trainer.log_process_memory(0, 'unused', False)
            with self.assertRaisesRegex(RuntimeError, 'Windows'):
                trainer.log_process_memory(0, 'unsupported', True)


class AdmissionTests(unittest.TestCase):
    def test_operation_specific_entrypoint_does_not_loosen_training(self):
        state = {'active_operation': {'required_entrypoint': 'train.py',
                 'operation_entrypoints': {'matched_eval': 'evaluate.py'}}}
        work_guard._check_entrypoint(state, 'train', 'train.py')
        work_guard._check_entrypoint(state, 'matched_eval', 'evaluate.py')
        for operation, entrypoint in (('train', 'evaluate.py'), ('matched_eval', 'train.py'), ('matched_eval', None)):
            with self.assertRaises(RuntimeError):
                work_guard._check_entrypoint(state, operation, entrypoint)

    def test_invalid_entrypoint_mapping_fails(self):
        for bindings in ([], {'matched_eval': False}, {'matched_eval': ''}):
            with self.assertRaises(RuntimeError):
                work_guard._check_entrypoint({'active_operation': {'operation_entrypoints': bindings}}, 'matched_eval', 'eval.py')

    def test_local_runtime_has_exact_limits(self):
        runtime = {'execution_mode': 'local_windows', 'world_size': 1, 'device_index': 0,
                   'gpu_uuid': guard.GPU_UUID, 'cpu_affinity': list(range(8, 16)),
                   'rss_limit_bytes': 32 * 2**30, 'gpu_memory_fraction_limit': .45,
                   'aggregate_gpu_fraction_limit': .85, 'repo': str(ROOT)}
        guard.validate_runtime(runtime)
        for key, value in (('world_size', 2), ('device_index', True), ('gpu_uuid', 'wrong'),
                           ('cpu_affinity', [0]), ('rss_limit_bytes', 64 * 2**30),
                           ('gpu_memory_fraction_limit', .9), ('execution_mode', 'ssh')):
            changed = copy.deepcopy(runtime)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_runtime(changed)

    def test_local_evaluation_is_scoped_to_explicit_component(self):
        with self.assertRaises(RuntimeError):
            guard.operation_record({'active_operation': {}})
        record = {'label': guard.SCOPE, 'explicit_human_authorized': True}
        self.assertEqual(guard.operation_record({'active_operation': {'baseline_matched_evaluation': record}}), record)

    def test_other_python_gpu_jobs_are_not_treated_as_idle(self):
        with patch.object(launcher.subprocess, 'check_output', return_value='123, C:/env/python.exe\n456, C:/app/chrome.exe\n789, C:/env/python.exe\n'):
            self.assertEqual(launcher.other_gpu_python_pids(ignore=(789,)), [123])


if __name__ == '__main__':
    unittest.main()
