"""The new-node recovery entry cannot bypass mode, resource or capacity gates."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = 'model_training/rigweave/scripts/recover_virtual_fusion_normal.py'


class RecoveryGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = {'runtime': {'allocation': 130309880, 'host': 'pcg01i', 'physical_gpus': [5, 6]}}
        path = root / 'config.json'
        path.write_text(json.dumps(self.config))
        capacity = root / 'capacity.json'
        capacity.write_text(json.dumps({'accepted': True}))
        digest = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
        self.checkpoint = {'path': '/checkpoints/50k.pt', 'sha256': 'fixed'}
        self.calls = []
        namespace = {'Path': Path, 'json': json, 'ENTRYPOINT': ENTRYPOINT, 'CHECKPOINT': self.checkpoint,
                     'normal': SimpleNamespace(sha256=digest,
                         validate_capacity=lambda config, receipt: self.calls.append((config, receipt)))}
        tree = ast.parse((ROOT / ENTRYPOINT).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'declared_operation')
        exec(compile(ast.Module(body=[function], type_ignores=[]), ENTRYPOINT, 'exec'), namespace)
        self.gate = namespace['declared_operation']
        self.args = SimpleNamespace(config=path, output=root / 'output', mode='profile')
        self.run = {'mode': 'profile', 'config': str(path), 'output': str(self.args.output),
                    'config_sha256': digest(path), 'approved': True, 'execution_allowed': True,
                    'checkpoint': deepcopy(self.checkpoint), **deepcopy(self.config['runtime']),
                    'capacity_receipt': {'path': str(capacity), 'sha256': digest(capacity)}}
        self.state = {'active_operation': {'mode': 'normal5000_interrupted_recovery',
                      'gpu_work_allowed': True, 'formal_training_allowed': False, 'complete': False,
                      'required_entrypoint': ENTRYPOINT, 'runs': [self.run]}}

    def reject(self, state=None):
        with self.assertRaises((ValueError, KeyError)):
            self.gate(self.state if state is None else state, self.args, self.config)

    def test_profile_is_separate_from_training_permission(self):
        self.gate(self.state, self.args, self.config)
        self.assertEqual(self.calls, [])
        self.args.mode = 'train'
        self.reject()

    def test_train_needs_reviewed_capacity_and_training_permission(self):
        self.args.mode = self.run['mode'] = 'train'
        self.reject()
        self.run['training_allowed'] = True
        self.gate(self.state, self.args, self.config)
        self.assertEqual(self.calls[-1], (self.config, {'accepted': True}))
        self.run['capacity_receipt']['sha256'] = 'changed'
        self.reject()

    def test_operation_flags_fail_closed(self):
        for key, value in [('gpu_work_allowed', False), ('formal_training_allowed', True),
                           ('complete', True), ('mode', 'normal5000_continuation'),
                           ('required_entrypoint', 'old-entrypoint')]:
            state = deepcopy(self.state)
            state['active_operation'][key] = value
            self.reject(state)

    def test_old_resources_and_altered_config_are_rejected(self):
        for key, value in [('host', 'pcg02i'), ('allocation', 130258195),
                           ('physical_gpus', [5, 7]), ('config_sha256', 'changed'),
                           ('checkpoint', {'path': '/checkpoints/best.pt'}),
                           ('approved', False), ('execution_allowed', False)]:
            state = deepcopy(self.state)
            state['active_operation']['runs'][0][key] = value
            self.reject(state)

    def test_duplicate_and_foreign_output_are_rejected(self):
        state = deepcopy(self.state)
        state['active_operation']['runs'].append(deepcopy(self.run))
        self.reject(state)
        self.args.output = self.args.output.parent / 'old-output'
        self.reject()


if __name__ == '__main__':
    unittest.main()
