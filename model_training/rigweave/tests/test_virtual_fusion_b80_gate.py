"""CPU proofs for the B80 predecessor and replacement-node continuation gate."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from virtual_fusion_continuation import B60_CHECKPOINT, B80_RUNTIME, continuation_contract


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'model_training/rigweave/scripts'
REJECTED = (ValueError, KeyError, TypeError, AttributeError)
RECOVERY_JOBDIR = Path('/home/wangyy/jobs/virtual_fusion_recovery_20261021_v1')
OLD_JOBDIR = Path('/home/wangyy/jobs/virtual_fusion_normal_20261014_v1')


def gate_namespace():
    namespace = {
        'Path': Path, 'json': json, 'continuation_contract': continuation_contract,
        'PURPOSE': 'normal5000_continuation',
        'ENTRYPOINT': 'model_training/rigweave/scripts/resume_virtual_fusion_normal.py',
    }
    tree = ast.parse((SCRIPTS / 'resume_virtual_fusion_normal.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'declared_operation')
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'resume_virtual_fusion_normal.py', 'exec'), namespace)
    return namespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class B80GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = {'runtime': deepcopy(B80_RUNTIME), 'route': 'b', 'training': {'target_assets': 80000}}
        config_path = root / 'model_training/experiments/fusion_normal_b_lr3e5_80000_v1.json'
        output_root = root / 'outputs'
        output = output_root / 'normal_b_lr3e5_80000_v1'
        capacity_path = root / 'reviewed_capacity.json'
        result_path = output_root / 'normal_b_recover60000_20261021_v1/result.json'
        config_path.parent.mkdir(parents=True)
        output.mkdir(parents=True)
        result_path.parent.mkdir(parents=True)
        capacity_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(config_path, self.config)
        write_json(capacity_path, {'accepted': True})
        contract = continuation_contract(self.config)
        write_json(result_path, deepcopy(contract['predecessor_result']))
        self.namespace = gate_namespace()
        calls = []
        self.namespace['normal'] = SimpleNamespace(
            OUTPUT_ROOT=output_root,
            sha256=digest,
            validate_capacity=lambda config, receipt: calls.append((config, receipt)),
        )
        run = {
            'name': contract['label'], 'mode': 'resume', 'approved': True,
            'training_allowed': True, 'config': str(config_path),
            'config_sha256': digest(config_path), 'output': str(output),
            'physical_gpus': [5, 6], 'resume_checkpoint': deepcopy(contract['checkpoint']),
            'allocation': B80_RUNTIME['allocation'], 'host': B80_RUNTIME['host'],
            'capacity_receipt': {'path': str(capacity_path), 'sha256': digest(capacity_path)},
        }
        self.state = {
            'architecture_contract': {'id': 'virtual_query_xyz_aux_ids_v1'},
            'active_operation': {
                'mode': self.namespace.get('PURPOSE', 'normal5000_continuation'),
                'gpu_work_allowed': True, 'formal_training_allowed': False, 'complete': False,
                'required_entrypoint': 'model_training/rigweave/scripts/resume_virtual_fusion_normal.py',
                'runs': [run],
                'completed_predecessor': {
                    'result': str(result_path),
                    'exit': str(RECOVERY_JOBDIR / (contract['predecessor_label'] + '.exit')),
                    'result_sha256': digest(result_path),
                },
            },
        }
        self.args = SimpleNamespace(config=config_path, output=output)
        self.config_path = config_path
        self.output = output
        self.result_path = result_path
        self.calls = calls

    def exit_text(self, value='0'):
        original = Path.read_text
        expected = RECOVERY_JOBDIR / 'normal_b_recover60000_20261021_v1.exit'

        def read_text(path, *args, **kwargs):
            if str(path).replace('\\', '/') == str(expected).replace('\\', '/'):
                return value
            return original(path, *args, **kwargs)

        return patch.object(Path, 'read_text', new=read_text)

    def test_b80_uses_recovered_predecessor_exit_and_final60_checkpoint(self):
        with self.exit_text():
            self.namespace['declared_operation'](self.state, self.args, self.config)
        self.assertEqual(self.calls, [({'runtime': self.config['runtime'], 'route': 'b',
                                       'training': {'target_assets': 80000}}, {'accepted': True})])
        changed = deepcopy(self.state)
        changed['active_operation']['completed_predecessor']['exit'] = str(
            OLD_JOBDIR / 'normal_b_recover60000_20261021_v1.exit')
        with self.exit_text(), self.assertRaises(ValueError):
            self.namespace['declared_operation'](changed, self.args, self.config)
        self.assertEqual(self.namespace['continuation_contract'](self.config)['checkpoint'], B60_CHECKPOINT)

    def test_b80_gate_rejects_changed_predecessor_result_or_checkpoint(self):
        with self.exit_text():
            changed = deepcopy(self.state)
            result = json.loads(self.result_path.read_text())
            result['mode'] = 'resume'
            write_json(self.result_path, result)
            changed['active_operation']['completed_predecessor']['result_sha256'] = digest(self.result_path)
            with self.assertRaises(ValueError):
                self.namespace['declared_operation'](changed, self.args, self.config)
        changed = deepcopy(self.state)
        changed['active_operation']['runs'][0]['resume_checkpoint']['sha256'] = '0' * 64
        with self.exit_text(), self.assertRaises(ValueError):
            self.namespace['declared_operation'](changed, self.args, self.config)

    def test_resume_source_uses_configured_replacement_host_and_gpu_list(self):
        source = (SCRIPTS / 'resume_virtual_fusion_normal.py').read_text()
        self.assertIn("physical_host = config['runtime']['host']", source)
        self.assertIn("','.join(map(str, physical_gpus))", source)
        self.assertNotIn("socket.gethostname().split('.')[0] != 'pcg02'", source)
        self.assertEqual(B80_RUNTIME, {'allocation': 130309880, 'host': 'pcg01i', 'physical_gpus': [5, 6]})


if __name__ == '__main__':
    unittest.main()
