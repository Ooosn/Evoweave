"""CPU tests for the published A-continuation operation gates."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from virtual_fusion_continuation import continuation_contract


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'model_training/rigweave/scripts'
REMOTE_C_EXIT = Path('/home/wangyy/jobs/virtual_fusion_normal_20261014_v1/normal_c_v1.exit')
REJECTED = (ValueError, KeyError, TypeError, AttributeError)


def functions(filename, names, namespace):
    tree = ast.parse((SCRIPTS / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('missing actual implementation')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, 'exec'), namespace)


def gate_namespace():
    namespace = {'Path': Path, 'json': json, 'continuation_contract': continuation_contract}
    entrypoint = ast.parse((SCRIPTS / 'resume_virtual_fusion_normal.py').read_text())
    helper = ast.parse((SCRIPTS / 'virtual_fusion_continuation.py').read_text())
    assignments = [node for node in entrypoint.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in {'ENTRYPOINT', 'C_SOURCE'}
                           for target in node.targets)]
    assignments += [node for node in helper.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in {'CHECKPOINT', 'PURPOSE'}
                            for target in node.targets)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])),
                 'resume_virtual_fusion_normal.py', 'exec'), namespace)
    functions('resume_virtual_fusion_normal.py', ['declared_operation', 'verify_operation'], namespace)
    namespace['sys'] = SimpleNamespace(executable='python')
    namespace['subprocess'] = SimpleNamespace()
    return namespace


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def make_fixture(root, target_assets=40000, route='a'):
    root = Path(root)
    config = {'runtime': {'physical_gpus': [5, 7], 'host': 'pcg02i',
                         'allocation': 130258195 if target_assets >= 120000 or route == 'b' else 129839822},
              'route': route, 'training': {'target_assets': target_assets}}
    contract = continuation_contract(config)
    label = contract['label']
    config_path = root / f'model_training/experiments/fusion_{label}.json'
    output_root = root / 'outputs'
    output = output_root / label
    capacity_path = root / 'reviewed_a_capacity.json'
    state_path = root / 'model_training/state/current.json'
    config_path.parent.mkdir(parents=True)
    output.mkdir(parents=True)
    capacity_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    capacity = {'accepted': True, 'world_size': 2, 'capacity_contract': {'fixture': 'reviewed-a'},
                'approved_cases': [{'frames': 24, 'batch_per_rank': 6}]}
    c_result = deepcopy(contract['predecessor_result'])
    write_json(config_path, config)
    write_json(capacity_path, capacity)
    result_path = output_root / contract['predecessor_label'] / 'result.json'
    result_path.parent.mkdir(parents=True)
    write_json(result_path, c_result)

    namespace = gate_namespace()
    calls = []

    def validate_capacity(received_config, receipt):
        calls.append((received_config, receipt))
        if received_config != config or receipt != capacity:
            raise ValueError('fixture capacity receipt differs')

    normal = SimpleNamespace(CODE_ROOT=root, OUTPUT_ROOT=output_root,
                             sha256=file_sha256, validate_capacity=validate_capacity)
    namespace.update(ROOT=root, normal=normal)
    run = {
        'name': label, 'mode': 'resume', 'approved': True,
        'training_allowed': True, 'config': str(config_path),
        'config_sha256': file_sha256(config_path), 'output': str(output),
        'physical_gpus': [5, 7], 'resume_checkpoint': deepcopy(contract['checkpoint']),
        'allocation': config['runtime']['allocation'], 'host': config['runtime']['host'],
        'capacity_receipt': {'path': str(capacity_path), 'sha256': file_sha256(capacity_path)},
    }
    predecessor = {
        'result': str(result_path),
        'exit': str(REMOTE_C_EXIT.parent / (contract['predecessor_label'] + '.exit')),
        'result_sha256': file_sha256(result_path),
    }
    operation = {
        'mode': namespace['PURPOSE'], 'gpu_work_allowed': True,
        'formal_training_allowed': False, 'complete': False,
        'required_entrypoint': namespace['ENTRYPOINT'], 'code_commit': 'approved-source',
        'runs': [run], 'completed_predecessor': predecessor,
    }
    state = {'architecture_contract': {'id': 'virtual_query_xyz_aux_ids_v1'},
             'active_operation': operation}
    write_json(state_path, state)
    return SimpleNamespace(root=root, namespace=namespace, normal=normal, calls=calls, config=config,
                           config_path=config_path, output=output, output_root=output_root,
                           capacity=capacity, capacity_path=capacity_path, result_path=result_path,
                           state_path=state_path, state=state)


def write_state(fixture, state):
    write_json(fixture.state_path, state)


def replace_c_result(fixture, state, **changes):
    result = json.loads(fixture.result_path.read_text())
    result.update(changes)
    write_json(fixture.result_path, result)
    state['active_operation']['completed_predecessor']['result_sha256'] = file_sha256(fixture.result_path)


def remote_exit_text(value, label='normal_c_v1'):
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        expected = REMOTE_C_EXIT.parent / (label + '.exit')
        if str(path).replace('\\', '/') == str(expected).replace('\\', '/'):
            return value
        return original(path, *args, **kwargs)

    return patch.object(Path, 'read_text', new=read_text)


def clean_subprocess():
    calls = []

    def run(command, check=False):
        calls.append(('run', command, check))
        return SimpleNamespace(returncode=0)

    def check_output(command, text=False):
        calls.append(('check_output', command, text))
        if command[-1] == '--porcelain':
            return ''
        if command[-2:] == ['rev-parse', 'HEAD']:
            return 'test-head\n'
        raise AssertionError(f'unexpected subprocess command: {command}')

    return SimpleNamespace(run=run, check_output=check_output), calls


class ContinuationGateTests(unittest.TestCase):
    def test_b60k_gate_requires_completed_final_b40k(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 60000, 'b')
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_b_extend40000_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                state = deepcopy(fixture.state)
                state['active_operation']['runs'][0]['resume_checkpoint'] = deepcopy(fixture.namespace['CHECKPOINT'])
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, route='a')
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_a150k_gate_requires_completed_final_a130k(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 150000)
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_a_lr3e5_130000_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                state = deepcopy(fixture.state)
                state['active_operation']['runs'][0]['resume_checkpoint'] = deepcopy(fixture.namespace['CHECKPOINT'])
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, asset_exposures=120000, updates=2520)
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_b50k_gate_requires_own_completed_final_b40k(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 50000, 'b')
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_b_extend40000_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                state = deepcopy(fixture.state)
                state['active_operation']['runs'][0]['resume_checkpoint'] = deepcopy(fixture.namespace['CHECKPOINT'])
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, asset_exposures=35000, updates=735)
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_a130k_gate_requires_completed_final_a120k(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 130000)
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_a_extend120000_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                state = deepcopy(fixture.state)
                state['active_operation']['runs'][0]['resume_checkpoint'] = deepcopy(fixture.namespace['CHECKPOINT'])
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, asset_exposures=80000, updates=1680)
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_b40k_gate_requires_own_completed_b20k_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 40000, 'b')
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_b_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                state = deepcopy(fixture.state)
                state['active_operation']['runs'][0]['resume_checkpoint'] = deepcopy(fixture.namespace['CHECKPOINT'])
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, route='a')
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_a120k_gate_requires_completed_a80k_on_the_replacement_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 120000)
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_a_extend80000_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                for mutate in (
                    lambda state: state['active_operation']['runs'][0].__setitem__('allocation', 129839822),
                    lambda state: state['active_operation']['runs'][0].__setitem__('host', 'pcg01i'),
                    lambda state: state['active_operation']['completed_predecessor'].__setitem__(
                        'exit', str(REMOTE_C_EXIT.parent / 'normal_a_extend40000_v1.exit')),
                ):
                    state = deepcopy(fixture.state)
                    mutate(state)
                    with self.assertRaises(ValueError):
                        fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, asset_exposures=40000, updates=840)
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_a80k_gate_requires_completed_a40k_and_its_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(directory, 80000)
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0', 'normal_a_extend40000_v1'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
                for mutate in (
                    lambda state: state['active_operation'].__setitem__('complete', True),
                    lambda state: state['active_operation']['runs'][0].__setitem__(
                        'resume_checkpoint', deepcopy(fixture.namespace['CHECKPOINT'])),
                    lambda state: state['active_operation']['completed_predecessor'].__setitem__(
                        'exit', str(REMOTE_C_EXIT)),
                ):
                    state = deepcopy(fixture.state)
                    mutate(state)
                    with self.assertRaises(ValueError):
                        fixture.namespace['declared_operation'](state, args, fixture.config)
                state = deepcopy(fixture.state)
                replace_c_result(fixture, state, asset_exposures=20000, updates=420)
                with self.assertRaises(ValueError):
                    fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_current_c_active_operation_cannot_authorize_a_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            state = deepcopy(fixture.state)
            state['active_operation']['mode'] = 'normal5000'
            state['active_operation']['required_entrypoint'] = (
                'model_training/rigweave/scripts/train_virtual_fusion_normal.py')
            write_state(fixture, state)
            with self.assertRaises(ValueError):
                fixture.namespace['declared_operation'](state, SimpleNamespace(
                    config=fixture.config_path, output=fixture.output), fixture.config)
            with self.assertRaises(ValueError):
                fixture.namespace['verify_operation'](SimpleNamespace(
                    config=fixture.config_path, output=fixture.output), fixture.config, 0)

    def test_explicit_a_operation_and_verify_gate_accept_only_published_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            with remote_exit_text('0'):
                operation = fixture.namespace['declared_operation'](fixture.state, args, fixture.config)
            self.assertIs(operation, fixture.state['active_operation'])
            self.assertEqual(len(fixture.calls), 1)

            subprocess, calls = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0'):
                self.assertEqual(fixture.namespace['verify_operation'](args, fixture.config, 0), 'test-head')
            self.assertEqual(len([call for call in calls if call[0] == 'run']), 3)

    def test_operation_and_run_bindings_are_hash_and_mode_bound(self):
        mutations = (
            ('architecture', lambda state: state['architecture_contract'].__setitem__('id', 'wrong')),
            ('purpose', lambda state: state['active_operation'].__setitem__('mode', 'normal5000')),
            ('entrypoint', lambda state: state['active_operation'].__setitem__('required_entrypoint',
                'model_training/rigweave/scripts/train_virtual_fusion_normal.py')),
            ('gpu authorization', lambda state: state['active_operation'].__setitem__('gpu_work_allowed', False)),
            ('formal authorization', lambda state: state['active_operation'].__setitem__('formal_training_allowed', True)),
            ('completion flag', lambda state: state['active_operation'].__setitem__('complete', True)),
            ('duplicate run', lambda state: state['active_operation']['runs'].append(
                deepcopy(state['active_operation']['runs'][0]))),
            ('config hash', lambda state: state['active_operation']['runs'][0].__setitem__('config_sha256', 'wrong')),
            ('output binding', lambda state: state['active_operation']['runs'][0].__setitem__('output', '/tmp/wrong')),
            ('run mode', lambda state: state['active_operation']['runs'][0].__setitem__('mode', 'train')),
            ('approval', lambda state: state['active_operation']['runs'][0].__setitem__('approved', False)),
            ('training authorization', lambda state: state['active_operation']['runs'][0].__setitem__('training_allowed', False)),
            ('physical devices', lambda state: state['active_operation']['runs'][0].__setitem__('physical_gpus', [5])),
            ('checkpoint binding', lambda state: state['active_operation']['runs'][0].__setitem__('resume_checkpoint', {})),
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            for label, mutate in mutations:
                with self.subTest(label=label):
                    state = deepcopy(fixture.state)
                    mutate(state)
                    with self.assertRaises(ValueError):
                        fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_reviewed_capacity_and_exact_c_completion_evidence_are_required(self):
        cases = (
            ('wrong exit code', lambda fixture, state: None, '1'),
            ('wrong exit path', lambda fixture, state: state['active_operation']['completed_predecessor'].__setitem__(
                'exit', '/tmp/wrong.exit'), '0'),
            ('missing exit path', lambda fixture, state: state['active_operation']['completed_predecessor'].pop('exit'), '0'),
            ('wrong source', lambda fixture, state: replace_c_result(fixture, state, code_commit='wrong'), '0'),
            ('missing source', lambda fixture, state: replace_c_result_missing(fixture, state, 'code_commit'), '0'),
            ('wrong route', lambda fixture, state: replace_c_result(fixture, state, route='a'), '0'),
            ('wrong assets', lambda fixture, state: replace_c_result(fixture, state, asset_exposures=10000), '0'),
            ('wrong updates', lambda fixture, state: replace_c_result(fixture, state, updates=419), '0'),
            ('wrong frame count', lambda fixture, state: replace_c_result(
                fixture, state, supervised_frame_exposures=1), '0'),
            ('wrong result hash', lambda fixture, state: state['active_operation']['completed_predecessor'].__setitem__(
                'result_sha256', 'wrong'), '0'),
            ('missing result hash', lambda fixture, state: state['active_operation']['completed_predecessor'].pop(
                'result_sha256'), '0'),
            ('wrong capacity hash', lambda fixture, state: state['active_operation']['runs'][0][
                'capacity_receipt'].__setitem__('sha256', 'wrong'), '0'),
            ('changed capacity receipt', change_capacity_receipt, '0'),
            ('missing capacity receipt', lambda fixture, state: state['active_operation']['runs'][0].pop(
                'capacity_receipt'), '0'),
        )
        with tempfile.TemporaryDirectory() as directory:
            for label, mutate, exit_text in cases:
                with self.subTest(label=label):
                    fixture = make_fixture(Path(directory) / label.replace(' ', '_'))
                    state = deepcopy(fixture.state)
                    mutate(fixture, state)
                    write_state(fixture, state)
                    args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
                    with remote_exit_text(exit_text), self.assertRaises(REJECTED):
                        fixture.namespace['declared_operation'](state, args, fixture.config)

    def test_verify_gate_rejects_unpublished_paths_or_dirty_source(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            valid_args = SimpleNamespace(config=fixture.config_path, output=fixture.output)
            subprocess, _ = clean_subprocess()
            fixture.namespace['subprocess'] = subprocess
            with remote_exit_text('0'):
                self.assertEqual(fixture.namespace['verify_operation'](valid_args, fixture.config, 0), 'test-head')

            for label, args in (
                    ('wrong config parent', SimpleNamespace(config=fixture.root / 'other.json', output=fixture.output)),
                    ('wrong output parent', SimpleNamespace(config=fixture.config_path,
                                                            output=fixture.root / 'other/normal_a_extend40000_v1')),
                    ('wrong output name', SimpleNamespace(config=fixture.config_path,
                                                          output=fixture.output_root / 'other')),
            ):
                with self.subTest(label=label), self.assertRaises(ValueError):
                    fixture.namespace['verify_operation'](args, fixture.config, 0)

            def dirty_output(command, text=False):
                if command[-1] == '--porcelain':
                    return ' M source.py\n'
                return 'test-head\n'

            fixture.namespace['subprocess'] = SimpleNamespace(run=subprocess.run, check_output=dirty_output)
            with remote_exit_text('0'), self.assertRaises(ValueError):
                fixture.namespace['verify_operation'](valid_args, fixture.config, 0)


def replace_c_result_missing(fixture, state, key):
    result = json.loads(fixture.result_path.read_text())
    result.pop(key)
    write_json(fixture.result_path, result)
    state['active_operation']['completed_predecessor']['result_sha256'] = file_sha256(fixture.result_path)


def change_capacity_receipt(fixture, state):
    changed = deepcopy(fixture.capacity)
    changed['capacity_contract'] = {'fixture': 'changed'}
    write_json(fixture.capacity_path, changed)
    state['active_operation']['runs'][0]['capacity_receipt']['sha256'] = file_sha256(fixture.capacity_path)


if __name__ == '__main__':
    unittest.main()
