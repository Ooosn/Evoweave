"""CPU-only regressions for the scoped ID/XYZ execution CPU override."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
sys.path.insert(0, str(ROOT / 'model_training/jobs'))
import id_xyz_launch_guard as guard

spec = importlib.util.spec_from_file_location(
    'cpu_budget_launcher', ROOT / 'model_training/jobs/launch_legacy_single_target.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

CPUS = [58, 63, 64, 65]
RESOURCES = dict(allocation=130309880, host='pcg01i', physical_gpus=[5, 6],
                 cpu_slots=1, rss_limit_bytes=68719476736,
                 gpu_memory_fraction_limit=.85, keeper_pid=1090013)


def fixture(override=True):
    runtime = dict(deepcopy(RESOURCES), execution_mode='explicit_ssh_exception',
                   require_scheduler_membership=False)
    exception = dict(deepcopy(RESOURCES), scope=guard.SCOPE, explicit_human_authorized=True,
                     kernel_scheduler_membership=False, authorization='Explicit CPU4 approval.')
    config = {'family': guard.FAMILY, 'runtime': runtime}
    if override:
        config['resource_transition'] = {'kind': 'cpu_budget_only_v1'}
        runtime.update(execution_cpu_budget=4, cpu_affinity=CPUS.copy())
        exception.update(execution_cpu_budget=4, cpu_affinity=CPUS.copy())
    state = {'active_operation': {'label': guard.SCOPE, 'ssh_exception': exception}}
    return config, state, exception


class CPUBudgetTests(unittest.TestCase):
    def test_authorized_four_cpus_preserve_inputs_and_scheduler_grant(self):
        config, state, exception = fixture()
        before = deepcopy((config, state))
        with patch.object(guard.os, 'sched_getaffinity', create=True) as get_affinity:
            self.assertIs(guard.validate_ssh_exception(state, config), exception)
            self.assertEqual(guard.execution_cpu_binding(config, exception, set(range(224))), set(CPUS))
        get_affinity.assert_not_called()
        self.assertEqual((config, state), before)
        self.assertEqual(config['runtime']['cpu_slots'], 1)

    def test_four_cpus_require_an_explicit_exception(self):
        config, _, _ = fixture()
        for exception in (None, {}, False):
            with self.subTest(exception=exception), self.assertRaises(RuntimeError):
                guard.execution_cpu_binding(config, exception, set(range(224)))

    def test_exception_authorization_scope_and_flags_are_strict(self):
        changes = [('scope', 'other'), ('explicit_human_authorized', 1),
                   ('explicit_human_authorized', False), ('kernel_scheduler_membership', 0),
                   ('authorization', ''), ('authorization', ' '), ('authorization', None)]
        for key, value in changes:
            config, state, exception = fixture()
            exception[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                guard.validate_ssh_exception(state, config)
        config, state, _ = fixture()
        state['active_operation']['label'] = 'other'
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception(state, config)

    def test_cpu_fields_must_match_in_both_directions(self):
        for owner in ('runtime', 'exception'):
            for key, value in [('execution_cpu_budget', 3), ('execution_cpu_budget', True),
                               ('execution_cpu_budget', 4.0), ('cpu_affinity', [58, 63, 64, 66])]:
                config, state, exception = fixture()
                target = config['runtime'] if owner == 'runtime' else exception
                target[key] = value
                with self.subTest(owner=owner, key=key, value=value), self.assertRaises(RuntimeError):
                    guard.validate_ssh_exception(state, config)
            for key in guard.CPU_OVERRIDE_FIELDS:
                config, state, exception = fixture()
                target = config['runtime'] if owner == 'runtime' else exception
                del target[key]
                with self.subTest(owner=owner, missing=key), self.assertRaises(RuntimeError):
                    guard.validate_ssh_exception(state, config)

    def test_affinity_requires_sorted_distinct_nonnegative_integers(self):
        for value in ([58, 58, 64, 65], [65, 64, 63, 58], [-1, 63, 64, 65],
                      [False, 63, 64, 65], [58.0, 63, 64, 65], [58, 63, 64],
                      [58, 63, 64, 65, 66], tuple(CPUS), None):
            config, state, exception = fixture()
            config['runtime']['cpu_affinity'] = value
            exception['cpu_affinity'] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                guard.validate_ssh_exception(state, config)
        config, state, exception = fixture()
        config['runtime']['cpu_affinity'] = [0, 63, 64, 65]
        exception['cpu_affinity'] = [False, 63, 64, 65]
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception(state, config)

    def test_absent_transition_rejects_extra_cpu_fields_in_either_owner(self):
        for owner in ('runtime', 'exception'):
            for key, value in [('execution_cpu_budget', 1), ('cpu_affinity', [58])]:
                config, state, exception = fixture(override=False)
                target = config['runtime'] if owner == 'runtime' else exception
                target[key] = value
                with self.subTest(owner=owner, key=key), self.assertRaises(RuntimeError):
                    guard.validate_ssh_exception(state, config)

    def test_transition_kind_and_idxyz_family_are_required(self):
        for transition in (None, {}, {'kind': 'other'}, 'cpu_budget_only_v1'):
            config, state, _ = fixture()
            config['resource_transition'] = transition
            with self.subTest(transition=transition), self.assertRaises(RuntimeError):
                guard.validate_ssh_exception(state, config)
        config, state, _ = fixture()
        del config['resource_transition']
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception(state, config)
        config, state, _ = fixture()
        config['family'] = 'another_experiment'
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception(state, config)

    def test_all_previous_resource_limits_remain_fixed(self):
        changes = dict(allocation=130309881, host='pcg02i', physical_gpus=[5, 7],
                       cpu_slots=4, rss_limit_bytes=68719476737,
                       gpu_memory_fraction_limit=.9, keeper_pid=1090014)
        for key, value in changes.items():
            config, state, exception = fixture()
            config['runtime'][key] = exception[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_ssh_exception(state, config)
        config, state, exception = fixture()
        config['runtime']['cpu_slots'] = exception['cpu_slots'] = True
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception(state, config)

    def test_cpu_budget_override_is_ssh_only_and_must_be_allowed(self):
        for key, value in [('execution_mode', 'scheduler'), ('require_scheduler_membership', True)]:
            config, state, _ = fixture()
            config['runtime'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate_ssh_exception(state, config)
        config, _, exception = fixture()
        with self.assertRaises(RuntimeError):
            guard.execution_cpu_binding(config, exception, {58, 63, 64})

    def test_old_one_cpu_path_keeps_keeper_intersection(self):
        config, state, exception = fixture(override=False)
        with patch.object(guard.os, 'sched_getaffinity', return_value={63, 64}, create=True) as get_affinity:
            self.assertIs(guard.validate_ssh_exception(state, config), exception)
            self.assertEqual(guard.execution_cpu_binding(config, exception, {58, 63, 65}), {63})
        get_affinity.assert_called_once_with(1090013)
        with patch.object(guard.os, 'sched_getaffinity', create=True) as get_affinity:
            self.assertEqual(guard.execution_cpu_binding(config, None, {65, 58}), {58})
        get_affinity.assert_not_called()

    def test_empty_old_intersection_is_rejected(self):
        config, _, exception = fixture(override=False)
        with patch.object(guard.os, 'sched_getaffinity', return_value={0}, create=True):
            with self.assertRaises(RuntimeError):
                guard.execution_cpu_binding(config, exception, set(CPUS))


class ReachedPostCPUGate(Exception):
    pass


class GuardBindingTests(unittest.TestCase):
    def call_guard(self, affinity, override=True, in_job=False, slots='1'):
        config, state, _ = fixture(override=override)
        fake_resource = SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1,
                                       getrlimit=Mock(side_effect=ReachedPostCPUGate))
        with ExitStack() as stack:
            stack.enter_context(patch.object(guard.Path, 'read_text', return_value=json.dumps(state)))
            stack.enter_context(patch.object(guard, 'require_job_membership',
                                            side_effect=None if in_job else RuntimeError('SSH')))
            stack.enter_context(patch.object(guard, 'verify_exception_keeper'))
            stack.enter_context(patch.object(guard, 'verify_supervisor'))
            stack.enter_context(patch.object(guard.os, 'sched_getaffinity', return_value=affinity, create=True))
            stack.enter_context(patch.object(guard.socket, 'gethostname', return_value='pcg01'))
            stack.enter_context(patch.dict(guard.os.environ, {guard.EXCEPTION_ENV: '1',
                'CUDA_VISIBLE_DEVICES': '', 'JOB_ID': '130309880', 'NSLOTS': slots}, clear=True))
            stack.enter_context(patch.dict(sys.modules, {'resource': fake_resource}))
            guard.verify_launch(SimpleNamespace(mode='audit'), config)

    def test_override_requires_exact_configured_affinity(self):
        with self.assertRaises(ReachedPostCPUGate):
            self.call_guard(set(CPUS))
        for actual in ({58}, {58, 63, 64}, {58, 63, 64, 66}, {58, 63, 64, 65, 66}):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                self.call_guard(actual)

    def test_old_guard_still_requires_one_cpu(self):
        with self.assertRaises(ReachedPostCPUGate):
            self.call_guard({58}, override=False)
        with self.assertRaises(RuntimeError):
            self.call_guard(set(CPUS), override=False)

    def test_normal_job_cannot_use_ssh_override(self):
        with self.assertRaises(RuntimeError):
            self.call_guard(set(CPUS), in_job=True)
        with self.assertRaises(ReachedPostCPUGate):
            self.call_guard({58}, override=False, in_job=True)
        for slots in ('0', '2', '4'):
            with self.subTest(slots=slots), self.assertRaises(RuntimeError):
                self.call_guard({58}, override=False, in_job=True, slots=slots)


class CommonLauncherTests(unittest.TestCase):
    def test_provider_receives_original_allowance_without_keeper_intersection(self):
        config, _, exception = fixture()
        allowed = {0, *CPUS}
        with patch.object(launcher.os, 'sched_getaffinity', return_value=allowed, create=True) as get_affinity:
            self.assertEqual(launcher.select_cpu_binding(guard, config, exception), set(CPUS))
        get_affinity.assert_called_once_with(0)

    def test_generic_legacy_binding_algorithm_is_unchanged(self):
        checks = SimpleNamespace()
        config = {'runtime': {'cpu_slots': 1}}
        with patch.object(launcher.os, 'sched_getaffinity', side_effect=[{1, 4, 6}, {4, 6}], create=True) as get_affinity:
            self.assertEqual(launcher.select_cpu_binding(checks, config, {'keeper_pid': 99}), {4})
        self.assertEqual([c.args[0] for c in get_affinity.call_args_list], [0, 99])
        config['runtime']['cpu_slots'] = 2
        with patch.object(launcher.os, 'sched_getaffinity', return_value={6, 1, 4}, create=True):
            self.assertEqual(launcher.select_cpu_binding(checks, config, None), {1, 4})
        with patch.object(launcher.os, 'sched_getaffinity', side_effect=[{1, 4}, {4}], create=True):
            with self.assertRaises(RuntimeError):
                launcher.select_cpu_binding(checks, config, {'keeper_pid': 99})

    def test_provider_cannot_escape_actual_allowance_or_authorization(self):
        config, _, exception = fixture()
        for value in ({58, 63, 64, 224}, {58}, CPUS, {False, 63, 64, 65}):
            checks = SimpleNamespace(execution_cpu_binding=Mock(return_value=value))
            with self.subTest(value=value), patch.object(launcher.os, 'sched_getaffinity', return_value=set(range(224)), create=True):
                with self.assertRaises(RuntimeError):
                    launcher.select_cpu_binding(checks, config, exception)
        checks = SimpleNamespace(execution_cpu_binding=Mock(return_value=set(CPUS)))
        with patch.object(launcher.os, 'sched_getaffinity', return_value=set(range(224)), create=True):
            with self.assertRaises(RuntimeError):
                launcher.select_cpu_binding(checks, config, None)

    def run_mocked_supervisor(self, escape=False):
        config, state, _ = fixture()
        config['runtime'].update(repo=str(ROOT), unirig_root='unirig', opt_config_root='opt',
                                 extra_pythonpath='extra', python='python')
        config_path, output_path = ROOT/'config.json', ROOT/'not_created'
        checks = SimpleNamespace(require_job_membership=Mock(side_effect=RuntimeError('SSH')),
                                 validate_ssh_exception=guard.validate_ssh_exception,
                                 verify_exception_keeper=Mock(), verify_launch=Mock(return_value='source'),
                                 execution_cpu_binding=guard.execution_cpu_binding)
        child = Mock(pid=900, returncode=0)
        child.poll.side_effect = [None, 0] if escape else [0, 0]
        def read(path, *args, **kwargs):
            if path == config_path:
                return json.dumps(config)
            if path.name == 'current.json':
                return json.dumps(state)
            return '1:devices:/user.slice\n2:cpuset:/\n'
        def affinity(pid):
            return set(range(224)) if pid == 0 else ({58} if pid == 901 else {66})
        grant = ('owner: wangyy\njob_state 1: r\nexec_host_list 1: pcg01i:1\n'
                 'granted_devices 1,0: pcg01i: /dev/nvidia5 /dev/nvidia6\n')
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher.sys, 'argv', ['launcher', '--config', str(config_path),
                '--output', str(output_path), '--mode', 'audit']))
            stack.enter_context(patch.object(launcher.sys, 'path', launcher.sys.path.copy()))
            stack.enter_context(patch.dict(launcher.os.environ, {}, clear=True))
            stack.enter_context(patch.object(launcher.importlib, 'import_module', return_value=checks))
            stack.enter_context(patch.object(launcher.Path, 'read_text', read))
            stack.enter_context(patch.object(launcher.Path, 'exists', return_value=False))
            mkdir = stack.enter_context(patch.object(launcher.Path, 'mkdir'))
            stack.enter_context(patch.object(launcher.os, 'sched_getaffinity', side_effect=affinity, create=True))
            bind = stack.enter_context(patch.object(launcher.os, 'sched_setaffinity', create=True))
            libc = stack.enter_context(patch.object(launcher.ctypes, 'CDLL'))
            libc.return_value.prctl.return_value = 0
            popen = stack.enter_context(patch.object(launcher.subprocess, 'Popen', return_value=child))
            stack.enter_context(patch.object(launcher.subprocess, 'check_output', return_value=grant))
            stack.enter_context(patch.object(launcher.signal, 'SIGHUP', 1, create=True))
            stack.enter_context(patch.object(launcher.signal, 'signal'))
            stack.enter_context(patch.object(launcher, 'descendants', return_value={900: (1, 100)}))
            stack.enter_context(patch.object(launcher, 'verify_child_cgroups', return_value=True))
            tasks = [Path('/proc/900/task/901'), Path('/proc/900/task/902')]
            stack.enter_context(patch.object(launcher.Path, 'iterdir', return_value=iter(tasks)))
            cleanup = stack.enter_context(patch.object(launcher, 'cleanup_payload', return_value=[]))
            write = stack.enter_context(patch.object(launcher, 'write_json'))
            if escape:
                with self.assertRaisesRegex(RuntimeError, 'child thread escaped'):
                    launcher.main(trainer_module_name='id_xyz_launch_guard', modes=('audit',), cpu_modes=('audit',))
            else:
                self.assertEqual(launcher.main(trainer_module_name='id_xyz_launch_guard', modes=('audit',), cpu_modes=('audit',)), 0)
            bind.assert_called_once_with(0, set(CPUS))
            popen.assert_called_once()
            mkdir.assert_called_once()
            cleanup.assert_called_once_with(child, 1090013)
            receipt = write.call_args.args[1]
            self.assertEqual(receipt['cpu_affinity'], CPUS)
            self.assertEqual(receipt['scheduler_cpu_slots'], 1)
            self.assertEqual(receipt['execution_cpu_budget'], 4)
            self.assertIs(receipt['kernel_scheduler_membership'], False)
            self.assertEqual(receipt['rss_limit_bytes'], 68719476736)
            self.assertEqual(receipt['physical_gpus'], [5, 6])
            self.assertEqual(receipt['passed'], not escape)
            self.assertIs(receipt['payload_cleanup_complete'], True)

    def test_supervisor_receipt_records_actual_binding_and_original_grant(self):
        self.run_mocked_supervisor()

    def test_supervisor_checks_every_descendant_thread_against_chosen_binding(self):
        self.run_mocked_supervisor(escape=True)


if __name__ == '__main__':
    unittest.main()
