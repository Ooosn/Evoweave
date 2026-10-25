"""CPU-only admission and supervisor regressions; all external operations mocked."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
sys.path.insert(0, str(ROOT / 'model_training/jobs'))
import virtual_pair_guard as guard
import launch_legacy_single_target as launcher


def fixture(legacy=False):
    runtime = dict(execution_mode='scheduler_job', require_scheduler_membership=True,
        allocation=130400001, host='pcg02i', hostname='pcg02', physical_gpus=[0, 2],
        gpu_uuids=['GPU-first', 'GPU-second'], cpu_slots=16, execution_cpu_budget=16,
        cpu_affinity=list(range(16)), world_size=2, rss_limit_bytes=68719476736,
        gpu_memory_fraction_limit=.85, repo=str(ROOT), output_root=str(ROOT / 'unused_outputs'),
        stop_before_utc='2099-01-01T00:00:00Z', python='python', unirig_root='unirig',
        opt_config_root='opt', extra_pythonpath='extra')
    config = {'family': guard.SCOPE, 'runtime': runtime}
    args = SimpleNamespace(config=ROOT / 'model_training/experiments/unused_scheduler.json',
                           output=ROOT / 'unused_outputs/stage', mode='train')
    op = dict(label=guard.SCOPE, explicit_human_authorized=True, complete=False,
        required_entrypoint=guard.ENTRYPOINT, gpu_work_allowed=True, formal_training_allowed=True,
        CPU_audit_accepted=True, capacity_accepted=True, code_commit='approved-code',
        runs=[dict(mode='train', config=str(args.config), output=str(args.output),
                   approved=True, config_sha256='config-digest')])
    if legacy:
        runtime.update(execution_mode='explicit_ssh_exception', require_scheduler_membership=False,
            allocation=130309880, host='pcg01i', hostname='pcg01', physical_gpus=[5, 6],
            cpu_slots=1, execution_cpu_budget=4, cpu_affinity=list(guard.CPUS), keeper_pid=1090013)
        names = ('allocation', 'host', 'physical_gpus', 'cpu_slots', 'execution_cpu_budget',
                 'cpu_affinity', 'rss_limit_bytes', 'gpu_memory_fraction_limit', 'keeper_pid')
        op['ssh_exception'] = dict({name: deepcopy(runtime[name]) for name in names},
            scope=guard.SCOPE, explicit_human_authorized=True, kernel_scheduler_membership=False,
            authorization='Historical explicit bounded SSH execution.')
    return config, {'active_operation': op}, args


def cgroups(config):
    job = config['runtime']['allocation']
    return f'12:devices:/AGE/{job}.1\n7:cpuset:/AGE/{job}.1\n'


def grant_text(config):
    runtime = config['runtime']
    devices = ' '.join(f'/dev/nvidia{gpu}' for gpu in runtime['physical_gpus'])
    return (f"job_number: {runtime['allocation']}\nowner: wangyy\njob_state 1: r\n"
            f"exec_host_list 1: {runtime['host']}:{runtime['cpu_slots']}\n"
            + ''.join(f"granted_devices 1,{slot}: {runtime['host']}: {devices}\n"
                      for slot in range(runtime['cpu_slots'])))


class SchedulerGuardTests(unittest.TestCase):
    def run_guard(self, config=None, state=None, args=None, *, affinity=None, membership=None,
                  environment=None, hostname=None, grant=None, uuids=None, pid=100, dirty=False):
        if config is None:
            config, state, args = fixture()
        runtime = config['runtime']
        env = dict(JOB_ID=str(runtime['allocation']), NSLOTS=str(runtime['cpu_slots']),
            EVOWEAVE_RESOURCE_SUPERVISOR_PID='100',
            CUDA_VISIBLE_DEVICES=','.join(map(str, runtime['physical_gpus'])))
        if runtime['execution_mode'] == 'explicit_ssh_exception':
            env[guard.EXCEPTION_ENV] = '1'
        env.update(environment or {})

        def read(path, *unused, **kwargs):
            if path.name == 'current.json':
                return json.dumps(state)
            if path.name == 'cgroup':
                return cgroups(config) if membership is None else membership
            if path.name == 'status':
                return 'PPid:\t100\n'
            raise AssertionError(f'unexpected read: {path}')

        def output(command, **kwargs):
            if command[0] == 'qstat':
                return grant_text(config) if grant is None else grant
            if command[0] == 'nvidia-smi':
                if '--query-gpu=uuid' in command:
                    self.assertEqual(command[1:3], ['-i', ','.join(map(str, runtime['physical_gpus']))])
                    return '\n'.join(runtime['gpu_uuids'] if uuids is None else uuids)
                return ''
            if 'status' in command:
                return ' M changed.py' if dirty else ''
            if 'rev-parse' in command:
                return 'new-source'
            raise AssertionError(f'unexpected command: {command}')

        resource = SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1, getrlimit=Mock(return_value=(-1, -1)))
        with ExitStack() as stack:
            stack.enter_context(patch.object(guard.Path, 'read_text', read))
            stack.enter_context(patch.object(guard.Path, 'read_bytes', return_value=str(ROOT / guard.LAUNCHER).encode()))
            stack.enter_context(patch.object(guard, 'file_sha256', return_value='config-digest'))
            keeper = stack.enter_context(patch.object(guard, 'verify_exception_keeper'))
            stack.enter_context(patch.object(guard.os, 'sched_getaffinity', create=True,
                return_value=set(runtime['cpu_affinity']) if affinity is None else affinity))
            stack.enter_context(patch.object(guard.os, 'getpid', return_value=pid))
            stack.enter_context(patch.object(guard.socket, 'gethostname', return_value=hostname or runtime.get('hostname', 'pcg01')))
            stack.enter_context(patch.dict(guard.os.environ, env, clear=True))
            stack.enter_context(patch.dict(sys.modules, {'resource': resource}))
            stack.enter_context(patch.object(guard.subprocess, 'check_output', side_effect=output))
            source_checks = stack.enter_context(patch.object(guard.subprocess, 'run'))
            value = guard.verify_launch(args, config)
            self.assertEqual(value, 'new-source')
            self.assertEqual(source_checks.call_count, 3)
            self.assertEqual(keeper.call_count, int(runtime['execution_mode'] == 'explicit_ssh_exception'))

    def test_scheduler_supervisor_and_descendant_accept_exact_sixteen_slots(self):
        self.run_guard()
        self.run_guard(pid=101)

    def test_reported_grant_accepts_master_cgroup_and_selected_sixteen_cpus(self):
        config, state, args = fixture()
        config['runtime'].update(allocation=130494177, host='pcg01i', hostname='pcg01',
            physical_gpus=[3, 4], gpu_uuids=['GPU-1fb4221b-91b0-0df6-8615-b36fbdd55bf7',
                                          'GPU-0544830a-4dac-9fe3-a099-95ddc2240b72'],
            cpu_affinity=[6, 12, 13, 15, 17, 18, 19, 20, 74, 79, 82, 87, 88, 95, 97, 108])
        membership = cgroups(config).replace('.1\n', '.1/master\n')
        self.assertEqual(guard.execution_cpu_binding(config, None, set(range(224))),
                         set(config['runtime']['cpu_affinity']))
        self.run_guard(config, state, args, membership=membership)
        self.run_guard(config, state, args, membership=membership, pid=101)
        with self.assertRaises(RuntimeError):
            self.run_guard(config, state, args, membership=membership, affinity=set(range(224)))

    def test_cgroup_paths_are_exact_and_controllers_agree(self):
        root = '/AGE/130494177.1'
        for path in (root, root + '/master'):
            guard.require_job_membership(f'12:devices:{path}\n7:cpuset:{path}\n', 130494177)
        for path in ('/AGE/130494178.1/master', root + '0/master', root + '/master-other',
                     root + '/master/child', root + '/other'):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                guard.require_job_membership(f'12:devices:{path}\n7:cpuset:{path}\n', 130494177)
        for text in (f'12:devices:{root}/master\n7:cpuset:{root}\n',
                     f'12:devices:{root}/master\n',
                     f'12:devices:{root}\n7:cpuset:{root}\n8:devices:{root}/master\n'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                guard.require_job_membership(text, 130494177)

    def test_hostname_override_and_documented_pcg01_alias(self):
        config, state, args = fixture()
        config['runtime']['hostname'] = 'compute-special'
        self.run_guard(config, state, args, hostname='compute-special.example')
        config['runtime']['host'] = 'pcg01i'
        del config['runtime']['hostname']
        self.run_guard(config, state, args, hostname='pcg01')
        with self.assertRaises(RuntimeError):
            self.run_guard(config, state, args, hostname='pcg02')

    def test_scheduler_requires_real_device_and_cpuset_membership(self):
        config, _, _ = fixture()
        texts = ['12:devices:/user.slice\n7:cpuset:/\n',
                 cgroups(config).replace('12:devices:/AGE/130400001.1', '12:devices:/AGE/1.1'),
                 cgroups(config).replace('7:cpuset:/AGE/130400001.1', '7:cpuset:/AGE/1.1')]
        for text in texts:
            with self.subTest(cgroup=text), self.assertRaises(RuntimeError):
                self.run_guard(membership=text)

    def test_inherited_job_slots_and_no_ssh_exception_are_required(self):
        for env in ({'JOB_ID': ''}, {'JOB_ID': '130309880'}, {'NSLOTS': '1'},
                    {'NSLOTS': '4'}, {'NSLOTS': ''}, {guard.EXCEPTION_ENV: '1'}):
            with self.subTest(environment=env), self.assertRaises(RuntimeError):
                self.run_guard(environment=env)

    def test_invalid_scheduler_config_is_rejected(self):
        for key, value in [('cpu_slots', 1), ('cpu_slots', 4), ('execution_cpu_budget', 4),
                           ('cpu_slots', 16.0), ('require_scheduler_membership', False),
                           ('world_size', 1), ('physical_gpus', [0, 0]),
                           ('cpu_affinity', list(range(15))), ('cpu_affinity', [0] * 16),
                           ('cpu_affinity', [False] + list(range(1, 16))),
                           ('gpu_uuids', ['same', 'same'])]:
            config, state, args = fixture()
            config['runtime'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.run_guard(config, state, args)

    def test_configured_cpus_must_be_available_and_actual_binding_exact(self):
        config, _, _ = fixture()
        self.assertEqual(guard.execution_cpu_binding(config, None, set(range(32))), set(range(16)))
        with self.assertRaises(RuntimeError):
            guard.execution_cpu_binding(config, {'keeper_pid': 1090013}, set(range(32)))
        for affinity in (set(range(15)), set(range(17)), set(range(1, 17))):
            with self.subTest(affinity=affinity), self.assertRaises(RuntimeError):
                self.run_guard(affinity=affinity)

    def test_live_grant_requires_exact_job_owner_state_host_slots_devices(self):
        config, _, _ = fixture()
        good = grant_text(config)
        changes = [('130400001', '130400002'), ('owner: wangyy', 'owner: other'),
                   ('job_state 1: r', 'job_state 1: qw'), ('pcg02i:16', 'pcg02i:1'),
                   ('pcg02i:16', 'pcg03i:16'), ('/dev/nvidia2', '/dev/nvidia3'),
                   ('/dev/nvidia2', '/dev/nvidia0'),
                   ('/dev/nvidia2', '/dev/nvidia2 /dev/nvidia3')]
        for old, new in changes:
            with self.subTest(change=(old, new)), self.assertRaises(RuntimeError):
                self.run_guard(grant=good.replace(old, new))
        with self.assertRaises(RuntimeError):
            self.run_guard(grant=good + 'exec_host_list 2: pcg03i:16\n')

    def test_per_slot_device_records_require_every_identity_exactly_once(self):
        config, _, _ = fixture()
        lines = grant_text(config).splitlines()
        header, records = lines[:4], lines[4:]
        self.assertEqual(len(records), 16)
        guard.verify_allocation(config, '\n'.join(header + records[::-1]))
        bad_records = [records[:slot] + records[slot + 1:] for slot in range(16)]
        bad_records += [records + [records[0]], records[:-1] + [records[14]],
                        records[:-1] + [records[15].replace('1,15:', '1,16:')],
                        records[:-1] + [records[15].replace('1,15:', '2,15:')],
                        records[:-1] + ['granted_devices malformed'],
                        records + ['granted_devices malformed'], records[:1]]
        for values in bad_records:
            with self.subTest(records=values), self.assertRaises(RuntimeError):
                guard.verify_allocation(config, '\n'.join(header + values))

    def test_mismatch_in_any_one_device_record_is_rejected(self):
        config, _, _ = fixture()
        lines = grant_text(config).splitlines()
        for slot in range(16):
            for old, new in [('pcg02i', 'pcg03i'), ('/dev/nvidia2', '/dev/nvidia3'),
                             ('/dev/nvidia2', '/dev/nvidia0')]:
                changed = list(lines)
                changed[4 + slot] = changed[4 + slot].replace(old, new)
                with self.subTest(slot=slot, changed=new), self.assertRaises(RuntimeError):
                    guard.verify_allocation(config, '\n'.join(changed))

    def test_gpu_visibility_and_uuid_order_match_configuration(self):
        with self.assertRaises(RuntimeError):
            self.run_guard(environment={'CUDA_VISIBLE_DEVICES': '5,6'})
        for uuids in (['GPU-second', 'GPU-first'], ['GPU-first', 'GPU-other']):
            with self.subTest(uuids=uuids), self.assertRaises(RuntimeError):
                self.run_guard(uuids=uuids)

    def test_historical_ssh_path_still_requires_its_exact_exception(self):
        self.run_guard(*fixture(legacy=True))
        config, state, args = fixture(legacy=True)
        with self.assertRaises(RuntimeError):
            self.run_guard(config, state, args, environment={guard.EXCEPTION_ENV: ''})
        config['runtime']['cpu_slots'] = 16
        with self.assertRaises(RuntimeError):
            self.run_guard(config, state, args)
        config, state, _ = fixture()
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception(state, config)

    def test_source_and_stage_gates_remain_enforced(self):
        with self.assertRaises(RuntimeError):
            self.run_guard(dirty=True)
        for key, value in [('required_entrypoint', 'other.py'), ('formal_training_allowed', False),
                           ('capacity_accepted', False), ('CPU_audit_accepted', False)]:
            config, state, args = fixture()
            state['active_operation'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.run_guard(config, state, args)


class SupervisorTests(unittest.TestCase):
    def test_provider_controls_allocation_and_legacy_fallback_is_unchanged(self):
        config, _, _ = fixture()
        provider = Mock(wraps=guard.verify_allocation)
        checks = SimpleNamespace(verify_allocation=provider)
        launcher.verify_allocation(checks, config, grant_text(config))
        provider.assert_called_once_with(config, grant_text(config))
        with self.assertRaises(RuntimeError):
            launcher.verify_allocation(checks, config, grant_text(config).replace('pcg02i:16', 'pcg02i:1'))
        legacy, _, _ = fixture(legacy=True)
        launcher.verify_allocation(SimpleNamespace(), legacy, grant_text(legacy))
        with self.assertRaises(RuntimeError):
            launcher.verify_allocation(SimpleNamespace(), config, grant_text(config))

    def test_memory_monitor_uses_configured_devices_and_fraction(self):
        config, _, _ = fixture()
        config['runtime']['gpu_memory_fraction_limit'] = .6
        with patch.object(launcher.subprocess, 'check_output', return_value='0, 600, 1000\n2, 599, 1000\n') as call:
            self.assertEqual(launcher.read_gpu_memory(config), [[0, 600, 1000], [2, 599, 1000]])
        self.assertEqual(call.call_args.args[0][1:3], ['-i', '0,2'])
        for text in ('0, 601, 1000\n2, 599, 1000\n', '5, 0, 1000\n6, 0, 1000\n'):
            with patch.object(launcher.subprocess, 'check_output', return_value=text), self.assertRaises(RuntimeError):
                launcher.read_gpu_memory(config)
        legacy, _, _ = fixture(legacy=True)
        with patch.object(launcher.subprocess, 'check_output', return_value='5, 850, 1000\n6, 850, 1000\n'):
            launcher.read_gpu_memory(legacy)
        with patch.object(launcher.subprocess, 'check_output', return_value='5, 851, 1000\n6, 850, 1000\n'):
            with self.assertRaises(RuntimeError):
                launcher.read_gpu_memory(legacy)

    def test_missing_scheduler_membership_never_enters_ssh_fallback(self):
        config, _, args = fixture()
        checks = SimpleNamespace(require_job_membership=guard.require_job_membership,
            validate_ssh_exception=Mock(), verify_exception_keeper=Mock(), verify_launch=Mock())
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher.sys, 'argv', ['launcher', '--config', str(args.config),
                '--output', str(args.output), '--mode', 'train']))
            stack.enter_context(patch.object(launcher.sys, 'path', launcher.sys.path.copy()))
            stack.enter_context(patch.object(launcher.importlib, 'import_module', return_value=checks))
            stack.enter_context(patch.object(launcher.Path, 'read_text',
                side_effect=[json.dumps(config), '12:devices:/user.slice\n7:cpuset:/\n']))
            popen = stack.enter_context(patch.object(launcher.subprocess, 'Popen'))
            with self.assertRaisesRegex(RuntimeError, 'actual job membership'):
                launcher.main()
            checks.validate_ssh_exception.assert_not_called()
            checks.verify_exception_keeper.assert_not_called()
            popen.assert_not_called()

    def test_scheduler_supervisor_records_membership_and_uses_live_provider(self):
        config, _, args = fixture()
        runtime = config['runtime']
        checks = SimpleNamespace(require_job_membership=guard.require_job_membership,
            validate_ssh_exception=Mock(), verify_exception_keeper=Mock(), verify_launch=Mock(return_value='new-source'),
            execution_cpu_binding=guard.execution_cpu_binding, verify_allocation=Mock(wraps=guard.verify_allocation))
        child = Mock(pid=900, returncode=0)
        child.poll.side_effect = [None, 0, 0, 0]

        def output(command, **kwargs):
            if command[0] == 'qstat':
                return grant_text(config)
            if '--query-gpu=index,memory.used,memory.total' in command:
                self.assertEqual(command[1:3], ['-i', '0,2'])
                return '0, 0, 1000\n2, 0, 1000\n'
            return ''

        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher.sys, 'argv', ['launcher', '--config', str(args.config),
                '--output', str(args.output), '--mode', 'train']))
            stack.enter_context(patch.object(launcher.sys, 'path', launcher.sys.path.copy()))
            stack.enter_context(patch.dict(launcher.os.environ,
                {'JOB_ID': str(runtime['allocation']), 'NSLOTS': '16'}, clear=True))
            stack.enter_context(patch.object(launcher.importlib, 'import_module', return_value=checks))
            stack.enter_context(patch.object(launcher.Path, 'read_text', side_effect=[json.dumps(config), cgroups(config)]))
            stack.enter_context(patch.object(launcher.Path, 'exists', return_value=False))
            stack.enter_context(patch.object(launcher.Path, 'mkdir'))
            stack.enter_context(patch.object(launcher.os, 'sched_getaffinity', create=True,
                side_effect=lambda pid: set(range(32)) if pid == 0 else set(range(16))))
            bind = stack.enter_context(patch.object(launcher.os, 'sched_setaffinity', create=True))
            libc = stack.enter_context(patch.object(launcher.ctypes, 'CDLL'))
            libc.return_value.prctl.return_value = 0
            stack.enter_context(patch.object(launcher.subprocess, 'Popen', return_value=child))
            stack.enter_context(patch.object(launcher.subprocess, 'check_output', side_effect=output))
            stack.enter_context(patch.object(launcher.signal, 'SIGHUP', 1, create=True))
            stack.enter_context(patch.object(launcher.signal, 'signal'))
            stack.enter_context(patch.object(launcher, 'descendants', return_value={900: (100, 1000)}))
            groups = stack.enter_context(patch.object(launcher, 'verify_child_cgroups', return_value=True))
            stack.enter_context(patch.object(launcher.Path, 'iterdir', return_value=[Path('/proc/900/task/901')]))
            stack.enter_context(patch.object(launcher, 'cpu_ticks', return_value=10))
            stack.enter_context(patch.object(launcher.os, 'sysconf', return_value=100, create=True))
            cleanup = stack.enter_context(patch.object(launcher, 'cleanup_payload', return_value=[]))
            write = stack.enter_context(patch.object(launcher, 'write_json'))
            self.assertEqual(launcher.main(trainer_module_name='virtual_pair_guard'), 0)
            bind.assert_called_once_with(0, set(range(16)))
            checks.verify_allocation.assert_called_once_with(config, grant_text(config))
            checks.validate_ssh_exception.assert_not_called()
            groups.assert_called_once_with(900, launcher.resource_cgroups(cgroups(config)))
            cleanup.assert_called_once_with(child, None)
            receipt = write.call_args.args[1]
            self.assertIs(receipt['kernel_scheduler_membership'], True)
            self.assertIsNone(receipt['ssh_exception'])
            self.assertEqual(receipt['scheduler_cpu_slots'], 16)
            self.assertEqual(receipt['execution_cpu_budget'], 16)
            self.assertEqual(receipt['physical_gpus'], [0, 2])
            self.assertTrue(receipt['passed'])
            self.assertTrue(receipt['payload_cleanup_complete'])


if __name__ == '__main__':
    unittest.main()
