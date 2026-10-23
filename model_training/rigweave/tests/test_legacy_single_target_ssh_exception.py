"""Pure CPU checks for the explicitly scoped single-target SSH exception."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))

import train_legacy_single_target as launch


SCOPE = 'legacy_single_target_100000_20261024_v1'
JOB = 130309880
ORDINARY_SSH = '12:devices:/user.slice\n7:cpuset:/\n'
RESOURCES = {
    'allocation': JOB,
    'host': 'pcg01i',
    'physical_gpus': [5, 6],
    'cpu_slots': 1,
    'rss_limit_bytes': 68719476736,
    'gpu_memory_fraction_limit': .85,
}


def fixture():
    exception = {
        **deepcopy(RESOURCES),
        'explicit_human_authorized': True,
        'kernel_scheduler_membership': False,
        'keeper_pid': 1090013,
        'scope': SCOPE,
        'authorization': 'Explicit human authorization for this run only.',
    }
    state = {'active_operation': {'label': SCOPE, 'ssh_exception': exception}}
    config = {'runtime': {
        **deepcopy(RESOURCES),
        'keeper_pid': 1090013,
        'execution_mode': 'explicit_ssh_exception',
        'require_scheduler_membership': False,
    }}
    return state, config


class SSHExceptionTests(unittest.TestCase):
    def setUp(self):
        self.validate = getattr(launch, 'validate_ssh_exception', None)
        self.assertTrue(callable(self.validate), 'Pending primary validate_ssh_exception implementation')

    def test_exact_authorized_exception_is_accepted_without_mutation(self):
        state, config = fixture()
        before = deepcopy((state, config))
        result = self.validate(state, config)
        self.assertEqual(result, state['active_operation']['ssh_exception'])
        self.assertEqual((state, config), before)

    def test_missing_exception_or_operation_cannot_authorize_ordinary_SSH(self):
        state, config = fixture()
        for candidate in ({}, {'active_operation': {}},
                          {'active_operation': {'label': SCOPE}},
                          {'active_operation': {'label': SCOPE, 'ssh_exception': None}}):
            with self.subTest(state=candidate), self.assertRaises(RuntimeError):
                self.validate(candidate, config)
        with self.assertRaises(RuntimeError):
            launch.require_job_membership(ORDINARY_SSH, JOB)

    def test_missing_or_invalid_authorization_text_is_rejected(self):
        for authorization in (None, '', '   ', False):
            state, config = fixture()
            state['active_operation']['ssh_exception']['authorization'] = authorization
            with self.subTest(authorization=authorization), self.assertRaises(RuntimeError):
                self.validate(state, config)
        state, config = fixture()
        del state['active_operation']['ssh_exception']['authorization']
        with self.assertRaises(RuntimeError):
            self.validate(state, config)

    def test_authorization_and_membership_flags_are_explicit_booleans(self):
        for key, bad_values in (
                ('explicit_human_authorized', (False, None, 1, 'true')),
                ('kernel_scheduler_membership', (True, None, 0, 'false'))):
            for value in bad_values:
                state, config = fixture()
                state['active_operation']['ssh_exception'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    self.validate(state, config)
            state, config = fixture()
            del state['active_operation']['ssh_exception'][key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                self.validate(state, config)

    def test_operation_label_and_exception_scope_are_both_required(self):
        for owner, key in (('active_operation', 'label'), ('ssh_exception', 'scope')):
            for value in (None, '', 'another_run', SCOPE + '_copy'):
                state, config = fixture()
                target = state['active_operation']
                if owner == 'ssh_exception':
                    target = target['ssh_exception']
                target[key] = value
                with self.subTest(owner=owner, value=value), self.assertRaises(RuntimeError):
                    self.validate(state, config)

    def test_each_wrong_or_missing_exception_resource_is_rejected(self):
        changes = {'allocation': JOB + 1, 'host': 'pcg02i', 'physical_gpus': [4, 5],
                   'cpu_slots': 2, 'rss_limit_bytes': 68719476737,
                   'gpu_memory_fraction_limit': .90, 'keeper_pid': 1090014}
        for key, value in changes.items():
            for missing in (False, True):
                state, config = fixture()
                exception = state['active_operation']['ssh_exception']
                if missing:
                    del exception[key]
                else:
                    exception[key] = value
                with self.subTest(key=key, missing=missing), self.assertRaises(RuntimeError):
                    self.validate(state, config)

    def test_runtime_resources_must_match_and_cannot_coordinately_expand_scope(self):
        changes = {'allocation': JOB + 1, 'host': 'pcg02i', 'physical_gpus': [6, 7],
                   'cpu_slots': 2, 'rss_limit_bytes': 137438953472,
                   'gpu_memory_fraction_limit': .90}
        for key, value in changes.items():
            for change_both in (False, True):
                state, config = fixture()
                config['runtime'][key] = deepcopy(value)
                if change_both:
                    state['active_operation']['ssh_exception'][key] = deepcopy(value)
                with self.subTest(key=key, change_both=change_both), self.assertRaises(RuntimeError):
                    self.validate(state, config)

    def test_GPU_order_duplicates_and_extra_cards_are_rejected(self):
        for value in ([6, 5], [5, 5], [5], [5, 6, 7], None):
            state, config = fixture()
            state['active_operation']['ssh_exception']['physical_gpus'] = value
            with self.subTest(gpus=value), self.assertRaises(RuntimeError):
                self.validate(state, config)

    def test_runtime_requires_explicit_exception_mode_and_false_membership_policy(self):
        for key, bad_values in (
                ('execution_mode', ('scheduler', '', None)),
                ('require_scheduler_membership', (True, None, 0, 'false'))):
            for value in bad_values:
                state, config = fixture()
                config['runtime'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    self.validate(state, config)
            state, config = fixture()
            del config['runtime'][key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                self.validate(state, config)

    def test_valid_exception_does_not_weaken_ordinary_membership_helper(self):
        state, config = fixture()
        self.validate(state, config)
        with self.assertRaises(RuntimeError):
            launch.require_job_membership(ORDINARY_SSH, JOB)


class OrdinaryMembershipTests(unittest.TestCase):
    def test_actual_membership_still_requires_both_matching_controllers(self):
        matching = f'12:devices:/AGE/{JOB}.1\n7:cpuset:/AGE/{JOB}.1\n'
        launch.require_job_membership(matching, JOB)
        for text in (ORDINARY_SSH,
                     f'12:devices:/AGE/{JOB}.1\n7:cpuset:/\n',
                     f'12:devices:/AGE/{JOB+1}.1\n7:cpuset:/AGE/{JOB+1}.1\n'):
            with self.subTest(cgroup=text), self.assertRaises(RuntimeError):
                launch.require_job_membership(text, JOB)


if __name__ == '__main__':
    unittest.main()
