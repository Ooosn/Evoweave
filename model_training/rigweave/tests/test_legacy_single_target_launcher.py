import importlib.util
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import Mock, patch

JOBS = Path(__file__).resolve().parents[2] / 'jobs'
sys.path.insert(0, str(JOBS))
spec = importlib.util.spec_from_file_location('legacy_launcher', JOBS / 'launch_legacy_single_target.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherCleanupTests(unittest.TestCase):
    def setUp(self):
        kill_signal = patch.object(launcher.signal, 'SIGKILL', 9, create=True)
        kill_signal.start()
        self.addCleanup(kill_signal.stop)

    def test_resource_cgroups_exclude_unrelated_controllers(self):
        self.assertEqual(launcher.resource_cgroups('1:devices:/user.slice\n2:cpuset:/\n3:cpu:/other'),
                         {'devices': '/user.slice', 'cpuset': '/'})

    def test_live_resource_cgroup_match(self):
        with patch.object(launcher, 'process_identity', return_value=(1, 'R')), \
                patch.object(launcher.Path, 'read_text', return_value='1:devices:/user.slice\n2:cpuset:/'):
            self.assertTrue(launcher.verify_child_cgroups(20, {'devices':'/user.slice','cpuset':'/'}))

    def test_existing_zombie_is_not_subject_to_live_cgroup_check(self):
        with patch.object(launcher, 'process_identity', return_value=(1, 'Z')), \
                patch.object(launcher.Path, 'read_text') as read:
            self.assertFalse(launcher.verify_child_cgroups(20, {'devices':'/user.slice'}))
        read.assert_not_called()

    def test_exit_race_does_not_misreport_root_cgroup_as_escape(self):
        with patch.object(launcher, 'process_identity', side_effect=[(1, 'R'), (1, 'Z')]), \
                patch.object(launcher.Path, 'read_text', return_value='1:devices:/\n2:cpuset:/'):
            self.assertFalse(launcher.verify_child_cgroups(20, {'devices':'/user.slice','cpuset':'/'}))

    def test_live_resource_cgroup_change_still_fails(self):
        with patch.object(launcher, 'process_identity', return_value=(1, 'R')), \
                patch.object(launcher.Path, 'read_text', return_value='1:devices:/\n2:cpuset:/'):
            with self.assertRaises(RuntimeError):
                launcher.verify_child_cgroups(20, {'devices':'/user.slice','cpuset':'/'})

    def test_cleanup_kills_adopted_rank_but_never_keeper_or_self(self):
        processes = {10: (1, 0), 20: (10, 0), 99: (1, 0)}
        with patch.object(launcher.os, 'getpid', return_value=10), \
                patch.object(launcher, 'descendants', return_value=processes), \
                patch.object(launcher, 'process_identity', side_effect=[(100, 'R'), (100, 'R'),
                                                                      (100, 'Z'), (100, 'Z')]), \
                patch.object(launcher.os, 'kill') as kill, \
                patch.object(launcher.time, 'sleep'), \
                patch.object(launcher, 'reap_exited_children'):
            self.assertEqual(launcher.cleanup_payload(Mock(), 99), [20])
        kill.assert_called_once_with(20, signal.SIGTERM)

    def test_cleanup_does_not_signal_reused_pid(self):
        with patch.object(launcher.os, 'getpid', return_value=10), \
                patch.object(launcher, 'descendants', return_value={20: (10, 0)}), \
                patch.object(launcher, 'process_identity', side_effect=[(100, 'R'), (101, 'R')]), \
                patch.object(launcher.os, 'kill') as kill, \
                patch.object(launcher, 'reap_exited_children'):
            self.assertEqual(launcher.cleanup_payload(Mock(), 99), [])
        kill.assert_not_called()

    def test_cleanup_tolerates_vanished_child(self):
        with patch.object(launcher.os, 'getpid', return_value=10), \
                patch.object(launcher, 'descendants', return_value={20: (10, 0)}), \
                patch.object(launcher, 'process_identity', side_effect=FileNotFoundError), \
                patch.object(launcher.os, 'kill') as kill, \
                patch.object(launcher, 'reap_exited_children'):
            self.assertEqual(launcher.cleanup_payload(Mock(), 99), [])
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
