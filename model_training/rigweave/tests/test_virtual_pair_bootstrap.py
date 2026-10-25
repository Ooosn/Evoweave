from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/jobs'))
import launch_virtual_pair as launcher


class BootstrapTests(unittest.TestCase):
    def test_reexec_preserves_argv_and_inherited_grant(self):
        args = ['launch_virtual_pair.py', '--config', '/path with spaces/config.json', '--mode', 'audit']
        with patch.dict(launcher.os.environ, {'JOB_ID': '130494177', 'NSLOTS': '16'}, clear=True), \
                patch.object(launcher.sys, 'argv', args), patch.object(launcher.os, 'execv') as execute:
            launcher.bootstrap_site_environment()
            program, command = execute.call_args.args
            self.assertEqual(program, '/bin/bash')
            self.assertEqual(command[7:], args[1:])
            self.assertIn('exec "$@"', command[2])
            self.assertEqual(launcher.os.environ['JOB_ID'], '130494177')
            self.assertEqual(launcher.os.environ['NSLOTS'], '16')

    def test_bootstrap_marker_prevents_recursion(self):
        with patch.dict(launcher.os.environ, {'EVOWEAVE_PAIR_SITE_BOOTSTRAPPED': '1'}), \
                patch.object(launcher.os, 'execv') as execute:
            launcher.bootstrap_site_environment()
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
