import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1]/'cleanup_personal_checkpoints_20261106.py'
SPEC = importlib.util.spec_from_file_location('personal_cleanup', FILE)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


class HistoricalCleanupTests(unittest.TestCase):
    def test_protects_all_six_requested_recovery_and_inference_files(self):
        for (scope, name) in cleanup.KEEPS:
            with self.subTest(scope=scope, name=name):
                self.assertEqual(cleanup.decision(scope, name), 'keep')

    def test_does_not_touch_evaluation_data_or_sources(self):
        for name in ('evaluation_010000/examples_rank_0.pt', 'examples_rank_0.pt',
                'checkpoint_100000.json', 'evidence.tar.gz', 'source.git/objects/weights_100000.pt',
                '../code_native_train/checkpoint_100000.pt', 'model.ckpt'):
            with self.subTest(name=name):
                self.assertEqual(cleanup.decision('OPT300k', name), 'untouched')

    def test_current_training_and_unknown_roots_are_not_admissible(self):
        self.assertEqual(cleanup.decision('native_full', 'checkpoint_010000.pt'), 'untouched')

    def test_old_intermediates_and_ended_capacity_weights_are_admissible(self):
        for scope, name in [('OPT300k', 'checkpoint_100000.pt'), ('OPT300k', 'capacity/checkpoint_000576.pt'),
                ('OPT330k', 'weights_020000.pt'), ('OPT340k400k', 'checkpoint_100000.pt'),
                ('Qwen250k', 'checkpoint_240000.pt')]:
            with self.subTest(scope=scope, name=name):
                self.assertEqual(cleanup.decision(scope, name), 'delete')

    def test_identical_numbers_in_other_runs_are_not_mistaken_for_protected_points(self):
        self.assertEqual(cleanup.decision('OPT340k400k', 'weights_030000.pt'), 'delete')
        self.assertEqual(cleanup.decision('OPT300k', 'weights_250000.pt'), 'delete')

    def test_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base/'old'
            root.mkdir()
            outside = base/'outside.pt'
            outside.touch()
            with patch.object(cleanup, 'BASE', base), patch.object(cleanup, 'EXCLUDED', base/'active'), \
                    patch.object(cleanup, 'LIVE', base/'live'):
                with self.assertRaises(RuntimeError):
                    cleanup.checked_path(outside, root)


if __name__ == '__main__':
    unittest.main()
