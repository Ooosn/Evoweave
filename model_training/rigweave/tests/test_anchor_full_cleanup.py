import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
import anchor_full_cleanup as cleanup
from reference_flow_protocol import sha256
from virtual_query_xyz_runtime import write_json


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.work, self.durable = root/'work', root/'durable'
        self.patches = [patch.object(cleanup, 'ROOT', self.work),
            patch.object(cleanup, 'RETAINED', self.work/'retained'),
            patch.object(cleanup, 'DURABLE', self.durable/'retained'),
            patch.object(cleanup, 'SOURCE_DURABLE', self.durable),
            patch.object(cleanup, 'STAGES', (2,)),
            patch.object(cleanup, 'KEEP_FULL', {'train_v2/control/checkpoint_020000.pt'})]
        for item in self.patches:
            item.start()
        self.output = root/'audit'
        self.output.mkdir()
        self.model = {'a': torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        for samples in (5000, 10000, 15000, 20000):
            path = self.work/'outputs/train_v2/control'/f'checkpoint_{samples:06d}.pt'
            backup = self.durable/'control'/path.name
            path.parent.mkdir(parents=True, exist_ok=True)
            backup.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': self.model, 'optimizer': {'large_state': torch.ones(100)},
                'rank_states': ['not_needed_for_inference'], 'asset_exposures': samples}, path)
            shutil.copyfile(path, backup)
            write_json(path.with_suffix('.json'), {'path': str(path), 'durable_path': str(backup),
                'bytes': path.stat().st_size, 'sha256': sha256(path), 'complete': True, 'both_copies_verified': True})
            evaluation = path.parent/f'evaluation_{samples:06d}'
            evaluation.mkdir()
            write_json(evaluation/'summary.json', {'complete': True,
                'groups': {'all': {'original_edge_f1': .9 if samples == 15000 else .5}}})

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def plan(self):
        plan = cleanup.plan_legacy()
        path = self.output/'plan.json'
        write_json(path, plan)
        return plan, {'path': str(path), 'sha256': sha256(path)}

    def test_best15k_final_and10k_preserved_but5k_removed(self):
        plan, spec = self.plan()
        self.assertEqual([r['action'] for r in plan['rows']],
            ['remove_intermediate', 'compact_model', 'compact_model', 'keep_full'])
        result = cleanup.execute_legacy(spec, self.output)
        self.assertEqual(result['retired_checkpoints'], 3)
        kept = self.work/'outputs/train_v2/control/checkpoint_020000.pt'
        self.assertTrue(kept.exists())
        self.assertTrue((self.durable/'control'/kept.name).exists())
        for n in (10000, 15000):
            exported = torch.load(self.work/'retained/train_v2/control'/f'weights_{n:06d}.pt', weights_only=False)
            torch.testing.assert_close(exported['model']['a'], self.model['a'], rtol=0, atol=0)
            self.assertNotIn('optimizer', exported)
            self.assertNotIn('rank_states', exported)
        second = cleanup.execute_legacy(spec, self.output)
        self.assertEqual(second['net_bytes_freed'], result['net_bytes_freed'])

    def test_modified_checkpoint_stops_before_any_deletion(self):
        _, spec = self.plan()
        path = self.work/'outputs/train_v2/control/checkpoint_005000.pt'
        path.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            cleanup.execute_legacy(spec, self.output)
        self.assertTrue(path.exists())
        self.assertTrue((self.durable/'control'/path.name).exists())

    def test_export_verification_failure_preserves_originals(self):
        plan, _ = self.plan()
        row = next(r for r in plan['rows'] if r['samples'] == 10000)
        with patch.object(cleanup.torch, 'equal', return_value=False):
            with self.assertRaises(RuntimeError):
                cleanup.compact(row)
        self.assertTrue(Path(row['checkpoint']['path']).exists())
        self.assertTrue(Path(row['checkpoint']['durable_path']).exists())

    def test_path_outside_model_outputs_is_rejected(self):
        plan, _ = self.plan()
        spec = dict(plan['rows'][0]['checkpoint'])
        spec['path'] = str(self.work/'dataset/checkpoint_005000.pt')
        with self.assertRaises(ValueError):
            cleanup.validated_paths(spec)

    def test_interrupted_two_copy_deletion_is_recoverable(self):
        _, spec = self.plan()
        backup = self.durable/'control/checkpoint_005000.pt'
        original_unlink = Path.unlink
        def interrupted(path, *args, **kwargs):
            if path == backup:
                raise OSError('simulated interruption')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', interrupted):
            with self.assertRaises(OSError):
                cleanup.execute_legacy(spec, self.output)
        self.assertTrue(backup.exists())
        result = cleanup.execute_legacy(spec, self.output)
        self.assertEqual(result['retired_checkpoints'], 3)
        self.assertFalse(backup.exists())


if __name__ == '__main__':
    unittest.main()
