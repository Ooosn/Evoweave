import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'model_training/rigweave/scripts'),str(ROOT/'model_training/rigweave/src')]
from tokenrig_native_continuation import (PARENT_ASSETS, PARENT_STEPS, completed_streams,
    parent_config, validate_config, verify_first_update, source_compatible)
from anchor_full_data import make_stream
from reference_flow_protocol import load_backbone_config
from virtual_pair_runtime import next_update_batches
from virtual_frame_runtime import exposure_learning_rate


class NativeContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'model_training/experiments/tokenrig_native_cont400k_20261107.json').read_text())

    def test_exact_suffix_and_tail_schedule(self):
        base = validate_config(self.config)
        parent = load_backbone_config(parent_config())
        states = []
        for rank in (0,1):
            stream = make_stream(parent,rank,2)
            stream.cursor = PARENT_ASSETS//2
            states.append({'rank':rank,'stream':stream.state_dict()})
        extended = completed_streams(self.config,{'rank_states':states})
        self.assertEqual(len(extended[0].global_ends)-PARENT_STEPS,2090)
        self.assertEqual(int(extended[0].global_ends[-1]),400000)
        all_indices = []
        for stream in extended:
            goal,batches = next_update_batches(stream,PARENT_ASSETS,2)
            self.assertEqual(goal,48)
            all_indices += [r.item_index for batch in batches for r in batch]
            self.assertEqual(stream.cursor,150024)
        self.assertEqual(all_indices,extended[0].order[300000:300048].tolist())
        lr,beta = exposure_learning_rate(PARENT_ASSETS,base['optimization']['schedule_samples'])
        self.assertAlmostEqual(lr,2e-6,places=15)
        self.assertAlmostEqual(beta,.95,places=15)

    def test_rejects_batch_and_evaluation_changes(self):
        for section,key,value in (('training','effective_assets_per_update',24),
                ('training','resume',None),('evaluation','ar_interval_assets',50000)):
            altered = copy.deepcopy(self.config)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(altered)

    def test_source_boundary(self):
        old = {'model_training/rigweave/src/model.py':'a','model_training/rigweave/scripts/anchor_full_runtime.py':'b'}
        new = {**old,'model_training/rigweave/scripts/tokenrig_native_continuation.py':'c'}
        new['model_training/rigweave/scripts/anchor_full_runtime.py'] = 'd'
        self.assertTrue(source_compatible(old,new))
        new['model_training/rigweave/src/model.py'] = 'changed'
        self.assertFalse(source_compatible(old,new))

    def test_actual_first_update_rejects_wrong_seed(self):
        expected = {'step':6271,'additional_asset_exposures':300048,'assets_in_update':48,
            'frames_in_update':576,'microbatches':[{'manifest_indices':[3],'asset_ids':['a'],
                'sampling_seeds':[42],'frames':12}]}
        actual = {**copy.deepcopy(expected),'lr':2e-6,'head_lr':6e-6,'beta1':.95,
            'learning_rate_asset_position':300000,'ar_loss':1.,'xyz_loss':.01,'bone_loss':.01,
            'gradient_norm':.5,'gradient_groups':{'head':.4}}
        verify_first_update(actual,expected)
        actual['microbatches'][0]['sampling_seeds'] = [43]
        with self.assertRaises(ValueError):
            verify_first_update(actual,expected)


if __name__ == '__main__':
    unittest.main()
