"""Weighted membership, rank accounting, continuation and scheduled admission."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from hardcase_adaptation_protocol import (COUNTS, OPTIMIZATION, SCOPE, SOURCE_SHA,
    SOURCE_SCIENTIFIC_SHA, assess_development, classify, initialize_from_source,
    learning_rate, validate_resume, weighted_order)
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_pair_runtime import next_update_batches, parameter_names, set_lr
from virtual_pair_guard import validate_stage


class HardcaseAdaptationTests(unittest.TestCase):
    def tiers(self):
        return [n for n in COUNTS for _ in range(COUNTS[n])]

    def stream(self, rank, order, state=None):
        return MatchedFrameStream(range(18002), rank=rank, world_size=2, seed=20261101,
            target_assets=30000, effective_assets=48, frame_policy={str(t): 24 if t<=6 else 12 if t<=12 else 6 for t in range(2,25)},
            manifest_sha256='a'*64, update_boundaries=[5000,10000,20000,30000],
            exposure_order=order, state=state)

    def test_thresholds_are_nested_and_use_training_only_frequency(self):
        self.assertEqual(classify({'train_family_assets':20,'joints':64,'leaves':5,'max_depth':8}), 'core')
        self.assertEqual(classify({'train_family_assets':50,'joints':48,'leaves':5,'max_depth':8}), 'extension')
        self.assertEqual(classify({'train_family_assets':51,'joints':100,'leaves':32,'max_depth':30}), 'background')
        self.assertEqual(classify({'train_family_assets':5688,'joints':52,'leaves':15,'max_depth':10}), 'background')

    def test_exact_weighted_cycle_and_full_coverage(self):
        tiers = self.tiers()
        order = weighted_order(tiers)
        self.assertEqual(order, weighted_order(tiers))
        self.assertNotEqual(order, weighted_order(tiers, seed=20261102))
        self.assertEqual(set(order), set(range(18002)))
        np.testing.assert_array_equal(np.bincount(order[:23470], minlength=18002),
            [3 if n=='core' else 2 if n=='extension' else 1 for n in tiers])

    def test_rank_accounting_and_resume_at_all_checkpoints(self):
        order = weighted_order(self.tiers())
        streams = [self.stream(rank, order) for rank in (0,1)]
        samples, updates = 0, 0
        while samples < 30000:
            consumed=[]
            for stream in streams:
                goal, batches = next_update_batches(stream,samples,2)
                consumed += [r.item_index for b in batches for r in b]
            self.assertEqual(consumed,order[samples:samples+goal])
            samples += goal
            updates += 1
            if samples in (5000,10000,20000,30000):
                for stream in streams:
                    restored=self.stream(stream.rank,order,stream.state_dict())
                    self.assertEqual(restored.state_dict(),stream.state_dict())
        self.assertEqual(updates,628)
        changed=order.copy(); changed[0],changed[1]=changed[1],changed[0]
        with self.assertRaises(ValueError):
            self.stream(0,changed,streams[0].state_dict())

    def test_low_lr_phase_keeps_adam_moments_and_step(self):
        parameter=torch.nn.Parameter(torch.ones(2))
        optimizer=torch.optim.AdamW([parameter],lr=1e-4)
        parameter.sum().backward(); optimizer.step()
        before=copy.deepcopy(optimizer.state_dict())
        for samples, expected in ((0,2e-6),(1000,1e-5),(30000,2e-6)):
            self.assertAlmostEqual(learning_rate(samples),expected)
            lr,beta=set_lr(optimizer,samples,{'optimization':OPTIMIZATION})
            self.assertAlmostEqual(lr,expected); self.assertEqual(beta,.95)
            for key,value in before['state'][0].items():
                torch.testing.assert_close(optimizer.state_dict()['state'][0][key],value,rtol=0,atol=0)

    def test_full300k_import_preserves_both_groups_and_actual_moments(self):
        model=torch.nn.Module()
        model.backbone=torch.nn.Linear(2,2); model.head=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.backbone.parameters(),lr=1e-4)
        optimizer.add_param_group({'params':list(model.head.parameters())})
        sum(p.sum() for p in model.parameters()).backward(); optimizer.step()
        weights=copy.deepcopy(model.state_dict()); moments=copy.deepcopy(optimizer.state_dict())
        saved={'scientific_sha256':SOURCE_SCIENTIFIC_SHA,'additional_asset_exposures':300000,
            'step':6270,'optimizer_parameter_names':parameter_names(model),'model':weights,'optimizer':moments}
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
        incoming=torch.optim.AdamW(model.backbone.parameters(),lr=1e-5)
        incoming.add_param_group({'params':list(model.head.parameters())})
        with patch('hardcase_adaptation_protocol.sha256',return_value=SOURCE_SHA), \
                patch.object(torch,'load',return_value=saved):
            initial=initialize_from_source({'backbone':{'checkpoint':{'path':'full300k.pt'}}},model,incoming)
        self.assertTrue(initial['project_optimizer_loaded'])
        for name,value in weights.items(): torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        for i,state in moments['state'].items():
            for key,value in state.items(): torch.testing.assert_close(incoming.state_dict()['state'][i][key],value,rtol=0,atol=0)

    def test_gpu_stages_cannot_run_before_user_scheduled_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'; path.write_text('{}')
            output=Path(directory)/'capacity'
            op={'label':SCOPE,'required_entrypoint':'model_training/rigweave/scripts/train_hardcase_adaptation.py',
                'explicit_human_authorized':True,'complete':False,'gpu_work_allowed':True,
                'not_before_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                'runs':[{'mode':'capacity','config':str(path),'output':str(output),
                    'approved':True,'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}
            args=SimpleNamespace(mode='capacity',config=path,output=output)
            with self.assertRaisesRegex(RuntimeError,'scheduled GPU start time'):
                validate_stage({'active_operation':op},{'family':SCOPE},args)

    def test_candidate_cannot_trade_background_quality_for_focus_gain(self):
        baseline={'samples':0,'stage1':{'failures':2},'stage1_strata':{
            n:{'assets':48 if n!='extension' else 16,'original_edge_f1':.7} for n in COUNTS},
            'groups':[{'split':'valid','method':'head','normalized_mpjpe':.004}]}
        candidate=copy.deepcopy(baseline); candidate['samples']=5000
        for n in ('core','extension'): candidate['stage1_strata'][n]['original_edge_f1']=.75
        self.assertTrue(assess_development(baseline,candidate)['eligible_for_official_and_locked_verification'])
        candidate['stage1_strata']['background']['original_edge_f1']=.69
        self.assertFalse(assess_development(baseline,candidate)['eligible_for_official_and_locked_verification'])


if __name__ == '__main__':
    unittest.main()
