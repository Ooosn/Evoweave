"""400k continuation changes future budget, not the consumed weighted prefix."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'model_training/rigweave/scripts'),str(ROOT/'model_training/rigweave/src')]
from hardcase_adaptation_protocol import COUNTS,weighted_order
from hardcase_continuation_protocol import (BOUNDARIES,NEW_SOURCE,restore_stream,source_compatible,validate_config)
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_pair_runtime import next_update_batches,set_lr
sys.path.insert(0,str(ROOT/'model_training/tools'))
import hardcase_adaptation_ops as ops


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'model_training/experiments/hardcase400k_20261101.json').read_text())
        tiers=[name for name,count in COUNTS.items() for _ in range(count)]
        cls.old=weighted_order(tiers)
        cls.new=weighted_order(tiers,budget=100000)

    def stream(self,rank):
        base=validate_config(self.config)
        return MatchedFrameStream(range(18002),rank=rank,world_size=2,seed=20261101,
            target_assets=30000,effective_assets=48,frame_policy=base['training']['frame_policy'],
            manifest_sha256=base['data']['manifests']['train']['sha256'],
            update_boundaries=[5000,10000,20000,30000],exposure_order=self.old)

    def plans(self,corrupt=False):
        new=self.new.copy()
        if corrupt: new[0],new[1]=new[1],new[0]
        return lambda spec: {'exposure_order':new if spec.get('continuation_target_assets') else self.old}

    def test_exact_recipe_and_no_implicit_weight_or_LR_change(self):
        validate_config(self.config)
        for key,value in [('additional_assets',70000),('head_lr_multiplier',1),('optimizer','fresh_AdamW_all_groups')]:
            candidate=copy.deepcopy(self.config); candidate['training'][key]=value
            with self.assertRaises(ValueError): validate_config(candidate)
        candidate=copy.deepcopy(self.config); candidate['training']['optimization']['peak_lr']=1e-5
        with self.assertRaises(ValueError): validate_config(candidate)
        candidate=copy.deepcopy(self.config); candidate['sampling']['weights']['core']=5
        with self.assertRaises(ValueError): validate_config(candidate)

    def test_full330k_checkpoint_and96h_required(self):
        candidate=copy.deepcopy(self.config)
        candidate['training']['resume']['path']=candidate['training']['resume']['path'].replace('checkpoint','weights')
        with self.assertRaises(ValueError): validate_config(candidate)
        candidate=copy.deepcopy(self.config); candidate['runtime']['required_h_rt_seconds']=86400
        with self.assertRaises(ValueError): validate_config(candidate)

    def test_both_rank_prefixes_and_exact_future_update_count(self):
        for rank in (0,1):
            old=self.stream(rank); old.cursor=old.local_total
            with patch('hardcase_adaptation_protocol.load_sampling',side_effect=self.plans()):
                new=restore_stream(self.config,old.state_dict(),rank,2,from_parent=True)
            self.assertEqual(new.cursor,15000)
            self.assertEqual(len(old.update_sizes),628)
            self.assertEqual(len(new.update_sizes),2091)
            self.assertEqual(len(new.update_sizes)-len(old.update_sizes),1463)
            np.testing.assert_array_equal(old.frame_counts,new.frame_counts[:628])
            self.assertEqual(int(np.dot(new.update_sizes[:628].astype(np.int64),new.frame_counts[:628])),392704)
            goal,batches=next_update_batches(new,30000,2)
            self.assertEqual(goal,48)
            indices=[r.item_index for batch in batches for r in batch]
            self.assertEqual(indices,self.new[30000+rank*24:30000+(rank+1)*24])

    def test_consumed_order_or_unfinished_source_is_rejected(self):
        old=self.stream(0); old.cursor=old.local_total
        with patch('hardcase_adaptation_protocol.load_sampling',side_effect=self.plans(corrupt=True)):
            with self.assertRaises(ValueError): restore_stream(self.config,old.state_dict(),0,2,from_parent=True)
        old.cursor-=1
        with patch('hardcase_adaptation_protocol.load_sampling',side_effect=self.plans()):
            with self.assertRaises(ValueError): restore_stream(self.config,old.state_dict(),0,2,from_parent=True)

    def test_source_transition_cannot_change_model_math(self):
        old={'model_training/rigweave/src/backbone.py':'old',
             'model_training/rigweave/scripts/anchor_full_runtime.py':'loader-old'}
        new=dict(old,**{NEW_SOURCE:'new-helper'})
        new['model_training/rigweave/scripts/anchor_full_runtime.py']='loader-new'
        self.assertTrue(source_compatible(old,new))
        new['model_training/rigweave/src/backbone.py']='changed'
        self.assertFalse(source_compatible(old,new))

    def test_constant_LR_keeps_existing_Adam_moments(self):
        parameter=torch.nn.Parameter(torch.ones(2))
        optimizer=torch.optim.AdamW([parameter],lr=2e-6,betas=(.95,.999))
        parameter.sum().backward(); optimizer.step()
        before=copy.deepcopy(optimizer.state_dict()['state'])
        for samples in (30000,40000,70000,100000):
            lr,beta=set_lr(optimizer,samples,{'optimization':self.config['training']['optimization']})
            self.assertEqual(lr,2e-6); self.assertEqual(beta,.95)
            for key,value in before[0].items():
                torch.testing.assert_close(value,optimizer.state_dict()['state'][0][key],rtol=0,atol=0)

    def test_CPU_acceptance_releases_only_the_exact_training_stage(self):
        config=copy.deepcopy(self.config)
        result={'complete':True,'passed':True,'scientific_sha256':'science','source_sha256':{'loader':'source'},
            'cuda_initialized':False,'plan':{'assets':100000,'updates':2091,'unique_assets':18002},
            'validation_panel':112,'initialization':{'project_optimizer_loaded':True}}
        resources={'complete':True,'passed':True,'exit_code':0,'payload_cleanup_complete':True,
            'kernel_scheduler_membership':True,'allocation':config['runtime']['allocation']}
        state={'active_operation':{'label':config['family'],'gpu_work_allowed':False,
            'formal_training_allowed':False,'CPU_audit_accepted':False,'capacity_accepted':False,
            'runs':[{'mode':'audit','approved':True,'complete':False},
                    {'mode':'train','approved':False,'complete':False}]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); output=root/'outputs/audit_v32'; output.mkdir(parents=True)
            (output/'result.json').write_text(json.dumps(result))
            output.with_name('audit_v32_resources.json').write_text(json.dumps(resources))
            state_path=root/'state.json'; state_path.write_text(json.dumps(state))
            with patch.object(ops,'RUN',root),patch.object(ops,'STATE',state_path),\
                    patch.object(ops,'CONFIG',root/'config.json'),\
                    patch('anchor_full_protocol.scientific_sha',return_value='science'),\
                    patch('anchor_full_protocol.source_hashes',return_value={'loader':'source'}),\
                    patch('hardcase_continuation_protocol.inherited_capacity',return_value={'passed':True}) as proof,\
                    patch('builtins.print'):
                ops.record(config,'audit','audit_v32')
            proof.assert_called_once_with(config)
            op=json.loads(state_path.read_text())['active_operation']
            self.assertTrue(all(op[key] for key in ('CPU_audit_accepted','capacity_accepted',
                'gpu_work_allowed','formal_training_allowed')))
            self.assertFalse(op['runs'][0]['approved']); self.assertTrue(op['runs'][0]['complete'])
            self.assertTrue(op['runs'][1]['approved']); self.assertFalse(op['runs'][1]['complete'])

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
