import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from ar_only_protocol import validate_config, validate_donor, validate_resume, scientific_sha
from ar_only_runtime import load_model, update, verify_actual_trace, full_decode_capacity
from reference_flow_protocol import load_backbone_config
from virtual_pair_runtime import make_stream, next_update_batches, target_selection
from stage1_feedback_analysis import compare, pair_rows


def configuration():
    return json.loads((ROOT/'model_training/experiments/ar_only_feedback_5k_20261029_v1.json').read_text())


class ProtocolTests(unittest.TestCase):
    def test_exact_budget_and_original_LR_horizon(self):
        base = validate_config(configuration())
        self.assertEqual(base['training']['target_assets'], 40000)
        self.assertEqual(base['optimization']['schedule_samples'], 100000)
        self.assertEqual(base['data']['stream_seed'], 20261028)

    def test_wrong_budget_random_start_or_precision_rejected(self):
        for section, key, value in [('training','additional_assets',60000),
                ('training','learning_rate_asset_offset',0), ('training','stream_seed',7),
                ('training','optimizer','fresh'), ('training','precision','fp32'),
                ('evaluation','precision','bf16'), ('evaluation','decoder_attention','flash_attention_2'),
                ('evaluation','reference_kind','GT'), ('architecture','stage2_enabled',True)]:
            c = configuration()
            c[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_donor_requires_complete_Adam_and_both_rank_states(self):
        payload = {'variant':'fraction20_40_query', 'asset_exposures':20000,
            'code_commit':'a0bda62bb8fba70320310b063da8c7898ddaf441', 'optimizer_parameter_names':['a'],
            'rank_states':[{'rank':0},{'rank':1}],
            'optimizer':{'param_groups':[{'params':[0]}], 'state':{0:{'step':420}}}}
        validate_donor(payload, ['a'], 2)
        for key, value in [('optimizer_parameter_names',['b']),('asset_exposures',40000),
                ('rank_states',[{'rank':1},{'rank':0}]),
                ('optimizer',{'param_groups':[{}], 'state':{}})]:
            with self.assertRaises(ValueError):
                validate_donor({**payload,key:value}, ['a'], 2)

    def test_resume_rejects_extra_optimizer_group_and_wrong_cumulative(self):
        c = configuration()
        payload = {'scientific_sha256':scientific_sha(c),'optimizer_parameter_names':['a'],
            'rank_states':[{'rank':0},{'rank':1}], 'optimizer':{'param_groups':[{}]},
            'additional_asset_exposures':10000,'global_backbone_asset_exposures':30000}
        validate_resume(c,payload,['a'],2)
        for key,value in [('global_backbone_asset_exposures',10000),('optimizer',{'param_groups':[{},{}]})]:
            with self.assertRaises(ValueError):
                validate_resume(c,{**payload,key:value},['a'],2)

    def test_two_rank_stream_matches_joint_prefix_exactly(self):
        c = configuration()
        base = validate_config(c)
        joint = json.loads((ROOT/'model_training/experiments/anchor_joint_binding_5k_20261027_v1.json').read_text())
        parent = load_backbone_config(joint)
        count = frames = 0
        for rank in range(2):
            a,b = make_stream(base,rank,2), make_stream(parent,rank,2)
            samples,updates = 0,0
            while samples < 40000:
                goal,x = next_update_batches(a,samples,2)
                old,y = next_update_batches(b,samples,2)
                self.assertEqual((goal,x),(old,y))
                for left,right in zip(x,y):
                    self.assertEqual(target_selection(base,left),target_selection(parent,right))
                    frames += sum(r.frames for r in left)
                samples += goal
                updates += 1
            self.assertEqual(updates,840)
            count += samples//2
        self.assertEqual((count,frames),(40000,527856))


class RuntimeTests(unittest.TestCase):
    def test_initial_load_retains_weights_Adam_and_rank_RNG_but_resets_stream(self):
        c = configuration()
        base = validate_config(c)
        model = torch.nn.Linear(2,1)
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.999))
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        payload = {'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'variant':'fraction20_40_query','asset_exposures':20000,
            'code_commit':'a0bda62bb8fba70320310b063da8c7898ddaf441',
            'optimizer_parameter_names':['weight','bias'],
            'rank_states':[{'rank':0,'stream':'old0'},{'rank':1,'stream':'old1'}]}
        with patch('ar_only_runtime.build_model',return_value=(torch.nn.Linear(2,1),None,None,{'checkpoint_sha256':'upstream'})), \
                patch('ar_only_runtime.VirtualPathAblationRig',side_effect=lambda x,**kw:x), \
                patch('ar_only_runtime.torch.load',return_value=payload), \
                patch('ar_only_runtime.dist.barrier'), patch('ar_only_runtime.apply_rng') as rng:
            result = load_model(c,torch.device('cpu'),1,2)
        loaded,actual = result[:2]
        for key,value in payload['model'].items():
            self.assertTrue(torch.equal(loaded.state_dict()[key],value))
        for key,state in payload['optimizer']['state'].items():
            for name,value in state.items():
                self.assertTrue(torch.equal(actual.state_dict()['state'][key][name],value))
        self.assertEqual(actual.param_groups[0]['betas'],(.9,.999))
        rng.assert_called_once_with(payload['rank_states'][1])
        self.assertEqual(result[6].cursor,0)
        self.assertEqual(result[7:],(0,0,0))

    def test_update_reuses_AR_kernel_with_offset_and_no_head(self):
        model,optimizer = SimpleNamespace(), SimpleNamespace(param_groups=[{}])
        result = {'assets_in_update':48}
        with patch('ar_only_runtime.ar_update',return_value=result) as kernel:
            value = update(model,None,optimizer,None,[[0]*24],None,None,{},configuration(),
                10000,torch.device('cpu'),2)
        self.assertEqual(kernel.call_args.args[9],30000)
        self.assertFalse(value['stage2_forward_applied'])
        model.head = object()
        with self.assertRaises(ValueError):
            update(model,None,optimizer,None,[],None,None,{},configuration(),0,torch.device('cpu'),2)

    def trace(self):
        batch = {'asset_ids':['a'],'manifest_indices':[2],'sampling_seeds':[7],
            'selected_raw_frames':[[4,8]],'frames':2,'target_frame_indices':[[0,1]]}
        left = {'step':1,'additional_asset_exposures':48,'input_frame_exposures':96,
            'assets_in_update':48,'frames_in_update':96,'beta1':.9,'learning_rate':3e-5,
            'supervised_frames_in_update':96,'microbatches':[batch]}
        right = copy.deepcopy(left)
        right['lr'] = right.pop('learning_rate')
        right['ar_targets'] = right.pop('supervised_frames_in_update')
        right['microbatches'][0]['ar_target_slots'] = right['microbatches'][0].pop('target_frame_indices')
        return left,right

    def test_actual_trace_matches_and_catches_raw_frame_target_LR_drift(self):
        a,b = self.trace()
        verify_actual_trace(a,b)
        for mutate in (lambda d:d.update(learning_rate=1e-5),
                lambda d:d['microbatches'][0].update(selected_raw_frames=[[4,9]]),
                lambda d:d['microbatches'][0].update(target_frame_indices=[[1,0]]),
                lambda d:d.update(step=2)):
            c = copy.deepcopy(a)
            mutate(c)
            with self.assertRaises(ValueError):
                verify_actual_trace(c,b)

    def test_full_decode_profile_reaches_limit_and_restores_TF32(self):
        class Model:
            def eval(self): return self
            def encode(self,surface): return torch.ones(1,1,1,1),None
            def conditions(self,real,virtual): return real
            def generate_condition(self,condition,start_tokens,**kwargs):
                self.assertions = kwargs['max_new_tokens'], kwargs['eos_token']
                values = kwargs['logits_processor'](None,torch.ones(1,8))
                assert not torch.isfinite(values[0,2])
                return {'status':'max_tokens','tokens':[1]*10}
        model = Model()
        tok = SimpleNamespace(bos=1,eos=2,cls_name_to_token=lambda s:7)
        old = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        with patch('ar_only_runtime.original_xyz_vocabulary_processor',return_value=lambda ids,x:x):
            result = full_decode_capacity(model,{},tok,8)
        self.assertEqual(result['new_tokens'],8)
        self.assertEqual(old,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))


class AnalysisTests(unittest.TestCase):
    def rows(self):
        return [{'asset_id':'a','frame_index':f,'raw_frame_index':f+10,'surface_sha256':'s',
            'normalization_sha256':'n','original_GT_sha256':str(f),'target_mesh_diagonal':2.,
            'gt_joint_count':3,'original_edge_f1':.5,'success':True,'exact_count':False,
            'original_normalized_j2j':.1} for f in range(2)]

    def identity(self,sha):
        return {'checkpoint_sha256':sha,'checkpoint_layout':'density','precision':'fp32'}

    def test_failure_stays_in_quality_denominator_not_geometry(self):
        a,b = self.rows(),self.rows()
        b[0].update(success=False,original_edge_f1=0.,original_normalized_j2j=None)
        result = compare(a,b,self.identity('a'),self.identity('b'))
        self.assertEqual(result['success']['mean_left_minus_right'],.5)
        self.assertEqual(result['original_edge_f1']['mean_left_minus_right'],.25)
        self.assertEqual(result['geometry_jointly_successful_views'],1)
        self.assertEqual(result['original_normalized_j2j']['mean_left_minus_right'],0.)

    def test_input_precision_coverage_drift_rejected(self):
        a,b = self.rows(),self.rows()
        for rows,identity in [(b[:1],self.identity('b')),
                (b,{**self.identity('b'),'precision':'bf16'}),
                ([{**b[0],'surface_sha256':'changed'},b[1]],self.identity('b'))]:
            with self.assertRaises(ValueError):
                pair_rows(a,rows,self.identity('a'),identity)

    def test_no_shared_successes_is_null_geometry(self):
        a,b = self.rows(),self.rows()
        for row in b:
            row.update(success=False,original_edge_f1=0.,original_normalized_j2j=None)
        result = compare(a,b,self.identity('a'),self.identity('b'))
        self.assertIsNone(result['original_normalized_j2j']['mean_left_minus_right'])


if __name__ == '__main__':
    unittest.main()
