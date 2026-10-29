import ast
import copy
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'model_training/rigweave/scripts'))
from stage1_surface_repair_protocol import LABELS, merge_rows, plan_inputs, validate_config
from stage1_surface_repair_runtime import canonical_identity, canonical_record, validate_canonical


class ProtocolTests(unittest.TestCase):
    def test_exact18asset_zero_update_config_and_no_train_mode(self):
        c = json.loads((ROOT/'model_training/experiments/stage1_surface_repair_20261029_v1.json').read_text())
        validate_config(c,'audit')
        validate_config(c,'capacity')
        for mode in ('train','resume'):
            with self.assertRaises(ValueError):
                validate_config(c,mode)
        for key,value in [('optimizer_updates',1),('recomputed_views',11264),('reuse_assets_per_checkpoint',0)]:
            changed = copy.deepcopy(c)
            changed['repair'][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed,'capacity')

    def examples(self):
        rows = [{'asset_id':f'a{a:03d}','frame_index':f,'raw_frame_index':f+10,
            'surface_sha256':f's{a}','normalization_sha256':'n','original_GT_sha256':f'g{a}_{f}',
            'target_mesh_diagonal':1.,'gt_joint_count':3} for a in range(128) for f in range(8)]
        summary = {'identity':{'checkpoint_sha256':'w','checkpoint_layout':'density','precision':'fp32'}}
        return {k:copy.deepcopy(summary) for k in LABELS},{k:copy.deepcopy(rows) for k in LABELS}

    def test_union_selects_inputs_only_and_retains_every_other_asset(self):
        summaries,rows = self.examples()
        for label in LABELS[1:]:
            for row in rows[label][:18*8]:
                row['surface_sha256'] += label
        plan = plan_inputs(summaries,rows)
        self.assertEqual(len(plan['affected_asset_ids']),18)
        self.assertEqual(len(plan['retained_asset_ids']),110)
        self.assertEqual(plan['affected_asset_ids'],[f'a{n:03d}' for n in range(18)])

    def test_changed_GT_frames_normalization_or_protocol_rejects_reuse(self):
        for field,value in [('raw_frame_index',99),('normalization_sha256','x'),('original_GT_sha256','x'),
                ('target_mesh_diagonal',2.),('gt_joint_count',4)]:
            summaries,rows = self.examples()
            rows[LABELS[1]][0][field] = value
            with self.assertRaises(ValueError):
                plan_inputs(summaries,rows)
        summaries,rows = self.examples()
        summaries[LABELS[1]]['identity']['precision'] = 'bf16'
        with self.assertRaises(ValueError):
            plan_inputs(summaries,rows)

    def test_duplicate_or_incomplete_parent_view_coverage_rejected(self):
        for change in ('duplicate','missing','within_asset_surface'):
            summaries,rows = self.examples()
            if change == 'duplicate': rows[LABELS[1]].append(rows[LABELS[1]][0])
            elif change == 'missing': rows[LABELS[1]].pop()
            else: rows[LABELS[1]][0]['surface_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                plan_inputs(summaries,rows)

    def test_merge_preserves_unaffected_objects_and_replaces_all8views(self):
        _,rows = self.examples()
        original = rows['donor']
        new = [{**r,'surface_sha256':'canonical'} for r in original[:8]]
        merged = merge_rows(original,new,['a000'])
        self.assertIs(merged[8],original[8])
        self.assertIs(merged[0],new[0])
        for replacement in (new[:7],new+new[:1],[{**new[0],'raw_frame_index':999},*new[1:]]):
            with self.assertRaises(ValueError):
                merge_rows(original,replacement,['a000'])


class CacheTests(unittest.TestCase):
    def test_serialization_roundtrip_and_transfer_identity(self):
        torch.manual_seed(7)
        surface = {'dense_points':torch.rand(1,8,4,3),'dense_normals':torch.rand(1,8,4,3),
            'query_points':torch.rand(1,8,2,3),'query_normals':torch.rand(1,8,2,3),
            'query_indices':torch.tensor([[0,1]])}
        batch = {'asset_id':'a','frame_vertices':torch.rand(1,8,6,3),'heldout_raw_ids':list(range(8,16))}
        geometry = {'joint_mask':torch.ones(1,2,dtype=torch.bool),
            'frame_target_joints':torch.rand(1,8,2,3),'target_parents':torch.tensor([[-1,0]]),
            'selected_frames':torch.arange(8)[None],'query_center':torch.zeros(1,3),'query_scale':torch.ones(1,1)}
        record = canonical_record(batch,surface,geometry)
        original = [canonical_identity(record,f) for f in range(8)]
        validate_canonical(record,original)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'cache.pt'
            torch.save(record,path)
            saved = torch.load(path,weights_only=True)
        validate_canonical(saved,original)
        saved['surface']['dense_points'][0,0,0,0] += .1
        with self.assertRaises(ValueError):
            validate_canonical(saved,original)

    def test_runtime_has_no_optimization_or_training_calls(self):
        path = ROOT/'model_training/rigweave/scripts/stage1_surface_repair_runtime.py'
        tree = ast.parse(path.read_text())
        called = {node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id
            for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Attribute,ast.Name))}
        self.assertFalse(called & {'backward','step','optimizer_for','load_model','train','ar_update'})
        self.assertFalse(torch.cuda.is_initialized())


class LauncherTests(unittest.TestCase):
    def test_wrapper_propagates_failure_exit_code(self):
        launcher = ROOT/'model_training/jobs/launch_virtual_pair.py'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            path.write_text(json.dumps({'family':'stage1_surface_repair_20261029_v1'}))
            calls = []
            def fake_main(**kwargs):
                calls.append(kwargs)
                return 7
            with patch.dict(os.environ,{'EVOWEAVE_PAIR_SITE_BOOTSTRAPPED':'1'}), \
                    patch.dict(sys.modules,{'launch_legacy_single_target':SimpleNamespace(main=fake_main)}), \
                    patch.object(sys,'argv',[str(launcher),'--config',str(path)]):
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(str(launcher),run_name='__main__')
            self.assertEqual(stopped.exception.code,7)
            self.assertEqual(calls[0]['trainer_script'],'evaluate_stage1_surface_repair.py')


if __name__ == '__main__':
    unittest.main()
