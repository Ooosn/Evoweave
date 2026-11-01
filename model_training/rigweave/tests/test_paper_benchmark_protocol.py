import copy
import importlib.util
from pathlib import Path
import unittest
import sys
import tempfile
import numpy as np

PATH = Path(__file__).parents[1] / 'scripts/paper_benchmark_protocol.py'
SPEC = importlib.util.spec_from_file_location('paper_protocol_test', PATH)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def example():
    return {'family':P.SCOPE,'runtime':{'physical_gpus':[4],'execution_cpu_budget':8,
        'world_size':1,'require_scheduler_membership':True,'allocation':131170166},
        'working_model':{'sha256':'abaa87aa7b377f17093850981024a4d63529449f68f082e7c6502ea7a5571018'},
        'evaluation':{'assets':24,'input_frames':8,'reference_slot':0,'selection_seed':20261102,
            'optimizer_updates':0,'payload_limit_seconds':10800,
            'methods':['evoweave330k','unirig_official','puppeteer_official'],
            'retry_failed_generation':False,'use_GT_reference':False,
            'case_selection':'hash_order_excluding_development128'},'audit_result':None}


class PaperProtocolTests(unittest.TestCase):
    def test_puppeteer_normalization_matches_official_function_and_inverse(self):
        points=np.array([[-2.,-1.,1.],[3.,2.,4.],[1.,0.,-2.]],dtype=np.float32)
        normalized,center,scale,pc_center,pc_scale=P.puppeteer_normalization(points)
        np.testing.assert_allclose(center,[.5,.5,1.])
        self.assertAlmostEqual(scale,.9995/3,places=7)
        np.testing.assert_allclose(normalized,(points-[.5,.5,1.])*(.9995/3),atol=1e-7)
        self.assertAlmostEqual(float(np.abs(normalized).max()),.9995,places=6)
        np.testing.assert_allclose(normalized/scale+center,points,atol=3e-7)

    def test_adapter_revision_changes_science_and_requires_retention(self):
        a=example();b=copy.deepcopy(a);b['baseline_adapter_revision']=P.ADAPTER_REVISION
        self.assertNotEqual(P.scientific_sha(a),P.scientific_sha(b))
        with self.assertRaises(ValueError):P.validate_config(b,'audit')
        b['runtime']['paper_completed_baselines']={'unirig_official':{'rows_sha256':'bound'}}
        P.validate_config(b,'audit')

    def test_audit(self):
        P.validate_config(example(),'audit')

    def test_training_closed(self):
        with self.assertRaises(ValueError): P.validate_config(example(),'train')

    def test_cpu_first(self):
        with self.assertRaises(ValueError): P.validate_config(example(),'capacity')

    def test_no_teacher_reference(self):
        c=example();c['evaluation']['use_GT_reference']=True
        with self.assertRaises(ValueError): P.validate_config(c,'audit')

    def test_no_extra_updates(self):
        c=example();c['evaluation']['optimizer_updates']=1
        with self.assertRaises(ValueError): P.validate_config(c,'audit')

    def test_no_new_checkpoint(self):
        c=example();c['working_model']['sha256']='different'
        with self.assertRaises(ValueError): P.validate_config(c,'audit')

    def test_no_card_switch(self):
        c=example();c['runtime']['physical_gpus']=[5]
        with self.assertRaises(ValueError): P.validate_config(c,'audit')

    def test_receipt_does_not_change_science(self):
        a=example();b=copy.deepcopy(a);b['audit_result']={'sha256':'verified'}
        self.assertEqual(P.scientific_sha(a),P.scientific_sha(b))

    def test_selection_changes_science(self):
        a=example();b=copy.deepcopy(a);b['evaluation']['selection_seed']=42
        self.assertNotEqual(P.scientific_sha(a),P.scientific_sha(b))

    def test_annotation_compatibility_never_edits_source(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'legacy_paper_module.py'
            text='value: UndefinedLegacyType = 7\n'
            path.write_text(text)
            sys.path.insert(0,root)
            finder=P.defer_official_annotations(root)
            try:
                import legacy_paper_module
                self.assertEqual(legacy_paper_module.value,7)
                self.assertEqual(legacy_paper_module.__annotations__['value'],'UndefinedLegacyType')
                self.assertEqual(path.read_text(),text)
            finally:
                sys.meta_path.remove(finder)
                sys.path.remove(root)
                sys.modules.pop('legacy_paper_module',None)

    def test_shared_policy_does_not_authorize_training_family(self):
        c=example()
        c['runtime']['paper_shared_inference']={'explicit_human_authorized':True,'scope':P.SCOPE,
            'own_memory_fraction':.30,'minimum_free_mib':32768,'preserve_existing_processes':True}
        P.shared_gpu_policy(c)
        c['family']='hardcase3608_20261101_v1'
        with self.assertRaises(ValueError): P.shared_gpu_policy(c)

    def test_baseline_extension_never_reprepares_inputs_or_regenerates_evoweave(self):
        c=example();c['release_extension']='skintokens_skeleton_prefix_20261102_v1'
        c['evaluation']['methods'].append('skintokens_official')
        with self.assertRaises(ValueError):P.validate_config(c,'audit')
        c['inherit_prepared_inputs']={'sha256':'bound'}
        c['runtime']['paper_completed_evoweave']={'files':'bound'}
        P.validate_config(c,'audit')


if __name__ == '__main__': unittest.main()
