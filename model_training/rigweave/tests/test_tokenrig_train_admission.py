"""The native frontend cannot inherit the old hybrid capacity receipts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tokenrig_train_admission as admission


class NativeCapacityAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.result_path, self.resources_path = root/'result.json', root/'resources.json'
        self.config = {'tokenrig_decoder': {'conditioning': {'kind': 'released_mesh_projection_v1'}},
            'runtime': {'allocation': 131351736, 'physical_gpus': [3, 4]},
            'capacity': {'optimizer_steps_per_case': 2, 'representative_updates': 12},
            'execution': {'capacity': {'path': str(self.result_path), 'sha256': 'fixed',
                'resources_path': str(self.resources_path), 'resources_sha256': 'fixed'}}}
        self.cases = [{'frames': t, 'batch_per_rank': 24 if t <= 12 else 12,
            'stress_group': group} for t in (6, 12, 24) for group in ('largest_joints', 'largest_geometry')]
        finite = {'assets_in_update': 48, 'ar_loss': 1., 'xyz_loss': .2,
            'bone_loss': .3, 'gradient_norm': 2.}
        extreme = [{**case, **finite, 'profile': 'full_manifest_extreme', 'optimizer_step': step}
            for case in self.cases for step in (1, 2)]
        first = {**finite, 'assets_in_update': 2, 'profile': 'per_rank_batch1_joint_contract'}
        rank = {'cases': [first]+extreme, 'resume_roundtrip_exact': True,
            'representative_updates': [finite]*12,
            'FP32_inference': {'teacher_prefix': {'passed': True}, 'XYZ_finite': True}}
        self.result = {'complete': True, 'passed': True, 'scientific_sha256': 'physics',
            'source_sha256': {'module': 'source'}, 'source': 'deployed', 'initialization': {'native': True},
            'profile_weights_not_training_initialization': True,
            'ranks': [{**copy.deepcopy(rank), 'rank': r} for r in (0, 1)],
            'resume_checkpoint': {'complete': True, 'both_copies_verified': True,
                'optimizer_and_rank_RNG_retained': True}}
        self.resources = {'complete': True, 'passed': True, 'mode': 'capacity', 'exit_code': 0,
            'source': 'deployed', 'allocation': 131351736, 'kernel_scheduler_membership': True,
            'payload_cleanup_complete': True, 'peak_gpu_memory_mib': {'3': 71000, '4': 72000}}
        patches = {
            'sha256': lambda _: 'fixed', 'scientific_sha': lambda _: 'physics',
            'source_hashes': lambda: {'module': 'source'},
            'verify_audit': lambda _: {'initialization': {'native': True}},
            'validate_config': lambda _: {'capacity': {'cases': self.cases}}}
        self.mockers = [patch.object(admission, name, function) for name, function in patches.items()]
        for mocker in self.mockers:
            mocker.start()
            self.addCleanup(mocker.stop)

    def verify(self):
        self.result_path.write_text(json.dumps(self.result))
        self.resources_path.write_text(json.dumps(self.resources))
        return admission.verify_capacity(self.config)

    def test_accepts_finite_native_stress_and_exact_recovery(self):
        self.assertTrue(self.verify()['passed'])

    def test_rejects_zero_update_probe(self):
        self.result['ranks'][0]['cases'] = []
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_nonfinite_or_missing_extreme_case(self):
        for defect in ('nonfinite', 'missing'):
            with self.subTest(defect=defect):
                saved = copy.deepcopy(self.result)
                if defect == 'nonfinite':
                    self.result['ranks'][1]['cases'][-1]['gradient_norm'] = float('nan')
                else:
                    self.result['ranks'][1]['cases'].pop()
                with self.assertRaises(RuntimeError):
                    self.verify()
                self.result = saved

    def test_rejects_over_ceiling_or_incomplete_cleanup(self):
        for field, value in [('peak_gpu_memory_mib', {'3': 79873, '4': 72000}),
                ('payload_cleanup_complete', False), ('kernel_scheduler_membership', False)]:
            with self.subTest(field=field):
                saved = self.resources[field]
                self.resources[field] = value
                with self.assertRaises(RuntimeError):
                    self.verify()
                self.resources[field] = saved


if __name__ == '__main__':
    unittest.main()
