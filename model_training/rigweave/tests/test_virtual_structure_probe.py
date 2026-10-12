import unittest
from pathlib import Path
import numpy as np
import torch
from diagnose_virtual_structure import attention_summary, contrast, difference
from motion_experiment_runtime import load_config


class StructureProbeTests(unittest.TestCase):
    def test_cohort_configuration_matches_checkpoint_on_both_ranks(self):
        path=Path(__file__).resolve().parents[2]/'experiments/virtual_structure_cohort_20261012.json'
        plan=load_config(path)
        self.assertEqual(plan['mode'],'cohort')
        self.assertEqual(plan['models'][0]['checkpoint'],plan['models'][1]['checkpoint'])
        self.assertEqual(plan['models'][0]['sha256'],plan['models'][1]['sha256'])
        self.assertTrue(plan['output'].endswith(plan['label']))

    def test_actual_launch_configuration_is_readable(self):
        path=Path(__file__).resolve().parents[2]/'experiments/virtual_structure_audit_20261012.json'
        plan=load_config(path)
        self.assertEqual(len(plan['models']),2)
        self.assertEqual(plan['manifest_indices'],[2724,1622])
        self.assertTrue(plan['output'].endswith(plan['label']))
        for spec in plan['models']:
            self.assertEqual(len(spec['sha256']),64)

    def test_attention_measurement_matches_actual_weights(self):
        torch.manual_seed(7)
        layer=torch.nn.MultiheadAttention(16,4,dropout=0,batch_first=True).eval()
        q,m=torch.randn(2,3,16),torch.randn(2,7,16)
        with torch.no_grad():
            _,weights=layer(q,m,m,need_weights=True,average_attn_weights=False)
            measured=attention_summary(layer,q,m)
        entropy=-(weights*weights.log()).sum(-1)/np.log(7)
        self.assertAlmostEqual(measured['normalized_entropy_mean'],float(entropy.mean()),places=6)
        self.assertAlmostEqual(measured['max_probability_mean'],float(weights.max(-1).values.mean()),places=6)

    def test_contrast_preserves_axis_meaning(self):
        x=torch.ones(1,3,4,5)
        x[:,1]=2
        self.assertEqual(contrast(x,2)['contrast_rms'],0.)
        self.assertGreater(contrast(x,1)['contrast_rms'],0.)

    def test_difference_is_elementwise_not_loss_difference(self):
        value=difference(torch.tensor([-1.,2.]),torch.tensor([1.,0.]))
        self.assertEqual(value,{'meanabs':2.,'maxabs':2.})


if __name__=='__main__':unittest.main()
