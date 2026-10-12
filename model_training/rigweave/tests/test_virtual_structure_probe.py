import unittest
import numpy as np
import torch
from diagnose_virtual_structure import attention_summary, contrast, difference


class StructureProbeTests(unittest.TestCase):
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
