from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from reference_xyz_protocol import validate_config
from reference_xyz_runtime import evaluate, select_generator_cohort
from rigweave.dynamic_rig.reference_skeleton_xyz import ReferenceXYZConfig
from rigweave.dynamic_rig.reference_coordinate_generator import (
    CoordinateGeneratorConfig, ReferenceCoordinateGenerator, quantize_xyz, dequantize_xyz,
)


def fixture(batch=1):
    torch.set_num_threads(1)
    torch.manual_seed(18)
    model = ReferenceCoordinateGenerator(ReferenceXYZConfig(input_dim=8, dim=32, heads=4,
        reference_depth=1, target_depth=1, max_joints=8), CoordinateGeneratorConfig(bins=16, decoder_layers=2))
    features = torch.randn(batch, 3, 8, 8)
    points, normals = torch.randn(batch, 3, 8, 3) * .3, torch.randn(batch, 3, 8, 3)
    parents = torch.tensor([[-1, 0, 1]]).expand(batch, -1).clone()
    mask = torch.ones(batch, 3, dtype=torch.bool)
    reference = torch.zeros(batch, dtype=torch.long)
    targets = dequantize_xyz(torch.randint(2, 14, (batch, 3, 3, 3)), 16)
    return model, (features, points, normals, targets[:, 0], parents, mask, reference), targets


class CoordinateGeneratorTest(unittest.TestCase):
    def test_quantization_bounds_and_no_silent_clipping(self):
        values = torch.tensor([-1., -.1, 0., .4, 1.])
        reconstructed = dequantize_xyz(quantize_xyz(values, 256), 256)
        self.assertLessEqual(float((values - reconstructed).abs().max()), 1. / 255. + 1.e-7)
        for bad in (float('nan'), -1.01, 1.01):
            with self.assertRaises(ValueError):
                quantize_xyz(torch.tensor([bad]), 256)

    def test_causal_teacher_prefix_has_no_future_label_leakage(self):
        model, inputs, targets = fixture()
        model.eval()
        condition, mask, query = model.prepare_condition(*inputs)
        tokens = quantize_xyz(targets, 16).reshape(3, 9)
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 3) % 16
        before = model.teacher_logits(condition, mask, query, tokens)
        after = model.teacher_logits(condition, mask, query, changed)
        torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0., atol=0.)
        self.assertGreater(float((before[:, 5:] - after[:, 5:]).detach().abs().max()), 0.)

    def test_free_generation_cache_reference_and_padding(self):
        model, inputs, targets = fixture(batch=2)
        inputs[4][0, 2] = -2
        inputs[5][0, 2] = False
        model.eval()
        cached = model.generate(*inputs)
        uncached = model.generate(*inputs, use_cache=False)
        torch.testing.assert_close(cached, uncached, rtol=0., atol=0.)
        self.assertTrue(torch.equal(cached[1, 0], inputs[3][1]))
        self.assertEqual(cached[0, :, 2].abs().sum().item(), 0.)
        self.assertTrue(bool(torch.isfinite(cached).all()))
        self.assertLessEqual(float(cached.abs().max()), 1.)
        self.assertEqual(cached.shape, targets.shape)

    def test_teacher_loss_and_all_conditioning_gradients(self):
        model, inputs, targets = fixture()
        inputs[0].requires_grad_()
        result = model(*inputs, targets=targets, return_predictions=True)
        result['loss_per_asset'].mean().backward()
        self.assertEqual(result['teacher_prediction'].shape, targets.shape)
        for module in (model.conditioner.reference_blocks, model.conditioner.target_blocks,
                       model.conditioner.target_geometry, model.decoder, model.axis_embedding):
            self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in module.parameters()))
        self.assertGreater(inputs[0].grad[:, 0].abs().sum().item(), 0.)
        self.assertGreater(inputs[0].grad[:, 1:].abs().sum().item(), 0.)

    def test_synthetic_tokens_and_free_generation_can_be_learned(self):
        model, inputs, targets = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
        first = float(model(*inputs, targets=targets)['loss_per_asset'].mean().detach())
        for _ in range(200):
            optimizer.zero_grad()
            loss = model(*inputs, targets=targets)['loss_per_asset'].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        result = model(*inputs, targets=targets)
        self.assertLess(float(result['loss_per_asset'].mean().detach()), first * .1)
        self.assertGreaterEqual(float(result['teacher_forced_accuracy'].mean()), .99)
        model.eval()
        torch.testing.assert_close(model.generate(*inputs)[:, 1:], targets[:, 1:], rtol=0., atol=1.e-6)

    def test_complete_generator_evaluation_keeps_free_and_teacher_results_separate(self):
        model, inputs, targets = fixture()
        row = {'asset_id': 'sample', 'features': inputs[0][0], 'points': inputs[1][0], 'normals': inputs[2][0],
               'targets': targets[0], 'parents': inputs[4][0], 'diagonal': torch.full((3,), 3.),
               'predicted_references': {0: {'success': True, 'xyz': inputs[3][0], 'parents': inputs[4][0]}}}
        config = {'generator': {'bins': 16, 'decoder_layers': 2}, 'pilot': {'evaluation_reference_slots': [0],
                  'frames': 3, 'local_rigid_neighbors': 8, 'output_baseline': 'generated_xyz'}}
        with tempfile.TemporaryDirectory() as directory, \
                patch('reference_xyz_runtime.torch.autocast', return_value=nullcontext()), \
                patch('reference_xyz_runtime.gather', side_effect=lambda value, world: [value]):
            groups = evaluate(model, {'train': [row], 'valid': [row]}, config, torch.device('cpu'), 0, 1, 0, Path(directory))
            self.assertEqual(groups['valid/GT/all/learned']['views'], 2)
            teacher = json.loads((Path(directory) / 'evaluation_0000_teacher_rank_0.json').read_text())
            self.assertEqual(len(teacher), 2)
            self.assertEqual({r['reference_kind'] for r in teacher}, {'GT'})
            summary = json.loads((Path(directory) / 'evaluation_0000_summary.json').read_text())
            self.assertIn('teacher_forced_accuracy', summary['teacher_forced']['train'])
            self.assertAlmostEqual(summary['teacher_forced']['train']['quantized_GT_normalized_mpjpe'], 0., places=6)
            self.assertEqual(summary['output_baseline'], 'generated_xyz')

    def test_subcohort_keeps_original_rank_membership(self):
        selected = {split: [{'asset_id': f'{split}_{group}_{i}', 'stratum': group}
                    for group in ('low', 'mid', 'gt52', 'high') for i in range(4)] for split in ('train', 'valid')}
        for rank in range(2):
            caches = {split: selected[split][rank::2] for split in selected}
            filtered, report = select_generator_cohort(caches, selected, {'pilot': {'train_assets': 8, 'valid_assets': 16}}, rank, 2)
            self.assertEqual(len(filtered['train']), 4)
            self.assertEqual(len(filtered['valid']), 8)
            self.assertEqual(filtered['train'], report['train'][rank::2])

    def test_generator_contract_excludes_geometric_baseline(self):
        config = json.loads((ROOT / 'model_training/experiments/reference_skeleton_xyz_ar_fit_20261027_v1.json').read_text())
        validate_config(config)
        config['pilot']['output_baseline'] = 'local_rigid'
        with self.assertRaisesRegex(ValueError, 'geometric'):
            validate_config(config)


if __name__ == '__main__':
    unittest.main()
