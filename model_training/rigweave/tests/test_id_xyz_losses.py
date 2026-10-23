"""CPU tests for detached leave-one-out structural CE and its averaging contract."""
import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rigweave.dynamic_rig.id_xyz_losses import structural_consistency_ce


def fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(2, 4, 7, 11, generator=generator, dtype=dtype, requires_grad=True)
    labels = torch.zeros(2, 4, 7, dtype=torch.long)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[0, :, [1, 4]] = True
    mask[1, :, [0, 2, 5]] = True
    labels[0, :, 1], labels[0, :, 4] = 5, 2
    labels[1, :, 0], labels[1, :, 2], labels[1, :, 5] = 4, 7, 2
    frames = torch.tensor([[False, True, True, False], [True, False, True, True]])
    return logits, labels, mask, frames


def explicit_pairwise(logits, labels, structure_mask, frame_mask):
    per_asset, entropy_assets = [], []
    for asset in range(logits.shape[0]):
        valid = frame_mask[asset].nonzero().flatten().tolist()
        positions = structure_mask[asset, valid[0]] & (labels[asset, valid[0]] != -100)
        logs = {frame: logits[asset, frame, positions].float().log_softmax(-1) for frame in valid}
        frame_losses, frame_entropies = [], []
        for student in valid:
            peers = [frame for frame in valid if frame != student]
            pair_losses = [-(logs[teacher].detach().exp() * logs[student]).sum(-1).mean()
                           for teacher in peers]
            frame_losses.append(torch.stack(pair_losses).mean())
            target = torch.stack([logs[teacher].detach().exp() for teacher in peers]).mean(0)
            frame_entropies.append(-torch.xlogy(target, target).sum(-1).mean())
        per_asset.append(torch.stack(frame_losses).mean())
        entropy_assets.append(torch.stack(frame_entropies).mean())
    per_asset = torch.stack(per_asset)
    return per_asset, (per_asset.detach() - torch.stack(entropy_assets)).mean()


class IdXYZLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_explicit_pairwise_ce_and_detached_teacher_gradients(self):
        logits, labels, mask, frames = fixture()
        output = structural_consistency_ce(logits, labels, mask, frames)
        expected, expected_kl = explicit_pairwise(logits, labels, mask, frames)
        torch.testing.assert_close(output['loss_per_asset'], expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(output['loss'], expected.mean(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(output['kl_diagnostic'], expected_kl, rtol=1e-5, atol=1e-6)
        actual_grad = torch.autograd.grad(output['loss'], logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.mean(), logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-8)
        self.assertFalse(output['kl_diagnostic'].requires_grad)
        self.assertEqual(output['structure_positions_per_asset'].tolist(), [2, 3])
        for asset in range(2):
            for frame in range(4):
                if frames[asset, frame]:
                    self.assertGreater(float(actual_grad[asset, frame].abs().sum()), 0.)
                else:
                    self.assertEqual(float(actual_grad[asset, frame].abs().sum()), 0.)
        self.assertEqual(float(actual_grad[~mask].abs().sum()), 0.)

    def test_identical_distributions_positive_ce_zero_kl_and_student_gradient(self):
        base = torch.tensor([-.7, .2, 1.1, 2.3])
        logits = base.expand(2, 4, 3, -1).clone().requires_grad_()
        labels = torch.ones(2, 4, 3, dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        frames = torch.tensor([[True, True, False, False], [True, True, True, True]])
        output = structural_consistency_ce(logits, labels, mask, frames)
        self.assertGreater(float(output['loss'].detach()), 0.)
        self.assertLess(abs(float(output['kl_diagnostic'])), 1e-6)
        output['loss'].backward()
        self.assertLess(float(logits.grad.abs().max()), 1e-7)

    def test_eos_participates_xyz_pad_and_invalid_frames_are_excluded(self):
        logits = torch.zeros(1, 3, 6, 5)
        labels = torch.tensor([[[1, 3, 3, 3, 2, -100], [1, 4, 4, 4, 2, -100],
                                [99, -5, 9, 9, 9, 9]]])
        mask = torch.tensor([[[True, False, False, False, True, True],
                              [True, False, False, False, True, True],
                              [False, True, True, True, False, False]]])
        frames = torch.tensor([[True, True, False]])
        logits[0, 0, 4, 2] = 5.
        logits[0, 1, 4, 4] = 5.
        logits[0, :2, 1:4] = float('nan')
        logits[0, :2, 5] = float('inf')
        logits[0, 2] = float('nan')
        logits.requires_grad_()
        output = structural_consistency_ce(logits, labels, mask, frames)
        self.assertTrue(torch.isfinite(output['loss']).item())
        self.assertGreater(float(output['kl_diagnostic']), 0.)
        self.assertEqual(output['structure_positions_per_asset'].tolist(), [2])
        output['loss'].backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(float(logits.grad[0, :2, 4].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0, :2, 1:4].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0, :, 5].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.)

    def test_equal_asset_weight_with_different_position_and_frame_counts(self):
        logits, labels, mask, frames = fixture()
        output = structural_consistency_ce(logits, labels, mask, frames)
        independent = [structural_consistency_ce(logits[i:i+1], labels[i:i+1], mask[i:i+1], frames[i:i+1])['loss']
                       for i in range(2)]
        torch.testing.assert_close(output['loss'], torch.stack(independent).mean())
        # Repeating every structural position must not give that asset more weight.
        repeated = structural_consistency_ce(logits.repeat_interleave(2, 2), labels.repeat_interleave(2, 2),
                                             mask.repeat_interleave(2, 2), frames)
        torch.testing.assert_close(output['loss_per_asset'], repeated['loss_per_asset'])

    def test_bfloat16_logits_have_finite_gradient_with_fp32_loss(self):
        logits, labels, mask, frames = fixture(torch.bfloat16)
        output = structural_consistency_ce(logits, labels, mask, frames)
        self.assertEqual(output['loss'].dtype, torch.float32)
        self.assertEqual(output['loss_per_asset'].dtype, torch.float32)
        output['loss'].backward()
        self.assertEqual(logits.grad.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_full_vocabulary_teacher_not_structure_only_renormalization(self):
        logits = torch.tensor([[[[0., 0., 6.]], [[0., 6., 0.]]]], requires_grad=True)
        labels = torch.zeros(1, 2, 1, dtype=torch.long)
        output = structural_consistency_ce(logits, labels, torch.ones_like(labels, dtype=torch.bool),
                                           torch.ones(1, 2, dtype=torch.bool))
        self.assertGreater(float(output['loss'].detach()), 5.)
        self.assertGreater(float(output['kl_diagnostic']), 5.)

    def test_mismatched_structural_masks_labels_and_ignored_slots_rejected(self):
        for kind in ('mask', 'label', 'ignored'):
            logits, labels, mask, frames = fixture()
            if kind == 'mask':
                mask[0, 2, 2] = True
            elif kind == 'label':
                labels[0, 2, 1] = 8
            else:
                labels[0, 2, 1] = -100
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'match across valid frames'):
                structural_consistency_ce(logits, labels, mask, frames)

    def test_minimum_frames_positions_vocabulary_and_finite_logits(self):
        for kind in ('frames', 'empty', 'ignored_all', 'label_range', 'nan'):
            logits, labels, mask, frames = fixture()
            if kind == 'frames':
                frames[0, 2] = False
            elif kind == 'empty':
                mask[0] = False
            elif kind == 'ignored_all':
                labels[0] = -100
            elif kind == 'label_range':
                labels[0, :, 1] = 11
            else:
                with torch.no_grad():
                    logits[0, 1, 1, 0] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                structural_consistency_ce(logits, labels, mask, frames)

    def test_input_shapes_and_dtypes_rejected(self):
        args = fixture()
        variants = [(0, args[0][0]), (0, args[0].long()), (1, args[1].float()),
                    (2, args[2].long()), (2, args[2][:, :, :-1]),
                    (3, args[3].float()), (3, args[3][:, :-1]), (0, None)]
        for index, value in variants:
            changed = list(args)
            changed[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                structural_consistency_ce(*changed)


if __name__ == '__main__':
    unittest.main()
