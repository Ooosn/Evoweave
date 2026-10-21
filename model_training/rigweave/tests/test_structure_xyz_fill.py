from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.structure_xyz_fill import (
    StructureXYZFillConfig, StructureXYZFillRig, allowed_structure_tokens,
    coordinate_token_loss, initialize_fill_embeddings,
)
from rigweave.dynamic_rig.structure_xyz_fill_data import build_structure_fill_targets
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig


class Surface(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(6, dim)

    def forward(self, dense, normals, query, query_normals):
        return self.linear(torch.cat((query, query_normals), dim=-1))


class FillModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        upstream = VirtualQueryXYZConfig(
            dim=16, heads=4, query_tokens=5, register_tokens=2, motion_depth=2,
            max_frames=4, auxiliary_depth=1, auxiliary_max_tokens=32,
            motion_evidence_fusion='off', motion_evidence_heads=0,
            gradient_checkpointing=False, temporal_chunk_size=100,
        )
        self.config = StructureXYZFillConfig(upstream=upstream, fill_depth=2,
                                             fill_max_tokens=64, fill_ffn_dim=64, coordinate_bins=16)
        self.codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=32)
        self.model = StructureXYZFillRig(Surface(16), self.config, self.codec)
        self.surface = {key: torch.randn(2, 2, 5, 3) for key in
                        ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        self.geometry = {
            'frame_target_joints': np.random.default_rng(4).uniform(-.8, .8, (2, 2, 4, 3)).astype(np.float32),
            'target_parents': np.array([[-1, 0, 0, 2], [-1, 0, 1, -2]], dtype=np.int64),
            'joint_mask': np.array([[True, True, True, True], [True, True, True, False]]),
        }
        self.targets = build_structure_fill_targets(self.geometry, self.codec, bins=16, max_fill_tokens=64)

    def test_shapes_and_each_layer_updates_all_xyz_slots(self):
        saved = []
        handles = [layer.register_forward_hook(
            lambda module, args, output: saved.append((args[0].detach().clone(), output.detach().clone()))
        ) for layer in self.model.fill_decoder.layers]
        output = self.model(self.surface, self.targets, return_logits=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(output['xyz_logits'].shape, (2, 2, 4, 3, 16))
        self.assertEqual(output['xyz_loss_per_frame'].shape, (2, 2))
        self.assertEqual(len(saved), 4)
        mask = self.targets['fill_joint_ids'] >= 0
        for before, after in saved:
            self.assertTrue(torch.isfinite(after).all())
            self.assertGreater(float((after[mask] - before[mask]).abs().max()), 1e-6)

    def test_losses_train_shared_upstream_but_xyz_does_not_train_structure_decoder(self):
        output = self.model(self.surface, self.targets)
        shared = [self.model.surface_tokenizer.linear.weight, self.model.motion_encoder.virtual_slots,
                  self.model.motion_encoder.blocks[0].anchor_temporal.self_attn.in_proj_weight]
        for loss_name in ('xyz_loss', 'structure_loss'):
            grads = torch.autograd.grad(output[loss_name], shared, retain_graph=True)
            for grad in grads:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.norm()), 0)
        structure_params = list(self.model.structure_decoder.parameters())
        grads = torch.autograd.grad(output['xyz_loss'], structure_params, allow_unused=True, retain_graph=True)
        self.assertTrue(all(value is None for value in grads))
        fill_params = list(self.model.fill_decoder.parameters())
        grads = torch.autograd.grad(output['structure_loss'], fill_params, allow_unused=True, retain_graph=True)
        self.assertTrue(all(value is None for value in grads))
        output['loss'].backward()
        for value in [self.model.fill_decoder.head.weight, self.model.fill_decoder.embedding.weight,
                      self.model.structure_decoder.head.weight,
                      self.model.fill_decoder.layers[0].multihead_attn.in_proj_weight]:
            self.assertGreater(float(value.grad.norm()), 0)

    def test_xyz_ignores_query_decoder_state_and_target_coordinate_answers(self):
        self.model.eval()
        real, virtual = self.model.encode(self.surface)
        first = self.model.forward_encoded(real, virtual, self.surface['query_points'], self.targets,
                                           return_logits=True)
        changed = dict(self.targets)
        changed['xyz_targets'] = torch.full_like(self.targets['xyz_targets'], .99)
        changed['xyz_labels'] = self.targets['xyz_labels'].clone()
        valid = changed['xyz_labels'] >= 0
        changed['xyz_labels'][valid] = (changed['xyz_labels'][valid] + 1) % 16
        with torch.no_grad():
            self.model.structure_decoder.embedding.weight.add_(torch.randn_like(
                self.model.structure_decoder.embedding.weight))
        second = self.model.forward_encoded(real, torch.randn_like(virtual), self.surface['query_points'], changed,
                                            return_logits=True)
        self.assertTrue(torch.equal(first['xyz_logits'], second['xyz_logits']))
        self.assertFalse(torch.equal(first['xyz_loss'], second['xyz_loss']))
        self.assertFalse(torch.allclose(first['structure_logits'], second['structure_logits']))

    def test_final_frame_features_and_real_anchor_positions_affect_only_their_frame(self):
        self.model.eval()
        real, _ = self.model.encode(self.surface)
        points = self.surface['query_points']
        first = self.model.fill_decoder(real, points, self.targets)
        other = real.clone()
        other[:, 1] = torch.randn_like(other[:, 1])
        second = self.model.fill_decoder(other, points, self.targets)
        self.assertTrue(torch.equal(first[:, 0], second[:, 0]))
        self.assertFalse(torch.allclose(first[:, 1], second[:, 1]))
        moved = points.clone()
        moved[:, 1] += 2
        third = self.model.fill_decoder(real, moved, self.targets)
        self.assertTrue(torch.equal(first[:, 0], third[:, 0]))
        self.assertFalse(torch.allclose(first[:, 1], third[:, 1]))

    def test_self_attention_reads_future_context_without_causal_mask(self):
        self.model.eval()
        real, _ = self.model.encode(self.surface)
        first = self.model.fill_decoder(real, self.surface['query_points'], self.targets)
        changed = dict(self.targets, fill_input_ids=self.targets['fill_input_ids'].clone())
        last = int(self.targets['fill_attention_mask'][0].sum()) - 1
        # Deliberately replace only a later context token to probe attention visibility.
        changed['fill_input_ids'][0, last] = self.codec.bos
        second = self.model.fill_decoder(real, self.surface['query_points'], changed)
        self.assertFalse(torch.allclose(first[0, :, 0], second[0, :, 0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_structure_decoder_remains_causal(self):
        self.model.eval()
        _, virtual = self.model.encode(self.surface)
        ids = self.targets['structure_input_ids']
        mask = self.targets['structure_attention_mask']
        first = self.model.structure_decoder(virtual, ids, mask)
        changed = ids.clone()
        changed[:, -1] = self.codec.joint_offset + 6
        second = self.model.structure_decoder(virtual, changed, mask)
        self.assertTrue(torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6))

    def test_coordinate_loss_normalizes_frames_and_ignores_padding(self):
        logits = torch.randn(2, 2, 4, 3, 16)
        labels, joints, frames = (self.targets[key] for key in ('xyz_labels', 'fill_joint_mask', 'frame_mask'))
        original = coordinate_token_loss(logits, labels, joints, frames)[0]
        changed = logits.clone()
        changed[1, :, 3] = 1000
        same = coordinate_token_loss(changed, labels, joints, frames)[0]
        self.assertTrue(torch.equal(original, same))
        one = coordinate_token_loss(logits[:, :1], labels[:, :1], joints, frames[:, :1])[0]
        repeat = coordinate_token_loss(logits[:, :1].repeat(1, 3, 1, 1, 1), labels[:, :1].repeat(1, 3, 1, 1),
                                       joints, frames[:, :1].repeat(1, 3))[0]
        self.assertTrue(torch.allclose(one, repeat, atol=1e-6))

    def test_checkpointed_fill_path_backpropagates(self):
        upstream = VirtualQueryXYZConfig(**{**asdict_upstream(self.config), 'gradient_checkpointing': True})
        cfg = StructureXYZFillConfig(upstream=upstream, fill_depth=2, fill_max_tokens=64,
                                     fill_ffn_dim=64, coordinate_bins=16)
        model = StructureXYZFillRig(Surface(16), cfg, self.codec)
        output = model(self.surface, self.targets)
        output['loss'].backward()
        self.assertGreater(float(model.fill_decoder.layers[-1].linear2.weight.grad.norm()), 0)

    def test_explicit_embedding_copy_has_no_parameter_alias_or_head_copy(self):
        old_head = self.model.fill_decoder.head.weight.detach().clone()
        old_ffn = self.model.fill_decoder.layers[0].linear1.weight.detach().clone()
        position = torch.randn(80, 16)
        source = self.model.structure_decoder.embedding.weight
        report = initialize_fill_embeddings(self.model, structure_embedding=source,
                                            position_embedding=position, position_row_offset=10,
                                            source_name='test')
        self.assertEqual(report['structure_rows'], self.codec.vocab_size)
        self.assertEqual(report['position_rows'], 64)
        self.assertTrue(torch.equal(self.model.fill_decoder.position.weight, position[10:74]))
        self.assertTrue(torch.equal(self.model.fill_decoder.head.weight, old_head))
        self.assertTrue(torch.equal(self.model.fill_decoder.layers[0].linear1.weight, old_ffn))
        original = source.detach().clone()
        with torch.no_grad():
            self.model.fill_decoder.embedding.weight[:self.codec.vocab_size].add_(1)
        self.assertTrue(torch.equal(source, original))

    def test_generation_grammar_and_natural_eos(self):
        self.model.eval()
        _, virtual = self.model.encode(self.surface)
        planned = [4, 5, 3, 4, 6, 2]
        def choose(memory, ids, mask):
            result = torch.full((1, ids.shape[1], self.codec.vocab_size), -10.)
            result[0, -1, planned[ids.shape[1] - 1]] = 10
            return result
        with patch.object(self.model.structure_decoder, 'forward', side_effect=choose):
            prediction = self.model.generate_structure(virtual[:1])
        self.assertTrue(prediction['success'])
        self.assertEqual(prediction['parents'], [-1, 0, 0])
        self.assertEqual(prediction['grammar_constrained_steps'], 0)
        with patch.object(self.model.structure_decoder, 'forward', side_effect=choose):
            capped = self.model.generate_structure(virtual[:1], max_new_tokens=2)
        self.assertFalse(capped['success'])
        self.assertEqual(capped['status'], 'max_tokens')
        self.assertNotIn(self.codec.eos, capped['tokens'])

    def test_no_hidden_structure_bridge_even_after_embedding_copy(self):
        initialize_fill_embeddings(self.model, structure_embedding=self.model.structure_decoder.embedding.weight)
        output = self.model(self.surface, self.targets)
        gradients = torch.autograd.grad(output['xyz_loss'], tuple(self.model.structure_decoder.parameters()),
                                        allow_unused=True)
        self.assertTrue(all(value is None for value in gradients))

    def test_nonfinite_structure_generation_aborts_instead_of_scoring_failure(self):
        self.model.eval()
        _, virtual = self.model.encode(self.surface)
        def nonfinite(memory, ids, mask):
            return torch.full((1, ids.shape[1], self.codec.vocab_size), float('nan'))
        with patch.object(self.model.structure_decoder, 'forward', side_effect=nonfinite):
            with self.assertRaises(FloatingPointError):
                self.model.generate_structure(virtual[:1])

    def test_tiny_synthetic_fit_is_trainable(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.015, weight_decay=0)
        losses = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(self.surface, self.targets)
            losses.append(float(output['loss'].detach()))
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        self.assertLess(losses[-1], .6 * losses[0])


def asdict_upstream(config):
    return dict(config.upstream.__dict__)


class GrammarTests(unittest.TestCase):
    def setUp(self):
        self.codec = AuxSkeletonTokens(max_joints=4, max_sequence_tokens=16)

    def test_allowed_states(self):
        self.assertEqual(allowed_structure_tokens([1], self.codec), [4])
        self.assertEqual(allowed_structure_tokens([1, 4], self.codec), [2, 5])
        self.assertEqual(allowed_structure_tokens([1, 4, 5], self.codec), [2, 6, 3])
        self.assertEqual(allowed_structure_tokens([1, 4, 5, 3], self.codec), [4])
        self.assertEqual(allowed_structure_tokens([1, 4, 5, 3, 4], self.codec), [6])
        self.assertEqual(allowed_structure_tokens([1, 4, 2], self.codec), [])

    def test_invalid_partial_structures(self):
        for prefix in ([], [4], [1, 5], [1, 4, 3], [1, 4, 5, 3, 5], [1, 4, 2, 5]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                allowed_structure_tokens(prefix, self.codec)


if __name__ == '__main__':
    unittest.main()
