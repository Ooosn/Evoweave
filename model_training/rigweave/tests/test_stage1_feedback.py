import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from stage1_feedback import (backbone_state, compare_backbone_weights, generate_asset,
                             score_generation, summarize_rows)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.donor = {'base.surface_tokenizer.weight': torch.ones(2),
                      'base.motion_encoder.weight': torch.ones(2),
                      'base.xyz_decoder.weight': torch.ones(3),
                      'base.condition_norm.bias': torch.zeros(3),
                      'base.auxiliary.buffer': torch.tensor(1)}
        self.anchor = {'backbone.'+name: value.clone() for name, value in self.donor.items()}
        self.anchor['head.weight'] = torch.ones(4)

    def test_complete_backbone_extraction_preserves_identity(self):
        state = backbone_state({'model': self.anchor}, self.donor, layout='anchor')
        self.assertEqual(set(state), set(self.donor))
        self.assertIs(state['base.xyz_decoder.weight'], self.anchor['backbone.base.xyz_decoder.weight'])
        report = compare_backbone_weights(self.donor, state, require_unchanged_nonshared=True)
        self.assertTrue(report['nonshared']['all_unchanged'])

    def test_key_shape_dtype_and_layout_fail_closed(self):
        for mutate in (lambda d: d.pop('backbone.base.xyz_decoder.weight'),
                       lambda d: d.update({'extra': torch.ones(1)}),
                       lambda d: d.update({'backbone.base.xyz_decoder.weight': torch.ones(4)}),
                       lambda d: d.update({'backbone.base.xyz_decoder.weight': torch.ones(3).double()})):
            state = copy.deepcopy(self.anchor)
            mutate(state)
            with self.assertRaises(ValueError):
                backbone_state({'model': state}, self.donor, layout='anchor')
        with self.assertRaises(ValueError):
            backbone_state({'model': self.anchor}, self.donor, layout='automatic')

    def test_shared_change_is_not_nonshared_change(self):
        state = copy.deepcopy(self.donor)
        state['base.motion_encoder.weight'][0] += 1
        report = compare_backbone_weights(self.donor, state, require_unchanged_nonshared=True)
        self.assertEqual(report['shared']['changed'], ['base.motion_encoder.weight'])
        self.assertTrue(report['nonshared']['all_unchanged'])

    def test_nonshared_parameters_and_buffers_must_be_bitwise_equal(self):
        for key in ('base.xyz_decoder.weight', 'base.auxiliary.buffer'):
            state = copy.deepcopy(self.donor)
            state[key] += 1
            with self.assertRaisesRegex(ValueError, 'nonshared'):
                compare_backbone_weights(self.donor, state, require_unchanged_nonshared=True)
        state = copy.deepcopy(self.donor)
        state['base.condition_norm.bias'][0] = -0.
        with self.assertRaises(ValueError):
            compare_backbone_weights(self.donor, state, require_unchanged_nonshared=True)


class Tokenizer:
    bos, eos = 1, 2

    def cls_name_to_token(self, name):
        assert name == 'articulationxl'
        return 7

    def detokenize(self, tokens):
        return SimpleNamespace(joints=np.array([[0., 0., 0.], [1., 0., 0.]]), parents=[None, 0])


class TinyGenerator(nn.Module):
    virtual_enabled = False

    def encode(self, surface):
        assert set(surface) == {'dense_points', 'dense_normals', 'query_points', 'query_normals', 'query_indices'}
        return surface['query_points'], None

    def conditions(self, real, virtual):
        return real

    def forward_encoded(self, *args, **kwargs):
        raise AssertionError('teacher forcing must not run')

    def generate_condition(self, condition, start, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        assert start.tolist() == [1, 7]
        assert kwargs['max_new_tokens'] == 2048
        return {'success': True, 'status': 'eos', 'tokens': [1, 7, 2]}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = TinyGenerator(), Tokenizer()
        self.surface = {key: torch.zeros(1, 2, 3, 3) for key in
                        ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        self.surface['query_indices'] = torch.arange(3)[None]

    @patch('stage1_feedback.original_xyz_vocabulary_processor', return_value=None)
    def test_free_generation_has_only_mesh_and_normal_start(self, processor):
        result = generate_asset(self.model, self.surface, self.tokenizer, frame_indices=[1, 0])
        self.assertEqual(list(result), [1, 0])
        self.assertEqual(processor.call_count, 2)
        self.assertTrue(self.model.training)

    @patch('stage1_feedback.original_xyz_vocabulary_processor', return_value=None)
    def test_generation_error_restores_model_mode(self, processor):
        with patch.object(self.model, 'generate_condition', side_effect=RuntimeError('decoder failure')):
            with self.assertRaises(RuntimeError):
                generate_asset(self.model, self.surface, self.tokenizer, frame_indices=[0])
        self.assertTrue(self.model.training)

    def test_invalid_frames_and_virtual_route_are_rejected(self):
        for frames in ([], [0, 0], [True], [-1], [2]):
            with self.assertRaises(ValueError):
                generate_asset(self.model, self.surface, self.tokenizer, frame_indices=frames)
        self.model.virtual_enabled = True
        with self.assertRaises(ValueError):
            generate_asset(self.model, self.surface, self.tokenizer, frame_indices=[0])

    def test_target_or_reference_fields_cannot_enter_generator(self):
        for field in ('target_joints', 'reference_xyz', 'main_input_ids'):
            with self.assertRaisesRegex(ValueError, 'mesh-surface'):
                generate_asset(self.model, {**self.surface, field: torch.zeros(1)},
                               self.tokenizer, frame_indices=[0])

    @patch('stage1_feedback.original_xyz_vocabulary_processor', return_value=None)
    def test_explicit_fp32_route_preserves_free_prompt(self, processor):
        result = generate_asset(self.model, self.surface, self.tokenizer, frame_indices=[0], precision='fp32')
        self.assertEqual(result[0]['tokens'], [1, 7, 2])
        self.assertTrue(self.model.training)

    def test_unknown_generation_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, 'precision'):
            generate_asset(self.model, self.surface, self.tokenizer, frame_indices=[0], precision='auto')

    @patch('stage1_feedback.original_xyz_vocabulary_processor', return_value=None)
    def test_real_Vminus_wrapper_uses_original_free_decoder_without_teacher(self, processor):
        from test_virtual_path_ablation import make_base, make_surface
        from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
        torch.set_num_threads(1)
        model = VirtualPathAblationRig(make_base(), virtual_enabled=False)
        with torch.no_grad():
            model.xyz_decoder.head.weight.zero_()
            model.xyz_decoder.head.bias.zero_()
            model.xyz_decoder.head.bias[self.tokenizer.eos] = 10.
        surface = make_surface(batch=1, frames=2)
        surface['query_indices'] = torch.arange(3)[None]
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch.object(model, 'forward_encoded', side_effect=AssertionError('teacher forcing')):
            result = generate_asset(model, surface, self.tokenizer, frame_indices=[0, 1])
        self.assertTrue(all(p['success'] and p['tokens'] == [1, 7, 2] for p in result.values()))
        report = compare_backbone_weights(before, model.state_dict(),
                                          require_unchanged_nonshared=True)
        self.assertTrue(report['shared']['all_unchanged'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertGreater(report['shared']['tensors'], 0)
        self.assertGreater(report['nonshared']['tensors'], 0)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.truth = np.array([[0., 0., 0.], [1., 0., 0.]])
        self.parents = [-1, 0]
        self.predicted = {'success': True, 'status': 'eos', 'tokens': [1, 7, 2]}

    def score(self, prediction=None, truth=None, diagonal=2.):
        return score_generation(prediction or self.predicted, self.tokenizer,
                                self.truth if truth is None else truth, self.parents, diagonal)

    def test_original_GT_not_quantized_target_and_geometric_units(self):
        row = self.score(truth=self.truth+[0., .5, 0.])
        self.assertTrue(row['success'])
        self.assertTrue(row['exact_count'])
        self.assertEqual(row['original_edge_f1'], 1.)
        self.assertAlmostEqual(row['original_normalized_j2j'], .25)

    def test_generation_failure_is_zero_quality_but_no_geometry(self):
        row = self.score({'success': False, 'status': 'max_tokens', 'tokens': [1, 7]})
        self.assertFalse(row['success'])
        self.assertFalse(row['exact_count'])
        self.assertEqual(row['original_edge_f1'], 0.)
        self.assertIsNone(row['original_normalized_j2j'])

    def test_bad_decoding_never_leaves_partial_success(self):
        for joints in (np.empty((0, 3)), np.full((2, 3), np.nan)):
            with patch.object(self.tokenizer, 'detokenize', return_value=SimpleNamespace(joints=joints, parents=[-1, 0])):
                row = self.score()
                self.assertFalse(row['success'])
                self.assertEqual(row['original_edge_f1'], 0.)
                self.assertIsNone(row['original_normalized_j2j'])

    def test_invalid_GT_scale_and_inconsistent_EOS_abort(self):
        for scale in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.score(diagonal=scale)
        with self.assertRaises(ValueError):
            self.score(truth=np.full((2, 3), np.nan))
        with self.assertRaises(ValueError):
            self.score({'success': True, 'status': 'eos', 'tokens': [1, 7]})

    def test_summary_weights_assets_and_exposes_geometry_denominators(self):
        success = self.score(truth=self.truth+[0., .5, 0.])
        failure = self.score({'success': False, 'status': 'max_tokens', 'tokens': []})
        rows = [{**success, 'asset_id': 'a', 'raw_frame_index': 1},
                {**failure, 'asset_id': 'a', 'raw_frame_index': 2},
                {**failure, 'asset_id': 'b', 'raw_frame_index': 1}]
        summary = summarize_rows(rows)
        self.assertEqual(summary['success'], .25)
        self.assertEqual(summary['original_edge_f1'], .25)
        self.assertEqual(summary['geometry_scored_views'], 1)
        self.assertEqual(summary['geometry_scored_assets'], 1)
        self.assertEqual(summary['original_normalized_j2j_successful_views'], .25)
        self.assertIsNone(summarize_rows(rows[1:])['original_normalized_j2j_successful_views'])

    def test_summary_rejects_duplicate_rows_and_fake_zero_failure_distance(self):
        row = {**self.score(), 'asset_id': 'a', 'raw_frame_index': 0}
        with self.assertRaises(ValueError):
            summarize_rows([row, row])
        row.update(success=False, exact_count=False, original_edge_f1=0.)
        with self.assertRaises(ValueError):
            summarize_rows([row])


if __name__ == '__main__':
    unittest.main()
