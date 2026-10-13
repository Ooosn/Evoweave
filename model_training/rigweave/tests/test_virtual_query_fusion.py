"""Tiny CPU fusion contracts; no pretrained decoder, data, or GPU is needed."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig


MODES = ('virtual_queries_real', 'real_queries_virtual', 'relative_position', 'token_film')
EXTRA_MODULES = ('.relative_position_bias.', '.token_film.')


class TinyXYZDecoder(nn.Module):
    def __init__(self, dim=8, vocab_size=17):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        positions = torch.arange(1, inputs_embeds.shape[1]+1, device=inputs_embeds.device)
        hidden = (inputs_embeds * attention_mask[..., None]).cumsum(1) / positions[None, :, None]
        return SimpleNamespace(logits=self.head(hidden.tanh()))


class TinySurface(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(6, dim)

    def forward(self, dense, normals, query, query_normals):
        return self.projection(torch.cat((query, query_normals), dim=-1))


def make_model(mode='virtual_queries_real', *, seed=71, model_class=VirtualQueryXYZRig, **overrides):
    torch.manual_seed(seed)
    config = VirtualQueryXYZConfig(**dict(dict(dim=8, heads=2, query_tokens=3, register_tokens=1,
        motion_depth=1, fusion_depth=2, auxiliary_depth=1, auxiliary_max_tokens=12,
        max_frames=3, motion_evidence_fusion='off', motion_evidence_heads=0,
        gradient_checkpointing=False, fusion_mode=mode), **overrides))
    return model_class(TinySurface(config.dim), TinyXYZDecoder(config.dim), config,
                       auxiliary_vocab_size=12)


def old_conditions(model, real, virtual):
    batch, frames, queries, dim = real.shape
    query = virtual[:, None].expand(-1, frames, -1, -1).reshape(batch*frames, queries, dim)
    memory = real.reshape(batch*frames, queries, dim)
    for layer in model.frame_fusion:
        normalized_memory = layer.memory_norm(memory)
        query = query + layer.cross(layer.query_norm(query), normalized_memory,
                                    normalized_memory, need_weights=False)[0]
        query = query + layer.ffn(layer.ffn_norm(query))
    if model.config.frame_residual_scale:
        query = query + model.config.frame_residual_scale * memory
    return model.condition_norm(query).reshape_as(real)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(812)
        self.real = torch.randn(2, 2, 3, 8)
        self.virtual = torch.randn(2, 3, 8)
        self.points = torch.randn(2, 2, 3, 3)
        self.targets = {'main_input_ids': torch.randint(1, 17, (2, 2, 6)),
                        'main_attention_mask': torch.ones(2, 2, 6, dtype=torch.bool)}

    def test_default_a_and_diagnostic_residual_equal_old_formula_exactly(self):
        self.assertEqual(VirtualQueryXYZConfig().fusion_mode, MODES[0])
        for scale, dropout in ((0., 0.), (.4, 0.), (0., .2)):
            model = make_model(frame_residual_scale=scale, dropout=dropout)
            state = torch.get_rng_state()
            expected = old_conditions(model, self.real, self.virtual)
            torch.set_rng_state(state)
            actual = model.conditions(self.real, self.virtual)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_a_b_entire_state_and_c_d_common_state_and_rng_match(self):
        baseline = make_model()
        baseline_state = baseline.state_dict()
        baseline_rng = torch.get_rng_state().clone()
        for mode in MODES[1:]:
            model = make_model(mode)
            self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng))
            state = model.state_dict()
            if mode == 'real_queries_virtual':
                self.assertEqual(state.keys(), baseline_state.keys())
            shared = {name: value for name, value in state.items() if not any(extra in name for extra in EXTRA_MODULES)}
            self.assertEqual(shared.keys(), baseline_state.keys())
            for name, value in baseline_state.items():
                with self.subTest(mode=mode, parameter=name):
                    self.assertEqual(shared[name].dtype, value.dtype)
                    self.assertEqual(shared[name].shape, value.shape)
                    self.assertTrue(torch.equal(shared[name].contiguous().reshape(-1).view(torch.uint8),
                                                value.contiguous().reshape(-1).view(torch.uint8)))

    def test_b_swaps_normalized_query_memory_and_uses_real_residual(self):
        model = make_model('real_queries_virtual', fusion_depth=1)
        layer = model.frame_fusion[0]
        model.condition_norm = nn.Identity()
        real = self.real.reshape(4, 3, 8)
        virtual = self.virtual[:, None].expand(-1, 2, -1, -1).reshape(4, 3, 8)
        with (patch.object(layer.cross, 'forward', return_value=(torch.zeros_like(real), None)) as cross,
              patch.object(layer.ffn, 'forward', return_value=torch.zeros_like(real))):
            result = model.conditions(self.real, self.virtual)
        query, key, value = cross.call_args.args
        torch.testing.assert_close(query, layer.query_norm(real), atol=0, rtol=0)
        torch.testing.assert_close(key, layer.memory_norm(virtual), atol=0, rtol=0)
        torch.testing.assert_close(value, key, atol=0, rtol=0)
        torch.testing.assert_close(result, self.real, atol=0, rtol=0)

    def test_c_zero_bias_matches_a(self):
        a, c = make_model().eval(), make_model('relative_position').eval()
        for layer in c.frame_fusion:
            self.assertEqual(torch.count_nonzero(layer.relative_position_bias[-1].weight).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.relative_position_bias[-1].bias).item(), 0)
        torch.testing.assert_close(c.conditions(self.real, self.virtual, query_points=self.points),
                                   a.conditions(self.real, self.virtual), atol=1e-7, rtol=1e-6)

    def test_c_input_anchor_delta_and_per_head_mask_orientation(self):
        model = make_model('relative_position', fusion_depth=1).eval()
        layer = model.frame_fusion[0]
        with torch.no_grad():
            layer.relative_position_bias[-1].weight.normal_(std=.1)
        with patch.object(layer.cross, 'forward', wraps=layer.cross.forward) as cross:
            actual = model.conditions(self.real, self.virtual, query_points=self.points)
        coords = self.points.reshape(4, 3, 3)
        expected = layer.relative_position_bias(coords[:, :, None] - coords[:, None, :])
        expected = expected.permute(0, 3, 1, 2).reshape(8, 3, 3)
        torch.testing.assert_close(cross.call_args.kwargs['attn_mask'], expected, atol=0, rtol=0)
        translated = model.conditions(self.real, self.virtual, query_points=self.points + 2.)
        torch.testing.assert_close(actual, translated, atol=2e-7, rtol=1e-6)

    def test_c_bias_last_layer_then_first_layer_receive_gradients(self):
        model = make_model('relative_position', fusion_depth=1)
        bias = model.frame_fusion[0].relative_position_bias
        optimizer = torch.optim.SGD(bias.parameters(), lr=.2)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model.forward_encoded(self.real, self.virtual, self.targets,
                auxiliary_weight=0., query_points=self.points)['main_loss']
            loss.backward()
            self.assertTrue(torch.isfinite(bias[-1].weight.grad).all())
            self.assertGreater(float(bias[-1].weight.grad.norm()), 0)
            if step == 0:
                self.assertEqual(float(bias[0].weight.grad.norm()), 0)
            else:
                self.assertTrue(torch.isfinite(bias[0].weight.grad).all())
                self.assertGreater(float(bias[0].weight.grad.norm()), 0)
            optimizer.step()

    def test_d_same_index_film_formula_and_no_cross_attention(self):
        model = make_model('token_film').eval()
        query = self.real.reshape(4, 3, 8)
        virtual = self.virtual[:, None].expand(-1, 2, -1, -1).reshape(4, 3, 8)
        for layer in model.frame_fusion:
            gamma, beta = layer.token_film(virtual).chunk(2, -1)
            query = query + gamma * layer.query_norm(query) + beta
            query = query + layer.ffn(layer.ffn_norm(query))
        expected = model.condition_norm(query).reshape_as(self.real)
        with patch.object(model.frame_fusion[0].cross, 'forward', side_effect=AssertionError('unused cross')):
            actual = model.conditions(self.real, self.virtual)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        changed = self.virtual.clone()
        changed[:, 1] *= -5
        other = model.conditions(self.real, changed)
        torch.testing.assert_close(actual[:, :, [0, 2]], other[:, :, [0, 2]], atol=0, rtol=0)
        self.assertFalse(torch.equal(actual[:, :, 1], other[:, :, 1]))

    def test_d_last_layer_has_declared_small_random_initialization(self):
        model = make_model('token_film')
        for layer in model.frame_fusion:
            weight, bias = layer.token_film[-1].weight, layer.token_film[-1].bias
            self.assertGreater(float(weight.detach().norm()), 0)
            self.assertAlmostEqual(float(weight.detach().std()), 1e-3 / 256**.5, delta=1e-5)
            self.assertEqual(torch.count_nonzero(bias).item(), 0)

    def test_all_modes_main_gradient_reaches_real_and_virtual_with_checkpointing(self):
        for mode in MODES:
            for enabled in (False, True):
                with self.subTest(mode=mode, checkpoint=enabled):
                    model = make_model(mode, gradient_checkpointing=enabled)
                    real = self.real.clone().requires_grad_()
                    virtual = self.virtual.clone().requires_grad_()
                    loss = model.forward_encoded(real, virtual, self.targets,
                        auxiliary_weight=0., query_points=self.points)['main_loss']
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for tensor in (real, virtual):
                        self.assertTrue(torch.isfinite(tensor.grad).all())
                        self.assertGreater(float(tensor.grad.norm()), 0)
                    if mode == 'token_film':
                        self.assertGreater(float(model.frame_fusion[0].token_film[0].weight.grad.norm()), 0)

    def test_forward_passes_input_points_and_trains_motion_virtual_slots(self):
        surface = {name: torch.randn(2, 2, 3, 3) for name in
                   ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        for mode in MODES:
            model = make_model(mode, gradient_checkpointing=True)
            with patch.object(model, 'forward_encoded', wraps=model.forward_encoded) as forward:
                result = model(surface, self.targets, auxiliary_weight=0.)
            if mode == 'relative_position':
                self.assertIs(forward.call_args.kwargs['query_points'], surface['query_points'])
            else:
                self.assertNotIn('query_points', forward.call_args.kwargs)
            result['loss'].backward()
            for parameter in (model.motion_encoder.virtual_slots, model.surface_tokenizer.projection.weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()), 0)

    def test_cpu_bfloat16_autocast_forward_and_backward_are_finite(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = make_model(mode)
                real = self.real.clone().requires_grad_()
                virtual = self.virtual.clone().requires_grad_()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = model.forward_encoded(real, virtual, self.targets,
                        auxiliary_weight=0., query_points=self.points)['main_loss']
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for tensor in (real, virtual):
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(float(tensor.grad.norm()), 0)

    def test_shapes_configuration_and_coordinate_restrictions(self):
        for invalid in ('unknown', '', None, True, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                VirtualQueryXYZConfig(fusion_mode=invalid)
        for mode in MODES[1:]:
            with self.assertRaises(ValueError):
                VirtualQueryXYZConfig(fusion_mode=mode, frame_residual_scale=.1)
        for mode in MODES:
            model = make_model(mode)
            self.assertEqual(model.configuration()['fusion_mode'], mode)
            for real, virtual in ((self.real[:, :, :2], self.virtual), (self.real, self.virtual[:, :2]),
                                  (self.real[0], self.virtual), (self.real[:, :0], self.virtual)):
                with self.assertRaises(ValueError):
                    model.conditions(real, virtual, query_points=self.points)
            self.assertEqual(model.conditions(self.real, self.virtual, query_points=self.points).shape,
                             self.real.shape)
        c = make_model('relative_position')
        invalid_points = [None, self.points[0], self.points[:, :, :2], self.points.to(torch.long),
                          torch.full_like(self.points, float('nan')), torch.full_like(self.points, float('inf'))]
        for points in invalid_points:
            with self.assertRaises(ValueError):
                c.conditions(self.real, self.virtual, query_points=points)
        with self.assertRaises(ValueError):
            c.forward_encoded(self.real, self.virtual, self.targets, auxiliary_weight=0.)
        for mode in (MODES[0], MODES[1], MODES[3]):
            model = make_model(mode)
            torch.testing.assert_close(model.conditions(self.real, self.virtual),
                model.conditions(self.real, self.virtual, query_points='unused'), atol=0, rtol=0)

    def test_positional_encoded_call_keeps_old_two_argument_condition_override(self):
        for mode in (MODES[0], MODES[1], MODES[3]):
            model = make_model(mode)
            original = model.conditions
            with patch.object(model, 'conditions', side_effect=lambda real, virtual: original(real, virtual)):
                result = model.forward_encoded(self.real, self.virtual, self.targets,
                    auxiliary_weight=0., query_points=self.points)
            self.assertTrue(torch.isfinite(result['loss']))

    def test_forward_preserves_both_old_override_signatures_outside_c(self):
        class OldInterface(VirtualQueryXYZRig):
            def conditions(self, real, virtual):
                return super().conditions(real, virtual)

            def forward_encoded(self, real, virtual, targets, *, auxiliary_weight):
                return super().forward_encoded(real, virtual, targets, auxiliary_weight=auxiliary_weight)

        surface = {name: torch.randn(2, 2, 3, 3) for name in
                   ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        for mode in (MODES[0], MODES[1], MODES[3]):
            model = make_model(mode, model_class=OldInterface)
            self.assertTrue(torch.isfinite(model(surface, self.targets, auxiliary_weight=0.)['loss']))


if __name__ == '__main__':
    unittest.main()
