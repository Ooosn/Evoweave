from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rigweave.dynamic_rig.shared_structure import (
    ParentSequenceTokens,
    SharedParentDecoder,
    SharedStructureConfig,
    SharedStructureMemory,
    SharedStructureRig,
    SkeletonSequence,
    StrictXYZQuantizer,
    frame_coordinate_loss,
)


class TinyBackbone(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.position = nn.Embedding(128, dim)
        block = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=2 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        length = inputs_embeds.shape[1]
        positions = self.position(torch.arange(length, device=inputs_embeds.device))
        causal = torch.ones(length, length, dtype=torch.bool, device=inputs_embeds.device).triu(1)
        hidden = self.blocks(inputs_embeds + positions, mask=causal,
                             src_key_padding_mask=~attention_mask.bool())
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


def settings() -> SharedStructureConfig:
    return SharedStructureConfig(dim=16, heads=4, shared_tokens=4, shared_depth=1,
                                 position_depth=1, max_joints=8, xyz_bins=16)


class SharedStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(733)

    def test_tree_roundtrip_padding_and_invalid_targets(self):
        tok = ParentSequenceTokens(8)
        parents = torch.tensor([[-1, 0, 0, 2], [-1, 0, -2, -2]])
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        encoded = tok.encode(parents, mask)
        self.assertEqual(encoded["input_ids"].tolist(), [[tok.bos, tok.root, 0, 0, 2],
                                                        [tok.bos, tok.root, 0, tok.pad, tok.pad]])
        self.assertEqual(encoded["labels"][1].tolist(), [tok.root, 0, tok.eos, -100, -100])
        for index, count in enumerate((4, 2)):
            self.assertEqual(tok.decode(encoded["labels"][index, :count + 1].tolist()),
                             parents[index, :count].tolist())
        invalid = parents.clone()
        invalid[0, 2] = 2
        with self.assertRaisesRegex(ValueError, "parent < child"):
            tok.encode(invalid, mask)
        with self.assertRaisesRegex(ValueError, "valid prefix"):
            tok.encode(parents, torch.tensor([[True, False, True, True], [True, True, False, False]]))
        with self.assertRaisesRegex(ValueError, "terminate naturally"):
            tok.decode([tok.root, 0])

    def test_grammar_never_exposes_future_parent_or_special_input_tokens(self):
        tok = ParentSequenceTokens(8)
        allowed = tok.allowed(torch.tensor([0, 1, 4, 8]))
        self.assertEqual(allowed[0].nonzero().flatten().tolist(), [tok.root])
        self.assertEqual(allowed[1].nonzero().flatten().tolist(), [0, tok.eos])
        self.assertEqual(allowed[2].nonzero().flatten().tolist(), [0, 1, 2, 3, tok.eos])
        self.assertFalse(bool(allowed[:, [tok.bos, tok.pad]].any()))

    def test_strict_quantizer_preserves_grid_and_rejects_clipping(self):
        quantizer = StrictXYZQuantizer(256, -1.0, 1.0)
        xyz = torch.tensor([[[[-1.0, 0.0, 1.0], [0.25, -0.25, 0.5]]]])
        valid = torch.tensor([[[True, False]]])
        labels = quantizer.encode(xyz, valid)
        self.assertEqual(labels[0, 0, 0].tolist(), [0, 128, 255])
        self.assertEqual(labels[0, 0, 1].tolist(), [-100, -100, -100])
        reconstructed = quantizer.decode(labels[valid])
        self.assertLessEqual(float((reconstructed - xyz[valid]).abs().max()), quantizer.step)
        xyz[0, 0, 0, 1] = 1.1
        with self.assertRaisesRegex(ValueError, "no clipping"):
            quantizer.encode(xyz, valid)
        xyz[0, 0, 0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            quantizer.encode(xyz, valid)

    def test_shared_pool_is_frame_permutation_invariant_and_input_dependent(self):
        memory = SharedStructureMemory(settings()).eval()
        anchors = torch.randn(2, 3, 7, 16)
        registers = torch.randn(2, 3, 2, 16)
        mask = torch.ones(2, 3, dtype=torch.bool)
        output = memory(anchors, registers, mask)
        order = torch.tensor([2, 0, 1])
        permuted = memory(anchors[:, order], registers[:, order], mask[:, order])
        torch.testing.assert_close(output, permuted, atol=2e-6, rtol=2e-6)
        changed = memory(anchors * -1, registers, mask)
        self.assertGreater(float((output - changed).abs().max()), 1e-3)

    def test_joint_features_use_own_token_not_future_structure(self):
        config = settings()
        decoder = SharedParentDecoder(TinyBackbone(config.dim), config).eval()
        memory = torch.randn(1, config.shared_tokens, config.dim)
        parents = torch.tensor([[-1, 0, 0, 1]])
        mask = torch.ones_like(parents, dtype=torch.bool)
        full = decoder.teacher_force(memory, parents, mask)
        changed = decoder.teacher_force(memory, torch.tensor([[-1, 0, 1, 2]]), mask)
        torch.testing.assert_close(full["joint_features"][:, :2], changed["joint_features"][:, :2])
        self.assertGreater(float((full["joint_features"][:, 2] - changed["joint_features"][:, 2]).abs().max()), 1e-4)
        for length in range(1, 6):
            logits, hidden = decoder.forward_prefix(memory, full["input_ids"][:, :length],
                                                      full["attention_mask"][:, :length])
            torch.testing.assert_close(logits[:, -1], full["logits"][:, length - 1], atol=2e-6, rtol=2e-6)
            if length > 1:
                torch.testing.assert_close(hidden[:, -1], full["joint_features"][:, length - 2],
                                           atol=2e-6, rtol=2e-6)

    def test_generation_eos_and_nontermination_are_distinct(self):
        config = settings()
        decoder = SharedParentDecoder(TinyBackbone(config.dim), config).eval()
        memory = torch.randn(1, config.shared_tokens, config.dim)
        with torch.no_grad():
            decoder.head.weight.zero_()
            decoder.head.bias.zero_()
            decoder.head.bias[decoder.tokens.eos] = 10.0
        complete = decoder.generate(memory)
        self.assertTrue(complete["success"])
        self.assertEqual(complete["parents"], [-1])
        self.assertEqual(complete["joint_features"].shape, (1, 1, config.dim))
        with torch.no_grad():
            decoder.head.bias[0] = 20.0
        failed = decoder.generate(memory)
        self.assertFalse(failed["success"])
        self.assertEqual(failed["status"], "max_joints")
        self.assertEqual(len(failed["tokens"]), config.max_joints)

    def test_all_frame_backward_reaches_memory_structure_and_position_modules(self):
        config = settings()
        model = SharedStructureRig(TinyBackbone(config.dim), config)
        anchors = torch.randn(2, 3, 7, config.dim, requires_grad=True)
        registers = torch.randn(2, 3, 2, config.dim, requires_grad=True)
        points = torch.randn(2, 3, 7, 3)
        parents = torch.tensor([[-1, 0, 1], [-1, 0, -2]])
        joint_mask = torch.tensor([[True, True, True], [True, True, False]])
        frame_mask = torch.ones(2, 3, dtype=torch.bool)
        targets = torch.rand(2, 3, 3, 3) - 0.5
        target_anchors = torch.randn(2, 3, 7, config.dim, requires_grad=True)
        out = model(anchors, registers, frame_mask, target_anchors, points,
                    parents, joint_mask, frame_mask, targets)
        self.assertEqual(out["xyz_logits"].shape, (2, 3, 3, 3, config.xyz_bins))
        out["loss"].backward()
        for parameter in (model.memory.queries, model.structure.embedding.weight,
                          model.structure.head.weight, model.positions.xyz_head.weight,
                          model.structure.backbone.blocks.layers[0].self_attn.in_proj_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for frame in range(3):
            self.assertGreater(float(anchors.grad[:, frame].abs().sum()), 0)
            self.assertGreater(float(registers.grad[:, frame].abs().sum()), 0)
            self.assertGreater(float(target_anchors.grad[:, frame].abs().sum()), 0)

    def test_coordinate_loss_does_not_reweight_assets_by_frame_count(self):
        logits = torch.randn(2, 1, 3, 3, 16, requires_grad=True)
        labels = torch.randint(0, 16, (2, 1, 3, 3))
        joints = torch.ones(2, 3, dtype=torch.bool)
        once = frame_coordinate_loss(logits, labels, joints, torch.ones(2, 1, dtype=torch.bool))
        repeated = frame_coordinate_loss(logits.repeat(1, 5, 1, 1, 1), labels.repeat(1, 5, 1, 1),
                                          joints, torch.ones(2, 5, dtype=torch.bool))
        torch.testing.assert_close(once, repeated)
        first = torch.autograd.grad(once.mean(), logits, retain_graph=True)[0]
        second = torch.autograd.grad(repeated.mean(), logits)[0]
        torch.testing.assert_close(first, second)

    def test_xyz_targets_do_not_enter_structure_predictions(self):
        config = settings()
        model = SharedStructureRig(TinyBackbone(config.dim), config).eval()
        anchors, regs, points = torch.randn(1, 2, 7, 16), torch.randn(1, 2, 2, 16), torch.randn(1, 2, 7, 3)
        parents, joints, frames = torch.tensor([[-1, 0]]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
        target = torch.zeros(1, 2, 2, 3)
        a = model(anchors, regs, frames, anchors, points, parents, joints, frames, target)
        b = model(anchors, regs, frames, anchors, points, parents, joints, frames, target + 0.5)
        torch.testing.assert_close(a["structure_logits"], b["structure_logits"])
        torch.testing.assert_close(a["joint_features"], b["joint_features"])
        self.assertNotEqual(float(a["xyz_loss"]), float(b["xyz_loss"]))

    def test_new_target_meshes_never_enter_structure_generation(self):
        config = settings()
        model = SharedStructureRig(TinyBackbone(config.dim), config).eval()
        source, regs = torch.randn(1, 2, 7, 16), torch.randn(1, 2, 2, 16)
        source_mask = torch.ones(1, 2, dtype=torch.bool)
        target, points = torch.randn(1, 3, 5, 16), torch.randn(1, 3, 5, 3)
        target_mask = torch.ones(1, 3, dtype=torch.bool)
        parents, mask = torch.tensor([[-1, 0]]), torch.ones(1, 2, dtype=torch.bool)
        xyz = torch.zeros(1, 3, 2, 3)
        a = model(source, regs, source_mask, target, points, parents, mask, target_mask, xyz)
        b = model(source, regs, source_mask, -target, points + 0.3, parents, mask, target_mask, xyz)
        torch.testing.assert_close(a["joint_features"], b["joint_features"])
        torch.testing.assert_close(a["structure_logits"], b["structure_logits"])
        self.assertGreater(float((a["xyz_logits"] - b["xyz_logits"]).abs().max()), 1e-4)

    def test_saved_sequence_rebinds_without_source_encoder_or_ar(self):
        config = settings()
        model = SharedStructureRig(TinyBackbone(config.dim), config).eval()
        with torch.no_grad():
            model.structure.head.weight.zero_()
            model.structure.head.bias.zero_()
            model.structure.head.bias[model.structure.tokens.eos] = 10.0
        generated = model.generate_structure(torch.randn(1, 2, 7, 16), torch.randn(1, 2, 2, 16),
                                             torch.ones(1, 2, dtype=torch.bool))
        sequence = generated["skeleton_sequence"]
        restored = SkeletonSequence.from_record(sequence.cpu_record())
        torch.testing.assert_close(sequence.joint_features, restored.joint_features)
        torch.testing.assert_close(sequence.parents, restored.parents)
        target, points = torch.randn(1, 3, 5, 16), torch.randn(1, 3, 5, 3)
        saved = restored.joint_features.clone()
        with patch.object(model.memory, "forward", side_effect=AssertionError("source was rerun")), \
             patch.object(model.structure, "forward_prefix", side_effect=AssertionError("AR was rerun")):
            batch = model.localize(restored, target, points)
            single = model.localize(restored, target[:, 1:2], points[:, 1:2])
            changed = model.localize(restored, target * -1, points)
        torch.testing.assert_close(batch["xyz_logits"][:, 1:2], single["xyz_logits"], atol=1e-6, rtol=1e-6)
        self.assertGreater(float((batch["xyz_logits"] - changed["xyz_logits"]).abs().max()), 1e-4)
        torch.testing.assert_close(saved, restored.joint_features)
        self.assertEqual(restored.joint_features.ndim, 3)

    def test_localization_loss_alone_trains_skeleton_representation(self):
        config = settings()
        model = SharedStructureRig(TinyBackbone(config.dim), config)
        source, regs = torch.randn(1, 2, 7, 16), torch.randn(1, 2, 2, 16)
        target, points = torch.randn(1, 3, 5, 16), torch.randn(1, 3, 5, 3)
        out = model(source, regs, torch.ones(1, 2, dtype=torch.bool), target, points,
                    torch.tensor([[-1, 0, 1]]), torch.ones(1, 3, dtype=torch.bool),
                    torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, 3, 3))
        out["xyz_loss"].backward()
        for parameter in (model.memory.queries, model.structure.embedding.weight,
                          model.structure.backbone.blocks.layers[0].self_attn.in_proj_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(model.structure.head.weight.grad)

    def test_configuration_validation(self):
        for kwargs in ({"dim": 15}, {"shared_tokens": 0}, {"xyz_bins": 1}, {"dropout": 1.0}):
            with self.assertRaises(ValueError):
                replace(settings(), **kwargs)


if __name__ == "__main__":
    unittest.main()
