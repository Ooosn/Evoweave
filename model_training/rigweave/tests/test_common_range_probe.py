import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from probe_common_range import require_formal_started, teacher_metrics, transform_batch


class CommonRangeProbeTests(unittest.TestCase):
    def batch(self):
        vertices = torch.randn(1, 8, 6, 3)
        return {"frame_vertices": vertices, "vertex_normals": torch.ones_like(vertices),
                "face_normals": torch.ones(1, 8, 4, 3), "input_ids": torch.tensor([[8, 9, 0, 1, 7]]),
                "target_joints": torch.zeros(1, 2, 3), "target_parents": torch.tensor([[-1, 0]]),
                "query_center": torch.zeros(1, 3), "query_scale": torch.ones(1)}

    def test_static_preserves_target_and_repeats_query(self):
        batch = self.batch()
        result = transform_batch(batch, {}, "static")
        self.assertIs(result["input_ids"], batch["input_ids"])
        self.assertTrue(torch.equal(result["frame_vertices"], batch["frame_vertices"][:, :1].expand_as(batch["frame_vertices"])))

    def test_mesh_gauge_translation_and_normal_behavior(self):
        batch = self.batch()
        gauge = {"selected_positions": list(range(8)), "gauges": {"rigid": {"available": True,
            "rotations": np.repeat(np.eye(3)[None], 8, 0).tolist(), "translations_common": np.ones((8, 3)).tolist()}}}
        result = transform_batch(batch, gauge, "rigid")
        self.assertTrue(torch.equal(result["frame_vertices"][:, 0], batch["frame_vertices"][:, 0]))
        self.assertTrue(torch.allclose(result["frame_vertices"][:, 1:], batch["frame_vertices"][:, 1:] + 1))
        self.assertTrue(torch.equal(result["vertex_normals"], batch["vertex_normals"]))
        self.assertIs(result["target_joints"], batch["target_joints"])

    def test_skeleton_oracle_cannot_enter_mesh_only_inference(self):
        with self.assertRaisesRegex(ValueError, "mesh-only"):
            transform_batch(self.batch(), {"gauges": {"root": {"available": True, "requires_gt_skeleton": True}}}, "root")

    def test_teacher_shift_and_token_groups(self):
        ids = torch.tensor([[5, 6, 1, 2, 4]])
        token_logits = torch.zeros(1, 7, 7)
        for index, label in enumerate(ids[0, 1:]):
            token_logits[0, 2 + index, label] = 9
        class Transformer:
            dtype = torch.float32
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=token_logits)
        model = SimpleNamespace(transformer=Transformer(), tokenizer=SimpleNamespace(num_discrete=3, eos=4, token_id_branch=3),
                                token_inputs_embeds=lambda token_ids, mask: torch.zeros(1, 5, 2))
        result = teacher_metrics(model, torch.zeros(1, 2, 2), {"input_ids": ids, "attention_mask": torch.ones(1, 5)})
        self.assertEqual(result["all"]["accuracy"], 1.)
        self.assertEqual(result["coordinate"]["tokens"], 2)
        self.assertEqual(result["eos"]["accuracy"], 1.)
        self.assertIsNone(result["branch"]["ce"])

    def test_formal_start_needs_a_real_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller, rank = Path(temporary) / "controller.json", Path(temporary) / "rank.json"
            controller.write_text(json.dumps({"status": "running", "plan": {"plan": "formal", "label": "formal_v1"}}))
            rank.write_text(json.dumps({"samples": 0, "initialization": {"old_method_trained_weights_loaded": False}}))
            with self.assertRaisesRegex(ValueError, "completed fresh-model"):
                require_formal_started(controller, rank)
            rank.write_text(json.dumps({"samples": 64, "initialization": {"old_method_trained_weights_loaded": False}}))
            self.assertEqual(require_formal_started(controller, rank)["observed_formal_samples"], 64)


if __name__ == "__main__":
    unittest.main()
