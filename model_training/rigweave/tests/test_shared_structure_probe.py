from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_shared_structure import evidence_indices, target_indices, validate_data_plan
from rigweave.dynamic_rig.model import DynamicRigConditioner
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingMotionEncoder
from rigweave.dynamic_rig.shared_structure import SharedStructureRig
from rigweave.dynamic_rig.shared_structure_dynamic import (
    SharedStructureDynamicRig, SharedSurfaceSequence, subset_frame_targets,
)
from test_shared_structure import TinyBackbone, settings


class SharedStructureProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        path = Path(__file__).resolve().parents[2] / "experiments/shared_structure_probe_20261011.json"
        return json.loads(path.read_text())["data"]

    def test_dynamic_views_exclude_reuse_meshes_and_retain_full_target_selection(self):
        data = self.data()
        validate_data_plan(data)
        for count in range(data["frames_min"], data["frames_max"] + 1):
            for seed in (0, 13, 999):
                source = evidence_indices(data, count, seed=seed)
                target = target_indices(data, source)
                self.assertEqual(source.numel(), count)
                self.assertEqual(source.unique().numel(), count)
                self.assertEqual(int(source[0]), 0)
                self.assertTrue(bool((source < data["source_pool_frames"]).all()))
                self.assertEqual(target.tolist()[:count], source.tolist())
                self.assertEqual(target.tolist()[count:], list(range(data["source_pool_frames"], data["pool_frames"])))
        with self.assertRaisesRegex(ValueError, "excluded target"):
            target_indices(data, torch.tensor([0, 24]))
        with self.assertRaisesRegex(ValueError, "unique"):
            target_indices(data, torch.tensor([0, 0]))

    def test_target_subset_keeps_affine_and_identity_aligned(self):
        parents = torch.tensor([[-1, 0]])
        batch = {"frame_target_joints": torch.arange(24.0).reshape(1, 4, 2, 3),
                 "selected_frames": torch.tensor([[19, 4, 11, 30]]),
                 "frame_mask": torch.ones(1, 4, dtype=torch.bool), "target_parents": parents}
        for name in ("target_mesh_centers", "target_source_centers"):
            batch[name] = torch.arange(12.0).reshape(1, 4, 3)
        for name in ("target_mesh_scales", "target_source_scales"):
            batch[name] = torch.arange(1.0, 5.0)[None]
        output = subset_frame_targets(batch, torch.tensor([2, 0]))
        self.assertEqual(output["selected_frames"].tolist(), [[11, 19]])
        self.assertEqual(output["target_source_scales"].tolist(), [[3.0, 1.0]])
        self.assertIs(output["target_parents"], parents)
        torch.testing.assert_close(output["target_joints"], batch["frame_target_joints"][:, 2])

    def test_dynamic_wrapper_uses_source_only_for_structure_and_independent_target_anchors(self):
        torch.manual_seed(54)
        cfg = settings()
        encoder = AnchorWiseAlternatingMotionEncoder(dim=cfg.dim, heads=cfg.heads, depth=1,
                                                      register_tokens=2, max_frames=4)
        model = SharedStructureDynamicRig(DynamicRigConditioner(nn.Identity(), encoder),
                                           SharedStructureRig(TinyBackbone(cfg.dim), cfg)).eval()
        source = SharedSurfaceSequence(torch.randn(1, 2, 7, 16), torch.randn(1, 2, 7, 3),
                                       torch.arange(7)[None], None)
        target = SharedSurfaceSequence(torch.randn(1, 3, 5, 16), torch.randn(1, 3, 5, 3),
                                       torch.arange(5)[None], None)
        batch = {"target_parents": torch.tensor([[-1, 0]]), "joint_mask": torch.ones(1, 2, dtype=torch.bool),
                 "frame_mask": torch.ones(1, 3, dtype=torch.bool), "frame_target_joints": torch.zeros(1, 3, 2, 3)}
        with patch.object(model, "encode_surface", wraps=model.encode_surface) as encode:
            output = model.forward_surface(source, target, batch)
            self.assertEqual(encode.call_count, 1)
            self.assertIs(encode.call_args.args[0], source)
        with patch.object(model, "encode_surface", side_effect=AssertionError("structure encoder reran")):
            rebound = model.localize_surface(output["skeleton_sequence"], target)
        torch.testing.assert_close(output["xyz_logits"], rebound["xyz_logits"])
        changed = SharedSurfaceSequence(target.tokens * -1, target.query_points, target.query_indices, None)
        altered = model.forward_surface(source, changed, batch)
        torch.testing.assert_close(output["joint_features"], altered["joint_features"])
        self.assertGreater(float((output["xyz_logits"] - altered["xyz_logits"]).detach().abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
