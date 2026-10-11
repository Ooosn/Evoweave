from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_shared_structure_medium as medium
import shared_structure_online_batch as batching
import test_shared_structure_medium_data as fixtures
from test_shared_structure_medium_probe import CausalBackbone
from rigweave.dynamic_rig.sampling import sample_trackable_surface
from rigweave.dynamic_rig.shared_structure import SharedStructureConfig, SharedStructureRig


class Surface(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, points, normals, query, query_normals):
        self.batch_sizes.append(query.shape[0])
        value = query + points.mean(1, keepdim=True) + query_normals * .1
        return torch.cat((value, value.square(), query_normals[..., :2]), dim=-1)


class OnlineBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(41)
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = {"data": {"pool_frames": 40, "heldout_frames": 8, "frame_seed": 21, "target_mesh_padding": 1.1}}
        fixture = fixtures.SharedStructureMediumDataTest()
        rows = []
        for index, joints in enumerate((4, 3)):
            raw = fixture.payload()
            raw["target_joints_rootspace"] = raw["target_joints_rootspace"][:, :joints]
            for key in ("target_parents", "target_has_skin", "target_is_synthetic_root"):
                raw[key] = raw[key][:joints]
            if index:
                vertices = raw["frame_vertices_rootspace"]
                raw["frame_vertices_rootspace"] = np.concatenate((vertices, vertices.mean(1, keepdims=True)), axis=1)
            path = fixture.write(root, raw, name=f"asset_{index}.npz")
            rows.append({"asset_id": f"fixture_{index}", "index": index, "path": str(path), "joint_count": joints})
        self.entries = medium.online_entries({"train": rows}, self.config)["train"]
        self.surface = Surface()
        conditioner = SimpleNamespace(surface_tokenizer=self.surface,
                                      motion_encoder=SimpleNamespace(motion_evidence_fusion="bias"))

        def references(batch):
            return sample_trackable_surface(batch["frame_vertices"][:, 0], batch["faces"], num_samples=12,
                vertex_samples=4, query_tokens=3, vertex_counts=batch["vertex_count"], face_counts=batch["face_count"])

        self.original = SimpleNamespace(conditioner=conditioner, sample_references=references)

    def tearDown(self):
        self.temp.cleanup()

    def cases(self):
        serial = [medium.select_online_input(self.original, entry, self.config, bank, 8,
            training=True, seed=seed, device="cpu")
            for entry, bank, seed in zip(self.entries, ("train_a", "train_b"), (117, 118))]
        self.surface.batch_sizes.clear()
        rng = torch.get_rng_state().clone()
        batched = batching.select_online_batch(self.original, self.entries, self.config,
            ["train_a", "train_b"], 8, training=True, seeds=[117, 118], device="cpu")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(self.surface.batch_sizes, [2] * 40)
        return serial, batched

    def test_true_batched_features_equal_serial_and_metadata_preserves_identity(self):
        serial, combined = self.cases()
        for field in ("tokens", "query_points", "query_indices", "dense_points"):
            expected = torch.cat([getattr(case[0], field) for case in serial])
            torch.testing.assert_close(getattr(combined[0], field), expected, atol=0, rtol=0)
        for field in ("tokens", "query_points", "query_indices"):
            torch.testing.assert_close(getattr(combined[1], field), torch.cat([getattr(case[1], field) for case in serial]), atol=0, rtol=0)
        batch = combined[2]
        self.assertEqual(batch["joint_mask"].tolist(), [[True] * 4, [True] * 3 + [False]])
        self.assertEqual(batch["joint_count"].tolist(), [4, 3])
        self.assertEqual(batch["target_parents"][1, -1].item(), -2)
        for index, case in enumerate(serial):
            count = case[2]["joint_count"].item()
            torch.testing.assert_close(batch["frame_target_joints"][index, :, :count], case[2]["frame_target_joints"][0])
            torch.testing.assert_close(batch["target_source_centers"][index], case[2]["target_source_centers"][0])
            self.assertEqual(combined[3][index], case[3])
            self.assertFalse(set(combined[3][index]["target_raw_frames"]) & set(self.entries[index]["partition"]["heldout_raw_ids"]))

    def test_batch_loss_and_all_gradients_equal_mean_of_serial_assets(self):
        serial, combined = self.cases()
        config = SharedStructureConfig(dim=8, heads=2, shared_tokens=2, shared_depth=1,
                                       position_depth=1, max_joints=8, xyz_bins=16)
        model = SharedStructureRig(CausalBackbone(8), config).train()

        def forward(case):
            source, target, batch, _ = case
            return model(source.tokens, source.tokens.mean(2, keepdim=True),
                torch.ones(source.tokens.shape[:2], dtype=torch.bool), target.tokens, target.query_points,
                batch["target_parents"], batch["joint_mask"], batch["frame_mask"], batch["frame_target_joints"])["loss"]

        loss = forward(combined)
        loss.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        individual = sum(forward(case) for case in serial) / len(serial)
        individual.backward()
        torch.testing.assert_close(loss, individual, atol=1e-6, rtol=1e-5)
        for name, p in model.named_parameters():
            torch.testing.assert_close(gradients[name], p.grad, atol=2e-5, rtol=2e-5, msg=name)

    def test_rejects_mismatched_lists_or_frame_counts(self):
        with self.assertRaises(ValueError):
            batching.select_online_batch(self.original, self.entries, self.config, ["train_a"], 8,
                                         training=True, seeds=[1, 2], device="cpu")
        serial, _ = self.cases()
        changed = deepcopy(serial[1][2])
        changed["frame_target_joints"] = changed["frame_target_joints"][:, :1]
        with self.assertRaises(ValueError):
            batching.collate_targets([serial[0][2], changed])


if __name__ == "__main__":
    unittest.main()
