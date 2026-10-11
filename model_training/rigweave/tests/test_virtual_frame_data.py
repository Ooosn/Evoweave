from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rigweave.dynamic_rig.data import _select_query_sequence, select_mesh_frame_ids
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_frame_data import (
    VirtualFrameDataset, collate_virtual_targets, prepare_virtual_surface, read_virtual_manifest,
)


def payload(frames=41, joints=4):
    base = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
    offset = np.arange(frames, dtype=np.float32)[:, None, None] * np.asarray([2, -1, .5], dtype=np.float32)
    vertices = base[None] + offset
    targets = np.linspace(.1, .8, joints * 3, dtype=np.float32).reshape(joints, 3)[None] + offset
    return {
        "canonical_schema_version": np.asarray("rootless_dynamic_npz_v3", dtype=object),
        "frame_vertices_rootspace": vertices, "target_joints_rootspace": targets,
        "faces": np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
        "target_parents": np.arange(-1, joints - 1, dtype=np.int64),
        "target_has_skin": np.arange(joints) < 2, "target_is_synthetic_root": np.zeros(joints, dtype=bool),
    }


class VirtualFrameDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_query_selector_preserves_legacy_selection(self):
        raw = payload()
        args = dict(frame_count=8, path=Path("synthetic.npz"), index=4, random_query=True, seed=73,
                    sampling_seed=391, motion_fps_ratio=.75, minimum_random_frames=1, motion_vertex_samples=4)
        selected = select_mesh_frame_ids(raw["frame_vertices_rootspace"], **args)
        meshes, joints, tails, legacy_ids = _select_query_sequence(
            raw["frame_vertices_rootspace"], raw["target_joints_rootspace"], None, **args)
        np.testing.assert_array_equal(selected, legacy_ids)
        np.testing.assert_array_equal(meshes, raw["frame_vertices_rootspace"][selected])
        np.testing.assert_array_equal(joints, raw["target_joints_rootspace"][selected[0]])
        self.assertIsNone(tails)

    def test_all_input_frames_have_exact_common_chart_supervision_without_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.npz"
            raw = payload()
            np.savez(path, **raw)
            dataset = VirtualFrameDataset([{"asset_id": "one", "path": str(path)}])
            sample = dataset[FrameSampleRequest(0, 12, 399)]
            ids = sample["selected_frames"][0].numpy()
            self.assertFalse(set(ids) & set(sample["heldout_raw_ids"]))
            self.assertEqual(sample["raw_frame_count"], 41)
            self.assertEqual(sample["frame_target_joints"].shape, (1, 12, 4, 3))
            center = sample["query_center"][0].numpy()
            scale = sample["query_scale"].item()
            np.testing.assert_allclose(sample["frame_target_joints"][0].numpy(),
                                       (raw["target_joints_rootspace"][ids] - center) / scale, rtol=1e-6)
            np.testing.assert_allclose(sample["frame_vertices"][0].numpy(),
                                       (raw["frame_vertices_rootspace"][ids] - center) / scale, rtol=1e-6)
            self.assertGreater(sample["frame_vertices"].abs().max().item(), 1)
            self.assertGreater(sample["frame_target_joints"].abs().max().item(), 1)
            self.assertEqual(sample["joint_count"].item(), 4)
            self.assertNotIn("target_mesh_scales", sample)
            heldout = VirtualFrameDataset(dataset.entries, mode="heldout_eval", frames=8)[0]
            self.assertEqual(set(heldout["selected_frames"][0].tolist()), set(sample["heldout_raw_ids"]))

    def test_surface_sampling_is_replayable_and_not_dependent_on_other_batch_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for index in range(2):
                path = Path(directory) / f"asset{index}.npz"
                np.savez(path, **payload(joints=4 - index))
                rows.append({"asset_id": str(index), "path": str(path)})
            dataset = VirtualFrameDataset(rows)
            batches = [dataset[FrameSampleRequest(i, 4, 91 + i)] for i in range(2)]
            before = torch.get_rng_state().clone()
            surface, target = prepare_virtual_surface(batches, "cpu", surface_samples=24, vertex_samples=8, query_tokens=7)
            torch.testing.assert_close(torch.get_rng_state(), before)
            self.assertEqual(surface["dense_points"].shape, (2, 4, 24, 3))
            self.assertEqual(surface["query_points"].shape, (2, 4, 7, 3))
            self.assertEqual(target["joint_mask"].tolist(), [[True] * 4, [True] * 3 + [False]])
            self.assertEqual(target["target_parents"][1, -1].item(), -2)
            self.assertEqual(target["frame_target_joints"][1, :, -1].abs().sum().item(), 0)
            for index in range(2):
                one, _ = prepare_virtual_surface([batches[index]], "cpu", surface_samples=24, vertex_samples=8, query_tokens=7)
                for key in surface:
                    torch.testing.assert_close(one[key], surface[key][index:index + 1])
                for frame in range(4):
                    torch.testing.assert_close(surface["query_points"][index, frame],
                                               surface["dense_points"][index, frame, surface["query_indices"][index]])
            with self.assertRaisesRegex(ValueError, "same real frame"):
                collate_virtual_targets([batches[0], dataset[FrameSampleRequest(1, 3, 92)]], "cpu")

    def test_manifest_identity_is_exact_and_does_not_rescan_backing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"asset_id": "not-opened", "path": str(root / "not_loaded.npz")}]
            path = root / "train.jsonl"
            path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(read_virtual_manifest(path, expected_count=1, expected_sha256=digest), rows)
            with self.assertRaisesRegex(ValueError, "hash"):
                read_virtual_manifest(path, expected_count=1, expected_sha256="wrong")
            with self.assertRaisesRegex(ValueError, "count"):
                read_virtual_manifest(path, expected_count=2, expected_sha256=digest)


if __name__ == "__main__":
    unittest.main()
