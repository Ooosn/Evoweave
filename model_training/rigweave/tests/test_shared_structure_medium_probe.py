from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_shared_structure_medium as medium
from rigweave.dynamic_rig.shared_structure import SharedStructureConfig, SharedStructureRig


class CausalBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        return SimpleNamespace(last_hidden_state=self.projection(inputs_embeds).cumsum(dim=1))


def sample_entry():
    def surface(frames, motion):
        return {"tokens": torch.randn(1, frames, 3, 8), "query_points": torch.randn(1, frames, 3, 3),
                "query_indices": torch.tensor([[0, 1, 2]]),
                "dense_points": torch.randn(1, frames, 5, 3) if motion else None}
    train, heldout = list(range(32)), list(range(32, 40))
    joints = torch.rand(1, 40, 3, 3) - .5
    return {"split": "train", "index": 7, "asset_id": "fixture", "joint_count": 3,
        "partition": {"train_raw_ids": train, "heldout_raw_ids": heldout},
        "source_frame_ids": {"train_a": train, "train_b": train[16:] + train[:16], "heldout": heldout},
        "source": {"train_a": surface(32, True), "train_b": surface(32, True), "heldout": surface(8, True)},
        "target": surface(40, False),
        "batch": {"path": ["fixture.npz"], "cls": ["articulationxl"], "target_parents": torch.tensor([[-1, 0, 0]]),
            "joint_count": torch.tensor([3]), "joint_mask": torch.ones(1, 3, dtype=torch.bool),
            "frame_mask": torch.ones(1, 40, dtype=torch.bool), "frame_target_joints": joints,
            "target_joints": joints[:, 0], "selected_frames": torch.arange(40)[None],
            "query_center": torch.zeros(1, 3), "query_scale": torch.ones(1),
            "target_mesh_centers": torch.zeros(1, 40, 3), "target_mesh_scales": torch.ones(1, 40),
            "target_source_centers": torch.zeros(1, 40, 3), "target_source_scales": torch.ones(1, 40)}}


class MediumProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(871)

    def test_raw_frame_lookup_preserves_order_and_rejects_duplicates(self):
        self.assertEqual(medium.raw_frame_slots(torch.tensor([[9, 2, 7]]), [7, 9]).tolist(), [2, 0])
        for available, requested in (([1, 1], [1]), ([1, 2], [1, 1]), ([1, 2], [3]), ([1, 2], [])):
            with self.subTest(available=available, requested=requested), self.assertRaises(ValueError):
                medium.raw_frame_slots(torch.tensor([available]), requested)

    def test_dynamic_evidence_slots_keep_query_and_are_replayable(self):
        for count in (2, 8, 24):
            a = medium.evidence_slots(32, count, seed=17)
            b = medium.evidence_slots(32, count, seed=17)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a[0].item(), 0)
            self.assertEqual(len(a.unique()), count)
        with self.assertRaises(ValueError):
            medium.evidence_slots(8, 24)

    def test_cached_training_physically_excludes_heldout_targets_and_inputs(self):
        entry = sample_entry()
        for bank in medium.TRAIN_BANKS:
            for frames in (2, 8, 24):
                source, target, batch, selected = medium.select_cached_input(entry, bank, frames,
                    training=True, seed=83, device="cpu")
                self.assertEqual(source.tokens.shape[1], frames)
                self.assertEqual(target.tokens.shape[1], 32)
                self.assertEqual(batch["selected_frames"].tolist(), [list(range(32))])
                self.assertFalse(set(selected["source_raw_frames"]) & set(range(32, 40)))
                self.assertEqual(selected["heldout_target_positions"], [])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            medium.select_cached_input(entry, "heldout", 8, training=True, device="cpu")
        _, _, _, selected = medium.select_cached_input(entry, "heldout", 8, training=False, device="cpu")
        self.assertEqual(selected["source_raw_frames"], list(range(32, 40)))
        self.assertEqual(selected["heldout_target_positions"], list(range(8, 16)))

    def test_heldout_feature_and_label_mutations_do_not_change_training_loss_or_gradients(self):
        entry = sample_entry()
        changed = deepcopy(entry)
        changed["target"]["tokens"][:, 32:] = 1e4
        changed["target"]["query_points"][:, 32:] = -1e4
        changed["batch"]["frame_target_joints"][:, 32:] = 1e4
        for value in changed["source"]["heldout"].values():
            if isinstance(value, torch.Tensor):
                value.zero_()
        config = SharedStructureConfig(dim=8, heads=2, shared_tokens=2, shared_depth=1,
                                       position_depth=1, max_joints=8, xyz_bins=16)
        model = SharedStructureRig(CausalBackbone(8), config).train()
        outputs = []
        for data in (entry, changed):
            source, target, batch, _ = medium.select_cached_input(data, "train_b", 8,
                training=True, seed=48, device="cpu")
            model.zero_grad(set_to_none=True)
            output = model(source.tokens, source.tokens.mean(dim=2, keepdim=True),
                torch.ones(1, 8, dtype=torch.bool), target.tokens, target.query_points,
                batch["target_parents"], batch["joint_mask"], batch["frame_mask"], batch["frame_target_joints"])
            output["loss"].backward()
            outputs.append((output["loss"].detach().clone(),
                            {n: p.grad.detach().clone() for n, p in model.named_parameters()}))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        for name, gradient in outputs[0][1].items():
            self.assertTrue(torch.equal(gradient, outputs[1][1][name]), name)

    def test_invalid_partition_or_same_query_fails(self):
        entry = sample_entry()
        entry["partition"]["heldout_raw_ids"][0] = 0
        with self.assertRaisesRegex(ValueError, "disjoint"):
            medium.validate_entry(entry)
        entry = sample_entry()
        entry["source_frame_ids"]["train_b"] = entry["source_frame_ids"]["train_a"]
        with self.assertRaisesRegex(ValueError, "different query"):
            medium.validate_entry(entry)

    def test_shard_roundtrip_and_checksum_reject_corruption(self):
        entry = sample_entry()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.pt"
            torch.save(entry, path)
            metadata = {key: entry[key] for key in ("split", "index", "asset_id", "partition", "source_frame_ids")}
            metadata.update(shard_path=str(path), shard_bytes=path.stat().st_size,
                            shard_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            restored = medium.load_entry(metadata)
            self.assertTrue(torch.equal(restored["target"]["tokens"], entry["target"]["tokens"]))
            self.assertEqual(medium.verify_shards({"train": [metadata], "valid": []})["shards"], 1)
            del restored
            metadata["shard_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "changed"):
                medium.verify_shards({"train": [metadata]})

    def test_identity_swaps_are_not_hidden_by_per_frame_matching(self):
        joints = torch.tensor([[[[0., 0., 0.], [.3, 0., 0.], [0., .3, 0.]],
                                [[0., 0., 0.], [.4, 0., 0.], [0., .4, 0.]]]])
        batch = {"target_parents": torch.tensor([[-1, 0, 0]]), "joint_mask": torch.ones(1, 3, dtype=torch.bool),
                 "frame_target_joints": joints}
        result = {"success": True, "status": "eos", "tokens": [1], "parents": [-1, 0, 0], "joints": joints.clone()}
        correct = medium.generation_metrics(result, batch, [0, 1], padding=1.1)
        self.assertEqual(correct["heldout_stable_identity_coverage_all"], 1.)
        self.assertEqual(correct["heldout_unique_joint_coverage"], 1.)
        result["joints"] = joints[:, :, [0, 2, 1]]
        swapped = medium.generation_metrics(result, batch, [0, 1], padding=1.1)
        self.assertEqual(swapped["heldout_unique_joint_coverage"], 1.)
        self.assertAlmostEqual(swapped["heldout_stable_identity_coverage_all"], 1 / 3, places=6)

    def test_failed_generation_stays_in_summary_denominator(self):
        failed = medium.generation_metrics({"success": False, "status": "max_joints", "tokens": [1]},
                                            {}, [0], padding=1.1)
        success = {**failed, "success": True, "status": "eos", "exact_tree": True, "count_correct": True,
                   "heldout_identity_mpjpe": .2, "heldout_unique_joint_coverage": 1.,
                   "heldout_located_edge_recall": 1., "heldout_stable_identity_coverage_all": .5}
        rows = [{"split": "valid", "view": "T8", "structure_ce": 1., "teacher_train_mpjpe": .1,
                 "teacher_heldout_mpjpe": .2, "generation": generation} for generation in (failed, success)]
        summary = medium.summarize_rows(rows)["valid/T8"]
        self.assertEqual(summary["assets"], 2)
        self.assertEqual(summary["natural_eos"], 1)
        self.assertEqual(summary["generated_heldout_identity_assets"], 1)
        self.assertEqual(summary["generated_heldout_unique_joint_coverage_all"], .5)
        self.assertEqual(summary["generated_heldout_stable_identity_coverage_all"], .25)

    def test_declared_budget_and_view_counts_are_explicit(self):
        config_path = Path(__file__).resolve().parents[2] / "experiments/shared_structure_medium_20261011.json"
        config = json.loads(config_path.read_text())
        medium.validate_medium_config(config)
        self.assertEqual(config["probe"]["steps"] * config["probe"]["accumulation"], 16384)
        initial, final = 0, 0
        for count in (config["probe"]["evaluate_train_assets"], config["data"]["valid_assets"]):
            initial += sum(len(medium.evaluation_views(config, i, "initialized")) for i in range(count))
            final += sum(len(medium.evaluation_views(config, i, "trained")) for i in range(count))
        self.assertEqual((initial, final), (96, 224))
        changed = deepcopy(config)
        changed["probe"]["steps"] = 4095
        with self.assertRaisesRegex(ValueError, "complete training-asset passes"):
            medium.validate_medium_config(changed)


if __name__ == "__main__":
    unittest.main()
