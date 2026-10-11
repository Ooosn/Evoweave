import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_shared_structure_full as full
import probe_shared_structure_medium as medium
from rigweave.dynamic_rig.shared_structure import SharedStructureConfig, SharedStructureRig, StrictXYZQuantizer
from test_shared_structure_medium_probe import CausalBackbone
from test_shared_structure_online_batch import OnlineBatchTest
from test_shared_structure_controller import SharedStructureControllerTest


class FullStreamTest(unittest.TestCase):
    def test_bounded_geometry_matches_existing_official_formulas(self):
        path = Path(full.__file__).with_name("eval_dynamic_rig_generation.py")
        names = {"_bones_from_parents", "_sample_bones_np", "_official_skeleton_metrics"}
        nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        rng = np.random.default_rng(21)
        a, b = rng.normal(size=(6, 3)).astype(np.float32), rng.normal(size=(8, 3)).astype(np.float32)
        pa, pb = [-1, 0, 0, 1, 1, 2], [-1, 0, 0, 1, 1, 2, 2, 3]
        bones = namespace["_bones_from_parents"]
        expected = namespace["_official_skeleton_metrics"](a, bones(a, pa), b, bones(b, pb), (-1/1.1, 1/1.1))
        with mock.patch.dict(sys.modules, {"eval_dynamic_rig_generation": SimpleNamespace(**namespace)}):
            actual = full.geometry_metrics(a, pa, b, pb, 1.1)
        np.testing.assert_allclose([actual[k] for k in actual], [expected[k] for k in actual], rtol=1e-6)

    def test_global_stream_crosses_epoch_without_dropping_or_padding(self):
        stream = full.AssetStream(37, 15)
        selected = []
        for _ in range(37):
            ids, frames, banks, seeds = stream.take(16, 2, 24)
            selected.extend(ids)
            self.assertTrue(2 <= frames <= 24)
            self.assertEqual(len(banks), 16)
            self.assertEqual(len(seeds), 16)
        for start in range(0, len(selected), 37):
            self.assertEqual(set(selected[start:start+37]), set(range(37)))
        np.testing.assert_array_equal(stream.exposures, np.full(37, 16))

    def test_rank_planners_are_identical_and_shards_disjoint_in_microbatch(self):
        a, b = full.AssetStream(97, 22), full.AssetStream(97, 22)
        for _ in range(5):
            left, right = a.take(16, 2, 24), b.take(16, 2, 24)
            self.assertEqual(left, right)
            self.assertFalse(set(left[0][:8]) & set(right[0][8:]))

    def test_warmstart_extension_preserves_old_centers_and_every_other_weight(self):
        old = SharedStructureConfig(dim=8, heads=2, shared_tokens=2, shared_depth=1,
                                    position_depth=1, max_joints=8, xyz_bins=16)
        new = SharedStructureConfig(**{**old.__dict__, "xyz_bins": 32, "xyz_min": -2., "xyz_max": 2.})
        source = SharedStructureRig(CausalBackbone(8), old)
        target = SharedStructureRig(CausalBackbone(8), new)
        identity = {"path": "checkpoint", "sha256": "a"*64}
        payload = {"configuration": {"architecture": old.__dict__}, "skeleton": source.state_dict(),
                   "step": 1024, "source_checkpoint": identity, "feature_encoder_frozen": True,
                   "training_accounting": {"assets": 4096, "heldout_frame_exposures": 0}}
        config = {"architecture": new.__dict__, "baseline": {"checkpoint": "checkpoint", "checkpoint_sha256": "a"*64},
                  "full": {"warmstart": {"step": 1024}}}
        info = full.warmstart_skeleton(SimpleNamespace(skeleton=target), payload, config)
        self.assertEqual(info["xyz_bin_offset"], 8)
        for key, value in source.state_dict().items():
            if key.startswith("positions.xyz_head."):
                actual = target.state_dict()[key].reshape(3, 32, *value.shape[1:])[:, 8:24]
                torch.testing.assert_close(actual, value.reshape(3, 16, *value.shape[1:]), atol=0, rtol=0)
            else:
                torch.testing.assert_close(target.state_dict()[key], value, atol=0, rtol=0)
        torch.testing.assert_close(source.quantizer.decode(torch.arange(16)), target.quantizer.decode(torch.arange(16)+8), atol=0, rtol=0)


class FullSamplingTest(OnlineBatchTest):
    def test_target_subsample_is_seeded_eligible_and_rotates(self):
        self.config["data"]["target_frames_per_exposure"] = 8
        entry = self.entries[0]
        seen = set()
        for seed in range(10):
            first = medium.prepare_online_batches(entry, self.config, "train_b", 8, training=True, seed=seed)
            second = medium.prepare_online_batches(entry, self.config, "train_b", 8, training=True, seed=seed)
            selection = first[3]
            self.assertEqual(selection, second[3])
            self.assertEqual(len(selection["target_raw_frames"]), 8)
            self.assertFalse(set(selection["target_raw_frames"]) & set(entry["partition"]["heldout_raw_ids"]))
            seen.update(selection["target_raw_frames"])
        self.assertEqual(seen, set(entry["partition"]["train_raw_ids"]))

    def test_41_frames_are_all_partitioned(self):
        row = {**self.entries[0], "frame_count": 41}
        entry = medium.online_entries({"train": [row]}, self.config)["train"][0]
        self.assertEqual(len(entry["partition"]["train_raw_ids"]), 33)
        self.assertEqual(set(entry["partition"]["train_raw_ids"] + entry["partition"]["heldout_raw_ids"]), set(range(41)))


class FullControllerTest(SharedStructureControllerTest):
    def test_full_training_guard_and_record_do_not_claim_medium_preflight(self):
        config = deepcopy(self.config)
        config["runtime"].update(probe_script="train_shared_structure_full.py", physical_gpus=[5, 7], feature_mode="online",
                                 inventory={"path": "inventory", "sha256": "b"*64})
        config["probe"].update(batch_size=8, accumulation=3)
        config["full"] = {"world_size": 2, "core_precision": "float32", "acceptance": "early full-data eval",
                          "warmstart": {"path": "warm", "sha256": "c"*64}}
        plan = full.build_stage_plan(config, "full", self.config_path)
        self.assertEqual(plan["guard_operations"], ["train"])
        import record_shared_structure_operation as recorder
        state = recorder.prepare_state({}, config, plan, self.source)
        self.assertEqual(state["active_operation"]["initialization"], "warm")
        self.assertIn("train", state["allowed_operations"])
        self.assertNotIn("preflight", state["allowed_operations"])
        import run_shared_structure_experiment as controller
        controller.validate_prepared_state(state, config, plan, self.source)
        config["probe"]["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "48"):
            full.build_stage_plan(config, "full", self.config_path)


if __name__ == "__main__":
    unittest.main()
