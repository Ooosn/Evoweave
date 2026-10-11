from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_virtual_frame_rig import TinyBackbone, TinySurface, settings
from rigweave.dynamic_rig.virtual_frame_rig import VirtualFrameRig
import train_virtual_frame as training


def fake_prepare(dataset, requests, config, device):
    batch, frames = len(requests), requests[0].frames
    surfaces, targets = [], []
    for request in requests:
        generator = torch.Generator().manual_seed(request.sampling_seed)
        surfaces.append({key: torch.randn(1, frames, points, 3, generator=generator)
                         for key, points in (("dense_points", 12), ("dense_normals", 12),
                                             ("query_points", 7), ("query_normals", 7))})
        targets.append(torch.randn(1, frames, 3, 3, generator=generator))
    surface = {key: torch.cat([value[key] for value in surfaces]) for key in surfaces[0]}
    surface["query_indices"] = torch.arange(7)[None].expand(batch, -1)
    return surface, {"target_parents": torch.tensor([[-1, 0, 1]]).expand(batch, -1),
                     "joint_mask": torch.ones(batch, 3, dtype=torch.bool),
                     "frame_mask": torch.ones(batch, frames, dtype=torch.bool),
                     "frame_target_joints": torch.cat(targets)}


class VirtualTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_optimizer_checkpoint_resume_matches_uninterrupted_training(self):
        config = {
            "data": {"seed": 72, "frames_min": 2, "frames_max": 4},
            "optimization": {"peak_lr": 1e-4, "div_factor": 5., "final_div_factor": 10.,
                             "warmup_fraction": .1, "schedule_samples": 32, "weight_decay": .04,
                             "clip_grad_norm": 1., "xyz_weight": 2.},
            "test": {"approved": True, "target_samples": 16, "evaluation_samples": [8, 16],
                     "frame_budget": 8, "batch_cap": 3, "accumulation": 2},
        }
        torch.manual_seed(17)
        original = VirtualFrameRig(TinySurface(), TinyBackbone(), settings())
        full, interrupted = copy.deepcopy(original), copy.deepcopy(original)
        initialization = {"source": "synthetic upstream fixture"}
        entries = {"train": list(range(23))}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(training, "dataset_for", side_effect=lambda entries, config: entries))
            stack.enter_context(patch.object(training, "prepare", side_effect=fake_prepare))
            stack.enter_context(patch.object(training, "source_metadata", return_value={"git_head": "test"}))
            stack.enter_context(patch.object(training, "evaluate"))
            stack.enter_context(patch("torch.autocast", side_effect=lambda *a, **k: nullcontext()))
            stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=0))
            stack.enter_context(patch("torch.cuda.get_rng_state", side_effect=torch.get_rng_state))
            stack.enter_context(patch("torch.cuda.set_rng_state", side_effect=torch.set_rng_state))
            base = Path(directory)
            args = SimpleNamespace(plan="test", candidate="tiny", output=base / "full", resume=None)
            training.run_training(full, initialization, entries, config, args, torch.device("cpu"), 0, 1)
            short = copy.deepcopy(config)
            short["test"].update(target_samples=8, evaluation_samples=[8])
            args.output = base / "resume"
            training.run_training(interrupted, initialization, entries, short, args, torch.device("cpu"), 0, 1)
            args.resume = args.output / "checkpoint_sample_000008.pt"
            resumed = copy.deepcopy(original)
            training.run_training(resumed, initialization, entries, config, args, torch.device("cpu"), 0, 1)
            left = torch.load(base / "full/checkpoint_sample_000016.pt", weights_only=False)
            right = torch.load(base / "resume/checkpoint_sample_000016.pt", weights_only=False)
            self.assertEqual((left["samples"], right["samples"]), (16, 16))
            self.assertEqual(left["input_frames"], right["input_frames"])
            self.assertEqual(left["rank_states"][0]["stream"], right["rank_states"][0]["stream"])
            self.assertEqual(left["step"], right["step"])
            for key, value in left["model"].items():
                torch.testing.assert_close(value, right["model"][key], atol=0, rtol=0)
            for index, state in left["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, right["optimizer"]["state"][index][key], atol=0, rtol=0)
            report = json.loads((base / "resume/rank_0.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["supervised_frames"], report["input_frames"])


if __name__ == "__main__":
    unittest.main()
