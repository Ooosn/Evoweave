from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from virtual_frame_runtime import (
    FrameBudgetStream, exposure_learning_rate, normalize_asset_gradients, virtual_runtime_environment,
)


class VirtualFrameRuntimeTests(unittest.TestCase):
    def test_sampler_roundtrip_partial_final_batch_and_epoch(self):
        plan = dict(rank=0, world_size=2, seed=37, frame_budget=48, batch_cap=7, min_frames=2, max_frames=24)
        stream = FrameBudgetStream(list(range(53)), **plan)
        for _ in range(7):
            stream.take()
        current = stream.state_dict()
        full = FrameBudgetStream(list(range(53)), state=current, **plan).take()
        first = stream.take(max_assets=1)
        self.assertEqual(first, full[:1])
        restored = FrameBudgetStream(list(range(53)), state=json.loads(json.dumps(stream.state_dict())), **plan)
        if len(full) > 1:
            self.assertEqual(restored.take(), full[1:])
            self.assertEqual(stream.take(), full[1:])
        for _ in range(80):
            self.assertEqual(restored.take(), stream.take())
        self.assertGreater(stream.epoch, 1)

    def test_ddp_schedules_and_exact_exposure_stop(self):
        plan = dict(world_size=2, seed=37, frame_budget=48, batch_cap=7, min_frames=2, max_frames=24)
        streams = [FrameBudgetStream(list(range(103)), rank=rank, **plan) for rank in range(2)]
        assets, target = 0, 316
        while assets < target:
            maximum = (target - assets) // 2
            batches = [x.take(maximum) for x in streams]
            self.assertEqual([(len(x), x[0].frames) for x in batches][0],
                             [(len(x), x[0].frames) for x in batches][1])
            assets += sum(map(len, batches))
        self.assertEqual(assets, target)
        with self.assertRaisesRegex(ValueError, "configuration"):
            FrameBudgetStream(list(range(103)), rank=1, state=streams[0].state_dict(), **plan)

    def test_unequal_microbatch_sum_matches_global_asset_mean_gradient(self):
        torch.manual_seed(41)
        reference = nn.Linear(3, 2).double()
        models = [copy.deepcopy(reference) for _ in range(2)]
        inputs = [torch.randn(7, 3, dtype=torch.double) for _ in range(2)]
        target = [torch.randn(7, 2, dtype=torch.double) for _ in range(2)]
        for model, x, y in zip(models, inputs, target):
            for chunk in (slice(0, 2), slice(2, 7)):
                (model(x[chunk]) - y[chunk]).square().mean(1).sum().backward()
        ddp = copy.deepcopy(reference)
        for p, left, right in zip(ddp.parameters(), models[0].parameters(), models[1].parameters()):
            p.grad = (left.grad + right.grad) / 2
        normalize_asset_gradients(ddp.parameters(), global_assets=14, world_size=2)
        (reference(torch.cat(inputs)) - torch.cat(target)).square().mean(1).mean().backward()
        for p, expected in zip(ddp.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, expected.grad, rtol=1e-12, atol=1e-12)

    def test_exposure_lr_warmup_peak_and_final_values(self):
        self.assertEqual(exposure_learning_rate(0, 80000), (2e-5, .95))
        lr, beta = exposure_learning_rate(8000, 80000)
        self.assertAlmostEqual(lr, 1e-4)
        self.assertAlmostEqual(beta, .85)
        lr, beta = exposure_learning_rate(80000, 80000)
        self.assertAlmostEqual(lr, 2e-6)
        self.assertAlmostEqual(beta, .95)
        for value in (-1, 80001):
            with self.assertRaises(ValueError):
                exposure_learning_rate(value, 80000)

    def test_launch_environment_cannot_inherit_old_method_weights(self):
        config = {"runtime": {"repo": "/repo", "conda_bin": "/env/bin", "unirig_root": "/upstream",
                              "opt_config_root": "/opt", "extra_pythonpath": "/runtime"}}
        with patch.dict(os.environ, {"EVOWEAVE_INIT_CHECKPOINT": "/old.pt", "RIGWEAVE_RESUME_CHECKPOINT": "/r3.pt"}):
            environment = virtual_runtime_environment(config, [5, 7])
        self.assertNotIn("EVOWEAVE_INIT_CHECKPOINT", environment)
        self.assertNotIn("RIGWEAVE_RESUME_CHECKPOINT", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "5,7")


if __name__ == "__main__":
    unittest.main()
