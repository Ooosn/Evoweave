import copy
import unittest

import torch

from train_virtual_readout_comparison import enable_coordinate_readout, paired_requests
from virtual_frame_runtime import FrameBudgetStream, normalize_asset_gradients
from rigweave.dynamic_rig.virtual_frame_rig import VirtualFrameRig
from test_virtual_frame_rig import TinyBackbone, TinySurface, settings


class ReadoutComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(59)

    def test_streams_resume_both_original_ranks_and_preserve_partial_batches(self):
        options = dict(world_size=2, seed=37, frame_budget=48, batch_cap=7, min_frames=2, max_frames=24)
        streams = [FrameBudgetStream(list(range(103)), rank=rank, **options) for rank in (0, 1)]
        for _ in range(9):
            paired_requests(streams, 100)
        restarted = [FrameBudgetStream(list(range(103)), rank=rank, state=stream.state_dict(), **options)
                     for rank, stream in enumerate(streams)]
        samples = 0
        while samples < 214:
            actual = paired_requests(streams, 214 - samples)
            self.assertEqual(actual, paired_requests(restarted, 214 - samples))
            samples += sum(map(len, actual))
        self.assertEqual(samples, 214)

    def test_attachment_preserves_existing_adam_state(self):
        model = VirtualFrameRig(TinySurface(), TinyBackbone(), settings())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
        state = {p: copy.deepcopy(value) for p, value in optimizer.state.items()}
        enable_coordinate_readout(model, optimizer, seed=119, attention_dim=8)
        self.assertEqual(len(optimizer.param_groups), 2)
        for parameter, expected in state.items():
            for key, value in expected.items():
                torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
        self.assertTrue(all(p not in optimizer.state for p in model.positions.coordinate_readout.parameters()))

    def test_single_gpu_two_microbatches_match_global_batch_mean(self):
        model = torch.nn.Linear(4, 3).double()
        reference = copy.deepcopy(model)
        x, y = torch.randn(9, 4).double(), torch.randn(9, 3).double()
        for part in (slice(0, 5), slice(5, 9)):
            (model(x[part]) - y[part]).abs().mean(1).sum().backward()
        normalize_asset_gradients(model.parameters(), global_assets=9, world_size=1)
        (reference(x) - y).abs().mean().backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
