"""Batch-consistent frame counts with replayable per-asset sampling seeds."""
from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
from torch.utils.data import BatchSampler, Sampler


class FrameSampleRequest(NamedTuple):
    item_index: int
    frames: int
    sampling_seed: int


class VariableFrameBatchSampler(Sampler[list[FrameSampleRequest]]):
    def __init__(
        self, sampler: Sampler[int], batch_size: int, *, min_frames: int, max_frames: int,
        seed: int, drop_last: bool = False,
    ) -> None:
        if not 2 <= min_frames <= max_frames:
            raise ValueError("frame bounds must satisfy 2 <= min_frames <= max_frames")
        self.batches = BatchSampler(sampler, batch_size, drop_last)
        self.min_frames, self.max_frames = int(min_frames), int(max_frames)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[FrameSampleRequest]]:
        # The schedule deliberately excludes rank, so DDP ranks use the same T.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 99011]))
        for indices in self.batches:
            frames = int(rng.integers(self.min_frames, self.max_frames + 1))
            requests = []
            for index in indices:
                sampling_seed = int(np.random.SeedSequence(
                    [self.seed, self.epoch, int(index), 129]).generate_state(1, dtype=np.uint32)[0])
                requests.append(FrameSampleRequest(int(index), frames, sampling_seed))
            yield requests
