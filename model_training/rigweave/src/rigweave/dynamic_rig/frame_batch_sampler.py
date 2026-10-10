"""Batch-consistent frame counts with replayable per-asset sampling seeds."""
from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from typing import NamedTuple

import numpy as np
from torch.utils.data import BatchSampler, Sampler


class FrameSampleRequest(NamedTuple):
    item_index: int
    frames: int
    sampling_seed: int


class VariableFrameBatchSampler(Sampler[list[FrameSampleRequest]]):
    """Keep fixed asset batches unless a per-rank frame budget and asset cap are set."""

    def __init__(
        self, sampler: Sampler[int], batch_size: int, *, min_frames: int, max_frames: int,
        seed: int, drop_last: bool = False, frame_budget: int = 0,
        max_batch_size: int | None = None,
    ) -> None:
        if not 2 <= min_frames <= max_frames:
            raise ValueError("frame bounds must satisfy 2 <= min_frames <= max_frames")
        if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget < 0:
            raise ValueError("frame_budget must be a non-negative integer")
        if max_batch_size is not None and (
            isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if frame_budget:
            if frame_budget < max_frames:
                raise ValueError("frame_budget must be at least max_frames")
            if max_batch_size is None:
                raise ValueError("frame-budget batching requires an explicit max_batch_size")
        self.batches = BatchSampler(sampler, batch_size, drop_last)
        self.min_frames, self.max_frames = int(min_frames), int(max_frames)
        self.seed = int(seed)
        self.epoch = 0
        self.frame_budget = frame_budget
        self.max_batch_size = max_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if not self.frame_budget:
            return len(self.batches)
        remaining = len(self.batches.sampler)
        count = 0
        schedule = self._frame_budget_schedule()
        while remaining > 0:
            _, batch_size = next(schedule)
            if remaining < batch_size and self.batches.drop_last:
                break
            remaining -= min(remaining, batch_size)
            count += 1
        return count

    def _frame_budget_schedule(self) -> Iterator[tuple[int, int]]:
        # T is uniform per microbatch, not per asset; short-T batches can contain more assets.
        assert self.max_batch_size is not None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 99011]))
        while True:
            frames = int(rng.integers(self.min_frames, self.max_frames + 1))
            yield frames, min(self.max_batch_size, self.frame_budget // frames)

    def _iter_frame_batches(self) -> Iterator[tuple[list[int], int]]:
        if self.frame_budget:
            sampler_iter = iter(self.batches.sampler)
            for frames, batch_size in self._frame_budget_schedule():
                batch = list(islice(sampler_iter, batch_size))
                if not batch or (len(batch) < batch_size and self.batches.drop_last):
                    return
                yield batch, frames
        else:
            # The schedule deliberately excludes rank, so DDP ranks use the same T.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 99011]))
            for indices in self.batches:
                frames = int(rng.integers(self.min_frames, self.max_frames + 1))
                yield indices, frames

    def __iter__(self) -> Iterator[list[FrameSampleRequest]]:
        for indices, frames in self._iter_frame_batches():
            requests = []
            for index in indices:
                sampling_seed = int(np.random.SeedSequence(
                    [self.seed, self.epoch, int(index), 129]).generate_state(1, dtype=np.uint32)[0])
                requests.append(FrameSampleRequest(int(index), frames, sampling_seed))
            yield requests
