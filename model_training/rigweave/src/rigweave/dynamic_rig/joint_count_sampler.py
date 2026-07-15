from __future__ import annotations

from math import ceil
from typing import Sequence

import torch
from torch.utils.data import Sampler


def parse_joint_count_bin_uppers(value: str) -> tuple[int, ...]:
    bounds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not bounds:
        raise ValueError("joint-count bins cannot be empty")
    if any(bound <= 0 for bound in bounds):
        raise ValueError(f"joint-count bin upper bounds must be positive, got {bounds}")
    if any(left >= right for left, right in zip(bounds, bounds[1:])):
        raise ValueError(f"joint-count bin upper bounds must be strictly increasing, got {bounds}")
    return bounds


class JointCountMixtureSampler(Sampler[int]):
    """Sample from a natural/uniform-over-joint-count-bins mixture.

    ``mixture_alpha=0`` is natural row-uniform sampling. ``mixture_alpha=1``
    first chooses a non-empty joint-count bin uniformly, then a row uniformly
    within that bin. Intermediate values mix the two distributions directly.
    Every distributed rank shards the same deterministic global draw.
    """

    def __init__(
        self,
        joint_counts: Sequence[int | None],
        *,
        bin_upper_bounds: Sequence[int],
        mixture_alpha: float,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= float(mixture_alpha) <= 1.0:
            raise ValueError(f"mixture_alpha must be in [0, 1], got {mixture_alpha}")
        if int(num_replicas) <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError(f"rank={rank} is outside num_replicas={num_replicas}")
        if not joint_counts:
            raise ValueError("joint-count sampler received no rows")
        if any(count is None for count in joint_counts):
            raise ValueError(
                "joint-count-balanced sampling requires target_joint_count in every manifest row"
            )

        self.bin_upper_bounds = tuple(int(value) for value in bin_upper_bounds)
        if not self.bin_upper_bounds:
            raise ValueError("joint-count sampler requires at least one bin")
        if self.bin_upper_bounds[0] <= 0:
            raise ValueError(f"joint-count bin upper bounds must be positive: {self.bin_upper_bounds}")
        if any(left >= right for left, right in zip(self.bin_upper_bounds, self.bin_upper_bounds[1:])):
            raise ValueError(f"joint-count bin upper bounds must be strictly increasing: {self.bin_upper_bounds}")

        self.mixture_alpha = float(mixture_alpha)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.dataset_size = len(joint_counts)
        self.num_samples = int(ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        bucket_ids = [self._bucket_index(int(count)) for count in joint_counts]
        self.bucket_counts = tuple(bucket_ids.count(index) for index in range(len(self.bin_upper_bounds)))
        empty = [index for index, count in enumerate(self.bucket_counts) if count == 0]
        if empty:
            raise ValueError(
                "joint-count-balanced sampling requires every configured bin to be non-empty; "
                f"empty bins={empty} bounds={self.bin_upper_bounds}"
            )

        natural_per_row = 1.0 / self.dataset_size
        bucket_mass = 1.0 / len(self.bucket_counts)
        weights = [
            (1.0 - self.mixture_alpha) * natural_per_row
            + self.mixture_alpha * bucket_mass / self.bucket_counts[bucket_id]
            for bucket_id in bucket_ids
        ]
        self.weights = torch.tensor(weights, dtype=torch.float64)
        self.expected_bucket_probabilities = tuple(
            (1.0 - self.mixture_alpha) * count / self.dataset_size
            + self.mixture_alpha * bucket_mass
            for count in self.bucket_counts
        )

    def _bucket_index(self, joint_count: int) -> int:
        if joint_count <= 0:
            raise ValueError(f"joint count must be positive, got {joint_count}")
        for index, upper in enumerate(self.bin_upper_bounds):
            if joint_count <= upper:
                return index
        raise ValueError(
            f"joint count {joint_count} exceeds final configured bin {self.bin_upper_bounds[-1]}"
        )

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(global_indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def report(self) -> dict[str, object]:
        return {
            "mode": "natural_uniform_joint_count_bin_mixture",
            "mixture_alpha": self.mixture_alpha,
            "bin_upper_bounds": list(self.bin_upper_bounds),
            "source_bucket_counts": list(self.bucket_counts),
            "expected_bucket_probabilities": list(self.expected_bucket_probabilities),
            "replacement": True,
            "samples_per_rank_per_epoch": self.num_samples,
            "global_samples_per_epoch": self.total_size,
        }
