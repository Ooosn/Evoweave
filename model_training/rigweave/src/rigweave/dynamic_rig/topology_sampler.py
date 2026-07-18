from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


TopologySignature = tuple[int, ...]


def _canonical_signature(values: Sequence[int]) -> TopologySignature:
    signature = tuple(int(value) for value in values)
    if not signature:
        raise ValueError("topology signature cannot be empty")
    roots = [index for index, parent in enumerate(signature) if parent < 0]
    if roots != [0]:
        raise ValueError(f"topology must have exactly one root at joint 0, got roots={roots}")
    for child, parent in enumerate(signature[1:], start=1):
        if not 0 <= parent < child:
            raise ValueError(
                "topology must be parent-before-child ordered, "
                f"got child={child} parent={parent}"
            )
    return signature


def _load_parent_topology(path: str | Path) -> TopologySignature:
    resolved = Path(path)
    try:
        with np.load(resolved, allow_pickle=False) as raw:
            if "target_parents" not in raw.files:
                raise KeyError("missing target_parents")
            parents = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
        return _canonical_signature(parents.tolist())
    except Exception as exc:
        raise ValueError(f"failed to read target topology from {resolved}: {exc}") from exc


def load_parent_topology_signatures(
    paths: Sequence[str | Path],
    *,
    num_workers: int = 0,
) -> list[TopologySignature]:
    """Read exact parent-tree signatures without loading mesh or animation arrays."""

    if not paths:
        raise ValueError("topology scan received no NPZ paths")
    workers = int(num_workers)
    if workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if workers <= 1:
        return [_load_parent_topology(path) for path in paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_load_parent_topology, paths))


def _frequency_bin(frequency: int) -> str:
    if frequency == 1:
        return "1"
    if frequency <= 9:
        return "2-9"
    if frequency <= 99:
        return "10-99"
    return "100+"


class TopologyFamilyMixtureSampler(Sampler[int]):
    """Mix natural row sampling with uniform-over-exact-topology sampling.

    A topology family is the complete ordered ``target_parents`` tuple. With
    ``mixture_alpha=1``, every family has equal probability and rows remain
    uniform within a family. Every distributed rank shards one deterministic
    global draw, matching the existing joint-count sampler contract.
    """

    def __init__(
        self,
        topology_signatures: Sequence[Sequence[int]],
        *,
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
        if not topology_signatures:
            raise ValueError("topology sampler received no rows")

        signatures = [_canonical_signature(values) for values in topology_signatures]
        family_counter = Counter(signatures)
        self.dataset_size = len(signatures)
        self.family_count = len(family_counter)
        self.family_frequencies = tuple(family_counter[signature] for signature in signatures)
        self.family_histogram = Counter(family_counter.values())
        self.mixture_alpha = float(mixture_alpha)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        natural_per_row = 1.0 / self.dataset_size
        uniform_family_mass = 1.0 / self.family_count
        weights = [
            (1.0 - self.mixture_alpha) * natural_per_row
            + self.mixture_alpha * uniform_family_mass / frequency
            for frequency in self.family_frequencies
        ]
        self.weights = torch.tensor(weights, dtype=torch.float64)

        labels = ("1", "2-9", "10-99", "100+")
        source_rows = Counter(
            _frequency_bin(frequency)
            for frequency in self.family_frequencies
        )
        source_families = Counter(
            _frequency_bin(frequency)
            for frequency in family_counter.values()
        )
        expected_mass = Counter()
        for weight, frequency in zip(weights, self.family_frequencies):
            expected_mass[_frequency_bin(frequency)] += weight
        self.source_row_mass_by_frequency = {
            label: source_rows[label] / self.dataset_size for label in labels
        }
        self.source_family_mass_by_frequency = {
            label: source_families[label] / self.family_count for label in labels
        }
        self.expected_row_mass_by_frequency = {
            label: expected_mass[label] for label in labels
        }

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
        family_frequencies = sorted(self.family_histogram.elements())
        middle = len(family_frequencies) // 2
        if len(family_frequencies) % 2:
            median_frequency = float(family_frequencies[middle])
        else:
            median_frequency = 0.5 * (
                family_frequencies[middle - 1] + family_frequencies[middle]
            )
        effective_sample_size = 1.0 / float(torch.square(self.weights).sum().item())
        return {
            "mode": "natural_uniform_exact_topology_mixture",
            "mixture_alpha": self.mixture_alpha,
            "topology_definition": "complete ordered target_parents tuple",
            "source_rows": self.dataset_size,
            "source_family_count": self.family_count,
            "singleton_family_count": int(self.family_histogram.get(1, 0)),
            "min_family_frequency": int(family_frequencies[0]),
            "median_family_frequency": median_frequency,
            "max_family_frequency": int(family_frequencies[-1]),
            "source_row_mass_by_family_frequency": self.source_row_mass_by_frequency,
            "source_family_mass_by_family_frequency": self.source_family_mass_by_frequency,
            "expected_row_mass_by_family_frequency": self.expected_row_mass_by_frequency,
            "sampling_effective_row_count": effective_sample_size,
            "replacement": True,
            "samples_per_rank_per_epoch": self.num_samples,
            "global_samples_per_epoch": self.total_size,
        }
