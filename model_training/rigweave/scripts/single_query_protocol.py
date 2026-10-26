"""Deterministic target selection without changing the input-frame stream."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from numbers import Integral
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest


SINGLE_SCOPE = 'real_query_single_target_5k_20261026_v1'
SINGLE_VARIANT = 'single_query'
SELECTOR = 'sha256_python_random_randrange_v1'
TWO_SCOPE = 'real_query_two_target_5k_20261026_v1'
TWO_VARIANT = 'two_query'
TWO_SELECTOR = 'sha256_python_random_sample2_v1'


def _nonnegative_integer(value: Integral, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer, not a boolean')
    return int(value)


def select_target_frames(requests: Iterable[FrameSampleRequest], seed: Integral) -> list[int]:
    """Select one uniform slot per request using an independent local RNG.

    Hash the ASCII bytes ``SELECTOR|seed|item_index|sampling_seed|frames``:
    numeric fields are full unsigned decimal integers, with no whitespace,
    leading zeros, trailing separator or newline. Interpret all 32 SHA-256
    digest bytes as an unsigned big-endian integer, then call
    ``random.Random(digest_integer).randrange(frames)`` once for each request.
    """
    outer_seed = _nonnegative_integer(seed, 'seed')
    selected = []
    for request in requests:
        index = _nonnegative_integer(request.item_index, 'request.item_index')
        sampling_seed = _nonnegative_integer(request.sampling_seed, 'request.sampling_seed')
        frames = _nonnegative_integer(request.frames, 'request.frames')
        if not 2 <= frames <= 24:
            raise ValueError('request.frames must be within 2..24')
        payload = f'{SELECTOR}|{outer_seed}|{index}|{sampling_seed}|{frames}'.encode('ascii')
        digest_seed = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
        selected.append(random.Random(digest_seed).randrange(frames))
    return selected


def select_two_target_frames(requests: Iterable[FrameSampleRequest], seed: Integral) -> list[list[int]]:
    """Select a uniform unordered pair, returned in ascending slot order.

    Use the single-target decimal encoding with the TWO_SELECTOR prefix and
    one ``random.Random(digest_integer).sample(range(frames), 2)`` per request.
    """
    outer_seed = _nonnegative_integer(seed, 'seed')
    selected = []
    for request in requests:
        index = _nonnegative_integer(request.item_index, 'request.item_index')
        sampling_seed = _nonnegative_integer(request.sampling_seed, 'request.sampling_seed')
        frames = _nonnegative_integer(request.frames, 'request.frames')
        if not 2 <= frames <= 24:
            raise ValueError('request.frames must be within 2..24')
        payload = f'{TWO_SELECTOR}|{outer_seed}|{index}|{sampling_seed}|{frames}'.encode('ascii')
        digest_seed = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
        selected.append(sorted(random.Random(digest_seed).sample(range(frames), 2)))
    return selected


def supervised_count(config: Mapping, assets: Integral, input_frames: Integral) -> int:
    """Count supervised targets separately from full encoder input frames."""
    assets = _nonnegative_integer(assets, 'assets')
    input_frames = _nonnegative_integer(input_frames, 'input_frames')
    if assets > input_frames:
        raise ValueError('assets must not exceed input_frames')
    if config.get('family') == TWO_SCOPE:
        if 2 * assets > input_frames:
            raise ValueError('two targets per asset must not exceed input_frames')
        return 2 * assets
    return assets if config.get('family') == SINGLE_SCOPE else input_frames
