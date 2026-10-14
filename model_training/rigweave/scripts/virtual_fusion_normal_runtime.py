"""Matched online exposure plans independent of physical microbatch capacity."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest


class MatchedFrameStream:
    """Share a random T per global update; resample each asset on every visit."""

    def __init__(self, dataset, *, rank, world_size, seed, target_assets, effective_assets,
                 frame_policy, manifest_sha256, min_frames=2, max_frames=24, state=None,
                 update_boundaries=None):
        integers = (rank, world_size, seed, target_assets, effective_assets, min_frames, max_frames)
        if any(type(value) is not int for value in integers):
            raise ValueError('sampling counts and seeds must be integers, not booleans')
        if not 0 <= rank < world_size or world_size < 1 or not 0 <= seed < 2**31:
            raise ValueError('invalid sampling rank, world size or seed')
        if (len(dataset) < 1 or not 0 < target_assets < 2**32 or effective_assets < 1
                or target_assets % world_size or effective_assets % world_size):
            raise ValueError('sampling needs rank-divisible global asset updates')
        if not 2 <= min_frames <= max_frames <= 24:
            raise ValueError('normal frame counts must stay within2..24')
        if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
                or any(char not in '0123456789abcdef' for char in manifest_sha256)):
            raise ValueError('sampling needs the immutable ordered manifest fingerprint')
        self.local_update = effective_assets // world_size
        expected_keys = {str(frames) for frames in range(min_frames, max_frames + 1)}
        if set(frame_policy) != expected_keys or any(
                type(size) is not int or not 1 <= size <= self.local_update for size in frame_policy.values()):
            raise ValueError('each frame count needs a measured physical batch within the local update budget')
        self.policy = dict(frame_policy)
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.target_assets, self.effective_assets = target_assets, effective_assets
        self.local_total = target_assets // world_size
        boundaries = [target_assets] if update_boundaries is None else list(update_boundaries)
        if (not boundaries or any(type(x) is not int or not 0 < x <= target_assets or x % world_size
                                  for x in boundaries)
                or boundaries != sorted(set(boundaries)) or boundaries[-1] != target_assets):
            raise ValueError('ordered update boundaries must include the final rank-divisible budget')
        sizes, start = [], 0
        for boundary in boundaries:
            while start < boundary:
                size = min(effective_assets, boundary - start)
                sizes.append(size)
                start += size
        self.update_sizes = np.asarray(sizes, dtype='<u4')
        self.global_ends = np.cumsum(self.update_sizes, dtype=np.int64)
        self.local_ends = self.global_ends // world_size
        self.contract = {'rank': rank, 'world_size': world_size, 'seed': seed,
                         'target_assets': target_assets, 'effective_assets': effective_assets,
                         'update_boundaries': boundaries,
                         'dataset_rows': len(dataset), 'manifest_sha256': manifest_sha256,
                         'min_frames': min_frames, 'max_frames': max_frames, 'frame_policy': dict(frame_policy)}
        permutations = [np.random.default_rng(np.random.SeedSequence([seed, epoch, 193])).permutation(len(dataset))
                        for epoch in range((target_assets + len(dataset) - 1) // len(dataset))]
        self.order = np.concatenate(permutations)[:target_assets].astype('<u4')
        self.frame_counts = np.random.default_rng(np.random.SeedSequence([seed, 99011])).integers(
            min_frames, max_frames + 1, size=len(sizes), dtype=np.uint8)
        semantic = {key: self.contract[key] for key in ('seed', 'target_assets', 'effective_assets',
                    'dataset_rows', 'manifest_sha256', 'min_frames', 'max_frames', 'update_boundaries')}
        digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode())
        digest.update(self.order.tobytes())
        digest.update(self.frame_counts.tobytes())
        digest.update(self.update_sizes.tobytes())
        self.plan_sha256 = digest.hexdigest()
        self.cursor = 0
        if state is not None:
            if state.get('contract') != self.contract or state.get('plan_sha256') != self.plan_sha256:
                raise ValueError('sampler replay contract or global exposure plan differs')
            cursor = state.get('cursor')
            if type(cursor) is not int or not 0 <= cursor <= self.local_total:
                raise ValueError('invalid saved exposure cursor')
            self.cursor = cursor

    def take(self, max_assets=None):
        if max_assets is not None and (type(max_assets) is not int or max_assets < 1):
            raise ValueError('a microbatch request needs a positive asset count')
        if self.cursor >= self.local_total:
            raise StopIteration('the complete exposure budget has been consumed')
        update = int(np.searchsorted(self.local_ends, self.cursor, side='right'))
        global_base = int(self.global_ends[update - 1]) if update else 0
        offset = self.cursor - global_base // self.world_size
        local_size = int(self.update_sizes[update]) // self.world_size
        frames = int(self.frame_counts[update])
        size = min(self.policy[str(frames)], max_assets or local_size, local_size - offset,
                   self.local_total - self.cursor)
        global_start = global_base + self.rank * local_size + offset
        requests = [FrameSampleRequest(int(self.order[position]), frames, (self.seed << 32) + position)
                    for position in range(global_start, global_start + size)]
        self.cursor += size
        return requests

    def state_dict(self):
        return {'contract': deepcopy(self.contract), 'plan_sha256': self.plan_sha256, 'cursor': self.cursor}

    def plan_report(self):
        frames, counts = np.unique(self.frame_counts, return_counts=True)
        return {'algorithm': 'matched_asset_exposure_v1', 'plan_sha256': self.plan_sha256,
                'contract': deepcopy(self.contract), 'sampling_seed_rule': '(seed << 32) + global_exposure_index',
                'frame_count_rule': 'uniform2..24 per global optimizer update, shared by both ranks',
                'asset_order_rule': 'seeded full permutations, independent of physical microbatch size',
                'update_sizes': self.update_sizes.tolist(),
                'supervised_frame_exposures': int(np.dot(self.update_sizes.astype(np.int64), self.frame_counts)),
                'updates_by_frame_count': {str(int(frame)): int(count) for frame, count in zip(frames, counts)},
                'no_fixed_pose_cache': True}
