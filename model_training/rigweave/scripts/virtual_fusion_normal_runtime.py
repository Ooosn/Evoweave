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

    @classmethod
    def extend_completed(cls, dataset, state, *, target_assets, update_boundaries):
        """Extend a fully consumed stream while preserving its generated prefix."""
        if not isinstance(state, dict):
            raise ValueError('completed sampler state must be a mapping')
        old_contract = state.get('contract')
        if not isinstance(old_contract, dict):
            raise ValueError('completed sampler state needs a contract mapping')
        expected_contract = {
            'rank', 'world_size', 'seed', 'target_assets', 'effective_assets',
            'update_boundaries', 'dataset_rows', 'manifest_sha256', 'min_frames',
            'max_frames', 'frame_policy',
        }
        if set(old_contract) != expected_contract:
            raise ValueError('completed sampler state contract fields differ')

        # Reconstruct through the normal constructor so the saved plan SHA and
        # every existing contract check remain authoritative for old resumes.
        try:
            old = cls(
                dataset,
                rank=old_contract['rank'],
                world_size=old_contract['world_size'],
                seed=old_contract['seed'],
                target_assets=old_contract['target_assets'],
                effective_assets=old_contract['effective_assets'],
                frame_policy=deepcopy(old_contract['frame_policy']),
                manifest_sha256=old_contract['manifest_sha256'],
                min_frames=old_contract['min_frames'],
                max_frames=old_contract['max_frames'],
                state=state,
                update_boundaries=old_contract['update_boundaries'],
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('invalid completed sampler state') from exc

        if old.cursor != old.local_total:
            raise ValueError('sampler extension requires a completed old exposure budget')
        if type(target_assets) is not int:
            raise ValueError('the extended target asset count must be an integer')
        if target_assets <= old.target_assets:
            raise ValueError('the extended target asset count must be larger than the old target')
        if target_assets % old.world_size:
            raise ValueError('the extended target asset count must be rank-divisible')
        try:
            new_boundaries = list(update_boundaries)
        except (TypeError, ValueError) as exc:
            raise ValueError('extended update boundaries must be an ordered sequence') from exc
        old_boundaries = old.contract['update_boundaries']
        if len(new_boundaries) <= len(old_boundaries):
            raise ValueError('extended update boundaries must add a suffix')
        if new_boundaries[:len(old_boundaries)] != old_boundaries:
            raise ValueError('extended update boundaries must preserve the old prefix')
        if (any(type(value) is not int or value <= old.target_assets
                for value in new_boundaries[len(old_boundaries):])
                or new_boundaries[-1] != target_assets):
            raise ValueError('extended update boundaries must end at the new target')

        new = cls(
            dataset,
            rank=old.contract['rank'],
            world_size=old.contract['world_size'],
            seed=old.contract['seed'],
            target_assets=target_assets,
            effective_assets=old.contract['effective_assets'],
            frame_policy=deepcopy(old.contract['frame_policy']),
            manifest_sha256=old.contract['manifest_sha256'],
            min_frames=old.contract['min_frames'],
            max_frames=old.contract['max_frames'],
            update_boundaries=new_boundaries,
        )
        for key in expected_contract - {'target_assets', 'update_boundaries'}:
            if new.contract[key] != old.contract[key]:
                raise ValueError('sampler extension changed an existing contract field')
        old_updates = len(old.update_sizes)
        if (not np.array_equal(old.order, new.order[:old.target_assets])
                or not np.array_equal(old.frame_counts, new.frame_counts[:old_updates])
                or not np.array_equal(old.update_sizes, new.update_sizes[:old_updates])):
            raise ValueError('sampler extension does not preserve the old generated prefix')
        new.cursor = old.local_total
        return new

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
