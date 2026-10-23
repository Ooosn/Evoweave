"""Single-stream source-stable DFS joint IDs and official UniRig XYZ bins."""
from __future__ import annotations

from dataclasses import dataclass, field
import numbers

import numpy as np
import torch

from .aux_skeleton_tokens import AuxSkeletonTokens
from .structure_xyz_fill_data import decode_coordinate_tokens, quantize_coordinates


def _integer(value, name, *, minimum=0):
    if (not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_))
            or int(value) < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _integers(values, name):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional')
        values = values.detach().cpu().tolist()
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional')
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{name} must be an integer sequence')
    try:
        return [_integer(value, f'{name}[{index}]') for index, value in enumerate(values)]
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer sequence') from exc


@dataclass(frozen=True)
class IdXYZTokens:
    """BOS CLS J0 XYZ ... BRANCH parentID newID XYZ ... EOS.

    Existing vocabulary IDs are unchanged; Jk = legacy_vocab_size + k.
    Encoding returns CPU NumPy int64 IDs and bool masks for all T poses.
    Decoding returns DFS-order joints/parents. ``structure_tokens`` is the
    full-vocabulary structural subsequence (IDs, BRANCH, EOS; no BOS/CLS/XYZ).
    max_sequence_tokens counts the complete sequence, including trailing PAD.
    """

    legacy_vocab_size: int
    coordinate_ids: tuple[int, ...]
    bos: int
    eos: int
    pad: int
    branch: int
    class_token: int
    max_joints: int = 256
    max_sequence_tokens: int = 2048
    _structure: AuxSkeletonTokens = field(init=False, repr=False, compare=False)
    _coordinate_to_bin: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('legacy_vocab_size', 'max_joints', 'max_sequence_tokens'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        if self.max_sequence_tokens < 7:
            raise ValueError('max_sequence_tokens must allow BOS, CLS, root ID, XYZ and EOS (7 tokens)')
        if self.vocab_size - 1 > np.iinfo(np.int64).max:
            raise ValueError('vocabulary IDs must fit int64')
        coordinates = tuple(_integers(self.coordinate_ids, 'coordinate_ids'))
        if len(coordinates) != 256 or len(set(coordinates)) != 256:
            raise ValueError('coordinate_ids must contain exactly 256 distinct IDs in bin order')
        special = []
        for name in ('bos', 'eos', 'pad', 'branch', 'class_token'):
            token = _integer(getattr(self, name), name)
            object.__setattr__(self, name, token)
            special.append(token)
        if any(token >= self.legacy_vocab_size for token in (*coordinates, *special)):
            raise ValueError('coordinate and special IDs must belong to the original vocabulary')
        if len(set(special)) != len(special) or set(coordinates).intersection(special):
            raise ValueError('coordinate IDs and all five special IDs must be disjoint')
        object.__setattr__(self, 'coordinate_ids', coordinates)
        object.__setattr__(self, '_coordinate_to_bin', {token: index for index, token in enumerate(coordinates)})
        object.__setattr__(self, '_structure', AuxSkeletonTokens(
            max_joints=self.max_joints, max_sequence_tokens=self.max_sequence_tokens))

    @property
    def joint_offset(self):
        return self.legacy_vocab_size

    @property
    def vocab_size(self):
        return self.legacy_vocab_size + self.max_joints

    @property
    def max_encoded_length(self):
        """Worst-case unpadded length: 3 + 4J + 2 max(J-2, 0); J=256 -> 1535."""
        return 3 + 4 * self.max_joints + 2 * max(self.max_joints - 2, 0)

    def configuration(self):
        """JSON-native constructor fields, sufficient to reconstruct this codec."""
        return {'legacy_vocab_size': self.legacy_vocab_size,
                'coordinate_ids': list(self.coordinate_ids), 'bos': self.bos,
                'eos': self.eos, 'pad': self.pad, 'branch': self.branch,
                'class_token': self.class_token, 'max_joints': self.max_joints,
                'max_sequence_tokens': self.max_sequence_tokens}

    def encode(self, parents, joints):
        if isinstance(parents, torch.Tensor):
            if parents.ndim != 1:
                raise ValueError('parents must be one-dimensional')
            parents = parents.detach().cpu().tolist()
        encoded = self._structure.encode(parents)
        bins = quantize_coordinates(joints, bins=256)
        count = len(encoded.dfs_order)
        if bins.ndim != 3 or bins.shape[0] < 1 or bins.shape[1:] != (count, 3):
            raise ValueError('joints must have shape [T>=1,J,3] matching every source parent')
        bins = bins[:, encoded.dfs_order]
        frames = bins.shape[0]
        columns = [np.full(frames, self.bos, dtype=np.int64),
                   np.full(frames, self.class_token, dtype=np.int64)]
        structural = [False, False]
        coordinate_ids = np.asarray(self.coordinate_ids, dtype=np.int64)
        created = 0
        for token in encoded.tokens[1:-1]:
            if token == self._structure.branch:
                columns.append(np.full(frames, self.branch, dtype=np.int64))
                structural.append(True)
                continue
            joint_id = int(token) - self._structure.joint_offset
            columns.append(np.full(frames, self.joint_offset + joint_id, dtype=np.int64))
            structural.append(True)
            if joint_id == created:
                columns.extend(coordinate_ids[bins[:, joint_id, axis]] for axis in range(3))
                structural.extend((False, False, False))
                created += 1
        columns.append(np.full(frames, self.eos, dtype=np.int64))
        structural.append(True)
        if len(columns) > self.max_sequence_tokens:
            raise ValueError(f'full ID+XYZ sequence length {len(columns)} exceeds max_sequence_tokens={self.max_sequence_tokens}')
        return {'input_ids': np.stack(columns, axis=1),
                'structure_mask': np.broadcast_to(np.asarray(structural, dtype=bool), (frames, len(columns))).copy(),
                'dfs_order': encoded.dfs_order.copy(),
                'dfs_parents': self._structure.decode(encoded.tokens)}

    def _parse(self, prefix):
        values = _integers(prefix, 'tokens')
        if len(values) > self.max_sequence_tokens:
            raise ValueError('token sequence exceeds max_sequence_tokens')
        parents, active, coordinates, structure = [], [], [], []
        phase, axes_left = 'bos', 0
        for position, token in enumerate(values):
            if token >= self.vocab_size:
                raise ValueError(f'token at position {position} is outside the vocabulary')
            if phase == 'ended':
                if token != self.pad:
                    raise ValueError('only trailing PAD may follow EOS')
                continue
            if phase in ('bos', 'class'):
                expected = self.bos if phase == 'bos' else self.class_token
                if token != expected:
                    raise ValueError(f'expected {phase.upper()} at position {position}')
                phase = 'class' if phase == 'bos' else 'root'
                continue
            if phase == 'xyz':
                if token not in self._coordinate_to_bin:
                    raise ValueError(f'expected XYZ coordinate token at position {position}')
                coordinates[-1].append(self._coordinate_to_bin[token])
                axes_left -= 1
                if axes_left == 0:
                    phase = 'joint_or_end'
                continue
            if phase == 'branch_parent':
                parent = token - self.joint_offset
                if parent not in active[:-1]:
                    raise ValueError('BRANCH parent must be a strict active ancestor')
                active = active[:active.index(parent) + 1]
                structure.append(token)
                phase = 'branch_child'
                continue
            if phase == 'joint_or_end' and token == self.eos:
                structure.append(token)
                phase = 'ended'
                continue
            if phase == 'joint_or_end' and token == self.branch:
                if len(active) < 2 or len(parents) >= self.max_joints:
                    raise ValueError('BRANCH requires a strict ancestor and remaining joint capacity')
                structure.append(token)
                phase = 'branch_parent'
                continue
            if token != self.joint_offset + len(parents) or len(parents) >= self.max_joints:
                raise ValueError(f'expected sequential new joint ID J{len(parents)} at position {position}')
            parents.append(active[-1] if active else -1)
            active.append(len(parents) - 1)
            coordinates.append([])
            structure.append(token)
            phase, axes_left = 'xyz', 3
        return values, phase, axes_left, parents, active, coordinates, structure

    def decode(self, fulltokens):
        _, phase, _, parents, _, coordinates, structure = self._parse(fulltokens)
        if phase != 'ended':
            raise ValueError('full sequence must contain a root with XYZ and terminate naturally with EOS')
        return {'joints': decode_coordinate_tokens(np.asarray(coordinates, dtype=np.int64), bins=256),
                'parents': np.asarray(parents, dtype=np.int64),
                'structure_tokens': np.asarray(structure, dtype=np.int64)}

    def allowed_tokens(self, prefix):
        """Grammar-legal next IDs with enough remaining room to finish with EOS."""
        values, phase, axes_left, parents, active, _, _ = self._parse(prefix)
        if phase == 'ended':
            return []
        remaining = self.max_sequence_tokens - len(values)
        minimum = {'bos': 7, 'class': 6, 'root': 5, 'xyz': axes_left + 1,
                   'joint_or_end': 1, 'branch_parent': 6, 'branch_child': 5}[phase]
        if remaining < minimum:
            raise ValueError('prefix cannot finish with natural EOS within max_sequence_tokens')
        if phase == 'bos':
            return [self.bos]
        if phase == 'class':
            return [self.class_token]
        if phase in ('root', 'branch_child'):
            return [self.joint_offset + len(parents)]
        if phase == 'xyz':
            return list(self.coordinate_ids)
        if phase == 'branch_parent':
            return [self.joint_offset + parent for parent in active[:-1]]
        allowed = [self.eos]
        if len(parents) < self.max_joints:
            if remaining >= 5:
                allowed.append(self.joint_offset + len(parents))
            if remaining >= 7 and len(active) > 1:
                allowed.append(self.branch)
        return allowed
