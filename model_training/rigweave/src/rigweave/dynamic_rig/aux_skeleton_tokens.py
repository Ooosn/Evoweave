"""CPU-only auxiliary skeleton token codec.

The auxiliary route serializes one static skeleton target per asset.  It does
not alter the main UniRig XYZ+BRANCH stream.  A sequence has this grammar::

    BOS J0 J1 J2 BRANCH J0 J3 EOS

``Jk`` creates the next joint in preorder DFS order.  A plain ``Jk`` makes it
the child of the current active joint.  ``BRANCH parent new`` returns to a
strictly active ancestor and creates ``new`` as that ancestor's next child.
Children are traversed in their original source-index order.  The decoder
therefore returns parent indices in the DFS ID space; callers can use
``dfs_order`` from :meth:`AuxSkeletonTokens.encode` to map them back to source
indices.

This module intentionally imports no ``rigweave`` package code and no neural
framework.  NumPy is used only for compact CPU-side integer arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class AuxSkeletonEncoding:
    """Encoded tokens and the source-index order used for DFS joint IDs."""

    tokens: np.ndarray
    dfs_order: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow ``tokens, dfs_order = codec.encode(...)``."""

        yield self.tokens
        yield self.dfs_order

    def __getitem__(self, index: int) -> np.ndarray:
        if index == 0:
            return self.tokens
        if index == 1:
            return self.dfs_order
        raise IndexError(index)

    @property
    def source_to_dfs(self) -> np.ndarray:
        """Return the inverse permutation, indexed by source joint ID."""

        result = np.empty(self.dfs_order.shape[0], dtype=np.int64)
        result[self.dfs_order] = np.arange(self.dfs_order.shape[0], dtype=np.int64)
        return result


@dataclass(frozen=True)
class AuxTeacherForcingBatch:
    """Padded NumPy arrays suitable for per-asset autoregressive CE.

    ``input_ids`` contains BOS and excludes the final EOS.  ``labels`` is the
    one-token-right-shifted target and uses ``-100`` outside each sequence, so
    it can be passed directly as ``ignore_index=-100`` labels.  ``token_ids``
    retains complete BOS-through-EOS sequences padded with ``pad``.
    """

    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    token_ids: np.ndarray
    lengths: np.ndarray
    dfs_orders: tuple[np.ndarray, ...]

    @property
    def inputs(self) -> np.ndarray:
        """Alias for ``input_ids``."""

        return self.input_ids

    @property
    def next_labels(self) -> np.ndarray:
        """Alias for the shifted CE labels."""

        return self.labels

    @property
    def tokens(self) -> np.ndarray:
        """Alias for complete padded token sequences."""

        return self.token_ids

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "inputs": "input_ids",
            "next_labels": "labels",
            "tokens": "token_ids",
        }
        return getattr(self, aliases.get(key, key))


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _sequence_values(values: Any, name: str) -> list[Any]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        return values.tolist()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    return result


def _integer_values(values: Any, name: str) -> list[int]:
    raw = _sequence_values(values, name)
    result: list[int] = []
    for index, value in enumerate(raw):
        if not _is_strict_int(value):
            raise ValueError(f"{name}[{index}] must be an integer, not {type(value).__name__}")
        result.append(int(value))
    return result


def _strict_config_int(value: Any, name: str) -> int:
    if not _is_strict_int(value):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _split_batch_rows(values: Any, name: str) -> list[Any]:
    """Normalize one row or a batch of rows without coercing dtypes."""

    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            return [values]
        if values.ndim == 2:
            if values.shape[0] == 0:
                raise ValueError(f"{name} must contain at least one row")
            return [values[index] for index in range(values.shape[0])]
        raise ValueError(f"{name} must be one- or two-dimensional")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain one or more rows")
    try:
        outer = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one or more rows") from exc
    if not outer:
        raise ValueError(f"{name} must contain at least one row")
    first = outer[0]
    if isinstance(first, np.ndarray) or isinstance(first, (list, tuple)):
        return outer
    return [outer]


def _normalize_mask(mask: Any, length: int, name: str) -> int:
    raw = _sequence_values(mask, name)
    if len(raw) != length:
        raise ValueError(f"{name} length must match parents length")
    values: list[bool] = []
    for index, value in enumerate(raw):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name}[{index}] must be boolean")
        values.append(bool(value))
    count = sum(values)
    if count == 0 or values[:count] != [True] * count or any(values[count:]):
        raise ValueError(f"{name} must contain a non-empty contiguous valid prefix")
    return count


@dataclass(frozen=True)
class AuxSkeletonTokens:
    """Explicit DFS-ID auxiliary skeleton tokenizer.

    Token IDs are fixed as ``pad=0``, ``bos=1``, ``eos=2``, ``branch=3`` and
    ``Jk = joint_offset + k`` with ``joint_offset=4``.  The default
    ``max_sequence_tokens=1024`` is this codec's independent target-sequence
    capacity; it is not the model's condition length.  The default 256-joint
    tree has a worst-case encoded length of 766.
    """

    max_joints: int = 256
    max_sequence_tokens: int = 1024

    def __post_init__(self) -> None:
        _strict_config_int(self.max_joints, "max_joints")
        sequence_limit = _strict_config_int(self.max_sequence_tokens, "max_sequence_tokens")
        if sequence_limit < 3:
            raise ValueError("max_sequence_tokens must allow BOS, one root joint, and EOS")

    @property
    def pad(self) -> int:
        return 0

    @property
    def bos(self) -> int:
        return 1

    @property
    def eos(self) -> int:
        return 2

    @property
    def branch(self) -> int:
        return 3

    @property
    def joint_offset(self) -> int:
        return 4

    @property
    def vocab_size(self) -> int:
        return self.joint_offset + int(self.max_joints)

    @property
    def max_tokens(self) -> int:
        """Alias for the configured complete-sequence length limit."""

        return int(self.max_sequence_tokens)

    def _joint_token(self, joint_id: int) -> int:
        if not 0 <= joint_id < int(self.max_joints):
            raise ValueError(f"joint ID {joint_id} is outside max_joints")
        return self.joint_offset + joint_id

    def _joint_id(self, token: int, name: str) -> int:
        if not self.joint_offset <= token < self.vocab_size:
            raise ValueError(f"{name} must be a joint token")
        return token - self.joint_offset

    def _resolve_joint_mask(self, joint_mask: Any, jointmask: Any) -> Any:
        if joint_mask is not None and jointmask is not None:
            raise ValueError("pass only one of joint_mask and jointmask")
        return joint_mask if joint_mask is not None else jointmask

    def _validate_parents(
        self,
        parents: Any,
        joint_mask: Any = None,
        *,
        name: str = "parents",
    ) -> tuple[list[int], list[int]]:
        raw = _sequence_values(parents, name)
        if not raw:
            raise ValueError(f"{name} must contain at least one joint")
        count = len(raw) if joint_mask is None else _normalize_mask(joint_mask, len(raw), "joint_mask")
        if count > int(self.max_joints):
            raise ValueError(
                f"joint count {count} exceeds max_joints={int(self.max_joints)}; targets must not be truncated"
            )

        normalized: list[int] = []
        for index, value in enumerate(raw[:count]):
            if value is None:
                normalized.append(-1)
                continue
            if not _is_strict_int(value):
                raise ValueError(f"{name}[{index}] must be an integer or None")
            normalized.append(int(value))

        roots = [index for index, parent in enumerate(normalized) if parent == -1]
        if len(roots) != 1:
            raise ValueError(f"parents must contain exactly one root, found {len(roots)}")
        root = roots[0]
        for child, parent in enumerate(normalized):
            if parent < -1 or parent >= count:
                raise ValueError(f"parent index {parent} for joint {child} is out of range")
            if parent == child:
                raise ValueError(f"joint {child} cannot parent itself")

        # Parent pointers must form one rooted acyclic component.  The explicit
        # walk also catches cycles before DFS could silently omit a component.
        state = [0] * count
        for start in range(count):
            current = start
            path: list[int] = []
            while current != -1 and state[current] == 0:
                state[current] = 1
                path.append(current)
                current = normalized[current]
            if current != -1 and state[current] == 1:
                raise ValueError(f"parent graph contains a cycle involving joint {current}")
            for node in path:
                state[node] = 2

        for start in range(count):
            current = start
            seen: set[int] = set()
            while current != -1:
                if current in seen:
                    raise ValueError(f"parent graph contains a cycle involving joint {current}")
                seen.add(current)
                current = normalized[current]
            if root not in seen:
                raise ValueError(f"joint {start} is disconnected from root {root}")

        children: list[list[int]] = [[] for _ in range(count)]
        for child, parent in enumerate(normalized):
            if parent >= 0:
                children[parent].append(child)

        order: list[int] = []
        stack = [root]
        while stack:
            current = stack.pop()
            order.append(current)
            stack.extend(reversed(children[current]))
        if len(order) != count:
            raise ValueError("parent graph is not one connected rooted tree")
        return normalized, order

    def encode(
        self,
        parents: Any,
        joint_mask: Any = None,
        *,
        jointmask: Any = None,
    ) -> AuxSkeletonEncoding:
        """Encode source-order parents as ``(tokens, dfs_order)``.

        ``dfs_order[d]`` is the source joint index assigned auxiliary ID
        ``Jd``.  ``joint_mask``/``jointmask`` may select a non-empty contiguous
        prefix from a padded parent row; padded values are ignored.
        """

        joint_mask = self._resolve_joint_mask(joint_mask, jointmask)
        normalized, order = self._validate_parents(parents, joint_mask)
        source_to_dfs = np.empty(len(order), dtype=np.int64)
        source_to_dfs[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)

        dfs_parents = np.full(len(order), -1, dtype=np.int64)
        for dfs_id, source_id in enumerate(order):
            source_parent = normalized[source_id]
            if source_parent >= 0:
                dfs_parents[dfs_id] = source_to_dfs[source_parent]

        tokens: list[int] = [self.bos]
        for dfs_id, parent_id in enumerate(dfs_parents.tolist()):
            if dfs_id == 0:
                tokens.append(self._joint_token(0))
                continue
            if parent_id != dfs_id - 1:
                tokens.extend((self.branch, self._joint_token(parent_id), self._joint_token(dfs_id)))
            else:
                tokens.append(self._joint_token(dfs_id))
        tokens.append(self.eos)
        if len(tokens) > int(self.max_sequence_tokens):
            raise ValueError(
                f"encoded sequence length {len(tokens)} exceeds max_sequence_tokens="
                f"{int(self.max_sequence_tokens)}"
            )
        return AuxSkeletonEncoding(
            tokens=np.asarray(tokens, dtype=np.int64),
            dfs_order=np.asarray(order, dtype=np.int64),
        )

    def decode(self, tokens: Any) -> np.ndarray:
        """Decode a complete token sequence into DFS-space parent indices.

        The function is fail-closed: a sequence must contain exactly one
        leading BOS, sequentially introduced joint IDs, valid branch triples,
        and a final EOS with no padding or trailing tokens.
        """

        values = _integer_values(tokens, "tokens")
        if not values:
            raise ValueError("tokens must contain a complete skeleton")
        if len(values) > int(self.max_sequence_tokens):
            raise ValueError(
                f"token sequence length {len(values)} exceeds max_sequence_tokens="
                f"{int(self.max_sequence_tokens)}"
            )
        if values[0] != self.bos:
            raise ValueError("a complete skeleton must start with BOS")
        if values[-1] != self.eos:
            raise ValueError("a complete skeleton must terminate naturally with EOS")
        if self.eos in values[:-1]:
            raise ValueError("EOS may occur only at the end of a complete skeleton")

        parents: list[int] = []
        active_path: list[int] = []
        cursor = 1
        while cursor < len(values) - 1:
            token = values[cursor]
            expected_id = len(parents)
            if token == self.branch:
                if cursor + 2 >= len(values) - 1:
                    raise ValueError("BRANCH must be followed by parent ID and new joint ID")
                parent_token = values[cursor + 1]
                new_token = values[cursor + 2]
                parent_id = self._joint_id(parent_token, "BRANCH parent ID")
                new_id = self._joint_id(new_token, "BRANCH new ID")
                if not active_path:
                    raise ValueError("BRANCH cannot occur before the root joint")
                if parent_id not in active_path:
                    raise ValueError("BRANCH parent ID must reference an active ancestor")
                if parent_id == active_path[-1]:
                    raise ValueError("BRANCH parent ID must be a strict active ancestor")
                if new_id != expected_id:
                    raise ValueError(
                        f"new joint IDs must be sequential; expected J{expected_id}, got J{new_id}"
                    )
                if new_id >= int(self.max_joints):
                    raise ValueError("decoded joint count exceeds max_joints")
                parents.append(parent_id)
                active_path = active_path[: active_path.index(parent_id) + 1] + [new_id]
                cursor += 3
                continue

            if token in (self.pad, self.bos, self.eos):
                raise ValueError("special tokens are not valid joint IDs in the body")
            joint_id = self._joint_id(token, "joint token")
            if joint_id != expected_id:
                raise ValueError(
                    f"new joint IDs must be sequential; expected J{expected_id}, got J{joint_id}"
                )
            if expected_id == 0:
                parents.append(-1)
                active_path = [0]
            else:
                if not active_path:
                    raise ValueError("non-root joint appears before the root")
                parents.append(active_path[-1])
                active_path.append(joint_id)
            cursor += 1

        if not parents:
            raise ValueError("a complete skeleton must contain a root joint")
        return np.asarray(parents, dtype=np.int64)

    def batch_teacher_forcing(
        self,
        parents_batch: Any,
        joint_masks: Any = None,
        *,
        jointmask: Any = None,
    ) -> AuxTeacherForcingBatch:
        """Encode a batch and return BOS-input/next-label arrays.

        The output has one row per asset, not one row per animation frame.  All
        inputs are padded with ``pad``; all ignored labels are ``-100``.
        """

        if joint_masks is not None and jointmask is not None:
            raise ValueError("pass only one of joint_masks and jointmask")
        if joint_masks is None:
            joint_masks = jointmask

        parent_rows = _split_batch_rows(parents_batch, "parents_batch")
        mask_rows = None if joint_masks is None else _split_batch_rows(joint_masks, "joint_masks")
        if mask_rows is not None and len(mask_rows) != len(parent_rows):
            raise ValueError("joint_masks batch size must match parents_batch")

        encoded = tuple(
            self.encode(row, None if mask_rows is None else mask_rows[index])
            for index, row in enumerate(parent_rows)
        )
        full_lengths = np.asarray([item.tokens.shape[0] for item in encoded], dtype=np.int64)
        max_full_length = int(full_lengths.max())
        max_steps = max_full_length - 1
        token_ids = np.full((len(encoded), max_full_length), self.pad, dtype=np.int64)
        input_ids = np.full((len(encoded), max_steps), self.pad, dtype=np.int64)
        labels = np.full((len(encoded), max_steps), -100, dtype=np.int64)
        attention_mask = np.zeros((len(encoded), max_steps), dtype=bool)
        for index, item in enumerate(encoded):
            length = int(item.tokens.shape[0])
            steps = length - 1
            token_ids[index, :length] = item.tokens
            input_ids[index, :steps] = item.tokens[:-1]
            labels[index, :steps] = item.tokens[1:]
            attention_mask[index, :steps] = True
        return AuxTeacherForcingBatch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            token_ids=token_ids,
            lengths=full_lengths,
            dfs_orders=tuple(item.dfs_order for item in encoded),
        )

    def teacher_forcing(
        self,
        parents_batch: Any,
        joint_masks: Any = None,
        *,
        jointmask: Any = None,
    ) -> AuxTeacherForcingBatch:
        """Readable alias for :meth:`batch_teacher_forcing`."""

        return self.batch_teacher_forcing(parents_batch, joint_masks, jointmask=jointmask)


__all__ = [
    "AuxSkeletonEncoding",
    "AuxSkeletonTokens",
    "AuxTeacherForcingBatch",
]
