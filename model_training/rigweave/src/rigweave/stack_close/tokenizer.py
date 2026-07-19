from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StackCloseSerialization:
    tokens: np.ndarray
    joints: np.ndarray
    parents: np.ndarray
    original_indices: np.ndarray
    coordinate_token_positions: np.ndarray


@dataclass(frozen=True)
class StackCloseDetokenizeOutput:
    tokens: np.ndarray
    joints: np.ndarray
    parents: list[int | None]
    bones: np.ndarray
    tails: np.ndarray
    cls: str | None

    @property
    def p_joints(self) -> np.ndarray:
        return self.bones[:, :3]

    @property
    def num_bones(self) -> int:
        return int(self.joints.shape[0])

    @property
    def J(self) -> int:
        return self.num_bones


class StackCloseTokenizer:
    """Lossless DFS tree tokenizer using one CLOSE token per real joint.

    The legacy UniRig BRANCH token id is reused as CLOSE. Coordinate, class,
    BOS, EOS, PAD, embedding, and output-head shapes therefore stay unchanged.
    """

    def __init__(self, legacy_tokenizer: Any) -> None:
        self._legacy = legacy_tokenizer
        self._num_discrete = int(legacy_tokenizer.num_discrete)
        self._continuous_range = tuple(float(x) for x in legacy_tokenizer.continuous_range)
        self._vocab_size = int(legacy_tokenizer.vocab_size)

        self.token_id_close = int(legacy_tokenizer.token_id_branch)
        self.token_id_branch = self.token_id_close
        self.token_id_bos = int(legacy_tokenizer.bos)
        self.token_id_eos = int(legacy_tokenizer.eos)
        self.token_id_pad = int(legacy_tokenizer.pad)
        self.token_id_cls_none = int(legacy_tokenizer.token_id_cls_none)
        self.cls_token_id = dict(legacy_tokenizer.cls_token_id)
        self.cls_token_to_name = {
            int(value): str(name) for name, value in self.cls_token_id.items()
        }
        self._class_tokens = {
            self.token_id_cls_none,
            *(int(value) for value in self.cls_token_id.values()),
        }

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad(self) -> int:
        return self.token_id_pad

    @property
    def bos(self) -> int:
        return self.token_id_bos

    @property
    def eos(self) -> int:
        return self.token_id_eos

    @property
    def num_discrete(self) -> int:
        return self._num_discrete

    @property
    def continuous_range(self) -> tuple[float, float]:
        return self._continuous_range

    def cls_name_to_token(self, cls: str | None) -> int:
        if cls is None or cls not in self.cls_token_id:
            return self.token_id_cls_none
        return int(self.cls_token_id[cls])

    def discretize(self, values: np.ndarray) -> np.ndarray:
        lo, hi = self.continuous_range
        scaled = (np.asarray(values, dtype=np.float32) - lo) / (hi - lo)
        scaled = scaled * self.num_discrete
        return np.clip(np.round(scaled), 0, self.num_discrete - 1).astype(np.int64)

    def undiscretize(self, values: np.ndarray) -> np.ndarray:
        lo, hi = self.continuous_range
        scaled = np.asarray(values, dtype=np.float32) + 0.5
        scaled = scaled / self.num_discrete
        return scaled * (hi - lo) + lo

    @staticmethod
    def _tree_children(parents: np.ndarray) -> tuple[int, list[list[int]]]:
        parent_array = np.asarray(parents, dtype=np.int64).reshape(-1)
        roots = np.flatnonzero(parent_array < 0)
        if roots.shape[0] != 1:
            raise ValueError(f"stack-close requires exactly one root, got {roots.tolist()}")
        children: list[list[int]] = [[] for _ in range(parent_array.shape[0])]
        for child, parent in enumerate(parent_array.tolist()):
            if parent < 0:
                continue
            if parent >= parent_array.shape[0] or parent == child:
                raise ValueError(f"invalid parent edge child={child} parent={parent}")
            children[parent].append(child)
        return int(roots[0]), children

    def serialize_tree(
        self,
        joints: np.ndarray,
        parents: np.ndarray,
        *,
        cls: str | None,
        sibling_rng: np.random.Generator | None = None,
    ) -> StackCloseSerialization:
        joint_array = np.asarray(joints, dtype=np.float32)
        parent_array = np.asarray(parents, dtype=np.int64).reshape(-1)
        if joint_array.ndim != 2 or joint_array.shape[1] != 3:
            raise ValueError(f"joints must be (J,3), got {joint_array.shape}")
        if parent_array.shape != (joint_array.shape[0],):
            raise ValueError(
                f"parents shape {parent_array.shape} does not match joints {joint_array.shape[0]}"
            )

        root, children = self._tree_children(parent_array)
        if sibling_rng is not None:
            for child_list in children:
                sibling_rng.shuffle(child_list)

        order: list[int] = []
        active: set[int] = set()

        def visit(node: int) -> None:
            if node in active:
                raise ValueError(f"cycle detected at joint {node}")
            active.add(node)
            order.append(node)
            for child in children[node]:
                visit(child)
            active.remove(node)

        visit(root)
        if len(order) != joint_array.shape[0]:
            missing = sorted(set(range(joint_array.shape[0])) - set(order))
            raise ValueError(f"tree is disconnected; unreachable joints={missing[:20]}")

        inverse = np.full((joint_array.shape[0],), -1, dtype=np.int64)
        inverse[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)
        ordered_joints = joint_array[np.asarray(order, dtype=np.int64)]
        ordered_parents = np.asarray(
            [
                -1 if parent_array[old] < 0 else int(inverse[parent_array[old]])
                for old in order
            ],
            dtype=np.int64,
        )
        ordered_children: list[list[int]] = [[] for _ in order]
        for child, parent in enumerate(ordered_parents.tolist()):
            if parent >= 0:
                ordered_children[parent].append(child)

        quantized = self.discretize(ordered_joints)
        tokens: list[int] = [
            self.bos,
            self.cls_name_to_token(cls),
        ]
        coordinate_positions = np.empty((len(order), 3), dtype=np.int64)

        def emit(node: int) -> None:
            start = len(tokens)
            coordinate_positions[node] = np.arange(start, start + 3, dtype=np.int64)
            tokens.extend(int(value) for value in quantized[node])
            for child in ordered_children[node]:
                emit(child)
            tokens.append(self.token_id_close)

        emit(0)
        tokens.append(self.eos)
        return StackCloseSerialization(
            tokens=np.asarray(tokens, dtype=np.int64),
            joints=ordered_joints,
            parents=ordered_parents,
            original_indices=np.asarray(order, dtype=np.int64),
            coordinate_token_positions=coordinate_positions,
        )

    def _prefix_state(self, ids: np.ndarray) -> tuple[str, int, bool]:
        sequence = np.asarray(ids, dtype=np.int64).reshape(-1)
        if sequence.shape[0] == 0:
            return "expect_bos", 0, False
        if int(sequence[0]) != self.bos:
            raise ValueError("stack-close prefix does not start with BOS")

        index = 1
        if index == sequence.shape[0]:
            return "expect_class_or_root", 0, False
        if int(sequence[index]) in self._class_tokens:
            index += 1
        if index == sequence.shape[0]:
            return "expect_root", 0, False

        stack_depth = 0
        root_started = False
        coords_remaining = 0
        while index < sequence.shape[0]:
            token = int(sequence[index])
            if coords_remaining > 0:
                if not 0 <= token < self.num_discrete:
                    raise ValueError(
                        f"expected coordinate with {coords_remaining} values remaining, got {token}"
                    )
                coords_remaining -= 1
                if coords_remaining == 0:
                    stack_depth += 1
                    root_started = True
                index += 1
                continue

            if 0 <= token < self.num_discrete:
                if root_started and stack_depth == 0:
                    raise ValueError("cannot add a second root after the root was closed")
                coords_remaining = 2
            elif token == self.token_id_close:
                if stack_depth <= 0:
                    raise ValueError("CLOSE encountered with an empty stack")
                stack_depth -= 1
            elif token == self.eos:
                if not root_started or stack_depth != 0:
                    raise ValueError("EOS is only legal after the root has been closed")
                if index != sequence.shape[0] - 1:
                    raise ValueError("tokens found after EOS")
                return "terminal", stack_depth, root_started
            else:
                raise ValueError(f"unexpected stack-close token {token}")
            index += 1

        if coords_remaining > 0:
            return f"expect_coord_{coords_remaining}", stack_depth, root_started
        if not root_started:
            return "expect_root", stack_depth, root_started
        if stack_depth > 0:
            return "expect_child_or_close", stack_depth, root_started
        return "expect_eos", stack_depth, root_started

    def next_posible_token(self, ids: np.ndarray) -> list[int]:
        state, _, _ = self._prefix_state(ids)
        coordinates = list(range(self.num_discrete))
        if state == "expect_bos":
            return [self.bos]
        if state == "expect_class_or_root":
            return sorted(self._class_tokens) + coordinates
        if state == "expect_root" or state.startswith("expect_coord_"):
            return coordinates
        if state == "expect_child_or_close":
            return coordinates + [self.token_id_close]
        if state == "expect_eos":
            return [self.eos]
        if state == "terminal":
            return []
        raise RuntimeError(f"unhandled stack-close parser state {state}")

    def next_possible_token(self, ids: np.ndarray) -> list[int]:
        return self.next_posible_token(ids)

    def bones_in_sequence(self, ids: np.ndarray) -> int:
        output = self.detokenize(ids)
        return output.num_bones

    def detokenize(self, ids: np.ndarray, **_: Any) -> StackCloseDetokenizeOutput:
        sequence = np.asarray(ids, dtype=np.int64).reshape(-1)
        while sequence.shape[0] > 0 and int(sequence[-1]) == self.pad:
            sequence = sequence[:-1]
        self._prefix_state(sequence)
        if sequence.shape[0] < 6 or int(sequence[-1]) != self.eos:
            raise ValueError("stack-close sequence is incomplete")

        index = 1
        cls: str | None = None
        if int(sequence[index]) in self._class_tokens:
            cls_token = int(sequence[index])
            cls = self.cls_token_to_name.get(cls_token)
            index += 1

        joints: list[np.ndarray] = []
        parents: list[int | None] = []
        stack: list[int] = []
        while index < sequence.shape[0]:
            token = int(sequence[index])
            if token == self.eos:
                break
            if token == self.token_id_close:
                stack.pop()
                index += 1
                continue
            if index + 3 > sequence.shape[0]:
                raise ValueError("truncated coordinate triple")
            triple = sequence[index : index + 3]
            if not np.all((0 <= triple) & (triple < self.num_discrete)):
                raise ValueError(f"invalid coordinate triple {triple.tolist()}")
            parent = None if not stack else int(stack[-1])
            joints.append(self.undiscretize(triple))
            parents.append(parent)
            stack.append(len(joints) - 1)
            index += 3

        joint_array = np.stack(joints, axis=0).astype(np.float32)
        parent_points = np.stack(
            [
                joint_array[i] if parent is None else joint_array[parent]
                for i, parent in enumerate(parents)
            ],
            axis=0,
        )
        bones = np.concatenate([parent_points, joint_array], axis=1)
        tails = joint_array.copy()
        children: list[list[int]] = [[] for _ in parents]
        for child, parent in enumerate(parents):
            if parent is not None:
                children[parent].append(child)
        for node, child_ids in enumerate(children):
            if child_ids:
                tails[node] = joint_array[child_ids[0]]
            elif parents[node] is not None:
                tails[node] = joint_array[node] + (
                    joint_array[node] - joint_array[parents[node]]
                )
        return StackCloseDetokenizeOutput(
            tokens=sequence,
            joints=joint_array,
            parents=parents,
            bones=bones.astype(np.float32),
            tails=tails.astype(np.float32),
            cls=cls,
        )
