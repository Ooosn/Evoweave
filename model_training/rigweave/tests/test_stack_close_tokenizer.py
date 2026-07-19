from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rigweave.stack_close.tokenizer import StackCloseTokenizer


@dataclass
class _LegacyTokenizer:
    num_discrete: int = 256
    continuous_range: tuple[float, float] = (-1.0, 1.0)
    vocab_size: int = 267
    token_id_branch: int = 256
    token_id_bos: int = 257
    token_id_eos: int = 258
    token_id_pad: int = 259
    token_id_spring: int = 260
    token_id_cls_none: int = 263

    def __post_init__(self) -> None:
        self.cls_token_id = {
            "vroid": 264,
            "mixamo": 265,
            "articulationxl": 266,
        }

    @property
    def bos(self) -> int:
        return self.token_id_bos

    @property
    def eos(self) -> int:
        return self.token_id_eos

    @property
    def pad(self) -> int:
        return self.token_id_pad


def _edges(parents: np.ndarray, original_indices: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0:
            edges.add(
                (
                    int(original_indices[parent]),
                    int(original_indices[child]),
                )
            )
    return edges


def test_round_trip_preserves_duplicate_joint_identity() -> None:
    tokenizer = StackCloseTokenizer(_LegacyTokenizer())
    joints = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    parents = np.asarray([-1, 0, 1, 0, 3], dtype=np.int64)

    serialized = tokenizer.serialize_tree(
        joints,
        parents,
        cls="articulationxl",
        sibling_rng=np.random.default_rng(7),
    )
    output = tokenizer.detokenize(serialized.tokens)
    recovered = np.asarray(
        [-1 if parent is None else parent for parent in output.parents],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(recovered, serialized.parents)
    assert _edges(serialized.parents, serialized.original_indices) == {
        (0, 1),
        (1, 2),
        (0, 3),
        (3, 4),
    }
    assert serialized.tokens.shape[0] == 4 * joints.shape[0] + 3
    assert np.count_nonzero(serialized.tokens == tokenizer.token_id_close) == 5


def test_sibling_permutations_change_order_not_tree() -> None:
    tokenizer = StackCloseTokenizer(_LegacyTokenizer())
    joints = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    parents = np.asarray([-1, 0, 0, 0], dtype=np.int64)
    orders = set()
    for seed in range(12):
        serialized = tokenizer.serialize_tree(
            joints,
            parents,
            cls="articulationxl",
            sibling_rng=np.random.default_rng(seed),
        )
        orders.add(tuple(serialized.original_indices.tolist()))
        assert _edges(serialized.parents, serialized.original_indices) == {
            (0, 1),
            (0, 2),
            (0, 3),
        }
        tokenizer.detokenize(serialized.tokens)
    assert len(orders) > 1


def test_prefix_grammar_closes_each_node_before_eos() -> None:
    tokenizer = StackCloseTokenizer(_LegacyTokenizer())
    root = tokenizer.discretize(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))[0]
    prefix = np.asarray(
        [
            tokenizer.bos,
            tokenizer.cls_name_to_token("articulationxl"),
            *root.tolist(),
        ],
        dtype=np.int64,
    )
    allowed = tokenizer.next_posible_token(prefix)
    assert tokenizer.token_id_close in allowed
    assert tokenizer.eos not in allowed

    closed = np.append(prefix, tokenizer.token_id_close)
    assert tokenizer.next_posible_token(closed) == [tokenizer.eos]
    tokenizer.detokenize(np.append(closed, tokenizer.eos))

    with pytest.raises(ValueError, match="empty stack"):
        tokenizer.next_posible_token(
            np.asarray(
                [
                    tokenizer.bos,
                    tokenizer.cls_name_to_token("articulationxl"),
                    tokenizer.token_id_close,
                ],
                dtype=np.int64,
            )
        )
