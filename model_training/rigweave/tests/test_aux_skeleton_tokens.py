"""CPU-only tests for the isolated auxiliary DFS skeleton codec."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "rigweave"
    / "dynamic_rig"
    / "aux_skeleton_tokens.py"
)
SPEC = importlib.util.spec_from_file_location("aux_skeleton_tokens_isolated", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUX
SPEC.loader.exec_module(AUX)
AuxSkeletonTokens = AUX.AuxSkeletonTokens


def joint(codec: AuxSkeletonTokens, index: int) -> int:
    return codec.joint_offset + index


def remapped_random_tree(rng: np.random.Generator, count: int) -> np.ndarray:
    """Create a valid DFS tree, then permute source node indices."""

    dfs_parents = np.full(count, -1, dtype=np.int64)
    for child in range(1, count):
        dfs_parents[child] = int(rng.integers(0, child))
    permutation = rng.permutation(count)
    source_parents = np.full(count, -1, dtype=np.int64)
    for dfs_id in range(1, count):
        source_parents[permutation[dfs_id]] = permutation[dfs_parents[dfs_id]]
    source_parents[permutation[0]] = -1
    return source_parents


class AuxSkeletonTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = AuxSkeletonTokens()

    def test_vocabulary_and_explicit_branch_example(self) -> None:
        self.assertEqual((self.codec.pad, self.codec.bos, self.codec.eos, self.codec.branch), (0, 1, 2, 3))
        self.assertEqual(self.codec.joint_offset, 4)
        self.assertEqual(self.codec.vocab_size, 4 + self.codec.max_joints)

        encoded = self.codec.encode([-1, 0, 1, 0])
        self.assertEqual(
            encoded.tokens.tolist(),
            [self.codec.bos, joint(self.codec, 0), joint(self.codec, 1), joint(self.codec, 2),
             self.codec.branch, joint(self.codec, 0), joint(self.codec, 3), self.codec.eos],
        )
        self.assertEqual(encoded.dfs_order.tolist(), [0, 1, 2, 3])
        np.testing.assert_array_equal(self.codec.decode(encoded.tokens), [-1, 0, 1, 0])

    def test_random_trees_roundtrip_through_source_remapping(self) -> None:
        rng = np.random.default_rng(20261013)
        for count in (1, 2, 3, 8, 31, 127, 256):
            for _ in range(8 if count < 128 else 3):
                source_parents = remapped_random_tree(rng, count)
                encoded = self.codec.encode(source_parents)
                decoded = self.codec.decode(encoded.tokens)
                order = encoded.dfs_order
                self.assertEqual(sorted(order.tolist()), list(range(count)))
                source_to_dfs = encoded.source_to_dfs
                expected_dfs_parents = np.full(count, -1, dtype=np.int64)
                for dfs_id, source_id in enumerate(order.tolist()):
                    source_parent = int(source_parents[source_id])
                    if source_parent >= 0:
                        expected_dfs_parents[dfs_id] = source_to_dfs[source_parent]
                np.testing.assert_array_equal(decoded, expected_dfs_parents)

                recovered_source = np.full(count, -1, dtype=np.int64)
                for dfs_id, source_id in enumerate(order.tolist()):
                    dfs_parent = int(decoded[dfs_id])
                    if dfs_parent >= 0:
                        recovered_source[source_id] = order[dfs_parent]
                np.testing.assert_array_equal(recovered_source, source_parents)

    def test_chain_and_star_hit_max_joint_boundary_without_sorting(self) -> None:
        chain = np.asarray([-1, *range(255)], dtype=np.int64)
        chain_encoded = self.codec.encode(chain)
        self.assertEqual(len(chain_encoded.tokens), 258)
        self.assertFalse(bool(np.any(chain_encoded.tokens == self.codec.branch)))
        np.testing.assert_array_equal(self.codec.decode(chain_encoded.tokens), chain)

        star = np.asarray([-1, *([0] * 255)], dtype=np.int64)
        star_encoded = self.codec.encode(star)
        self.assertEqual(len(star_encoded.tokens), 766)
        self.assertEqual(int(np.count_nonzero(star_encoded.tokens == self.codec.branch)), 254)
        self.assertEqual(star_encoded.dfs_order.tolist(), list(range(256)))
        np.testing.assert_array_equal(self.codec.decode(star_encoded.tokens), np.asarray([-1, *([0] * 255)]))

    def test_sibling_order_and_multiple_returns_to_one_ancestor(self) -> None:
        # Source index order is root children [1, 2, 5], and child 1 children
        # [3, 4].  The DFS order must preserve those source-index orders.
        parents = np.asarray([-1, 0, 0, 1, 1, 0], dtype=np.int64)
        encoded = self.codec.encode(parents)
        self.assertEqual(encoded.dfs_order.tolist(), [0, 1, 3, 4, 2, 5])
        tokens = encoded.tokens.tolist()
        branch_positions = [i for i, token in enumerate(tokens) if token == self.codec.branch]
        self.assertEqual(len(branch_positions), 3)
        self.assertEqual(
            [(tokens[i + 1] - self.codec.joint_offset, tokens[i + 2] - self.codec.joint_offset)
             for i in branch_positions],
            [(1, 3), (0, 4), (0, 5)],
        )
        np.testing.assert_array_equal(self.codec.decode(tokens), [-1, 0, 1, 1, 0, 0])

    def test_none_root_and_prefix_joint_mask_ignore_padding(self) -> None:
        parents = np.asarray([None, 0, 1, -2, -2], dtype=object)
        mask = np.asarray([True, True, True, False, False], dtype=bool)
        encoded = self.codec.encode(parents, mask)
        np.testing.assert_array_equal(self.codec.decode(encoded.tokens), [-1, 0, 1])

        with self.assertRaisesRegex(ValueError, "contiguous valid prefix"):
            self.codec.encode(parents, [True, False, True, False, False])

    def test_padded_batch_teacher_forcing_is_shifted_per_asset(self) -> None:
        parent_batch = np.asarray([[-1, 0, 0, -2], [-1, 0, -2, -2]], dtype=np.int64)
        masks = np.asarray([[True, True, True, False], [True, True, False, False]], dtype=bool)
        batch = self.codec.batch_teacher_forcing(parent_batch, masks)
        first = self.codec.encode([-1, 0, 0]).tokens
        second = self.codec.encode([-1, 0]).tokens

        self.assertEqual(batch.input_ids.shape, (2, len(first) - 1))
        np.testing.assert_array_equal(batch.input_ids[0], first[:-1])
        np.testing.assert_array_equal(batch.labels[0], first[1:])
        np.testing.assert_array_equal(batch.input_ids[1, : len(second) - 1], second[:-1])
        np.testing.assert_array_equal(batch.labels[1, : len(second) - 1], second[1:])
        self.assertTrue(np.all(batch.labels[1, len(second) - 1 :] == -100))
        self.assertTrue(np.all(batch.input_ids[1, len(second) - 1 :] == self.codec.pad))
        self.assertEqual(batch.input_ids[0, 0], self.codec.bos)
        self.assertEqual(batch["inputs"].shape, batch.input_ids.shape)
        self.assertEqual(batch["next_labels"].shape, batch.labels.shape)
        self.assertEqual(batch.lengths.tolist(), [len(first), len(second)])

    def test_invalid_parent_dtypes_and_graphs_fail_closed(self) -> None:
        invalid_parents = (
            [],
            [-1, -1],
            [0],
            [-1, 1],
            [-1, 2, 1],
            [-1, 0, 2],
            [-1, 0, 4],
            np.asarray([-1.0, 0.0], dtype=np.float64),
            np.asarray([-1, True], dtype=bool),
            ["-1", 0],
        )
        for parents in invalid_parents:
            with self.subTest(parents=repr(parents)):
                with self.assertRaises(ValueError):
                    self.codec.encode(parents)
        with self.assertRaises(ValueError):
            self.codec.encode(np.asarray([[-1, 0]], dtype=np.int64))
        with self.assertRaises(ValueError):
            self.codec.encode(-1)

    def test_malformed_sequences_never_succeed_by_forcing_eos(self) -> None:
        valid = self.codec.encode([-1, 0, 0]).tokens.tolist()
        invalid_sequences = [
            [],
            [self.codec.bos, self.codec.eos],
            valid[:-1],
            [joint(self.codec, 0), self.codec.eos],
            [self.codec.bos, joint(self.codec, 1), self.codec.eos],
            [self.codec.bos, joint(self.codec, 0), joint(self.codec, 2), self.codec.eos],
            [self.codec.bos, joint(self.codec, 0), self.codec.branch, joint(self.codec, 0), self.codec.eos],
            [self.codec.bos, joint(self.codec, 0), self.codec.branch, joint(self.codec, 0),
             joint(self.codec, 1), self.codec.eos, self.codec.pad],
            [self.codec.bos, joint(self.codec, 0), self.codec.branch, joint(self.codec, 0),
             self.codec.eos],
            [self.codec.bos, joint(self.codec, 0), self.codec.branch, joint(self.codec, 0),
             joint(self.codec, 1), self.codec.branch, joint(self.codec, 1),
             joint(self.codec, 2), self.codec.eos],
            [self.codec.bos, joint(self.codec, 0), self.codec.eos, joint(self.codec, 1)],
        ]
        for tokens in invalid_sequences:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    self.codec.decode(tokens)

        inactive_ancestor = [
            self.codec.bos, joint(self.codec, 0), joint(self.codec, 1), joint(self.codec, 2),
            self.codec.branch, joint(self.codec, 1), joint(self.codec, 3),
            self.codec.branch, joint(self.codec, 0), joint(self.codec, 4),
            self.codec.branch, joint(self.codec, 1), joint(self.codec, 5), self.codec.eos,
        ]
        with self.assertRaisesRegex(ValueError, "active ancestor"):
            self.codec.decode(inactive_ancestor)

        for tokens in (np.asarray(valid, dtype=np.float64), np.asarray(valid, dtype=bool), valid + ["bad"]):
            with self.subTest(tokens=repr(tokens)):
                with self.assertRaises(ValueError):
                    self.codec.decode(tokens)

    def test_joint_and_sequence_length_limits_are_explicit(self) -> None:
        small = AuxSkeletonTokens(max_joints=3)
        with self.assertRaisesRegex(ValueError, "max_joints"):
            small.encode([-1, 0, 1, 2])
        with self.assertRaises(ValueError):
            small.decode([small.bos, joint(small, 0), joint(small, 1), joint(small, 2),
                          small.joint_offset + 3, small.eos])

        limited = AuxSkeletonTokens(max_joints=4, max_sequence_tokens=5)
        with self.assertRaisesRegex(ValueError, "max_sequence_tokens"):
            limited.encode([-1, 0, 1, 2])
        with self.assertRaisesRegex(ValueError, "max_sequence_tokens"):
            limited.decode([limited.bos, joint(limited, 0), joint(limited, 1), joint(limited, 2),
                            joint(limited, 3), limited.eos])
        with self.assertRaises(ValueError):
            AuxSkeletonTokens(max_joints=0)
        with self.assertRaises(ValueError):
            AuxSkeletonTokens(max_sequence_tokens=2)


if __name__ == "__main__":
    unittest.main()
