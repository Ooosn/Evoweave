"""CPU codec checks; no model, remote assets or GPU required."""
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.id_xyz_tokens import IdXYZTokens
from rigweave.dynamic_rig.structure_xyz_fill_data import decode_coordinate_tokens, quantize_coordinates


def codec(**overrides):
    config = dict(legacy_vocab_size=271, coordinate_ids=tuple(range(256)),
                  branch=256, bos=257, eos=258, pad=259, class_token=270)
    config.update(overrides)
    return IdXYZTokens(**config)


class IdXYZTokenTests(unittest.TestCase):
    def test_original_ids_and_json_configuration_roundtrip(self):
        tokens = codec()
        self.assertEqual(tokens.joint_offset, 271)
        self.assertEqual(tokens.vocab_size, 527)
        self.assertEqual(tokens.class_token, 270)
        config = json.loads(json.dumps(tokens.configuration(), allow_nan=False))
        self.assertEqual(codec().configuration(), IdXYZTokens(**config).configuration())
        config['coordinate_ids'][0] = 99
        self.assertEqual(tokens.coordinate_ids[0], 0)

    def test_direct_joint_ids_and_explicit_branch_no_parent_xyz(self):
        tokens = codec()
        result = tokens.encode([-1, 0, 1, 0], np.zeros((2, 4, 3), dtype=np.float32))
        j = tokens.joint_offset
        expected = [257, 270, j, 128, 128, 128, j+1, 128, 128, 128,
                    j+2, 128, 128, 128, 256, j, j+3, 128, 128, 128, 258]
        np.testing.assert_array_equal(result['input_ids'], [expected, expected])
        self.assertEqual(result['input_ids'].dtype, np.int64)
        self.assertEqual(result['structure_mask'].dtype, np.bool_)
        structure = [j, j+1, j+2, 256, j, j+3, 258]
        np.testing.assert_array_equal(result['input_ids'][0, result['structure_mask'][0]], structure)
        decoded = tokens.decode(expected)
        np.testing.assert_array_equal(decoded['parents'], [-1, 0, 1, 0])
        np.testing.assert_array_equal(decoded['structure_tokens'], structure)
        self.assertEqual(decoded['joints'].dtype, np.float32)
        self.assertEqual(decoded['parents'].dtype, np.int64)

    def test_nonsorted_source_dfs_and_all_pose_structure_alignment(self):
        tokens = codec()
        parents = [3, 3, 0, -1]
        joints = np.random.default_rng(4).uniform(-.9, .9, (5, 4, 3)).astype(np.float32)
        result = tokens.encode(parents, joints)
        np.testing.assert_array_equal(result['dfs_order'], [3, 0, 2, 1])
        np.testing.assert_array_equal(result['dfs_parents'], [-1, 0, 1, 0])
        self.assertEqual(result['dfs_order'].dtype, np.int64)
        for frame in range(5):
            np.testing.assert_array_equal(result['structure_mask'][frame], result['structure_mask'][0])
            decoded = tokens.decode(result['input_ids'][frame])
            np.testing.assert_array_equal(decoded['parents'], result['dfs_parents'])
            np.testing.assert_array_equal(decoded['structure_tokens'],
                                          result['input_ids'][0, result['structure_mask'][0]])
            expected = decode_coordinate_tokens(quantize_coordinates(joints[frame, result['dfs_order']]))
            np.testing.assert_array_equal(decoded['joints'], expected)
        self.assertFalse(np.array_equal(result['input_ids'][0], result['input_ids'][1]))

    def test_duplicate_quantized_coordinates_do_not_change_parents(self):
        parents = [-1, 0, 0, 1, 1, 2]
        result = codec().encode(parents, np.zeros((2, 6, 3), dtype=np.float32))
        decoded = codec().decode(result['input_ids'][0])
        self.assertEqual(np.unique(decoded['joints'], axis=0).shape[0], 1)
        np.testing.assert_array_equal(decoded['parents'], [-1, 0, 1, 1, 0, 4])
        self.assertEqual(len(decoded['parents']), 6)

    def test_random_source_permutations_match_aux_codec(self):
        rng = np.random.default_rng(23)
        tokens = codec()
        aux = AuxSkeletonTokens()
        for count in (1, 2, 9, 32, 256):
            parents = np.array([-1] + [int(rng.integers(i)) for i in range(1, count)], dtype=np.int64)
            permutation = rng.permutation(count)
            source = np.full(count, -1, dtype=np.int64)
            source[permutation[1:]] = permutation[parents[1:]]
            encoded = tokens.encode(source, rng.uniform(-1, 1, (2, count, 3)).astype(np.float32))
            reference = aux.encode(source)
            np.testing.assert_array_equal(encoded['dfs_order'], reference.dfs_order)
            np.testing.assert_array_equal(encoded['dfs_parents'], aux.decode(reference.tokens))
            np.testing.assert_array_equal(tokens.decode(encoded['input_ids'][0])['parents'], encoded['dfs_parents'])

    def test_float32_rounding_extremes_and_noncontiguous_coordinate_ids(self):
        boundary = np.nextafter(np.float32(.00390625), np.float32(1.))
        joints = np.array([[[-1., 1., boundary]]], dtype=np.float32)
        tokens = codec(legacy_vocab_size=1024, coordinate_ids=tuple(range(600, 344, -1)))
        result = tokens.encode([-1], joints)
        np.testing.assert_array_equal(result['input_ids'][0, 3:6], [600, 345, 472])
        np.testing.assert_array_equal(tokens.decode(result['input_ids'][0])['joints'],
                                      np.array([[-.99609375, .99609375, .00390625]], dtype=np.float32))
        torch_result = codec().encode(torch.tensor([-1]), torch.from_numpy(joints))
        self.assertEqual(torch_result['input_ids'][0, 5], 128)

    def test_invalid_gt_is_rejected_not_clipped(self):
        arrays = [np.array([[[value, 0., 0.]]], dtype=np.float64)
                  for value in (np.nan, np.inf, -np.inf, 1.+1e-12, -1.-1e-12)]
        arrays += [np.array([[[np.nextafter(np.float32(1.), np.float32(2.)), 0., 0.]]], dtype=np.float32),
                   np.zeros((1, 1, 3), dtype=np.int64), np.zeros((0, 1, 3), dtype=np.float32),
                   np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        for joints in arrays:
            with self.subTest(shape=joints.shape, dtype=joints.dtype), self.assertRaises(ValueError):
                codec().encode([-1], joints)

    def test_malformed_parents_fail_without_deleting_joints(self):
        for parents in ([], [-1, -1], [-1, 3], [-1, 2, 1], [-1, 1], [-1., 0.], [-1, True]):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                codec().encode(parents, np.zeros((1, len(parents), 3), dtype=np.float32))

    def test_max_joint_formula_and_full_sequence_capacity(self):
        tokens = codec()
        self.assertEqual(tokens.max_encoded_length, 1535)
        self.assertLess(tokens.max_encoded_length, 2048)
        star = [-1] + [0]*255
        encoded = tokens.encode(star, np.zeros((1, 256, 3), dtype=np.float32))
        self.assertEqual(encoded['input_ids'].shape, (1, 1535))
        np.testing.assert_array_equal(tokens.decode(encoded['input_ids'][0])['parents'], star)
        self.assertEqual(tokens.allowed_tokens(encoded['input_ids'][0, :-1]), [tokens.eos])
        chain = tokens.encode([-1] + list(range(255)), np.zeros((1, 256, 3), dtype=np.float32))
        self.assertEqual(chain['input_ids'].shape[1], 1027)
        with self.assertRaisesRegex(ValueError, 'max_joints'):
            tokens.encode([-1] + [0]*256, np.zeros((1, 257, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'max_sequence_tokens'):
            codec(max_sequence_tokens=1534).encode(star, np.zeros((1, 256, 3), dtype=np.float32))

    def test_allowed_tokens_every_phase_and_natural_eos(self):
        tokens = codec()
        full = tokens.encode([-1, 0, 1, 0], np.zeros((1, 4, 3), dtype=np.float32))['input_ids'][0].tolist()
        for index, token in enumerate(full):
            self.assertIn(token, tokens.allowed_tokens(full[:index]))
        self.assertEqual(tokens.allowed_tokens([]), [tokens.bos])
        self.assertEqual(tokens.allowed_tokens([tokens.bos]), [tokens.class_token])
        self.assertEqual(tokens.allowed_tokens(full[:3]), list(tokens.coordinate_ids))
        self.assertEqual(tokens.allowed_tokens(full[:15]), [271, 272])
        self.assertEqual(tokens.allowed_tokens(full[:16]), [274])
        self.assertEqual(tokens.allowed_tokens(full), [])
        np.testing.assert_array_equal(tokens.decode(full + [tokens.pad]*3)['parents'], [-1, 0, 1, 0])
        self.assertEqual(tokens.allowed_tokens(full + [tokens.pad]), [])

    def test_generation_budget_always_reserves_eos(self):
        tokens = codec(max_joints=12, max_sequence_tokens=31)
        rng = np.random.default_rng(9)
        for _ in range(15):
            prefix = []
            while not prefix or prefix[-1] != tokens.eos:
                allowed = tokens.allowed_tokens(prefix)
                self.assertTrue(allowed)
                prefix.append(int(rng.choice(allowed)))
                self.assertLessEqual(len(prefix), tokens.max_sequence_tokens)
            self.assertGreaterEqual(len(tokens.decode(prefix)['parents']), 1)
        root = codec(max_sequence_tokens=7)
        prefix = [root.bos, root.class_token, root.joint_offset, 128, 128, 128]
        self.assertEqual(root.allowed_tokens(prefix), [root.eos])
        with self.assertRaisesRegex(ValueError, 'cannot finish'):
            codec(max_sequence_tokens=10).allowed_tokens(prefix + [272, 128, 128, 128])

    def test_invalid_grammar_and_token_types(self):
        c = codec()
        root = [c.bos, c.class_token, c.joint_offset, 128, 128, 128]
        invalid = [[], [c.bos, c.eos], root, root + [c.branch, c.joint_offset, 272, 128, 128, 128, c.eos],
                   root + [273, 128, 128, 128, c.eos], root + [c.pad, c.eos],
                   root + [c.eos, 128], root + [c.eos, c.eos],
                   [c.bos, c.class_token, c.joint_offset, 128, c.eos],
                   root + [c.eos, -1], [float(value) for value in root + [c.eos]],
                   [True] + root[1:] + [c.eos]]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                c.decode(values)
        # J1 was closed when returning to J0, and cannot be referenced again.
        prefix = root + [272, 128, 128, 128, c.branch, 271, 273, 128, 128, 128, c.branch, 272]
        with self.assertRaisesRegex(ValueError, 'strict active ancestor'):
            c.allowed_tokens(prefix)
        with self.assertRaises(ValueError):
            c.decode(np.array([[257, 270, 271]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, 'max_sequence_tokens'):
            codec(max_sequence_tokens=7).decode(root + [c.eos, c.pad])

    def test_invalid_vocabulary_configuration(self):
        invalid = [dict(legacy_vocab_size=True), dict(legacy_vocab_size=270), dict(max_joints=0),
                   dict(max_sequence_tokens=6), dict(bos=1.5), dict(eos=257), dict(class_token=128),
                   dict(coordinate_ids=tuple(range(255))), dict(coordinate_ids=(0,)*256),
                   dict(coordinate_ids=tuple(range(255)) + (True,)), dict(branch=-1)]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                codec(**config)


if __name__ == '__main__':
    unittest.main()
