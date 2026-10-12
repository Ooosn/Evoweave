from __future__ import annotations

import itertools
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_root_rt_overlay import QualityError, align_points, pair_window_bounds, root_rigid_transforms, publish, sha256_file


class RootRTTest(unittest.TestCase):
    def test_identity_and_rigid_roundtrip(self):
        a = .73
        r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        d = np.repeat(np.eye(4)[None], 2, axis=0)
        d[1, :3, :3] = r
        d[1, :3, 3] = [3, -4, 2]
        rs, ts, info = root_rigid_transforms(d, np.array([1., 2., 3.]))
        self.assertFalse(info["relative_nonrigid"])
        np.testing.assert_array_equal(rs[0], np.eye(3))
        np.testing.assert_allclose(rs[1], r, atol=1e-12)
        np.testing.assert_allclose(ts[1], d[1, :3, 3], atol=1e-12)
        x = np.array([[1., 0, 0], [0, 1., 0]])
        y = x @ r.T + ts[1]
        np.testing.assert_allclose(align_points(y, rs[1], ts[1]), x, atol=1e-6)

    def test_nonrigid_scale_is_preserved_and_root_fixed(self):
        d = np.repeat(np.eye(4)[None], 2, axis=0)
        d[1, :3, :3] *= .5
        d[1, :3, 3] = [2., 0, 0]
        root = np.array([1., 0, 0])
        r, t, info = root_rigid_transforms(d, root)
        self.assertTrue(info["relative_nonrigid"])
        np.testing.assert_allclose(r, np.repeat(np.eye(3)[None], 2, axis=0))
        posed = np.array([[2.5, 0, 0], [3., 0, 0]])
        aligned = align_points(posed, r[1], t[1])
        np.testing.assert_allclose(aligned[0], root)
        self.assertAlmostEqual(float(np.linalg.norm(aligned[1] - aligned[0])), .5)

    def test_singular_and_reflected_motion_rejected(self):
        for value in [0., -1.]:
            d = np.repeat(np.eye(4)[None], 2, axis=0)
            d[1, 0, 0] = value
            with self.assertRaises(QualityError):
                root_rigid_transforms(d, np.zeros(3))

    def test_pair_gate_catches_single_and_union_normalization_gap(self):
        lo = np.array([[5., -5, 0], [-5, -5, 0]], dtype=np.float32)
        hi = np.array([[5.1, 5, .1], [-4.9, 5, .1]], dtype=np.float32)
        j = np.array([[9., 0, 0], [-5., 0, 0]], dtype=np.float32)
        self.assertLess(pair_window_bounds(lo[:1], hi[:1], j[:1], j[:1])["pair_window_normalized_joint_maxabs"], 1.)
        self.assertGreater(pair_window_bounds(lo, hi, j, j)["pair_window_normalized_joint_maxabs"], 1.)

    def test_pair_bound_dominates_all_subsets(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            lo = rng.normal(size=(6, 3)).astype(np.float32)
            hi = lo + rng.uniform(.1, 3, (6, 3)).astype(np.float32)
            jlo = lo - rng.uniform(0, 1, (6, 3)).astype(np.float32)
            jhi = hi + rng.uniform(0, 1, (6, 3)).astype(np.float32)
            bound = pair_window_bounds(lo, hi, jlo, jhi)["pair_window_normalized_joint_maxabs"]
            for size in range(1, 7):
                for selected in itertools.combinations(range(6), size):
                    ids = list(selected)
                    low, high = lo[ids].min(0), hi[ids].max(0)
                    center = (low + high) * .5
                    scale = 1.1 * np.max((high - low) * .5)
                    value = max(np.abs((jlo[ids] - center) / scale).max(), np.abs((jhi[ids] - center) / scale).max())
                    self.assertLessEqual(float(value), bound + 1e-6)

    def test_publish_roundtrip_partition_and_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = []
            for index in range(3):
                path = base / f"asset_{index}.npz"
                d = np.repeat(np.eye(4)[None, None], 2, axis=0)
                if index == 1:
                    d[1, 0, :3, :3] *= .5
                verts = np.array([[[-1., -1, -1], [1., 1, 1]]]*2, dtype=np.float32)
                np.savez(path, frame_vertices_rootspace=verts, target_joints_rootspace=np.zeros((2,1,3),np.float32),
                         target_parents=np.array([-1]), target_raw_indices=np.array([0]), target_entry_raw_indices=np.array([0]),
                         bone_transforms=d, rest_joints=np.zeros((1,3)), frame_numbers=np.array([1,2]))
                paths.append(path)
            before = [sha256_file(p) for p in paths]
            train, valid = base/'train.jsonl', base/'valid.jsonl'
            train.write_text(''.join(json.dumps({'path':str(p),'untouched':'yes'})+'\n' for p in paths[:2]))
            valid.write_text(json.dumps({'path':str(paths[2]),'untouched':'yes'})+'\n')
            excluded = base/'excluded.jsonl'
            excluded.write_text(json.dumps({'path':str(paths[2]),'reason':'selected_mesh_hidden_at_sampled_frame'})+'\n')
            args = argparse.Namespace(output=base/'output', train_manifest=train, valid_manifest=valid,
                train_sha256=sha256_file(train), valid_sha256=sha256_file(valid), exclusions=excluded,
                selection_json=None,workers=1,shard_size=1)
            receipt = publish(args)
            self.assertEqual(receipt['counts']['accepted'], {'train':2})
            self.assertEqual(receipt['counts']['rejected'], {'valid':1})
            self.assertEqual(receipt['counts']['nonrigid_accepted'], {'train':1})
            self.assertTrue((args.output/'.READY').is_file())
            self.assertEqual(before,[sha256_file(p) for p in paths])
            rows = [json.loads(s) for s in (args.output/'quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['untouched']=='yes' for r in rows))
            for row in rows:
                with np.load(row['root_rt']['cache_path'],allow_pickle=False) as z:
                    self.assertEqual(z['rotations'].shape,(1,2,3,3))
                    self.assertEqual(z['frame_counts'][0],2)
            with self.assertRaises(FileExistsError):
                publish(args)


if __name__ == "__main__":
    unittest.main()
