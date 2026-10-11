from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rigweave.dynamic_rig.data import _face_normals, _vertex_normals
from rigweave.dynamic_rig.sampling import materialize_trackable_surface, sample_trackable_surface
from rigweave.dynamic_rig.shared_structure_coordinates import normalize_target_meshes, restore_target_coordinates
from rigweave.dynamic_rig.shared_structure_medium_data import (
    load_medium_batch, partition_raw_frames, select_source_view, source_bank_indices,
)


class SharedStructureMediumDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def payload(self, frames: int = 40) -> dict[str, np.ndarray]:
        mesh = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
        joints = np.asarray([[0.25, 0.25, 0.25], [0.5, 0.25, 0.25],
                             [0.75, 0.5, 0.25], [1.25, 0.5, 0.5]], dtype=np.float32)
        scale = 1 + np.arange(frames, dtype=np.float32) / 8
        offset = np.stack([np.arange(frames) * 2, -np.arange(frames), np.arange(frames) / 2], axis=-1)
        deformation = np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0)
        deformation[:, 0, 0] += np.arange(frames) % 5 / 16
        deformation[:, 1, 1] += np.arange(frames) % 7 / 20
        deformation[:, 0, 2] = np.arange(frames) % 3 / 8
        mesh_frames = np.einsum("fij,vj->fvi", deformation, mesh)
        joint_frames = np.einsum("fij,vj->fvi", deformation, joints)
        return {
            "canonical_schema_version": np.asarray("rootless_dynamic_npz_v3", dtype=object),
            "frame_vertices_rootspace": (mesh_frames * scale[:, None, None] + offset[:, None]).astype(np.float32),
            "faces": np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
            "target_joints_rootspace": (joint_frames * scale[:, None, None] + offset[:, None]).astype(np.float32),
            "target_parents": np.asarray([-1, 0, 1, 1], dtype=np.int64),
            "target_has_skin": np.asarray([True, True, False, False]),
            "target_is_synthetic_root": np.zeros(4, dtype=bool),
            "frame_numbers": np.arange(frames, dtype=np.int64) * 3 + 100,
        }

    def write(self, root: Path, payload: dict[str, np.ndarray], name: str = "asset.npz") -> Path:
        path = root / name
        np.savez(path, **payload)
        return path

    def assert_batch_unchanged(self, before: dict, after: dict) -> None:
        self.assertEqual(before.keys(), after.keys())
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(after[key], value, atol=0, rtol=0)
            else:
                self.assertEqual(after[key], value)

    def test_partition_is_deterministic_disjoint_and_geometry_free(self) -> None:
        a, h = partition_raw_frames(40, 20261012)
        state = np.random.get_state()
        try:
            np.random.seed(11)
            np.random.random(10)
            b, k = partition_raw_frames(40, 20261012)
        finally:
            np.random.set_state(state)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(h, k)
        self.assertEqual((len(a), len(h)), (32, 8))
        self.assertEqual(a.dtype, np.int64)
        self.assertEqual(h.dtype, np.int64)
        self.assertFalse(set(a) & set(h))
        self.assertEqual(set(a) | set(h), set(range(40)))
        self.assertFalse(np.array_equal(a, partition_raw_frames(40, 20261013)[0]))
        self.assertEqual(tuple(map(len, partition_raw_frames(32, 0))), (24, 8))
        self.assertEqual(tuple(map(len, partition_raw_frames(40, 0, 16))), (24, 16))

    def test_partition_rejects_invalid_counts_or_seed(self) -> None:
        for args in ((31, 0), (40, -1), (40, True), (40.0, 0), (40, 0, 7), (40, 0, 17)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                partition_raw_frames(*args)

    def test_source_banks_have_same_eligible_ids_and_different_query(self) -> None:
        train, heldout = partition_raw_frames(40, 37)
        before = train.copy()
        a, b = source_bank_indices(train, 0), source_bank_indices(torch.from_numpy(train), 1)
        np.testing.assert_array_equal(a, train)
        np.testing.assert_array_equal(b, np.roll(train, -16))
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(set(a), set(b))
        self.assertFalse(set(b) & set(heldout))
        a[0] = 999
        np.testing.assert_array_equal(train, before)
        for ids, bank in (([], 0), ([1], 1), ([1, 1], 0), ([1.0, 2.0], 0), ([1, 2], 2), ([1, 2], True)):
            with self.subTest(ids=ids, bank=bank), self.assertRaises(ValueError):
                source_bank_indices(ids, bank)

    def test_load_preserves_order_all_joints_and_source_affine(self) -> None:
        raw = self.payload()
        ids = np.asarray([19, 2, 31, 7])
        before = deepcopy(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), raw)
            batch = load_medium_batch(path, ids, cls_name="test")
        self.assertEqual(batch["path"], [str(path)])
        self.assertEqual(batch["cls"], ["test"])
        self.assertEqual(batch["selected_frames"].tolist(), [ids.tolist()])
        self.assertEqual(batch["frame_numbers"].tolist(), [raw["frame_numbers"][ids].tolist()])
        self.assertEqual(batch["joint_count"].tolist(), [4])
        self.assertTrue(bool(batch["joint_mask"].all()))
        self.assertEqual(batch["target_has_skin"].tolist(), [[True, True, False, False]])
        self.assertEqual(batch["target_parents"].tolist(), [[-1, 0, 1, 1]])
        self.assertNotIn("input_ids", batch)
        query = raw["frame_vertices_rootspace"][ids[0]]
        center = (query.min(axis=0) + query.max(axis=0)) * 0.5
        scale = np.max((query.max(axis=0) - query.min(axis=0)) * 0.5)
        torch.testing.assert_close(batch["query_center"], torch.from_numpy(center)[None])
        torch.testing.assert_close(batch["query_scale"], torch.tensor([scale]))
        for key, raw_key in (("frame_vertices", "frame_vertices_rootspace"),
                             ("frame_target_joints", "target_joints_rootspace")):
            restored = batch[key] * batch["query_scale"][:, None, None, None] + batch["query_center"][:, None, None]
            torch.testing.assert_close(restored[0], torch.from_numpy(raw[raw_key][ids]), atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(batch["target_joints"], batch["frame_target_joints"][:, 0], atol=0, rtol=0)
        for key in raw:
            np.testing.assert_array_equal(raw[key], before[key])

    def test_load_normals_match_existing_helpers_and_sampling_accepts_batch(self) -> None:
        raw = self.payload()
        with tempfile.TemporaryDirectory() as directory:
            batch = load_medium_batch(self.write(Path(directory), raw), [5, 1, 9])
        faces = raw["faces"]
        self.assertFalse(torch.equal(batch["vertex_normals"][:, 0], batch["vertex_normals"][:, 1]))
        for frame, vertices in enumerate(batch["frame_vertices"][0].numpy()):
            face_normals = _face_normals(vertices, faces)
            vertex_normals = _vertex_normals(vertices, faces, face_normals)
            np.testing.assert_allclose(batch["face_normals"][0, frame], face_normals, atol=1e-6)
            np.testing.assert_allclose(batch["vertex_normals"][0, frame], vertex_normals, atol=1e-6)
            np.testing.assert_allclose(face_normals[0], [0, 0, -1], atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(vertex_normals, axis=1), 1, atol=1e-6)
        refs = sample_trackable_surface(batch["frame_vertices"][:, 0], batch["faces"],
            num_samples=8, vertex_samples=4, query_tokens=3,
            vertex_counts=batch["vertex_count"], face_counts=batch["face_count"],
            generator=torch.Generator().manual_seed(3))
        sampled = materialize_trackable_surface(batch["frame_vertices"][:, 1], batch["faces"], refs,
            vertex_normals=batch["vertex_normals"][:, 1], face_normals=batch["face_normals"][:, 1])
        self.assertEqual(tuple(sampled.query_points.shape), (1, 3, 3))
        self.assertTrue(bool(torch.isfinite(sampled.dense_normals).all()))

    def test_source_view_rebases_raw_ids_and_preserves_every_aligned_field(self) -> None:
        raw = self.payload()
        ids, selected = [12, 3, 27, 8, 16], [27, 12, 16]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), raw)
            batch = load_medium_batch(path, ids)
            direct = load_medium_batch(path, selected)
        before = deepcopy(batch)
        view = select_source_view(batch, selected)
        for key, value in direct.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(view[key], value, atol=1e-5, rtol=1e-5)
            else:
                self.assertEqual(view[key], value)
        self.assert_batch_unchanged(before, batch)
        nested = select_source_view(view, [16, 27])
        normalized = normalize_target_meshes(nested)
        restored = restore_target_coordinates(normalized["frame_target_joints"],
            normalized["target_source_centers"], normalized["target_source_scales"])
        torch.testing.assert_close(restored[0], torch.from_numpy(raw["target_joints_rootspace"][[16, 27]]),
                                   atol=2e-5, rtol=1e-6)
        self.assertEqual(normalized["selected_frames"].tolist(), [[16, 27]])
        self.assertEqual(normalized["target_parents"].tolist(), [[-1, 0, 1, 1]])

    def test_heldout_mutations_do_not_change_train_sources_or_target_geometry(self) -> None:
        raw = self.payload()
        train, heldout = partition_raw_frames(40, 20261012)
        changed = deepcopy(raw)
        changed["frame_vertices_rootspace"][heldout] *= 7
        changed["frame_vertices_rootspace"][heldout] += 900
        changed["target_joints_rootspace"][heldout] -= 1234
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = load_medium_batch(self.write(root, raw), np.concatenate([train, heldout]))
            altered = load_medium_batch(self.write(root, changed, "changed.npz"), np.concatenate([train, heldout]))
        for bank in (0, 1):
            ids = source_bank_indices(train, bank)
            a, b = select_source_view(original, ids), select_source_view(altered, ids)
            for key, value in a.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(b[key], value, atol=0, rtol=0)
            target_a, target_b = normalize_target_meshes(a), normalize_target_meshes(b)
            for key in ("frame_vertices", "frame_target_joints", "target_source_centers", "target_source_scales"):
                torch.testing.assert_close(target_a[key], target_b[key], atol=0, rtol=0)

    def test_gt_cannot_change_query_chart_or_mesh_geometry(self) -> None:
        raw, changed = self.payload(), self.payload()
        changed["target_joints_rootspace"] += 5000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = load_medium_batch(self.write(root, raw), [7, 1, 4])
            b = load_medium_batch(self.write(root, changed, "changed.npz"), [7, 1, 4])
        for a, b in ((a, b), (select_source_view(a, [4, 1]), select_source_view(b, [4, 1]))):
            for key in ("frame_vertices", "vertex_normals", "face_normals", "query_center", "query_scale"):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
            self.assertGreater(float(b["frame_target_joints"].abs().max()), 1)
            target_a, target_b = normalize_target_meshes(a), normalize_target_meshes(b)
            torch.testing.assert_close(target_a["frame_vertices"], target_b["frame_vertices"], atol=0, rtol=0)

    def test_optional_frame_numbers_and_single_frame_selection(self) -> None:
        raw = self.payload()
        del raw["frame_numbers"]
        with tempfile.TemporaryDirectory() as directory:
            batch = load_medium_batch(self.write(Path(directory), raw), [4])
        self.assertNotIn("frame_numbers", batch)
        view = select_source_view(batch, [4])
        self.assertEqual(tuple(view["frame_vertices"].shape), (1, 1, 4, 3))

    def test_load_rejects_missing_fields_and_malformed_canonical_arrays(self) -> None:
        raw = self.payload()
        cases = []
        for key in raw.keys() - {"frame_numbers"}:
            altered = deepcopy(raw)
            del altered[key]
            cases.append((f"missing_{key}", altered))
        malformed = {
            "canonical_schema_version": np.asarray("legacy"),
            "frame_vertices_rootspace": np.zeros((40, 4, 2)),
            "faces": np.asarray([[0, 1, 4]]),
            "target_joints_rootspace": np.zeros((39, 4, 3)),
            "target_parents": np.asarray([-1, 0, 1]),
            "target_has_skin": np.ones(3, dtype=bool),
            "target_is_synthetic_root": np.ones(4, dtype=bool),
            "frame_numbers": np.arange(39),
        }
        for key, value in malformed.items():
            altered = deepcopy(raw)
            altered[key] = value
            cases.append((key, altered))
        for parents in ([-1, -1, 0, 0], [-1, 0, 2, 1], [0, 0, 1, 1], [-1.0, 0.0, 1.0, 1.0]):
            altered = deepcopy(raw)
            altered["target_parents"] = np.asarray(parents)
            cases.append((f"parents_{parents}", altered))
        altered = deepcopy(raw)
        altered["target_parents"] = np.asarray([2**64 - 1, 0, 1, 1], dtype=np.uint64)
        cases.append(("unsigned_parent_overflow", altered))
        for key in ("frame_vertices_rootspace", "target_joints_rootspace"):
            altered = deepcopy(raw)
            altered[key][0, 0, 0] = np.nan
            cases.append((f"nonfinite_{key}", altered))
        altered = deepcopy(raw)
        altered["frame_vertices_rootspace"][0] = 0
        cases.append(("degenerate_query", altered))
        with tempfile.TemporaryDirectory() as directory:
            for label, payload in cases:
                with self.subTest(label=label), self.assertRaises((ValueError, KeyError)):
                    load_medium_batch(self.write(Path(directory), payload), [0, 1])

    def test_load_and_view_reject_invalid_ids_or_partial_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), self.payload())
            for ids in ([], [-1, 0], [0, 0], [0.0, 1.0], [True, False], [[0, 1]], [0, 40]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    load_medium_batch(path, ids)
            batch = load_medium_batch(path, [9, 1, 4])
        for ids in ([], [9, 9], [0, 1], [-1], [9.0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_source_view(batch, ids)
        for key in ("frame_target_joints", "query_scale", "vertex_normals", "target_parents"):
            partial = dict(batch)
            del partial[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                select_source_view(partial, [9, 1])
        changed = deepcopy(batch)
        changed["target_joints"] += 1
        with self.assertRaisesRegex(ValueError, "first selected"):
            select_source_view(changed, [9, 1])
        changed = deepcopy(batch)
        changed["frame_mask"][0, 2] = False
        with self.assertRaisesRegex(ValueError, "unpadded"):
            select_source_view(changed, [9, 1])
        with self.assertRaisesRegex(ValueError, "per-mesh"):
            select_source_view(normalize_target_meshes(batch), [9, 1])


if __name__ == "__main__":
    unittest.main()
