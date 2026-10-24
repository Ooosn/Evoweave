from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rigweave.dynamic_rig import data, root_rt
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.shared_structure_medium_data import load_medium_batch, select_source_view
from rigweave.dynamic_rig.virtual_frame_data import (
    VirtualFrameDataset, collate_virtual_targets, prepare_virtual_surface,
)


def rotation(x: float, z: float) -> np.ndarray:
    cx, sx, cz, sz = np.cos(x), np.sin(x), np.cos(z), np.sin(z)
    return np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.asarray(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


class RecordingTokenizer:
    def tokenize(self, value):
        self.value = value
        return np.rint((value.bones.reshape(-1) + 1) * 100).astype(np.int64)


@contextmanager
def tokenizer_spec():
    # Test the real loader/token-input assembly without the vendor tokenizer.
    spec = types.ModuleType("src.tokenizer.spec")
    spec.TokenizeInput = types.SimpleNamespace
    with patch.dict(sys.modules, {"src.tokenizer.spec": spec}):
        yield


class RootRTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "asset.npz"
        self.cache = self.directory / "rt.npz"
        count = 40
        base = np.asarray([[x, y, z] for x in (-2, 2) for y in (-1.5, 1.5) for z in (-1, 1)])
        targets = np.asarray([[0, 0, 0], [.3, .3, .3], [-.4, .5, -.2], [.7, -.3, .2]])
        residual = np.stack([np.diag([1 + .1 * np.sin(i), 1 + .03 * i, 1]) for i in range(count)])
        residual[:, 0, 1] = .07 * np.sin(np.arange(count))
        self.aligned_vertices = np.einsum("vi,tji->tvj", base, residual).astype(np.float32)
        self.aligned_targets = np.einsum("vi,tji->tvj", targets, residual).astype(np.float32)
        self.rotations = np.stack([rotation(.13 * i, -.09 * i) for i in range(count)])
        self.translations = np.arange(count)[:, None] * np.asarray([[.7, -.4, .2]])
        self.numbers = 100 + 3 * np.arange(count, dtype=np.int64)
        self.raw = {
            "canonical_schema_version": np.asarray("rootless_dynamic_npz_v3", dtype=object),
            "frame_vertices_rootspace": self.to_original(self.aligned_vertices),
            "target_joints_rootspace": self.to_original(self.aligned_targets),
            "faces": np.asarray([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
                                 [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7]], dtype=np.int64),
            "target_parents": np.asarray([-1, 0, 1, 0], dtype=np.int64),
            "target_has_skin": np.ones(4, dtype=bool),
            "target_is_synthetic_root": np.zeros(4, dtype=bool),
            "rest_joints": np.zeros((6, 3), dtype=np.float32),
            "raw_root_index": np.asarray(0, dtype=np.int64),
            "target_raw_indices": np.arange(2, 6, dtype=np.int64),
            "frame_numbers": self.numbers,
        }
        self.shard = {
            "schema_version": np.asarray(root_rt.VERSION),
            "rotations": self.rotations[None], "translations": self.translations[None],
            "frame_counts": np.asarray([count], dtype=np.int32),
            "path_hashes": np.asarray([hashlib.sha256(str(self.path).encode()).hexdigest()], dtype="U64"),
            "frame_numbers": self.numbers[None], "root_indices": np.asarray([2], dtype=np.int32),
            "relative_nonrigid": np.asarray([True], dtype=bool),
        }
        self.row = {"asset_id": "synthetic", "path": str(self.path), "root_rt": {
            "version": root_rt.VERSION, "cache_path": str(self.cache), "index": 0,
            "frame_count": count, "reference_index": 0, "normalization": root_rt.NORMALIZATION,
            "padding": root_rt.PADDING,
        }}
        self.write_asset()
        self.write_shard()
        root_rt._load_shard.cache_clear()
        self.addCleanup(root_rt._load_shard.cache_clear)

    def to_original(self, aligned):
        return (np.einsum("tpi,tji->tpj", aligned, self.rotations) + self.translations[:, None]).astype(np.float32)

    def write_asset(self):
        np.savez(self.path, **self.raw)

    def write_shard(self, values=None, path=None):
        np.savez(self.cache if path is None else path, **(self.shard if values is None else values))

    def load_rt(self, row=None, numbers=None, count=40, root_index=2):
        return root_rt.load_root_rt(self.row if row is None else row, count,
                                    self.numbers if numbers is None else numbers, root_index=root_index)

    def test_relocated_files_preserve_canonical_identity_and_exact_geometry(self):
        original = VirtualFrameDataset([self.row], mode="seen_eval")[0]
        canonical = deepcopy(self.row)
        canonical["path"] = "/original/server/asset.npz"
        canonical["root_rt"]["cache_path"] = "/original/server/rt.npz"
        saved = deepcopy(canonical)
        shard = deepcopy(self.shard)
        shard["path_hashes"] = np.asarray([hashlib.sha256(canonical["path"].encode()).hexdigest()], dtype="U64")
        self.write_shard(shard)
        mapping = {canonical["path"]: self.path, canonical["root_rt"]["cache_path"]: self.cache}
        moved = VirtualFrameDataset([canonical], mode="seen_eval", file_resolver=mapping.__getitem__)[0]
        self.assertEqual(canonical, saved)
        self.assertEqual(moved["path"], [canonical["path"]])
        self.assertEqual(moved["resolved_path"], [str(self.path)])
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, moved[key]), key)
            elif key != "path":
                self.assertEqual(value, moved[key], key)
        bad = deepcopy(canonical)
        bad["path"] = "/original/server/wrong_asset.npz"
        mapping[bad["path"]] = self.path
        with self.assertRaisesRegex(ValueError, "path hash mismatch"):
            load_medium_batch(bad, [0, 1], file_resolver=mapping.__getitem__)

    def test_canonical_posix_manifest_paths_survive_windows_reading(self):
        from rigweave.dynamic_rig.virtual_frame_data import read_virtual_manifest
        row = deepcopy(self.row)
        row["path"] = "/original/server/asset.npz"
        row["root_rt"]["cache_path"] = "/original/server/rt.npz"
        manifest = self.directory / 'remote_manifest.jsonl'
        manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')
        self.assertEqual(read_virtual_manifest(manifest, expected_count=1,
            expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()), [row])
        root_rt.validate_descriptor(row)
        row['path'] = 'relative/asset.npz'
        with self.assertRaisesRegex(ValueError, 'original absolute'):
            root_rt.validate_descriptor(row)

    def flat(self, row=None, global_rotation=None, legacy=False):
        manifest = self.directory / "manifest.jsonl"
        manifest.write_text(json.dumps(self.row if row is None else row) + "\n", encoding="utf-8")
        tokenizer = RecordingTokenizer()
        module = sys.modules["rigweave.dynamic_rig._legacy_data"] if legacy else data
        kwargs = {} if legacy else {"global_rotation": global_rotation}
        dataset = module.DynamicRigManifestDataset(
            manifest, tokenizer, frame_count=8, random_query=False, seed=71, **kwargs)
        with tokenizer_spec():
            sample = dataset[0]
        return sample, tokenizer

    def test_alignment_retains_deformation_and_roundtrips(self):
        rt = self.load_rt()
        for key, expected in (("frame_vertices_rootspace", self.aligned_vertices),
                              ("target_joints_rootspace", self.aligned_targets)):
            actual = root_rt.align_points(self.raw[key], rt)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        self.assertTrue(rt.relative_nonrigid)
        self.assertGreater(np.max(np.abs(self.aligned_vertices[-1] - self.aligned_vertices[0])), .5)
        ids = np.asarray([33, 0, 7, 19])
        q = rotation(.37, -.49)
        vertices, targets, center, scale, saved_q = root_rt.normalize_inputs(
            root_rt.align_points(self.raw["frame_vertices_rootspace"], rt)[ids],
            root_rt.align_points(self.raw["target_joints_rootspace"], rt)[ids], q)
        meta = rt.selected_metadata(ids, saved_q)
        for normalized, key in ((vertices, "frame_vertices_rootspace"), (targets, "target_joints_rootspace")):
            restored = root_rt.restore_points(normalized, meta["root_rt_rotations"],
                meta["root_rt_translations"], center, scale, meta["root_rt_augmentation"])
            np.testing.assert_allclose(restored, self.raw[key][ids], atol=4e-6, rtol=0)
        self.assertFalse(rt.rotations.flags.writeable)
        self.assertTrue(meta["root_rt_rotations"].flags.writeable)

    def test_descriptor_fail_closed(self):
        bad = {"version": "old", "normalization": "query_bbox", "padding": 1.,
               "reference_index": 1, "index": -1, "frame_count": 0, "cache_path": "relative.npz"}
        for key, value in bad.items():
            row = deepcopy(self.row)
            row["root_rt"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load_rt(row)
        for key in self.row["root_rt"]:
            row = deepcopy(self.row)
            del row["root_rt"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.load_rt(row)
        for value in (None, [], "cache", True):
            with self.subTest(spec=value), self.assertRaises(ValueError):
                self.load_rt({**self.row, "root_rt": value})
        for key in ("index", "frame_count", "reference_index"):
            for value in (True, 1.5):
                row = deepcopy(self.row)
                row["root_rt"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load_rt(row)
        with self.assertRaises(ValueError):
            self.load_rt({**self.row, "path": "asset.npz"})
        self.assertIsNone(root_rt.load_root_rt({"path": str(self.path)}, 0, None))

    def test_asset_binding_frame_count_numbers_and_root_checked(self):
        row = deepcopy(self.row)
        row["path"] = str(self.directory / "wrong.npz")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load_rt(row)
        row = deepcopy(self.row)
        row["root_rt"]["index"] = 1
        with self.assertRaisesRegex(ValueError, "index"):
            self.load_rt(row)
        with self.assertRaisesRegex(ValueError, "frame_count"):
            self.load_rt(count=39)
        with self.assertRaisesRegex(ValueError, "frame_numbers"):
            self.load_rt(numbers=self.numbers[::-1])
        with self.assertRaisesRegex(ValueError, "frame_numbers"):
            self.load_rt(numbers=self.numbers.astype(float))
        with self.assertRaisesRegex(ValueError, "frame_numbers"):
            root_rt.load_root_rt(self.row, 40, None)
        with self.assertRaisesRegex(ValueError, "root index"):
            self.load_rt(root_index=0)
        changed = deepcopy(self.shard)
        changed["frame_counts"][0] = 39
        changed["frame_numbers"][0, 39] = -1
        self.write_shard(changed)
        with self.assertRaisesRegex(ValueError, "frame_count"):
            self.load_rt()

    def test_shard_schema_dtypes_shapes_and_rigid_checks(self):
        mutations = [
            ("schema", lambda x: x.update(schema_version=np.asarray("wrong"))),
            ("schema_shape", lambda x: x.update(schema_version=np.asarray([root_rt.VERSION]))),
            ("float32", lambda x: x.update(rotations=x["rotations"].astype(np.float32))),
            ("count_type", lambda x: x.update(frame_counts=x["frame_counts"].astype(np.int64))),
            ("hash_type", lambda x: x.update(path_hashes=x["path_hashes"].astype("S64"))),
            ("shape", lambda x: x.update(translations=x["translations"][:, :-1])),
            ("reflection", lambda x: x["rotations"].__setitem__((0, 1), np.diag([-1., 1., 1.]))),
            ("scale", lambda x: x["rotations"].__setitem__((0, 1), np.eye(3) * 1.1)),
            ("nonfinite", lambda x: x["translations"].__setitem__((0, 1, 0), np.nan)),
            ("ref_rotation", lambda x: x["rotations"].__setitem__((0, 0), rotation(.2, .3))),
            ("ref_translation", lambda x: x["translations"].__setitem__((0, 0, 0), .1)),
            ("bad_hash", lambda x: x["path_hashes"].__setitem__(0, "g" * 64)),
            ("bad_root", lambda x: x["root_indices"].__setitem__(0, -1)),
            ("bad_count", lambda x: x["frame_counts"].__setitem__(0, 41)),
            ("missing", lambda x: x.pop("relative_nonrigid")),
        ]
        for name, mutate in mutations:
            changed = deepcopy(self.shard)
            mutate(changed)
            self.write_shard(changed)
            root_rt._load_shard.cache_clear()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load_rt()

    def test_cache_rotation_tolerance_is_not_source_nonrigid_threshold(self):
        self.assertEqual(root_rt.RIGID_ATOL, 1e-6)
        changed = deepcopy(self.shard)
        changed["rotations"][0, 1] *= 1 + 1e-10
        self.write_shard(changed)
        self.load_rt()
        changed["rotations"][0, 1] = self.rotations[1] * (1 + 1e-5)
        self.write_shard(changed)
        root_rt._load_shard.cache_clear()
        with self.assertRaisesRegex(ValueError, "orthogonality"):
            self.load_rt()

    def test_padded_shards_cache_invalidation_and_archive_closure(self):
        changed = deepcopy(self.shard)
        for key in ("rotations", "translations", "frame_counts", "path_hashes", "frame_numbers",
                    "root_indices", "relative_nonrigid"):
            changed[key] = np.concatenate([changed[key], changed[key]], axis=0)
        changed["frame_counts"][1] = 3
        changed["frame_numbers"][1, 3:] = -1
        changed["rotations"][1, 3:] = 0
        self.write_shard(changed)
        archives = []
        original_load = np.load

        def tracked_load(*args, **kwargs):
            result = original_load(*args, **kwargs)
            archives.append(result)
            return result

        with patch.object(root_rt.np, "load", side_effect=tracked_load):
            self.load_rt()
            self.load_rt()
        self.assertEqual(len(archives), 1)
        self.assertIsNone(archives[0].zip)
        replacement = self.directory / "replacement.npz"
        changed["translations"][0, 1, 0] += 10
        self.write_shard(changed, replacement)
        replacement.replace(self.cache)
        self.assertEqual(self.load_rt().translations[1, 0], 10.7)
        changed["frame_numbers"][1, 4] = 0
        self.write_shard(changed)
        with self.assertRaisesRegex(ValueError, "padding"):
            self.load_rt()
        for i in range(3):
            path = self.directory / f"cache{i}.npz"
            self.write_shard(path=path)
            row = deepcopy(self.row)
            row["root_rt"]["cache_path"] = str(path)
            self.load_rt(row)
        self.assertLessEqual(root_rt._load_shard.cache_info().currsize, 2)

    def test_bbox_uses_input_union_only_and_no_gt_rescale_or_clip(self):
        vertices = self.aligned_vertices[[0, 39]]
        targets = self.aligned_targets[[0, 39]]
        first = root_rt.normalize_inputs(vertices, targets)
        second = root_rt.normalize_inputs(vertices, targets * .5)
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(first[3], second[3])
        expected_scale = np.float32(1.1 * np.max(np.ptp(vertices, axis=(0, 1))) * .5)
        self.assertEqual(first[3], expected_scale)
        self.assertGreater(first[3], root_rt.normalize_inputs(vertices[:1], targets[:1])[3])
        outlier = targets.copy()
        outlier[1, 0, 0] = 100
        with self.assertRaisesRegex(ValueError, r"outside \[-1,1\]"):
            root_rt.normalize_inputs(vertices, outlier)

    def test_rotation_before_bbox_fail_closes_on_out_of_range_gt(self):
        vertices = np.asarray([[[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]])
        targets = np.asarray([[[.9, .9, 0]]])
        self.assertLess(root_rt.normalize_inputs(vertices, targets)[1].max(), 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            root_rt.normalize_inputs(vertices, targets, rotation(0, np.pi / 4))
        for q in (np.eye(3) * 2, np.diag([1., 1., -1.]), np.full((3, 3), np.nan), np.eye(4)):
            with self.subTest(q=q), self.assertRaises(ValueError):
                root_rt.normalize_inputs(vertices, targets, q)

    def test_flat_loader_matches_medium_and_tokenizes_aligned_targets(self):
        q = rotation(.2, .3)
        sample, tokenizer = self.flat(global_rotation=q)
        ids = sample.selected_frames.numpy()
        medium = load_medium_batch(self.row, ids, global_rotation=q)
        for key in ("frame_vertices", "target_joints", "vertex_normals", "face_normals", "query_center", "query_scale"):
            torch.testing.assert_close(getattr(sample, key), medium[key][0], rtol=0, atol=0)
        np.testing.assert_array_equal(tokenizer.value.bones[:, 3:], sample.target_joints.numpy())
        self.assertEqual(sample.alignment_root_index, 2)
        self.assertEqual(sample.root_rt["root_rt_root_index"].item(), 2)
        torch.testing.assert_close(sample.root_rt["root_rt_rotations"], torch.from_numpy(self.rotations[ids]))
        batch = data.dynamic_rig_collate([sample, sample], pad_token=0)
        for key in root_rt.BATCH_KEYS:
            self.assertEqual(batch[key].shape[0], 2)
        restored = root_rt.restore_points(sample.target_joints.numpy()[None],
            sample.root_rt["root_rt_rotations"].numpy()[:1], sample.root_rt["root_rt_translations"].numpy()[:1],
            sample.query_center.numpy(), sample.query_scale.item(), sample.root_rt["root_rt_augmentation"].numpy())
        np.testing.assert_allclose(restored[0], self.raw["target_joints_rootspace"][ids[0]], atol=3e-6, rtol=0)

    def test_flat_selection_receives_aligned_geometry_before_fps(self):
        with patch.object(data, "_select_query_sequence", wraps=data._select_query_sequence) as select:
            sample, _ = self.flat()
        seen_vertices, seen_targets = select.call_args.args[:2]
        np.testing.assert_allclose(seen_vertices, self.aligned_vertices, atol=2e-6, rtol=0)
        np.testing.assert_allclose(seen_targets, self.aligned_targets, atol=2e-6, rtol=0)
        self.assertEqual(sample.frame_vertices.shape[0], 8)

    def test_flat_checks_all_selected_gt_not_only_query(self):
        sample, _ = self.flat()
        altered = self.aligned_targets.copy()
        altered[sample.selected_frames[1].item(), 0, 0] = 100
        self.raw["target_joints_rootspace"] = self.to_original(altered)
        self.write_asset()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.flat()

    @unittest.skipUnless("rigweave.dynamic_rig._legacy_data" in sys.modules,
                         "original implementation is supplied by the staging runner")
    def test_flat_legacy_is_exactly_unchanged(self):
        row = {"path": str(self.path)}
        current, tokenizer = self.flat(row)
        legacy, old_tokenizer = self.flat(row, legacy=True)
        for field in fields(legacy):
            actual, expected = getattr(current, field.name), getattr(legacy, field.name)
            with self.subTest(field=field.name):
                if isinstance(expected, torch.Tensor):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                else:
                    self.assertEqual(actual, expected)
        np.testing.assert_array_equal(tokenizer.value.bones, old_tokenizer.value.bones)
        self.assertEqual(current.alignment_root_index, 0)
        self.assertIsNone(current.root_rt)
        self.assertFalse(set(root_rt.BATCH_KEYS) & data.dynamic_rig_collate([current], pad_token=0).keys())
        rt_sample, _ = self.flat()
        with self.assertRaisesRegex(ValueError, "mix"):
            data.dynamic_rig_collate([rt_sample, current], pad_token=0)
        with self.assertRaisesRegex(ValueError, "requires root_rt"):
            self.flat(row, global_rotation=np.eye(3))

    def test_medium_callable_sees_aligned_meshes_and_validates_mapping(self):
        seen = []

        def select(vertices):
            seen.append(vertices.copy())
            return [30, 2, 17]

        sample = load_medium_batch(self.row, select)
        np.testing.assert_allclose(seen[0], self.aligned_vertices, atol=2e-6, rtol=0)
        self.assertEqual(sample["frame_numbers"][0].tolist(), self.numbers[[30, 2, 17]].tolist())
        self.raw["target_raw_indices"][0] = 0
        self.write_asset()
        with self.assertRaisesRegex(ValueError, "root index"):
            load_medium_batch(self.row, [0, 1])

    def test_source_view_matches_direct_subset_and_inverse(self):
        q = rotation(.3, .2)
        full = load_medium_batch(self.row, [0, 2, 8, 17, 30, 39], global_rotation=q)
        before = full["frame_vertices"].clone()
        ids = [17, 2, 30]
        selected = select_source_view(full, ids)
        direct = load_medium_batch(self.row, ids, global_rotation=q)
        for key in ("frame_vertices", "frame_target_joints", "query_center", "query_scale", *root_rt.BATCH_KEYS):
            torch.testing.assert_close(selected[key], direct[key], atol=5e-7, rtol=1e-6)
        torch.testing.assert_close(full["frame_vertices"], before, atol=0, rtol=0)
        restored = root_rt.restore_points(selected["frame_target_joints"][0].numpy(),
            selected["root_rt_rotations"][0].numpy(), selected["root_rt_translations"][0].numpy(),
            selected["query_center"][0].numpy(), selected["query_scale"].item(),
            selected["root_rt_augmentation"][0].numpy())
        np.testing.assert_allclose(restored, self.raw["target_joints_rootspace"][ids], atol=3e-6, rtol=0)
        incomplete = dict(full)
        del incomplete["root_rt_rotations"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select_source_view(incomplete, ids)

    def test_virtual_loader_metadata_heldout_isolation_and_surface(self):
        dataset = VirtualFrameDataset([self.row], frames=4, global_rotation=rotation(.2, -.3))
        request = FrameSampleRequest(0, 4, 91)
        batch = dataset[request]
        ids = batch["selected_frames"][0].numpy()
        heldout = batch["heldout_raw_ids"]
        self.assertFalse(set(ids) & set(heldout))
        self.raw["frame_vertices_rootspace"][heldout] *= 1000
        self.raw["target_joints_rootspace"][heldout] -= 1000
        self.write_asset()
        after = dataset[request]
        for key in ("selected_frames", "frame_vertices", "frame_target_joints", "query_center", "query_scale"):
            torch.testing.assert_close(batch[key], after[key], atol=0, rtol=0)
        surface, target = prepare_virtual_surface([batch], "cpu", surface_samples=24, vertex_samples=8, query_tokens=7)
        self.assertEqual(surface["dense_points"].shape, (1, 4, 24, 3))
        for key in root_rt.BATCH_KEYS:
            torch.testing.assert_close(target[key], batch[key], atol=0, rtol=0)
        for key in root_rt.BATCH_KEYS:
            incomplete = dict(batch)
            del incomplete[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mix"):
                collate_virtual_targets([incomplete], "cpu")
        legacy = {key: value for key, value in batch.items() if key not in root_rt.BATCH_KEYS}
        with self.assertRaisesRegex(ValueError, "mix"):
            collate_virtual_targets([batch, legacy], "cpu")
        self.raw["target_joints_rootspace"][heldout] = 1e9
        self.write_asset()
        with self.assertRaisesRegex(ValueError, "outside"):
            VirtualFrameDataset([self.row], frames=8, mode="heldout_eval")[0]

    def test_virtual_fps_uses_aligned_inputs_and_gt_does_not_change_chart(self):
        import rigweave.dynamic_rig.virtual_frame_data as virtual
        dataset = VirtualFrameDataset([self.row], frames=4)
        with patch.object(virtual, "select_mesh_frame_ids", wraps=virtual.select_mesh_frame_ids) as select:
            before = dataset[0]
        observed = select.call_args.args[0]
        expected = root_rt.align_points(self.raw["frame_vertices_rootspace"], self.load_rt())
        for frame in observed:
            self.assertTrue(any(np.array_equal(frame, candidate) for candidate in expected))
        altered = self.aligned_targets * .5
        self.raw["target_joints_rootspace"] = self.to_original(altered)
        self.write_asset()
        after = dataset[0]
        for key in ("selected_frames", "frame_vertices", "query_center", "query_scale"):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)

    def test_bad_metadata_cannot_silently_fall_back_in_loaders(self):
        row = deepcopy(self.row)
        row["root_rt"]["version"] = "wrong"
        for loader in (lambda: self.flat(row), lambda: load_medium_batch(row, [0, 1]),
                       lambda: VirtualFrameDataset([row], frames=2)[0]):
            with self.subTest(loader=loader), self.assertRaisesRegex(ValueError, "unsupported"):
                loader()

    def test_puppeteer_rejects_rt_even_when_joint_limit_would_filter_it(self):
        from rigweave.dynamic_rig.puppeteer_dynamic import _load_puppeteer_manifest
        manifest = self.directory / "puppet.jsonl"
        manifest.write_text(json.dumps({**self.row, "target_joint_count": 1000}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "root_rt.*not supported"):
            _load_puppeteer_manifest(manifest, max_joints=1)
        manifest.write_text(json.dumps({"path": str(self.path), "target_joint_count": 4}) + "\n", encoding="utf-8")
        self.assertEqual(_load_puppeteer_manifest(manifest)[0], [self.path])

    def test_legacy_shared_wrapper_rejects_new_rt_coordinates(self):
        from rigweave.dynamic_rig.shared_structure_data import _load_frame_target_joints
        sample, _ = self.flat()
        with self.assertRaisesRegex(ValueError, "does not support root_rt"):
            _load_frame_target_joints(sample)
        legacy, _ = self.flat({"path": str(self.path)})
        result = _load_frame_target_joints(legacy)
        expected = (self.raw["target_joints_rootspace"][legacy.selected_frames.numpy()]
                    - legacy.query_center.numpy()) / legacy.query_scale.item()
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
