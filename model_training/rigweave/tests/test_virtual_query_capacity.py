"""Focused CPU checks with the real auxiliary codec and manifest reader."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_virtual_query_capacity as capacity

SOURCE_ROOT = None
TEST_OUTPUT = None


def chain(count):
    return np.arange(-1, count - 1, dtype=np.int64)


def star(count):
    return np.array([-1] + [0] * (count - 1), dtype=np.int64)


def independent_length(parents):
    return 3 + sum(3 if i == 0 or int(parent) == i - 1 else 7 for i, parent in enumerate(parents))


def row(asset_id, parents, vertices=10, faces=10):
    root = Path(TEST_OUTPUT).resolve() / "unopened"
    return {"asset_id": asset_id, "path": str(root / f"{asset_id}.npz"), "split": "train",
            "integration_source": "base", "keep_original_payload": {"not_in_selection": True},
            "rootless_target_strict_metrics": {"target_joint_count": len(parents), "vertex_count": vertices,
                                                "face_count": faces},
            "canonical_metrics": {"target_joint_count": len(parents), "filtered_vertex_count": vertices,
                                  "filtered_face_count": faces},
            "root_rt": {"version": "retained_root_rigid_v1", "cache_path": str(root / "unread_rt.npz"),
                        "index": 0, "frame_count": 32, "reference_index": 0,
                        "normalization": "input_sequence_mesh_bbox", "padding": 1.1}}


def population():
    pairs = [("big_chain", chain(100), 10, 10), ("branch60", star(60), 10, 10),
             ("branch59", star(59), 10, 10), ("branch58", star(58), 10, 10),
             ("unread57", chain(57), 10, 10), ("max_vertices", chain(3), 300, 10),
             ("max_faces", chain(4), 10, 700), ("max_geometry", chain(5), 290, 699)]
    return [row(name, parents, v, f) for name, parents, v, f in pairs], {name: p for name, p, _, _ in pairs}


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = capacity.source_api(Path(SOURCE_ROOT))
        cls.api, cls.source_hashes = cls.context.__enter__()
        cls.addClassCleanup(cls.context.__exit__, None, None, None)
        cls.codec = cls.api["codec"].AuxSkeletonTokens()

    def select(self, rows, parents):
        paths = {item["path"]: item["asset_id"] for item in rows}
        reads = []
        def read(path):
            reads.append(paths[path])
            return parents[paths[path]].copy()
        result = capacity.select_capacity(rows, codec=self.codec,
            validate_descriptor=self.api["root_rt"].validate_descriptor, parents_reader=read)
        return result, reads

    def test_bound_certificate_matches_exhaustive_not_largest_joints(self):
        rows, parents = population()
        before = copy.deepcopy(rows)
        result, reads = self.select(rows, parents)
        exhaustive = sorted(rows, key=lambda r: (-independent_length(parents[r["asset_id"]]), r["asset_id"], r["path"]))
        self.assertEqual([r["asset_id"] for r in result["groups"]["longest_main"]], [r["asset_id"] for r in exhaustive[:3]])
        self.assertNotIn("big_chain", [r["asset_id"] for r in result["groups"]["longest_main"]])
        cert = result["census_certificate"]
        self.assertEqual(cert["length_search_parent_arrays_read"], 4)
        self.assertEqual(cert["next_max_bound"], 398)
        self.assertEqual(cert["third_measured_main_tokens"], 401)
        self.assertLess(cert["next_max_bound"], cert["third_measured_main_tokens"])
        self.assertNotIn("unread57", reads)
        self.assertEqual(rows, before)

    def test_non_topological_parents_preserve_original_order(self):
        parents = np.array([-1, 2, 0], dtype=np.int64)
        item = capacity.row_metadata(row("permuted", parents), 0, self.api["root_rt"].validate_descriptor)
        measured, stored = capacity.measure_parents(item, parents, self.codec)
        self.assertEqual(measured["branch_count"], 2)
        self.assertEqual(measured["main_tokens"], 20)
        self.assertEqual(capacity.main_token_bound(3), 20)
        self.assertEqual(stored, [-1, 2, 0])
        self.assertEqual(self.codec.encode(parents).dfs_order.tolist(), [0, 2, 1])

    def test_equality_bound_is_read_and_third_place_ties_retained(self):
        parents = {name: chain(8) for name in ("z1", "z2", "z3")}
        parents.update(a_tied=np.array([-1, 2, 3, 0], dtype=np.int64), skipped=chain(3))
        rows = [row(name, p) for name, p in parents.items()]
        result, reads = self.select(rows, parents)
        self.assertEqual(result["census_certificate"]["length_search_parent_arrays_read"], 4)
        self.assertEqual(result["census_certificate"]["third_measured_main_tokens"], 27)
        self.assertEqual(result["census_certificate"]["next_max_bound"], 20)
        self.assertEqual([r["asset_id"] for r in result["groups"]["longest_main"]], ["a_tied", "z1", "z2"])
        self.assertEqual(len(result["longest_main_with_ties"]), 4)
        self.assertIn("a_tied", reads)
        self.assertNotIn("skipped", reads)

    def test_manifest_permutations_are_deterministic_except_row_indices(self):
        rows, parents = population()
        first, reads = self.select(rows, parents)
        rng = np.random.default_rng(20261013)
        for _ in range(12):
            shuffled = [rows[int(i)] for i in rng.permutation(len(rows))]
            result, permuted_reads = self.select(shuffled, parents)
            self.assertEqual(reads, permuted_reads)
            for group, entries in first["groups"].items():
                actual = result["groups"][group]
                self.assertEqual([r["asset_id"] for r in entries], [r["asset_id"] for r in actual])
                for entry in actual:
                    original = shuffled[entry["manifest_row_index"]]
                    self.assertEqual((original["asset_id"], original["path"]), (entry["asset_id"], entry["path"]))

    def test_random_valid_permuted_trees_match_exhaustive(self):
        rng = np.random.default_rng(20261013)
        for trial in range(12):
            rows, parents = [], {}
            for index in range(35):
                count = int(rng.integers(1, 81))
                old = [-1] + [int(rng.integers(i)) for i in range(1, count)]
                permutation = np.concatenate(([0], rng.permutation(np.arange(1, count))))
                inverse = np.argsort(permutation)
                p = np.array([-1 if old[i] < 0 else inverse[old[i]] for i in permutation], dtype=np.int64)
                name = f"random_{trial:02}_{index:02}"
                parents[name] = p
                rows.append(row(name, p))
                self.assertLessEqual(independent_length(p), capacity.main_token_bound(count))
            result, _ = self.select(rows, parents)
            expected = sorted(rows, key=lambda r: (-independent_length(parents[r["asset_id"]]), r["asset_id"]))[:3]
            self.assertEqual([r["asset_id"] for r in result["groups"]["longest_main"]], [r["asset_id"] for r in expected])

    def test_geometry_groups_and_selected_parent_payload_hashes(self):
        rows, parents = population()
        result, reads = self.select(rows, parents)
        for group, name in (("largest_vertices", "max_vertices"), ("largest_faces", "max_faces"),
                            ("largest_geometry", "max_geometry")):
            self.assertEqual([r["asset_id"] for r in result["groups"][group]], [name])
        cert = result["census_certificate"]
        self.assertEqual(cert["geometry_additional_parent_arrays_read"], 3)
        self.assertEqual(cert["total_unique_parent_arrays_read"], 7)
        self.assertEqual(len(reads), len(set(reads)))
        self.assertEqual(result["groups"]["largest_geometry"][0]["geometry_proxy"], 385128)
        selected_ids = {r["asset_id"] for group in result["groups"].values() for r in group}
        self.assertEqual(set(result["selected_parent_arrays"]), selected_ids)
        for name, item in result["selected_parent_arrays"].items():
            self.assertEqual(item["target_parents"], parents[name].tolist())
            digest = hashlib.sha256(parents[name].astype("<i8").tobytes()).hexdigest()
            self.assertEqual(item["target_parents_sha256"], digest)
        self.assertEqual(result["selected_parent_arrays_sha256"], capacity.json_digest(result["selected_parent_arrays"]))

    def test_geometry_maxima_switch_with_frames_and_share_parent_reads(self):
        parents = {"long60": star(60), "long59": star(59), "long58": star(58),
                   "vertices": chain(3), "faces": chain(3), "middle": chain(3), "unused": chain(2)}
        rows = [row(name, parents[name]) for name in ("long60", "long59", "long58", "unused")]
        rows.extend([row("vertices", parents["vertices"], 2000, 10),
                     row("faces", parents["faces"], 100, 2000),
                     row("middle", parents["middle"], 1500, 800)])
        result, reads = self.select(rows, parents)
        self.assertEqual(set(result["geometry_by_frames"]), {str(t) for t in range(2, 25)})
        for t in range(2, 25):
            item = result["geometry_by_frames"][str(t)]
            expected = "middle" if t <= 7 else "vertices"
            self.assertEqual(item["asset_id"], expected)
            self.assertEqual(item["geometry_proxy"], 12 * t * (2 * item["vertex_count"] + item["face_count"]) + 24 * item["face_count"])
            self.assertEqual(item["main_tokens"], independent_length(parents[expected]))
            self.assertIn("branch_count", item)
            self.assertIn(expected, result["selected_parent_arrays"])
        self.assertNotIn("middle", {r["asset_id"] for group in result["groups"].values() for r in group})
        self.assertEqual(reads.count("middle"), 1)
        self.assertNotIn("unused", reads)
        self.assertEqual(len(reads), len(set(reads)))
        self.assertEqual(result["census_certificate"]["length_search_parent_arrays_read"], 3)
        self.assertEqual(result["census_certificate"]["geometry_additional_parent_arrays_read"], 3)
        permuted, reversed_reads = self.select(list(reversed(rows)), parents)
        self.assertEqual(reads, reversed_reads)
        self.assertEqual([r["asset_id"] for r in result["geometry_by_frames"].values()],
                         [r["asset_id"] for r in permuted["geometry_by_frames"].values()])

    def test_geometry_ties_choose_one_identity_and_reuse_parent_reads(self):
        parents = {name: chain(4) for name in ("c", "a", "b")}
        result, reads = self.select([row(name, p) for name, p in parents.items()], parents)
        self.assertEqual(len(reads), 3)
        for group in ("largest_vertices", "largest_faces", "largest_geometry"):
            self.assertEqual(result["groups"][group][0]["asset_id"], "a")
            self.assertEqual(result["geometry_maximum_tie_counts"][group], 3)
        self.assertIsNone(result["census_certificate"]["next_max_bound"])

    def test_wrong_parents_rejected_with_real_aux_codec(self):
        invalid = [np.array([2**64 - 1, 0, 1], dtype=np.uint64), np.array([-1., 0., 1.]),
                   np.array([[-1, 0, 1]]), np.array([-1, 0]), np.array([1, -1, 1]),
                   np.array([-1, -1, 0]), np.array([-1, 2, 1]), np.array([-1, 3, 0]), np.array([-1, 1, 0])]
        metadata = capacity.row_metadata(row("invalid", chain(3)), 0, self.api["root_rt"].validate_descriptor)
        for parents in invalid:
            with self.subTest(parents=str(parents)), self.assertRaises(ValueError):
                capacity.measure_parents(metadata, parents, self.codec)

    def test_parent_hash_validation_and_signed_byte_canonicalization(self):
        r = row("hash", chain(3))
        correct = hashlib.sha256(chain(3).astype("<i8").tobytes()).hexdigest()
        r["target_parents_sha256"] = correct
        meta = capacity.row_metadata(r, 0, self.api["root_rt"].validate_descriptor)
        measured, _ = capacity.measure_parents(meta, chain(3).astype(">i4"), self.codec)
        self.assertEqual(measured["target_parents_sha256"], correct)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            capacity.measure_parents(meta, star(3), self.codec)
        r["canonical_metrics"]["target_parents_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            capacity.row_metadata(r, 0, self.api["root_rt"].validate_descriptor)

    def test_duplicate_ids_paths_missing_rootrt_rejected_before_io(self):
        rows, parents = population()
        changes = [(lambda data: data[1].update(asset_id=data[0]["asset_id"])),
                   (lambda data: data[1].update(path=data[0]["path"])),
                   (lambda data: data[-1].pop("root_rt")),
                   (lambda data: data[-1]["root_rt"].update(reference_index=1))]
        for change in changes:
            data = copy.deepcopy(rows)
            change(data)
            with self.assertRaises(ValueError):
                capacity.select_capacity(data, codec=self.codec, validate_descriptor=self.api["root_rt"].validate_descriptor,
                    parents_reader=lambda path: self.fail("input validation must precede asset IO"))

    def test_ambiguous_or_missing_counts_fail_even_on_unread_rows(self):
        rows, parents = population()
        edits = [lambda r: r["canonical_metrics"].update(target_joint_count=99),
                 lambda r: r["canonical_metrics"].update(filtered_vertex_count=99),
                 lambda r: r["canonical_metrics"].update(filtered_face_count=99),
                 lambda r: r["rootless_target_strict_metrics"].pop("face_count"),
                 lambda r: r["rootless_target_strict_metrics"].update(target_joint_count=True),
                 lambda r: r.update(canonical_metrics=None)]
        for edit in edits:
            data = copy.deepcopy(rows)
            edit(data[4])
            with self.assertRaises(ValueError):
                self.select(data, parents)

    def make_files(self, root):
        rows, parents = population()
        for r in rows:
            path = root / f"{r['asset_id']}.npz"
            r["path"] = str(path)
            # Object arrays would fail allow_pickle=False if the selector touched them.
            np.savez(path, target_parents=parents[r["asset_id"]], frame_vertices=np.array([object()], dtype=object),
                     target_joints=np.array([object()], dtype=object))
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        manifest_hash = capacity.sha256(manifest)
        config = {"data": {"global_rt_removal": True, "coordinate_clipping": False,
            "normalization": "input_sequence_mesh_bbox", "manifests": {"train": {"path": str(manifest),
            "rows": len(rows), "sha256": manifest_hash}}}, "auxiliary_codec": {"max_joints": 256, "max_sequence_tokens": 1024}}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        kwargs = {"expected_config_sha256": capacity.sha256(config_path),
                  "expected_manifest_sha256": manifest_hash, "expected_train_count": len(rows)}
        return rows, config, config_path, manifest, kwargs

    def test_prepare_reads_only_parents_and_matches_parent_runner_interface(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            rows, config, config_path, _, kwargs = self.make_files(Path(tmp))
            accesses = []
            original = np.lib.npyio.NpzFile.__getitem__
            def checked_getitem(archive, key):
                accesses.append(key)
                return original(archive, key)
            with patch.object(np.lib.npyio.NpzFile, "__getitem__", checked_getitem):
                report = capacity.prepare(SOURCE_ROOT, config_path, **kwargs)
            self.assertEqual(set(accesses), {"target_parents"})
            self.assertEqual(len(accesses), 7)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["manifest"], config["data"]["manifests"]["train"])
            self.assertEqual(report["config_sha256"], kwargs["expected_config_sha256"])
            self.assertEqual({name: len(items) for name, items in report["groups"].items()},
                {"longest_main": 3, "largest_vertices": 1, "largest_faces": 1, "largest_geometry": 1})
            self.assertFalse(report["capacity_approved"])
            self.assertFalse(report["main_length_contract"]["vendor_tokenizer_executed"])
            self.assertTrue(report["config_manifest_source_hashes_rechecked"])
            self.assertNotIn("torch", sys.modules)
            for group in report["groups"].values():
                for r in group:
                    self.assertEqual(rows[r["manifest_row_index"]]["asset_id"], r["asset_id"])
                    self.assertIn("main_tokens", r)
                    self.assertIn("branch_count", r)
            (Path(TEST_OUTPUT) / "synthetic_selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def test_config_manifest_hashes_and_production_pin_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            _, _, config_path, manifest, kwargs = self.make_files(Path(tmp))
            with self.assertRaisesRegex(ValueError, "config SHA256 mismatch"):
                capacity.prepare(SOURCE_ROOT, config_path, **{**kwargs, "expected_config_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "pinned"):
                capacity.prepare(SOURCE_ROOT, config_path, expected_config_sha256=kwargs["expected_config_sha256"])
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                capacity.prepare(SOURCE_ROOT, config_path, **kwargs)

    def test_mid_selection_config_mutation_detected(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            _, _, config_path, _, kwargs = self.make_files(Path(tmp))
            changed = False
            def reader(path):
                nonlocal changed
                if not changed:
                    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    changed = True
                return capacity.read_target_parents(path)
            with self.assertRaisesRegex(ValueError, "changed during selection"):
                capacity.prepare(SOURCE_ROOT, config_path, parents_reader=reader, **kwargs)

    def test_cli_success_failure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            root = Path(tmp)
            _, _, config_path, _, kwargs = self.make_files(root)
            output = root / "selection.json"
            args = ["--source-root", str(SOURCE_ROOT), "--config", str(config_path), "--config-sha256",
                    kwargs["expected_config_sha256"], "--output", str(output)]
            original = capacity.prepare
            def synthetic_prepare(source, config, *, expected_config_sha256):
                return original(source, config, **{**kwargs, "expected_config_sha256": expected_config_sha256})
            with patch.object(capacity, "prepare", side_effect=synthetic_prepare):
                self.assertEqual(capacity.main(args), 0)
                before = output.read_bytes()
                with self.assertRaises(SystemExit):
                    capacity.main(args)
                self.assertEqual(output.read_bytes(), before)
            failed = root / "failure.json"
            args[-1] = str(failed)
            args[5] = "0" * 64
            self.assertEqual(capacity.main(args), 1)
            self.assertEqual(json.loads(failed.read_text())["status"], "failed")


def main():
    global SOURCE_ROOT, TEST_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SOURCE_ROOT, TEST_OUTPUT = args.source_root.resolve(), args.output.resolve()
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CapacityTests))
    evidence = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
                "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
                "source_root": str(SOURCE_ROOT), "source_sha256": getattr(CapacityTests, "source_hashes", {}),
                "script_sha256": capacity.sha256(capacity.__file__), "tests_sha256": capacity.sha256(__file__),
                "real_assets_read": False, "vendor_tokenizer_executed": False, "gpu_used": False}
    (TEST_OUTPUT / "test_results.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
