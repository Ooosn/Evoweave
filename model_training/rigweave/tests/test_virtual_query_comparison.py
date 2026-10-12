"""Synthetic metadata and injected-parent CLI tests; no real assets or models."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_virtual_query_comparison as preparation
from virtual_query_comparison_selection import SelectionError, select_comparison, topology_key

SOURCE_ROOT = None
TEST_OUTPUT = None
CLI_EXAMPLE = None
JOINTS = {"low": 6, "mid": 28, "high": 70}


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def item(split, source, size, index, unseen=False):
    return {"asset_id": f"{split}_{source}_{size}_{index}",
            "path": f"/unopened/{split}/{source}_{size}_{index}.npz", "split": split,
            "integration_source": source, "target_joint_count": JOINTS[size],
            "joint_count_field": "canonical_metrics.target_joint_count",
            "topology_sha256": key(f"{'new' if unseen else 'chain'}_{size}")}


def population():
    train = [item("train", source, size, i) for source in ("base", "addition")
             for size, count in (("low", 10), ("mid", 18), ("high", 10)) for i in range(count)]
    census = {"addition": {"low": 1, "mid": 113, "high": 1}, "base": {"low": 20, "mid": 725, "high": 107}}
    valid = [item("valid", source, size, i, unseen=i % 3 == 0)
             for source, sizes in census.items() for size, count in sizes.items() for i in range(count)]
    return train, valid


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(SOURCE_ROOT or os.environ.get("EVOWEAVE_TEST_SOURCE_ROOT") or Path(__file__).parents[3]).resolve()
        cls.context = preparation.source_api(cls.source)
        cls.api, cls.source_hashes = cls.context.__enter__()
        cls.addClassCleanup(cls.context.__exit__, None, None, None)
        cls.codec = cls.api["codec"].AuxSkeletonTokens()

    def select(self, train=None, valid=None, seed=20261013):
        if train is None:
            train, valid = population()
        return select_comparison(train, valid, choose_fit_rows=self.api["fit"].choose_fit_rows, seed=seed)

    def test_exact_train_quotas_heldout_subset_and_valid_constraints(self):
        result = self.select()
        groups = result["groups"]
        self.assertEqual({k: len(v) for k, v in groups.items()}, {"train": 64, "train_heldout": 8, "valid": 32})
        cells = Counter((r["integration_source"], r["joint_count_stratum"]) for r in groups["train"])
        for source in ("base", "addition"):
            self.assertEqual([cells[(source, size)] for size in ("low", "mid", "high")], [8, 16, 8])
        train_ids = {r["asset_id"] for r in groups["train"]}
        self.assertLessEqual({r["asset_id"] for r in groups["train_heldout"]}, train_ids)
        self.assertFalse(train_ids & {r["asset_id"] for r in groups["valid"]})
        self.assertFalse({r["path"] for r in groups["train"]} & {r["path"] for r in groups["valid"]})
        self.assertEqual(len({r["asset_id"] for r in groups["valid"]}), 32)
        self.assertGreaterEqual(sum(r["joint_count_stratum"] == "low" for r in groups["valid"]), 10)
        self.assertGreaterEqual(sum(r["unseen_to_full_train"] for r in groups["valid"]), 8)
        self.assertGreaterEqual(sum(r["integration_source"] == "addition" for r in groups["valid"]), 8)
        self.assertEqual(len({(r["integration_source"], r["joint_count_stratum"]) for r in groups["valid"]}), 6)
        self.assertTrue(any(r["asset_id"] == "valid_addition_low_0" for r in groups["valid"]))
        self.assertTrue(any(r["asset_id"] == "valid_addition_high_0" for r in groups["valid"]))

    def test_determinism_reordering_and_no_input_mutation(self):
        train, valid = population()
        before = copy.deepcopy((train, valid))
        a, b = self.select(train, valid), self.select(train[::-1], valid[::-1])
        self.assertEqual(a, b)
        self.assertEqual((train, valid), before)
        self.assertNotEqual(a["groups"]["train"], self.select(train, valid, seed=7)["groups"]["train"])

    def test_fulltrain_unseen_is_not_unseen_to_selected64(self):
        train, valid = population()
        initial = self.select(train, valid)
        selected_ids = {r["asset_id"] for r in initial["groups"]["train"]}
        excluded = next(r for r in train if r["target_joint_count"] == 6 and r["asset_id"] not in selected_ids)
        excluded["topology_sha256"] = key("rare_fulltrain_low")
        edge = next(r for r in valid if r["asset_id"] == "valid_addition_low_0")
        edge["topology_sha256"] = excluded["topology_sha256"]
        result = self.select(train, valid)
        record = next(r for r in result["groups"]["valid"] if r["asset_id"] == edge["asset_id"])
        self.assertEqual(record["train_topology_frequency"], 1)
        self.assertEqual(record["selected_train_topology_frequency"], 0)
        self.assertFalse(record["unseen_to_full_train"])
        self.assertTrue(record["unseen_to_selected64"])

    def test_missing_strata_low_unseen_and_total_capacities_fail(self):
        train, valid = population()
        with self.assertRaisesRegex(SelectionError, "missing source/size cells: addition/high"):
            self.select(train, [r for r in valid if r["asset_id"] != "valid_addition_high_0"])
        with self.assertRaisesRegex(SelectionError, "fewer than 10 low") as error:
            self.select(train, [r for r in valid if r["target_joint_count"] != 6 or r["asset_id"] == "valid_addition_low_0"])
        self.assertEqual(error.exception.counts["valid"]["by_size"]["low"], 1)
        only_seven = copy.deepcopy(valid)
        for i, row in enumerate(only_seven):
            size = next(size for size, count in JOINTS.items() if count == row["target_joint_count"])
            row["topology_sha256"] = key(f"{'new' if i < 7 else 'chain'}_{size}")
        with self.assertRaisesRegex(SelectionError, "fewer than 8.*FULL train") as error:
            self.select(train, only_seven)
        self.assertEqual(error.exception.counts["valid"]["globally_unseen"], 7)
        with self.assertRaisesRegex(SelectionError, "fewer than 32"):
            self.select(train, valid[:20])
        with self.assertRaisesRegex(SelectionError, "train64 source/size quota"):
            self.select([r for r in train if not (r["integration_source"] == "base" and r["asset_id"].startswith("train_base_low_") and int(r["asset_id"].rsplit("_", 1)[1]) >= 7)], valid)

    def test_addition_minimum_reports_reduced_capacity_only_when_unavailable(self):
        train, valid = population()
        reduced = [r for r in valid if r["integration_source"] == "base" or not r["asset_id"].startswith("valid_addition_mid_") or r["asset_id"].endswith("_0")]
        result = self.select(train, reduced)
        self.assertEqual(result["policy"]["valid_addition_capacity_adjusted_target"], 3)
        self.assertEqual(result["counts"]["valid32"]["by_source"]["addition"], 3)

    def test_duplicate_and_cross_split_identity_errors(self):
        train, valid = population()
        for field in ("asset_id", "path"):
            duplicate = copy.deepcopy(train)
            duplicate[1][field] = duplicate[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "duplicate"):
                self.select(duplicate, valid)
            overlap = copy.deepcopy(valid)
            overlap[0][field] = train[0][field]
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.select(train, overlap)

    def test_actual_codec_parent_validation_and_byte_contract(self):
        for parents in ([-1, -1, 0], [-1, 2, 1], [-1, 0, 99]):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                self.codec.encode(np.asarray(parents, dtype=np.int64))
        tokens = self.codec.encode(np.asarray([-1, 0, 1], dtype=np.int64)).tokens
        self.assertEqual(tokens.tolist(), [1, 4, 5, 6, 2])
        expected = hashlib.sha256(struct.pack("<5q", 1, 4, 5, 6, 2)).hexdigest()
        self.assertEqual(topology_key(tokens), expected)
        self.assertEqual(topology_key(tokens.astype(">i8")), expected)

    def test_npz_reader_requests_only_numeric_target_parents(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
            path = Path(directory) / "numeric_parents_poison_mesh.npz"
            np.savez_compressed(path, target_parents=np.asarray([-1, 0, 1]),
                frame_vertices_rootspace=np.asarray([{"must_not_unpickle": True}], dtype=object))
            original_get = np.lib.npyio.NpzFile.__getitem__
            requested = []

            def guarded(archive, name):
                requested.append(name)
                self.assertEqual(name, "target_parents")
                return original_get(archive, name)

            with patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded), patch.object(preparation.np, "load", wraps=np.load) as opened:
                np.testing.assert_array_equal(preparation.read_target_parents(path), [-1, 0, 1])
            self.assertEqual(requested, ["target_parents"])
            self.assertIs(opened.call_args.kwargs["allow_pickle"], False)

    def cli_fixture(self, directory):
        root = Path(directory)
        train = [item("train", source, size, i) for source in ("base", "addition")
                 for size, count in (("low", 8), ("mid", 16), ("high", 8)) for i in range(count)]
        limits = {"addition": {"low": 1, "mid": 6, "high": 1}, "base": {"low": 9, "mid": 8, "high": 7}}
        valid = [item("valid", source, size, i, unseen=True) for source, sizes in limits.items()
                 for size, count in sizes.items() for i in range(count)]
        mapping, specs = {}, {}
        for split, rows in (("train", train), ("valid", valid)):
            manifest_rows = []
            for row in rows:
                count = row["target_joint_count"]
                path = str(root / "unopened_assets" / f"{row['asset_id']}.npz")
                mapping[path] = np.asarray([-1] + (list(range(count - 1)) if split == "train" else [0] * (count - 1)), dtype=np.int64)
                manifest_rows.append({"asset_id": row["asset_id"], "path": path, "split": split,
                    "integration_source": row["integration_source"], "canonical_metrics": {"target_joint_count": count},
                    "root_rt": {"version": "retained_root_rigid_v1", "normalization": "input_sequence_mesh_bbox",
                        "padding": 1.1, "reference_index": 0, "index": 0, "frame_count": 40, "cache_path": str(root / "unopened_rt.npz")}})
            manifest = root / f"{split}.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
            specs[split] = {"path": str(manifest), "rows": len(rows), "sha256": preparation.sha256(manifest)}
        config = {"probe": {"selection_seed": 20261013}, "auxiliary_codec": {"max_joints": 256, "max_sequence_tokens": 1024},
            "data": {"global_rt_removal": True, "coordinate_clipping": False, "manifests": specs}}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, config, mapping

    def test_cli_groups_provenance_actual_manifest_codec_and_no_torch_import(self):
        global CLI_EXAMPLE
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
            config_path, config, mapping = self.cli_fixture(directory)
            output = Path(directory) / "declaration.json"
            before = set(sys.modules)
            with patch.object(preparation, "EXPECTED_COUNTS", {"train": 64, "valid": 32}), \
                 patch.object(preparation, "read_target_parents", side_effect=lambda path: mapping[path].copy()) as reader:
                code = preparation.main(["--source-root", str(self.source), "--config", str(config_path), "--output", str(output)])
            self.assertEqual(code, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["manifests"], config["data"]["manifests"])
            self.assertEqual(result["config_sha256"], preparation.sha256(config_path))
            self.assertEqual({k: len(v) for k, v in result["groups"].items()}, {"train": 64, "train_heldout": 8, "valid": 32})
            required = {"asset_id", "path", "integration_source", "target_joint_count", "topology_sha256",
                        "train_topology_frequency", "selected_train_topology_frequency"}
            self.assertTrue(all(required <= r.keys() for rows in result["groups"].values() for r in rows))
            self.assertEqual(reader.call_count, 96)
            self.assertEqual({call.args[0] for call in reader.call_args_list}, set(mapping))
            self.assertFalse(any(name == "torch" or name.startswith(("torch.", "src.model")) for name in set(sys.modules) - before))
            self.assertTrue(result["config_manifest_source_hashes_rechecked"])
            CLI_EXAMPLE = result

    def test_cli_unseen_shortage_emits_counts_without_selected_groups(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
            config_path, _, mapping = self.cli_fixture(directory)
            valid_paths = [p for p in mapping if Path(p).name.startswith("valid_")]
            for path in valid_paths[7:]:
                count = len(mapping[path])
                mapping[path] = np.asarray([-1] + list(range(count - 1)), dtype=np.int64)
            output = Path(directory) / "failure.json"
            with patch.object(preparation, "EXPECTED_COUNTS", {"train": 64, "valid": 32}), \
                 patch.object(preparation, "read_target_parents", side_effect=lambda path: mapping[path]):
                code = preparation.main(["--source-root", str(self.source), "--config", str(config_path), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(report["counts"]["valid"]["globally_unseen"], 7)
            self.assertEqual(report["groups"], {})
            self.assertEqual(report["config_sha256"], preparation.sha256(config_path))
            self.assertEqual(set(report["manifests"]), {"train", "valid"})
            self.assertTrue(report["source_sha256"])

    def test_cli_malformed_parents_and_wrong_manifest_hash_fail(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
            config_path, config, mapping = self.cli_fixture(directory)
            first = next(iter(mapping))
            mapping[first][2] = 2
            with patch.object(preparation, "EXPECTED_COUNTS", {"train": 64, "valid": 32}), \
                 patch.object(preparation, "read_target_parents", side_effect=lambda path: mapping[path]):
                with self.assertRaisesRegex(ValueError, "train row 0"):
                    preparation.prepare(self.source, config_path)
            config["data"]["manifests"]["train"]["sha256"] = "0" * 64
            config_path.write_text(json.dumps(config))
            with patch.object(preparation, "EXPECTED_COUNTS", {"train": 64, "valid": 32}), \
                 patch.object(preparation, "read_target_parents") as reader:
                with self.assertRaisesRegex(ValueError, "manifest hash differs"):
                    preparation.prepare(self.source, config_path)
                reader.assert_not_called()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SOURCE_ROOT, TEST_OUTPUT = args.source_root, args.output.resolve()
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests))
    result = {"status": "passed" if outcome.wasSuccessful() else "failed", "tests": outcome.testsRun,
        "failures": [(str(case), message) for case, message in outcome.errors + outcome.failures],
        "scope": "Synthetic metadata; actual source manifest function, descriptor validator, aux codec and fit chooser. CLI mock uses 64/32 row counts and injected parents; no real data scan.",
        "files_sha256": {str(path): preparation.sha256(path) for path in (Path(__file__),
            SCRIPTS / "prepare_virtual_query_comparison.py", SCRIPTS / "virtual_query_comparison_selection.py")}}
    (TEST_OUTPUT / "test_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if CLI_EXAMPLE:
        (TEST_OUTPUT / "synthetic_cli_example.json").write_text(json.dumps(CLI_EXAMPLE, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
