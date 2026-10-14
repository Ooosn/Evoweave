"""CPU fixture tests for the bounded normal5000 preparation contract."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_virtual_fusion_normal as normal
import prepare_virtual_query_capacity as capacity


SOURCE_ROOT: Path | None = None
TEST_OUTPUT: Path | None = None


def chain(count: int) -> np.ndarray:
    return np.arange(-1, count - 1, dtype=np.int64)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_row(root: Path, split: str, source: str, index: int, joints: int, frames: int = 32) -> dict:
    asset_id = f"{split}_{source}_{index:02d}"
    path = root / "assets" / f"{asset_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if split == "train":
        np.savez(path, target_parents=chain(joints),
                 frame_vertices_rootspace=np.zeros((frames, 2, 3), dtype=np.float32))
    return {
        "asset_id": asset_id,
        "path": str(path),
        "split": split,
        "integration_source": source,
        "keep_original_payload": {"line": index, "root_rt_must_survive": True},
        "canonical_metrics": {
            "target_joint_count": joints, "filtered_vertex_count": 100 + index,
            "filtered_face_count": 200 + index,
        },
        "rootless_target_strict_metrics": {
            "target_joint_count": joints, "vertex_count": 100 + index, "face_count": 200 + index,
        },
        "root_rt": {
            "version": "retained_root_rigid_v1", "cache_path": str(root / "rt_cache.npz"),
            "index": index, "frame_count": frames, "reference_index": 0,
            "normalization": "input_sequence_mesh_bbox", "padding": 1.1,
        },
    }


@contextmanager
def fixture_files():
    assert TEST_OUTPUT is not None
    with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as directory:
        root = Path(directory)
        train_specs = [
            ("base", 0, 4, 32), ("base", 1, 8, 36), ("base", 2, 11, 40),
            ("base", 3, 20, 48), ("base", 4, 60, 32), ("base", 5, 61, 36),
            ("base", 6, 80, 40), ("base", 7, 100, 48),
            ("addition", 0, 4, 32), ("addition", 1, 12, 36),
            ("addition", 2, 70, 40), ("addition", 3, 90, 48),
        ]
        valid_specs = [("base", 0, 5, 32), ("base", 1, 15, 32),
                       ("addition", 0, 25, 32), ("addition", 1, 65, 32)]
        train_rows = [make_row(root, "train", source, index, joints, frames)
                      for source, index, joints, frames in train_specs]
        valid_rows = [make_row(root, "valid", source, index, joints, frames)
                      for source, index, joints, frames in valid_specs]
        # The valid fixture needs no NPZ bodies; only its immutable JSONL bytes
        # and identities are inspected by the preparation code.
        for row in valid_rows:
            Path(row["path"]).unlink(missing_ok=True)
        train_manifest = root / "train.jsonl"
        valid_manifest = root / "valid.jsonl"
        train_lines = []
        for index, row in enumerate(train_rows):
            encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":"))
            train_lines.append((encoded + ("\r\n" if index % 2 == 0 else "\n")).encode("utf-8"))
        valid_lines = [(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
                       for row in valid_rows]
        train_manifest.write_bytes(b"".join(train_lines))
        valid_manifest.write_bytes(b"".join(valid_lines))
        config = {
            "auxiliary_codec": {"max_joints": 256, "max_sequence_tokens": 1024},
            "data": {
                "global_rt_removal": True, "coordinate_clipping": False,
                "normalization": "input_sequence_mesh_bbox", "heldout_frames": 8,
                "manifests": {
                    "train": {"path": str(train_manifest), "rows": len(train_rows), "sha256": digest(train_manifest)},
                    "valid": {"path": str(valid_manifest), "rows": len(valid_rows), "sha256": digest(valid_manifest)},
                },
            },
        }
        config_path = root / "config.json"
        config_path.write_bytes((json.dumps(config, indent=2) + "\n").encode("utf-8"))
        kwargs = {
            "expected_counts": {"train": len(train_rows), "valid": len(valid_rows)},
            "expected_manifest_sha256": {
                "train": digest(train_manifest), "valid": digest(valid_manifest),
            },
            "expected_source_counts": {"base": 8, "addition": 4},
            "target_count": 8,
            "seed": normal.SELECTION_SEED,
        }
        yield root, config_path, train_manifest, valid_manifest, train_rows, valid_rows, kwargs


class NormalSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert SOURCE_ROOT is not None
        cls.source_root = SOURCE_ROOT

    def run_prepare(self, config_path: Path, output: Path, kwargs: dict, **extra):
        return normal.prepare(self.source_root, config_path, output_dir=output, **kwargs, **extra)

    def test_production_quotas_and_fixture_bin_coverage_are_exact(self):
        quotas = normal.proportional_quotas(normal.SOURCE_COUNTS, 5000, order=list(normal.SOURCE_ORDER))
        self.assertEqual(quotas, {"base": 4397, "addition": 603})
        with fixture_files() as (_, config, train_manifest, _, _, _, kwargs):
            records = normal.read_manifest_bytes(
                train_manifest, expected_count=12, expected_sha256=digest(train_manifest)
            )
            selected, report = normal.select_train_records(
                records, target_count=8, seed=normal.SELECTION_SEED, source_counts={"base": 8, "addition": 4}
            )
            self.assertEqual(len(selected), 8)
            self.assertEqual(report["source_quotas"], {"base": 5, "addition": 3})
            self.assertTrue(all(report["selected_bin_counts"][f"{source}/{name}"] >= 1
                                for source in normal.SOURCE_ORDER for name in ("low", "mid", "high")))
            self.assertEqual(kwargs["expected_counts"], {"train": 12, "valid": 4})

    def test_selection_is_deterministic_without_replacement_and_ordered(self):
        with fixture_files() as (_, _, train_manifest, _, _, _, _):
            records = normal.read_manifest_bytes(
                train_manifest, expected_count=12, expected_sha256=digest(train_manifest)
            )
            first, first_report = normal.select_train_records(
                records, target_count=8, seed=normal.SELECTION_SEED, source_counts={"base": 8, "addition": 4}
            )
            second, second_report = normal.select_train_records(
                records, target_count=8, seed=normal.SELECTION_SEED, source_counts={"base": 8, "addition": 4}
            )
            self.assertEqual([r["manifest_row_index"] for r in first], [r["manifest_row_index"] for r in second])
            self.assertEqual(first_report, second_report)
            indices = [r["manifest_row_index"] for r in first]
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(len(indices), len(set(indices)))

    def test_success_preserves_jsonl_line_bytes_root_rt_and_safe_npz_reads(self):
        with fixture_files() as (_, config, train_manifest, _, _, _, kwargs):
            output = Path(config).parent / "normal5000"
            accessed = []
            original = np.lib.npyio.NpzFile.__getitem__

            def guarded(archive, name):
                accessed.append(name)
                if name != "target_parents":
                    raise AssertionError(f"unexpected NPZ body read: {name}")
                return original(archive, name)

            with patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded):
                summary = self.run_prepare(config, output, kwargs)
            records = normal.read_manifest_bytes(train_manifest, expected_count=12, expected_sha256=digest(train_manifest))
            selected, _ = normal.select_train_records(
                records, target_count=8, seed=normal.SELECTION_SEED, source_counts={"base": 8, "addition": 4}
            )
            expected_bytes = b"".join(record["raw"] for record in selected)
            self.assertEqual((output / normal.DERIVED_MANIFEST_NAME).read_bytes(), expected_bytes)
            self.assertEqual(summary["rows_unchanged"]["derived_manifest_sha256_matches_selected_source_lines"], True)
            self.assertTrue(summary["rows_unchanged"]["root_rt_preserved_for_selected_rows"])
            self.assertEqual(set(accessed), {"target_parents"})
            self.assertEqual(summary["actual_frame_count"]["minimum_actual_frame_count"], 32)
            self.assertIn("32", summary["actual_frame_count"]["histogram"])
            self.assertFalse(summary["execution_scope"]["vertex_array_bodies_read"])

    def test_actual_codec_length_parity_and_capacity_schema_index_identity(self):
        with fixture_files() as (_, config, _, _, _, _, kwargs):
            output = Path(config).parent / "normal5000"
            summary = self.run_prepare(config, output, kwargs)
            capacity_report = json.loads((output / normal.CAPACITY_NAME).read_text(encoding="utf-8"))
            subset_rows = [json.loads(line) for line in (output / normal.DERIVED_MANIFEST_NAME).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(capacity_report["manifest"], summary["derived_manifest"])
            required = {"longest_main", "largest_vertices", "largest_faces", "largest_geometry"}
            self.assertTrue(required <= set(capacity_report["groups"]))
            self.assertEqual(len(capacity_report["groups"]["largest_auxiliary"]), 1)
            self.assertEqual(len(capacity_report["groups"]["longest_main"]), 3)
            self.assertEqual(set(capacity_report["geometry_by_frames"]), {str(frames) for frames in range(2, 25)})
            for group in capacity_report["groups"].values():
                for value in group:
                    index = value["manifest_row_index"]
                    self.assertEqual((value["asset_id"], value["path"]),
                                     (subset_rows[index]["asset_id"], subset_rows[index]["path"]))
                    self.assertEqual(value["auxiliary_tokens"], value["auxiliary_token_count"])
            for value in capacity_report["geometry_by_frames"].values():
                self.assertEqual(value["auxiliary_tokens"], value["auxiliary_token_count"])
            context = capacity.source_api(self.source_root)
            api, _ = context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)
            codec = api["codec"].AuxSkeletonTokens(max_joints=256, max_sequence_tokens=1024)
            by_asset = {row["asset_id"]: row for row in subset_rows}
            for item in summary["selected_rows"]:
                metadata = capacity.row_metadata(by_asset[item["asset_id"]], item["subset_manifest_row_index"],
                                                 api["root_rt"].validate_descriptor)
                measured, _ = capacity.measure_parents(metadata, capacity.read_target_parents(item["path"]), codec)
                self.assertEqual(item["original_xyz_token_length"], measured["main_tokens"])
                self.assertEqual(item["branch_count"], measured["branch_count"])
                self.assertEqual(item["auxiliary_tokens"], measured["auxiliary_token_count"])
            self.assertFalse(capacity_report["capacity_approved"])
            self.assertFalse(capacity_report["gpu_capacity_passed"])
            self.assertEqual(len(capacity_report["max_joint_cases"]), 3)
            largest_auxiliary = capacity_report["groups"]["largest_auxiliary"][0]
            self.assertEqual(
                largest_auxiliary["auxiliary_tokens"],
                max(item["auxiliary_tokens"] for item in summary["selected_rows"]),
            )

    def test_train_valid_overlap_fails_before_parent_or_header_io(self):
        with fixture_files() as (root, config, train_manifest, valid_manifest, train_rows, valid_rows, kwargs):
            overlap = copy.deepcopy(valid_rows[0])
            overlap["asset_id"] = train_rows[0]["asset_id"]
            overlap["path"] = train_rows[0]["path"]
            valid_lines = [json.dumps(overlap) + "\n"] + [json.dumps(row) + "\n" for row in valid_rows[1:]]
            valid_manifest.write_text("".join(valid_lines), encoding="utf-8")
            config_payload = json.loads(config.read_text(encoding="utf-8"))
            config_payload["data"]["manifests"]["valid"]["sha256"] = digest(valid_manifest)
            config.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
            kwargs = dict(kwargs)
            kwargs["expected_manifest_sha256"] = {
                **kwargs["expected_manifest_sha256"], "valid": digest(valid_manifest),
            }
            reads = []

            def should_not_read(path):
                reads.append(path)
                raise AssertionError("asset I/O occurred before train/valid overlap validation")

            with self.assertRaisesRegex(ValueError, "overlap"):
                normal.prepare(self.source_root, config, parents_reader=should_not_read,
                               header_reader=should_not_read, **kwargs)
            self.assertEqual(reads, [])

    def test_insufficient_and_malformed_headers_fail_without_resampling(self):
        with fixture_files() as (root, config, train_manifest, _, _, _, kwargs):
            records = normal.read_manifest_bytes(train_manifest, expected_count=12, expected_sha256=digest(train_manifest))
            selected, _ = normal.select_train_records(
                records, target_count=8, seed=normal.SELECTION_SEED, source_counts={"base": 8, "addition": 4}
            )
            bad_path = Path(selected[0]["row"]["path"])
            np.savez(bad_path, target_parents=chain(selected[0]["joint_count"]),
                     frame_vertices_rootspace=np.zeros((31, 2, 3), dtype=np.float32))
            reads = []

            def parent_reader(path):
                reads.append(path)
                return capacity.read_target_parents(path)

            with self.assertRaisesRegex(ValueError, "requires at least 24"):
                normal.prepare(self.source_root, config, parents_reader=parent_reader,
                               target_count=8, expected_counts=kwargs["expected_counts"],
                               expected_manifest_sha256=kwargs["expected_manifest_sha256"],
                               expected_source_counts=kwargs["expected_source_counts"])
            self.assertEqual(reads, [])
            self.assertEqual([r["manifest_row_index"] for r in selected],
                             [r["manifest_row_index"] for r in normal.select_train_records(
                                 records, target_count=8, seed=normal.SELECTION_SEED,
                                 source_counts={"base": 8, "addition": 4})[0]])
            malformed = root / "malformed.npz"
            np.savez(malformed, target_parents=chain(3))
            with self.assertRaisesRegex(ValueError, "expected one NPZ member"):
                normal.read_npz_array_header(malformed)

    def test_output_directory_is_exclusive_and_never_overwritten(self):
        with fixture_files() as (_, config, _, _, _, _, kwargs):
            output = Path(config).parent / "normal5000"
            self.run_prepare(config, output, kwargs)
            before = {path.relative_to(output): path.read_bytes()
                      for path in output.rglob("*") if path.is_file()}
            with self.assertRaises(FileExistsError):
                self.run_prepare(config, output, kwargs)
            after = {path.relative_to(output): path.read_bytes()
                     for path in output.rglob("*") if path.is_file()}
            self.assertEqual(before, after)


def main() -> int:
    global SOURCE_ROOT, TEST_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    if args.output is None:
        TEST_OUTPUT = Path(tempfile.mkdtemp(prefix="normal5000_selection_tests_"))
    else:
        TEST_OUTPUT = args.output.resolve()
        TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NormalSelectionTests)
    )
    evidence = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "source_root": str(SOURCE_ROOT),
        "source_sha256": {},
        "script_sha256": normal.sha256(normal.__file__),
        "tests_sha256": digest(Path(__file__)),
        "real_assets_read": False,
        "server_or_gpu_used": False,
        "fixture_npz_only": True,
    }
    (TEST_OUTPUT / "test_results.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
