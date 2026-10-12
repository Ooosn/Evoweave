"""Focused CPU-only tests, with parity against the actual read-only probe source."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
HELPER = Path(__file__).parents[1] / "scripts/fit_cohort_selection.py"
spec = importlib.util.spec_from_file_location("fit_cohort_selection", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
choose_fit_rows = module.choose_fit_rows
SOURCE_ROOT = None
LEGACY = None
LEGACY_SOURCE = None
LEGACY_SOURCE_SHA256 = None


def row(asset_id, source, count, *, fallback=False):
    group = "rootless_target_strict_metrics" if fallback else "canonical_metrics"
    return {"asset_id": asset_id, "path": f"/unopened/train/{asset_id}.npz",
            "integration_source": source, group: {"target_joint_count": count}, "split": "train", "source_split": "train"}


def config(mode=None, assets=1, seed=20261013, frames=2):
    probe = {"assets": assets, "selection_seed": seed, "frames": frames}
    if mode is not None:
        probe["cohort_mode"] = mode
    return {"probe": probe}


def diverse_config(seed=20261013):
    return config("diverse8", 8, seed, 4)


def population(non_base="addition", per_cell=4):
    return [row(f"{source}_{size}_{i}", source, count, fallback=bool(i % 2))
            for source in ("base", non_base)
            for size, count in (("low", 6), ("mid", 28), ("high", 90))
            for i in range(per_cell)]


class CohortSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global LEGACY, LEGACY_SOURCE, LEGACY_SOURCE_SHA256
        if LEGACY is None:
            configured = SOURCE_ROOT or os.environ.get("EVOWEAVE_TEST_SOURCE_ROOT")
            source_root = Path(configured) if configured else Path(__file__).parents[3]
            LEGACY_SOURCE = source_root.resolve() / "model_training/rigweave/scripts/probe_virtual_query_xyz.py"
            source = LEGACY_SOURCE.read_bytes()
            LEGACY_SOURCE_SHA256 = hashlib.sha256(source).hexdigest()
            function = next(node for node in ast.parse(source, filename=str(LEGACY_SOURCE)).body
                            if isinstance(node, ast.FunctionDef) and node.name == "choose_rows")
            namespace = {"np": np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                         str(LEGACY_SOURCE), "exec"), namespace)
            LEGACY = namespace["choose_rows"]

    def test_default_parity_with_actual_legacy_for_seeds_orders_and_sizes(self):
        rows = [row(f"{source}_{count}", source, count, fallback=bool(count % 2))
                for source in ("base", "addition") for count in (4, 10, 11, 19, 20, 21, 28, 42, 60, 61, 100)]
        self.assertIsNotNone(LEGACY)
        for order in (rows, list(reversed(rows))):
            for seed in (0, 1, 17, 20261013):
                for assets in (1, 2, 5):
                    with self.subTest(seed=seed, assets=assets, reverse=order is not rows):
                        cfg = config(assets=assets, seed=seed)
                        actual = choose_fit_rows(order, cfg)
                        expected = LEGACY(order, cfg)
                        self.assertEqual(actual, expected)
                        self.assertTrue(all(a is b for a, b in zip(actual, expected)))
                        self.assertEqual(actual, choose_fit_rows(order, config("base20_60", assets, seed)))

    def test_default_fallback_and_canonical_precedence(self):
        rows = [row("canonical_wins", "base", 28), row("fallback", "base", 30, fallback=True),
                {"asset_id": "missing_count", "path": "/unopened/missing.npz", "integration_source": "base"}]
        rows[0]["rootless_target_strict_metrics"] = {"target_joint_count": 90}
        selected, report = choose_fit_rows(rows, config(assets=2), return_report=True)
        self.assertEqual(selected, LEGACY(rows, config(assets=2)))
        fields = {item["asset_id"]: item["joint_count_field"] for item in report["selected"]}
        self.assertEqual(fields["canonical_wins"], "canonical_metrics.target_joint_count")
        self.assertEqual(fields["fallback"], "rootless_target_strict_metrics.target_joint_count")

    def test_diverse_covers_six_cells_and_balances_two_extras(self):
        selected, report = choose_fit_rows(population(), diverse_config(), return_report=True)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({r["asset_id"] for r in selected}), 8)
        self.assertEqual(len({r["path"] for r in selected}), 8)
        cells = {(r["integration_source"], r["joint_count_stratum"]) for r in report["selected"][:6]}
        self.assertEqual(cells, {(source, size) for source in ("base", "addition") for size in ("low", "mid", "high")})
        self.assertEqual(report["selected_source_counts"], {"base": 4, "addition": 4})
        self.assertEqual(sorted(report["selected_size_counts"].values()), [2, 3, 3])
        self.assertEqual([r["selection_phase"] for r in report["selected"]].count("stratified_extra"), 2)
        json.dumps(report, allow_nan=False)

    def test_diverse_discovers_nonbase_value_not_a_hardcoded_name(self):
        _, report = choose_fit_rows(population("observed_nonbase_value"), diverse_config(), return_report=True)
        self.assertEqual(report["source_cohorts"], ["base", "observed_nonbase_value"])

    def test_parent_reported_population_counts_with_synthetic_ids(self):
        counts = {"addition": {"low": 36, "mid": 2082, "high": 53},
                  "base": {"low": 380, "mid": 13412, "high": 2039}}
        joints = {"low": 4, "mid": 28, "high": 61}
        rows = [row(f"{source}_{size}_{i}", source, joints[size])
                for source, sizes in counts.items() for size, count in sizes.items() for i in range(count)]
        selected, report = choose_fit_rows(rows, diverse_config(), return_report=True)
        self.assertEqual(len(rows), 18002)
        self.assertEqual(len(selected), 8)
        self.assertEqual(report["frames_per_asset"], 4)
        self.assertEqual(report["selected_source_counts"], {"base": 4, "addition": 4})
        for item in report["candidate_strata"]:
            self.assertEqual(item["count"], counts[item["integration_source"]][item["joint_count_stratum"]])

    def test_diverse_deterministic_and_independent_of_input_order(self):
        rows = population()
        first = choose_fit_rows(rows, diverse_config(), return_report=True)
        self.assertEqual(first, choose_fit_rows(list(reversed(rows)), diverse_config(), return_report=True))
        self.assertEqual(first, choose_fit_rows(rows, diverse_config(), return_report=True))
        ids = lambda seed: [r["asset_id"] for r in choose_fit_rows(rows, diverse_config(seed))]
        self.assertNotEqual(ids(1), ids(2))

    def test_exact_size_boundaries_and_below_four_exclusion(self):
        rows = [row(f"{source}_{count}", source, count) for source in ("base", "addition") for count in (4, 10, 11, 60, 61)]
        _, report = choose_fit_rows(rows, diverse_config(), return_report=True)
        sizes = {(r["integration_source"], r["joint_count_stratum"]): r["count"] for r in report["candidate_strata"]}
        for source in ("base", "addition"):
            self.assertEqual([sizes[(source, size)] for size in ("low", "mid", "high")], [2, 2, 1])
        rows.append(row("under_four", "base", 3))
        _, report = choose_fit_rows(rows, diverse_config(), return_report=True)
        self.assertEqual(report["excluded_below_four_joints"], 1)
        self.assertNotIn("under_four", {r["asset_id"] for r in report["selected"]})

    def test_extras_use_available_strata_without_silent_shrinking(self):
        rows = population(per_cell=1) + [row("extra_1", "addition", 5), row("extra_2", "addition", 61)]
        selected, report = choose_fit_rows(rows, diverse_config(), return_report=True)
        self.assertEqual(len(selected), 8)
        self.assertEqual(report["selected_source_counts"], {"base": 3, "addition": 5})

    def test_missing_strata_and_insufficient_total_fail_clearly(self):
        rows = [r for r in population() if not r["asset_id"].startswith("addition_high_")]
        with self.assertRaisesRegex(ValueError, "missing required strata: addition/high"):
            choose_fit_rows(rows, diverse_config())
        with self.assertRaisesRegex(ValueError, "needs 8 distinct eligible assets; found 6"):
            choose_fit_rows(population(per_cell=1), diverse_config())

    def test_missing_or_ambiguous_sources_fail(self):
        for rows in ([], [r for r in population() if r["integration_source"] == "base"],
                     [r for r in population() if r["integration_source"] == "addition"],
                     population() + [row("third", "third_source", 6)]):
            with self.subTest(sources={r["integration_source"] for r in rows}), self.assertRaisesRegex(ValueError, "exactly one discovered non-base"):
                choose_fit_rows(rows, diverse_config())

    def test_duplicate_id_and_path_fail_for_both_modes(self):
        for cfg in (config(), diverse_config()):
            for field in ("asset_id", "path"):
                rows = population()
                rows.append(copy.deepcopy(rows[4]))
                rows[-1]["asset_id"] = "unique_duplicate_test"
                rows[-1]["path"] = "/unopened/unique_duplicate_test.npz"
                rows[-1][field] = rows[4][field]
                with self.subTest(mode=cfg["probe"].get("cohort_mode"), field=field), self.assertRaisesRegex(ValueError, f"Duplicate candidate {field}"):
                    choose_fit_rows(rows, cfg)

    def test_empty_and_insufficient_default_match_legacy_errors(self):
        for rows, cfg, message in (([], config(), "fixed fit cohort is empty"),
                                   ([row("only", "base", 28)], config(assets=2), "not enough fixed assets")):
            for function in (choose_fit_rows, LEGACY):
                with self.assertRaisesRegex(ValueError, message):
                    function(rows, cfg)

    def test_bad_counts_sources_and_candidate_identity_fail(self):
        for value in (None, "28", 28.0, True, -1):
            rows = population()
            rows[0]["canonical_metrics"]["target_joint_count"] = value
            with self.subTest(count=value), self.assertRaises(ValueError):
                choose_fit_rows(rows, diverse_config())
        for field, value in (("integration_source", None), ("integration_source", ""),
                             ("asset_id", ""), ("path", None)):
            rows = population()
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                choose_fit_rows(rows, diverse_config())
        rows = population()
        del rows[0]["canonical_metrics"]
        with self.assertRaisesRegex(ValueError, "Missing positive target_joint_count"):
            choose_fit_rows(rows, diverse_config())

    def test_config_contract_errors(self):
        for cfg in (config("unknown"), config(seed=True), config(seed=-1), config(assets=0),
                    config("diverse8", assets=7, frames=4), config("diverse8", assets=8, frames=2),
                    config("diverse8", assets=8, frames=4.0)):
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                choose_fit_rows(population(), cfg)

    def test_training_only_and_no_file_io(self):
        for field in ("split", "source_split"):
            rows = population()
            rows[0][field] = "valid"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Training rows only"):
                choose_fit_rows(rows, diverse_config())
        with self.assertRaisesRegex(ValueError, "not a mapping of train/valid"):
            choose_fit_rows({"train": population(), "valid": []}, diverse_config())
        with patch("builtins.open", side_effect=AssertionError("No asset/file reads allowed")), \
             patch.object(Path, "open", side_effect=AssertionError("No path reads allowed")):
            self.assertEqual(len(choose_fit_rows(population(), diverse_config())), 8)

    def test_inputs_unchanged_and_reports_do_not_advance_selection_rng(self):
        for cfg in (config(), diverse_config()):
            rows = population()
            snapshot = copy.deepcopy(rows)
            chosen = choose_fit_rows(rows, cfg)
            with_report, _ = choose_fit_rows(rows, cfg, return_report=True)
            self.assertEqual(chosen, with_report)
            self.assertEqual(rows, snapshot)
            self.assertTrue(all(any(value is original for original in rows) for value in chosen))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        help="Oracle checkout; otherwise EVOWEAVE_TEST_SOURCE_ROOT or this file's repo root")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CohortSelectionTests))
    probe, digest = LEGACY_SOURCE, LEGACY_SOURCE_SHA256
    unchanged = probe is not None and digest is not None and hashlib.sha256(probe.read_bytes()).hexdigest() == digest
    _, example = choose_fit_rows(population(), diverse_config(), return_report=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "example_selection_report.json").write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    report = {"status": "passed" if outcome.wasSuccessful() and unchanged else "failed", "tests": outcome.testsRun,
        "failures": [(str(case), text) for case, text in outcome.failures + outcome.errors],
        "legacy_source": str(probe), "legacy_source_sha256": digest, "legacy_source_unchanged": unchanged,
        "helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(), "numpy_version": np.__version__,
        "scope": "Synthetic train metadata; legacy oracle extracted from current actual choose_rows, not reimplemented",
        "real_accepted_manifest_reselection": False}
    (args.output / "test_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)
