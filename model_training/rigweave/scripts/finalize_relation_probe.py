"""Combine unchanged motion cases with the verified exact-static correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_relation_probe import paired_measurement
from motion_experiment_runtime import write_json
from relation_probe_metrics import file_hash
from run_relation_probe import summarize


def finalize(anchor, static):
    if not anchor["complete"] or not static["complete"]:
        raise ValueError("both recorded stages must be complete")
    if static["nonstatic_cache_cases_proved_not_exactly_static"] != 32:
        raise ValueError("cannot reuse unchanged motion cases")
    if static["changed_base_parameters"] or static["changed_adapter_parameters"]:
        raise ValueError("read-only audit changed model parameters")
    if len(static["rows"]) != 16 or not all(row["evidence_exact_unknown"]
        and row["base_condition_vs_previous"]["max_abs"] == 0 for row in static["rows"]):
        raise ValueError("old static baseline cannot be reused")
    base = anchor["baseline"]["valid"]
    actual = [row for row in anchor["evaluation"]["actual"] if row["view"] != "static8"] + static["static_rows"]
    if len(actual) != 48 or any(row["condition_vs_corrected_base"]["max_abs"] != 0 for row in static["static_rows"]):
        raise ValueError("corrected anchored static identity failed")
    metrics = ("ce", "f1", "quiet_strict:joint_coverage_005", "quiet_strict:edge_recall",
               "hidden_demonstrated_strict:joint_coverage_005", "hidden_demonstrated_strict:edge_recall")
    paired, transitions = {}, {}
    for view in ("all", "normal8", "weak2", "static8"):
        aa = [row for row in base if view == "all" or row["view"] == view]
        bb = [row for row in actual if view == "all" or row["view"] == view]
        paired[view] = {metric: paired_measurement(aa, bb, metric) for metric in metrics}
        amap = {(row["asset_id"], row["view"]): row for row in aa}
        matches = [(amap[(row["asset_id"], row["view"])], row) for row in bb]
        good = [(a, b) for a, b in matches if a["generation"]["success"] and b["generation"]["success"]]
        j2j_a = [{**a, "ce": a["generation"]["metrics"]["official"]["j2j"]} for a, _ in good]
        j2j_b = [{**b, "ce": b["generation"]["metrics"]["official"]["j2j"]} for _, b in good]
        transitions[view] = {
            "matched_success_cases": len(good),
            "j2j_matched_success": paired_measurement(j2j_a, j2j_b, "ce"),
            "recovered": [{"index": b["index"], "view": b["view"]} for a, b in matches
                          if not a["generation"]["success"] and b["generation"]["success"]],
            "regressed": [{"index": b["index"], "view": b["view"]} for a, b in matches
                          if a["generation"]["success"] and not b["generation"]["success"]],
            "identical_generated_tokens": sum(a["generation"]["generated_ids"] == b["generation"]["generated_ids"]
                                               for a, b in matches),
            "successful_predictions_with_cycle": sum(bool(b["generation"].get("tree", {}).get("nodes_reaching_cycle", 0))
                                                      for _, b in matches if b["generation"]["success"])}
    nonstatic = [row for row in actual if row["view"] != "static8"]
    permuted = [row for row in anchor["ce_controls"]["permuted"] if row["view"] != "static8"]
    return {"summary": {"base": summarize(base), "anchored_exact_static": summarize(actual)},
            "paired": paired, "generation_transitions": transitions,
            "permuted_minus_actual_nonstatic_ce": paired_measurement(nonstatic, permuted, "ce"),
            "scope": "Fixed cached32 motion cases plus16 actual-data rebuilt static cases. No further training or fresh validation assets. Static generation reused only after exact old-base/corrected-base/trained-anchor condition identity; two greedy prefixes also checked. J2J conditions on paired successful cases; all-row F1 separately includes failed generations as zero. Exploratory asset-cluster intervals, not multiplicity corrected."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(json.loads(args.anchor.read_text()), json.loads(args.static.read_text()))
    result["sources"] = [{"path": str(path), "sha256": file_hash(path)} for path in (args.anchor, args.static)]
    write_json(args.output, result)


if __name__ == "__main__":
    main()
