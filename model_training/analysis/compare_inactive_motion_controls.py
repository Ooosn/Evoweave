"""Compare original evidence controls on identical target-joint masks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from report_inactive_joint_capability import count_mask, unique_matches


def paired_score(records, metric, denominator, repetitions=4000):
    assets = sorted({row["index"] for row in records if row["base"][denominator] > 0})
    values = np.asarray([[sum(row[side][metric] for row in records if row["index"] == index)
                          for side in ("base", "control")]
                         + [sum(row["base"][denominator] for row in records if row["index"] == index)]
                         for index in assets], dtype=float)
    if not len(assets):
        return {"assets": 0, "count": 0, "base": None, "control": None, "delta": None, "delta_95_interval": None}
    total = values.sum(0)
    interval = None
    if len(assets) >= 3:
        sample = np.random.default_rng(20261011).integers(0, len(assets), (repetitions, len(assets)))
        boot = values[sample].sum(1)
        interval = np.quantile((boot[:, 1] - boot[:, 0]) / boot[:, 2], [0.025, 0.975]).tolist()
    return {"assets": len(assets), "count": int(total[2]), "base_correct": int(total[0]),
            "control_correct": int(total[1]), "base": total[0] / total[2], "control": total[1] / total[2],
            "delta": (total[1] - total[0]) / total[2], "delta_95_interval": interval}


def compare(baseline, probe):
    expected_counts = {"base_replays": 48, "prefix_checks": 4, "new_full_generations": 64, "static_reuses": 32}
    if (probe.get("complete") is not True or probe["counts"] != expected_counts
            or probe["changed_base_parameters"] or probe["parameters_with_grad"] or probe["trainable_parameters"]
            or not probe["cache_tensor_versions_unchanged"] or not probe["source_hashes_verified_after"]):
        raise ValueError("probe is incomplete or violates the frozen input/model contract")
    if baseline["checkpoint"] != probe["plan"]["checkpoint"]:
        raise ValueError("baseline checkpoint mismatch")
    originals = {(row["index"], row["view"]): row for row in baseline["rows"]}
    base_metrics = {(row["index"], row["view"]): row for row in probe["baseline"]["valid"]}
    if len(originals) != 48 or set(originals) != set(base_metrics):
        raise ValueError("expected all48 original asset/view pairs")
    masks = {(row["index"], row["mask"]): row["joint_indices"] for row in baseline["same_joint_comparisons"]}
    output = {"controls": {}, "per_case": []}
    for control in ("unknown", "permuted"):
        controlled = {(row["index"], row["view"]): row for row in probe["evaluation"][control]}
        if len(probe["evaluation"][control]) != 48 or set(controlled) != set(originals):
            raise ValueError("control cases do not match baseline")
        records = []
        for key, original in originals.items():
            source = controlled[key]
            row = deepcopy(original)
            for field in ("pred_joints", "pred_parents", "joint_matches"):
                row[field] = source[field]
            row["success"] = source["generation"]["success"]
            row["unique_matches"] = unique_matches(row["target_joints"], row["target_parents"],
                                                     row["pred_joints"], row["pred_parents"])
            if row["asset_id"] != source["asset_id"] or original["success"] != base_metrics[key]["generation"]["success"]:
                raise ValueError("asset or baseline-generation identity mismatch")
            group_masks = {name: np.asarray(mask, dtype=bool) for name, mask in row["observation"]["strata"].items()}
            common = np.zeros(len(row["target_joints"]), dtype=bool)
            common[masks[(key[0], "quiet_in_both_with_activity_elsewhere")]] = True
            group_masks["same_quiet_both_inputs"] = common
            for name, mask in group_masks.items():
                records.append({"index": key[0], "view": key[1], "stratum": name,
                                "base": count_mask(original, mask), "control": count_mask(row, mask)})
            output["per_case"].append({"control": control, "index": key[0], "view": key[1],
                "ce_delta": source["ce"] - base_metrics[key]["ce"],
                "base_success": original["success"], "control_success": row["success"],
                "identical_tokens": source["generation"]["generated_ids"] == base_metrics[key]["generation"]["generated_ids"],
                "quiet_base": count_mask(original, common), "quiet_control": count_mask(row, common)})
        summary = {}
        for view in ("normal8", "weak2", "nonstatic"):
            included = ("normal8", "weak2") if view == "nonstatic" else (view,)
            subset = [row for row in records if row["view"] in included]
            summary[view] = {}
            for stratum in ("quiet_strict", "hidden_demonstrated_strict", "observed_active", "same_quiet_both_inputs"):
                group = [row for row in subset if row["stratum"] == stratum]
                summary[view][stratum] = {}
                for cohort in ("all_cases", "paired_success"):
                    matched = group if cohort == "all_cases" else [row for row in group if row["base"]["success"] and row["control"]["success"]]
                    summary[view][stratum][cohort] = {
                        "one_to_one_joint_coverage": paired_score(matched, "unique_covered", "joints"),
                        "one_to_one_located_edges": paired_score(matched, "unique_located_edges", "edges")}
        output["controls"][control] = summary
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_base, raw_probe = args.baseline.read_bytes(), args.probe.read_bytes()
    result = compare(json.loads(raw_base), json.loads(raw_probe))
    result.update(complete=True, inputs={"baseline_sha256": hashlib.sha256(raw_base).hexdigest(),
                                        "probe_sha256": hashlib.sha256(raw_probe).hexdigest()},
                  scope="Paired original-model interventions on identical input/GT. Whole-asset bootstrap, same16 reused assets, exploratory and not multiplicity corrected. Existing implicit multiframe processing stays active. One-to-one located matches are geometric proxies, not guaranteed anatomical identities.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(args.output),
                      "same_quiet_nonstatic": {control: result["controls"][control]["nonstatic"]["same_quiet_both_inputs"]
                                                for control in result["controls"]}}, indent=2))


if __name__ == "__main__":
    main()
