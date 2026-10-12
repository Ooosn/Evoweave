"""Pure predeclared G3 selection on verified metadata and topology keys."""
from collections import Counter
import hashlib
import json
from numbers import Integral

import numpy as np


SOURCES = ("base", "addition")
SIZES = ("low", "mid", "high")
TRAIN_QUOTAS = {"low": 8, "mid": 16, "high": 8}
SEMANTICS = "exact DFS auxiliary token structure, not unordered graph isomorphism"


class SelectionError(ValueError):
    def __init__(self, message, counts):
        super().__init__(message)
        self.counts = counts


def size_band(count):
    if count < 4:
        return None
    return "low" if count <= 10 else "mid" if count <= 60 else "high"


def topology_key(tokens):
    values = np.asarray(tokens)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("Topology tokens must be a one-dimensional integer sequence")
    return hashlib.sha256(values.astype("<i8").tobytes(order="C")).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def validate_identities(train, valid):
    identities = {}
    for split, rows in (("train", train), ("valid", valid)):
        ids, paths = set(), set()
        for row in rows:
            for field, seen in (("asset_id", ids), ("path", paths)):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip() or value in seen:
                    raise ValueError(f"{split}: missing/duplicate {field}: {value!r}")
                seen.add(value)
            for field in ("split",):
                if field in row and row[field] != split:
                    raise ValueError(f"{split}: mismatched {field} for {row['asset_id']}")
        identities[split] = (ids, paths)
    for index, field in enumerate(("asset_id", "path")):
        overlap = identities["train"][index] & identities["valid"][index]
        if overlap:
            raise ValueError(f"Train/valid {field} overlap: {sorted(overlap)[:3]}")


def _counts(rows):
    return {"total": len(rows), "by_source": dict(Counter(r["integration_source"] for r in rows)),
            "by_size": dict(Counter(r["joint_count_stratum"] for r in rows)),
            "by_source_size": {f"{source}/{size}": sum(r["integration_source"] == source and
                r["joint_count_stratum"] == size for r in rows) for source in SOURCES for size in SIZES},
            "globally_unseen": sum(r.get("unseen_to_full_train", False) for r in rows),
            "unique_globally_unseen_topologies": len({r["topology_sha256"] for r in rows if r.get("unseen_to_full_train", False)})}


def select_comparison(train, valid, *, choose_fit_rows, seed=20261013):
    """No I/O; frequencies use ALL train rows before any cohort selection."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("selection seed must be a nonnegative integer")
    validate_identities(train, valid)
    rows_by_split = {}
    for split, rows in (("train", train), ("valid", valid)):
        prepared = []
        for row in rows:
            count = row["target_joint_count"]
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError(f"Invalid joint count: {row['asset_id']}")
            if row["integration_source"] not in SOURCES:
                raise ValueError(f"Unexpected integration_source: {row['integration_source']!r}")
            key = row["topology_sha256"]
            if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError(f"Invalid topology SHA256: {row['asset_id']}")
            prepared.append({**row, "joint_count_stratum": size_band(count)})
        rows_by_split[split] = sorted(prepared, key=lambda r: (r["asset_id"], r["path"]))
    train, valid = rows_by_split["train"], rows_by_split["valid"]
    frequency_full = Counter(r["topology_sha256"] for r in train)
    rng_train, rng_valid = [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(2)]
    train_eligible = [r for r in train if r["joint_count_stratum"] is not None]
    train_counts = _counts(train_eligible)
    shortages = {f"{source}/{size}": {"available": train_counts["by_source_size"][f"{source}/{size}"], "required": quota}
                 for source in SOURCES for size, quota in TRAIN_QUOTAS.items()
                 if train_counts["by_source_size"][f"{source}/{size}"] < quota}
    if shortages:
        raise SelectionError("Insufficient train64 source/size quota capacity", {"train": train_counts, "shortages": shortages})
    chosen_train = []
    for source in SOURCES:
        for size, quota in TRAIN_QUOTAS.items():
            pool = [r for r in train_eligible if r["integration_source"] == source and r["joint_count_stratum"] == size]
            chosen_train.extend({**pool[int(i)], "selection_reason": f"train64_exact_quota:{source}/{size}"}
                                for i in rng_train.permutation(len(pool))[:quota])
    frequency_selected = Counter(r["topology_sha256"] for r in chosen_train)

    def annotate(row):
        full, selected = frequency_full[row["topology_sha256"]], frequency_selected[row["topology_sha256"]]
        return {**row, "train_topology_frequency": full, "selected_train_topology_frequency": selected,
                "unseen_to_full_train": full == 0, "unseen_to_selected64": selected == 0}

    chosen_train = [annotate(r) for r in chosen_train]
    valid = [annotate(r) for r in valid]
    valid_eligible = [r for r in valid if r["joint_count_stratum"] is not None]
    capacities = _counts(valid_eligible)
    missing = [cell for cell, count in capacities["by_source_size"].items() if count == 0]
    problems = []
    if missing:
        problems.append("missing source/size cells: " + ", ".join(missing))
    if capacities["total"] < 32:
        problems.append("fewer than 32 eligible validation assets")
    if capacities["by_size"].get("low", 0) < 10:
        problems.append("fewer than 10 low-joint validation assets")
    if capacities["globally_unseen"] < 8:
        problems.append("fewer than 8 validation assets unseen to FULL train")
    if problems:
        raise SelectionError("; ".join(problems), {"train": train_counts, "valid": capacities})

    addition_target = min(8, capacities["by_source"].get("addition", 0))
    ranks = {valid_eligible[int(i)]["asset_id"]: rank for rank, i in enumerate(rng_valid.permutation(len(valid_eligible)))}
    remaining = {r["asset_id"]: r for r in valid_eligible}
    chosen_valid = []
    by_source, by_size, by_cell = Counter(), Counter(), Counter()
    unseen_count = 0

    def take(predicate, reason):
        nonlocal unseen_count
        candidates = [r for r in remaining.values() if predicate(r)]
        if not candidates:
            raise SelectionError(f"Cannot satisfy {reason}", {"valid": capacities, "selected": _counts(chosen_valid)})

        def score(r):
            source, size = r["integration_source"], r["joint_count_stratum"]
            covers = int(size == "low" and by_size["low"] < 10)
            covers += int(r["unseen_to_full_train"] and unseen_count < 8)
            covers += int(source == "addition" and by_source["addition"] < addition_target)
            return (-covers, by_size[size], by_source[source], by_cell[(source, size)], ranks[r["asset_id"]])

        row = min(candidates, key=score)
        chosen_valid.append({**row, "selection_reason": reason})
        del remaining[row["asset_id"]]
        source, size = row["integration_source"], row["joint_count_stratum"]
        by_source[source] += 1
        by_size[size] += 1
        by_cell[(source, size)] += 1
        unseen_count += int(row["unseen_to_full_train"])

    for source in SOURCES:
        for size in SIZES:
            take(lambda r: r["integration_source"] == source and r["joint_count_stratum"] == size,
                 f"valid32_required_cell:{source}/{size}")
    while by_size["low"] < 10:
        take(lambda r: r["joint_count_stratum"] == "low", "valid32_low_minimum_10")
    while unseen_count < 8:
        take(lambda r: r["unseen_to_full_train"], "valid32_fulltrain_unseen_minimum_8")
    while by_source["addition"] < addition_target:
        take(lambda r: r["integration_source"] == "addition", "valid32_addition_minimum_when_available")
    if len(chosen_valid) > 32:
        raise SelectionError("Validation lower bounds exceed 32 slots", {"valid": capacities})
    while len(chosen_valid) < 32:
        take(lambda r: True, "valid32_balanced_size_fill")

    fit_rows = []
    for r in chosen_train:
        group = r.get("joint_count_field", "canonical_metrics.target_joint_count").split(".")[0]
        fit_rows.append({"asset_id": r["asset_id"], "path": r["path"], "split": "train",
            "integration_source": r["integration_source"], group: {"target_joint_count": r["target_joint_count"]}})
    heldout_rows, heldout_report = choose_fit_rows(fit_rows, {"probe": {"cohort_mode": "diverse8", "assets": 8,
        "frames": 4, "selection_seed": int(seed)}}, return_report=True)
    lookup = {r["asset_id"]: r for r in chosen_train}
    heldout = [{**lookup[r["asset_id"]], "selection_reason": "train_heldout8_actual_choose_fit_rows_diverse8"} for r in heldout_rows]
    return {"groups": {"train": chosen_train, "train_heldout": heldout, "valid": chosen_valid},
        "counts": {"full_train": len(train), "full_valid": len(valid), "unique_topologies_fulltrain": len(frequency_full),
            "train_capacity": train_counts, "valid_capacity": capacities, "train64": _counts(chosen_train),
            "valid32": _counts(chosen_valid), "valid_unseen_fulltrain_total": sum(r["unseen_to_full_train"] for r in valid),
            "valid_unseen_selected64_total": sum(r["unseen_to_selected64"] for r in valid)},
        "policy": {"seed": int(seed), "rng": "SeedSequence(seed).spawn(2): train64, valid32; heldout uses original seed",
            "train_per_source_size": TRAIN_QUOTAS, "valid_low_minimum": 10, "valid_fulltrain_unseen_minimum": 8,
            "valid_addition_requested_minimum": 8, "valid_addition_capacity_adjusted_target": addition_target,
            "valid_fill": "least represented size, then source, then cell; seeded rank tie-break",
            "topology_semantics": SEMANTICS, "performance_or_G2_outputs_used": False},
        "proof": {"fulltrain_topology_frequency_sha256": json_digest(dict(frequency_full)),
            "selected64_topology_frequency_sha256": json_digest(dict(frequency_selected)),
            "train_inventory_sha256": json_digest(train), "valid_inventory_sha256": json_digest(valid),
            "train_valid_ids_disjoint": True, "train_valid_paths_disjoint": True,
            "train_heldout8_subset_of_train64": set(r["asset_id"] for r in heldout) <= set(lookup)},
        "heldout_selection_report": heldout_report}
