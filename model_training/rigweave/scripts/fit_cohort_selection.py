"""Metadata-only deterministic fixed-fit cohorts; no manifest or asset file I/O."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from numbers import Integral

import numpy as np


STRATA = (("low", 4, 10), ("mid", 11, 60), ("high", 61, None))


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _joint_count(row):
    # Preserve the existing probe's precedence; fallback only when the key is absent.
    for group in ("canonical_metrics", "rootless_target_strict_metrics"):
        metrics = row.get(group, {})
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{group} must be a mapping for asset {row.get('asset_id')!r}")
        if "target_joint_count" in metrics:
            return _integer(metrics["target_joint_count"], f"{group}.target_joint_count", 0), f"{group}.target_joint_count"
    return 0, None


def _stratum(count):
    return next((name for name, low, high in STRATA if count >= low and (high is None or count <= high)), None)


def _training_rows(rows):
    if isinstance(rows, Mapping):
        raise ValueError("Pass training rows, not a mapping of train/valid splits")
    values = list(rows)
    for index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError(f"Row {index} must be a metadata mapping")
        for field in ("split", "source_split"):
            if field in row and row[field] != "train":
                raise ValueError(f"Training rows only: row {index} has {field}={row[field]!r}")
    return values


def _distinct(candidates):
    ids, paths = set(), set()
    for row, _, _ in candidates:
        asset_id, path = row.get("asset_id"), row.get("path")
        if not isinstance(asset_id, str) or not asset_id.strip() or not isinstance(path, str) or not path.strip():
            raise ValueError("Candidate rows require nonempty string asset_id and path")
        if asset_id in ids:
            raise ValueError(f"Duplicate candidate asset_id: {asset_id}")
        if path in paths:
            raise ValueError(f"Duplicate candidate path: {path}")
        ids.add(asset_id)
        paths.add(path)


def choose_fit_rows(rows, config, *, return_report=False):
    """Return original row objects, optionally paired with a JSON-safe report.

    Default probe.cohort_mode='base20_60' preserves the existing probe's ordered
    pool and NumPy RNG selection exactly for valid, unique training metadata.
    Explicit 'diverse8' requires assets=8 and frames=4. It discovers the sole
    non-base integration_source value and requires all six source/size cells.
    Caller owns immutable train-manifest loading, provenance and report writing.
    """
    rows = _training_rows(rows)
    probe = config["probe"]
    mode = probe.get("cohort_mode", "base20_60")
    seed = _integer(probe["selection_seed"], "probe.selection_seed", 0)
    assets = _integer(probe["assets"], "probe.assets", 1)
    if mode not in {"base20_60", "diverse8"}:
        raise ValueError(f"Unknown cohort_mode: {mode!r}")
    rng = np.random.default_rng(seed)
    metadata = [(row, *_joint_count(row)) for row in rows]
    discovered = sorted({row["integration_source"] for row in rows
                         if isinstance(row.get("integration_source"), str) and row["integration_source"]})
    report = {"schema_version": 1, "mode": mode, "selection_seed": seed, "numpy_version": np.__version__,
        "requested_assets": assets, "input_rows": len(rows), "discovered_source_values": discovered,
        "io_policy": "metadata only; caller supplies verified training-manifest rows",
        "count_precedence": ["canonical_metrics.target_joint_count", "rootless_target_strict_metrics.target_joint_count"],
        "strata_definition": [{"name": name, "min": low, "max": high} for name, low, high in STRATA]}
    if mode == "base20_60":
        candidates = [item for item in metadata if 20 <= item[1] <= 60 and item[0].get("integration_source") == "base"]
        if not candidates:
            raise ValueError("fixed fit cohort is empty")
        _distinct(candidates)
        # Intentionally do not sort: changing manifest order must retain legacy semantics.
        selected = [candidates[int(i)] for i in rng.permutation(len(candidates))[:assets]]
        if len(selected) != assets:
            raise ValueError("not enough fixed assets")
        phases = ["legacy_base20_60"] * len(selected)
        sources = ["base"]
        report["selection_policy"] = "unchanged ordered base20..60 pool; default_rng(seed).permutation"
        report["candidate_count"] = len(candidates)
    else:
        frames = _integer(probe.get("frames"), "probe.frames", 1)
        if assets != 8 or frames != 4:
            raise ValueError("diverse8 requires probe.assets=8 and probe.frames=4")
        for row, count, field in metadata:
            source = row.get("integration_source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Missing/non-string integration_source for asset {row.get('asset_id')!r}")
            if field is None or count == 0:
                raise ValueError(f"Missing positive target_joint_count for asset {row.get('asset_id')!r}")
        non_base = [name for name in discovered if name != "base"]
        if "base" not in discovered or len(non_base) != 1:
            raise ValueError(f"diverse8 needs base plus exactly one discovered non-base source; found {discovered}")
        sources = ["base", non_base[0]]
        candidates = [item for item in metadata if item[1] >= 4]
        _distinct(candidates)
        cells = {(source, name): [] for source in sources for name, _, _ in STRATA}
        for item in sorted(candidates, key=lambda x: (x[0]["asset_id"], x[0]["path"])):
            cells[(item[0]["integration_source"], _stratum(item[1]))].append(item)
        missing = [f"{source}/{name}" for (source, name), pool in cells.items() if not pool]
        if missing:
            raise ValueError("diverse8 missing required strata: " + ", ".join(missing))
        if len(candidates) < 8:
            raise ValueError(f"diverse8 needs 8 distinct eligible assets; found {len(candidates)} after covering six strata")
        queues = {key: [pool[int(i)] for i in rng.permutation(len(pool))] for key, pool in cells.items()}
        selected = [pool.pop() for pool in queues.values()]
        phases = ["required_source_size_cell"] * 6
        source_counts = Counter(item[0]["integration_source"] for item in selected)
        size_counts = Counter(_stratum(item[1]) for item in selected)
        cell_counts = Counter((item[0]["integration_source"], _stratum(item[1])) for item in selected)
        while len(selected) < 8:
            available = [key for key, pool in queues.items() if pool]
            scores = {key: (source_counts[key[0]], size_counts[key[1]], cell_counts[key]) for key in available}
            best = min(scores.values())
            tied = [key for key in available if scores[key] == best]
            key = tied[int(rng.integers(len(tied)))]
            selected.append(queues[key].pop())
            phases.append("stratified_extra")
            source_counts[key[0]] += 1
            size_counts[key[1]] += 1
            cell_counts[key] += 1
        report["selection_policy"] = "one per source/size cell, then least represented source, size, cell; seeded tie-break"
        report["candidate_count"] = len(candidates)
        report["frames_per_asset"] = frames
        report["excluded_below_four_joints"] = len(metadata) - len(candidates)
        report["candidate_strata"] = [{"integration_source": source, "joint_count_stratum": name, "count": len(pool)}
                                     for (source, name), pool in cells.items()]
        report["row_order_policy"] = "sort candidates by asset_id/path before seeded within-cell permutations"
    report["source_cohorts"] = sources
    report["selected"] = [{"selection_index": index, "asset_id": row["asset_id"], "path": row["path"],
        "integration_source": row["integration_source"], "target_joint_count": count, "joint_count_field": field,
        "joint_count_stratum": _stratum(count), "selection_phase": phase}
        for index, ((row, count, field), phase) in enumerate(zip(selected, phases))]
    report["selected_source_counts"] = dict(Counter(row["integration_source"] for row, _, _ in selected))
    report["selected_size_counts"] = dict(Counter(_stratum(count) for _, count, _ in selected))
    chosen = [row for row, _, _ in selected]
    return (chosen, report) if return_report else chosen
