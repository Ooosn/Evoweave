"""Prepare the bounded normal5000 manifest and CPU capacity declaration.

This module is deliberately CPU-only.  It selects complete rows from the
current train manifest, validates the selected rows without loading geometry,
and writes a fresh derived manifest plus the existing capacity-selection
schema.  It does not create learned-feature caches or approve GPU capacity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_virtual_query_capacity as capacity
from fit_cohort_selection import _joint_count, _stratum


SELECTION_SEED = 20261013
TARGET_COUNT = 5000
HELDOUT_FRAMES = 8
MIN_TRAIN_FRAMES = 24
VERTEX_ARRAY_KEY = "frame_vertices_rootspace"
VERTEX_ARRAY_SOURCE = "rigweave.dynamic_rig.shared_structure_medium_data.load_medium_batch"
SOURCE_ORDER = ("base", "addition")
SOURCE_COUNTS = {"base": 15831, "addition": 2171}
EXPECTED_COUNTS = {"train": 18002, "valid": 967}
EXPECTED_MANIFEST_SHA256 = {
    "train": "653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27",
    "valid": "fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a",
}
DERIVED_MANIFEST_NAME = "train5000_manifest.jsonl"
CAPACITY_NAME = "capacity_selection.json"
SELECTION_NAME = "selection.json"

# The first three bins are the existing fit-cohort bins.  ``under4`` is kept
# explicit so a malformed or unusual current manifest cannot silently vanish
# from a proportional quota calculation.
JOINT_BINS = ("under4", "low", "mid", "high")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    return bytes_sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False).encode("ascii"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def joint_count_bin(count: int) -> str:
    count = _integer(count, "target_joint_count", 1)
    if count < 4:
        return "under4"
    value = _stratum(count)
    if value not in {"low", "mid", "high"}:
        raise ValueError(f"no joint-count bin for {count}")
    return value


def _manifest_joint_count(row: Mapping[str, Any], index: int) -> tuple[int, str]:
    try:
        count, field = _joint_count(row)
    except Exception as exc:
        raise ValueError(f"manifest row {index}: invalid target_joint_count: {exc}") from exc
    if field is None or count <= 0:
        raise ValueError(f"manifest row {index}: positive target_joint_count is required")
    return int(count), str(field)


def read_manifest_bytes(path: str | Path, *, expected_count: int, expected_sha256: str) -> list[dict[str, Any]]:
    """Read JSON objects while retaining each original JSONL line byte-for-byte."""

    path = Path(path)
    raw = path.read_bytes()
    digest = bytes_sha256(raw)
    if digest != expected_sha256:
        raise ValueError(f"manifest hash differs from the selected immutable data version: {path}")
    lines = raw.splitlines(keepends=True)
    if len(lines) != expected_count:
        raise ValueError(f"manifest row count differs for {path}: expected {expected_count}, got {len(lines)}")
    records = []
    for index, line in enumerate(lines):
        payload = line.rstrip(b"\r\n")
        if not payload.strip():
            raise ValueError(f"manifest row {index} is blank")
        try:
            row = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"manifest row {index} is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be a JSON object")
        records.append({"manifest_row_index": index, "row": row, "raw": line})
    return records


def _identity_sets(records: list[dict[str, Any]], split: str) -> tuple[set[str], set[str]]:
    ids, paths = set(), set()
    normalized_paths = set()
    for record in records:
        index, row = record["manifest_row_index"], record["row"]
        for field, seen in (("asset_id", ids), ("path", paths)):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{split} row {index}: missing/non-string {field}")
            if value in seen:
                raise ValueError(f"{split} row {index}: duplicate {field} {value!r}")
            seen.add(value)
        if row.get("split", split) != split:
            raise ValueError(f"{split} row {index}: mismatched split field")
        normalized = os.path.normcase(os.path.normpath(row["path"]))
        if normalized in normalized_paths:
            raise ValueError(f"{split} row {index}: duplicate normalized path {row['path']!r}")
        normalized_paths.add(normalized)
    return ids, paths


def validate_train_valid_disjointness(
    train_records: list[dict[str, Any]], valid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_ids, train_paths = _identity_sets(train_records, "train")
    valid_ids, valid_paths = _identity_sets(valid_records, "valid")
    id_overlap = sorted(train_ids & valid_ids)
    path_overlap = sorted(train_paths & valid_paths)
    if id_overlap:
        raise ValueError(f"train/valid asset_id overlap: {id_overlap[:3]}")
    if path_overlap:
        raise ValueError(f"train/valid path overlap: {path_overlap[:3]}")
    return {
        "train_valid_ids_disjoint": True,
        "train_valid_paths_disjoint": True,
        "train_id_count": len(train_ids),
        "valid_id_count": len(valid_ids),
        "train_path_count": len(train_paths),
        "valid_path_count": len(valid_paths),
        "id_overlap_count": 0,
        "path_overlap_count": 0,
    }


def _largest_remainder(counts: Mapping[str, int], total: int, order: list[str]) -> dict[str, int]:
    available = sum(int(counts[name]) for name in order)
    if total < 0 or total > available:
        raise ValueError(f"quota {total} is outside available population {available}")
    if not available:
        if total:
            raise ValueError("cannot allocate a positive quota from an empty population")
        return {name: 0 for name in order}
    ideals = {name: total * int(counts[name]) / available for name in order}
    result = {name: min(int(counts[name]), int(np.floor(ideals[name]))) for name in order}
    remainder = total - sum(result.values())
    ranked = sorted(order, key=lambda name: (-(ideals[name] - result[name]), order.index(name)))
    for name in ranked[:remainder]:
        if result[name] >= int(counts[name]):
            raise AssertionError("largest-remainder allocation exceeded a cell")
        result[name] += 1
    return result


def proportional_quotas(
    counts: Mapping[str, int], total: int, *, minimum_one_nonempty: bool = False,
    order: list[str] | tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Allocate an exact integer total by largest remainder, deterministically."""

    total = _integer(total, "quota total", 0)
    names = list(counts) if order is None else list(order)
    if set(names) != set(counts) or len(names) != len(set(names)):
        raise ValueError("quota order must list each count key exactly once")
    values = {name: _integer(counts[name], f"available[{name}]", 0) for name in names}
    nonempty = [name for name in names if values[name] > 0]
    if minimum_one_nonempty and nonempty and total >= len(nonempty):
        base = {name: int(values[name] > 0) for name in names}
        remaining = total - len(nonempty)
        residual = {name: values[name] - base[name] for name in names}
        extra = _largest_remainder(residual, remaining, names) if remaining else {name: 0 for name in names}
        return {name: base[name] + extra[name] for name in names}
    return _largest_remainder(values, total, names)


def select_train_records(
    records: list[dict[str, Any]], *, target_count: int = TARGET_COUNT,
    seed: int = SELECTION_SEED, source_counts: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select without replacement, then restore original manifest order."""

    target_count = _integer(target_count, "target_count", 1)
    seed = _integer(seed, "selection seed", 0)
    if target_count > len(records):
        raise ValueError(f"target_count {target_count} exceeds train rows {len(records)}")
    expected_sources = dict(SOURCE_COUNTS if source_counts is None else source_counts)
    if set(expected_sources) != set(SOURCE_ORDER):
        raise ValueError(f"source counts must contain exactly {SOURCE_ORDER}")
    available_sources = Counter()
    available_bins = Counter()
    indexed = []
    for record in records:
        index, row = record["manifest_row_index"], record["row"]
        source = row.get("integration_source")
        if source not in SOURCE_ORDER:
            raise ValueError(f"train row {index}: unexpected integration_source {source!r}")
        joint_count, joint_field = _manifest_joint_count(row, index)
        joint_bin = joint_count_bin(joint_count)
        available_sources[source] += 1
        available_bins[(source, joint_bin)] += 1
        indexed.append({**record, "integration_source": source, "joint_count": joint_count,
                        "joint_count_field": joint_field, "joint_count_bin": joint_bin})
    if dict(available_sources) != expected_sources:
        raise ValueError(f"train source counts differ: expected {expected_sources}, got {dict(available_sources)}")

    source_quotas = proportional_quotas(expected_sources, target_count, order=list(SOURCE_ORDER))
    chosen_indices = []
    bin_quotas: dict[str, int] = {}
    for source_index, source in enumerate(SOURCE_ORDER):
        source_bin_counts = {name: available_bins[(source, name)] for name in JOINT_BINS}
        source_bin_quotas = proportional_quotas(
            source_bin_counts, source_quotas[source], minimum_one_nonempty=True, order=list(JOINT_BINS)
        )
        for name in JOINT_BINS:
            bin_quotas[f"{source}/{name}"] = source_bin_quotas[name]
            pool = [item for item in indexed if item["integration_source"] == source and item["joint_count_bin"] == name]
            quota = source_bin_quotas[name]
            if quota > len(pool):
                raise ValueError(f"insufficient train rows for quota {source}/{name}: {quota}>{len(pool)}")
            if quota:
                rng = np.random.default_rng(np.random.SeedSequence([seed, source_index, JOINT_BINS.index(name)]))
                positions = rng.permutation(len(pool))[:quota]
                chosen_indices.extend(pool[int(position)]["manifest_row_index"] for position in positions)
    if len(chosen_indices) != target_count or len(set(chosen_indices)) != target_count:
        raise AssertionError("selection quota accounting did not produce an exact without-replacement subset")
    chosen = [indexed[index] for index in sorted(chosen_indices)]
    selected_sources = Counter(item["integration_source"] for item in chosen)
    selected_bins = Counter(f"{item['integration_source']}/{item['joint_count_bin']}" for item in chosen)
    return chosen, {
        "seed": seed,
        "target_count": target_count,
        "source_counts": {name: int(available_sources[name]) for name in SOURCE_ORDER},
        "source_quotas": source_quotas,
        "selected_source_counts": {name: int(selected_sources[name]) for name in SOURCE_ORDER},
        "joint_count_bins": list(JOINT_BINS),
        "joint_count_bin_definition": {
            "under4": {"min": 1, "max": 3}, "low": {"min": 4, "max": 10},
            "mid": {"min": 11, "max": 60}, "high": {"min": 61, "max": None},
        },
        "available_bin_counts": {f"{source}/{name}": int(available_bins[(source, name)])
                                  for source in SOURCE_ORDER for name in JOINT_BINS},
        "bin_quotas": bin_quotas,
        "selected_bin_counts": {key: int(selected_bins[key]) for key in bin_quotas},
        "without_replacement": True,
        "row_order_policy": "seeded within source/joint-count cells; selected rows restored to original manifest order",
        "selection_algorithm": "numpy SeedSequence(seed, source_index, bin_index) plus permutation",
    }


def _stat_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return tuple(int(getattr(stat, name)) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns"))


def read_npz_array_header(path: str | Path, array_key: str = VERTEX_ARRAY_KEY) -> dict[str, Any]:
    """Read one NPZ member header without materializing its array body."""

    path = Path(path)
    before = _stat_stamp(path)
    member_name = f"{array_key}.npy"
    try:
        with zipfile.ZipFile(path, "r") as archive:
            matches = [name for name in archive.namelist()
                       if name.replace("\\", "/").rsplit("/", 1)[-1] == member_name]
            if len(matches) != 1:
                raise ValueError(f"{path}: expected one NPZ member {member_name!r}, found {len(matches)}")
            with archive.open(matches[0], "r") as stream:
                version = np.lib.format.read_magic(stream)
                reader = getattr(np.lib.format, f"read_array_header_{version[0]}_{version[1]}", None)
                if reader is None:
                    raise ValueError(f"{path}: unsupported NumPy .npy header version {version}")
                shape, fortran_order, dtype = reader(stream)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path}: cannot read NPZ member header: {exc}") from exc
    after = _stat_stamp(path)
    if before != after:
        raise ValueError(f"asset changed during NPZ header read: {path}")
    shape = tuple(_integer(value, "vertex header dimension", 0) for value in shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[2] != 3:
        raise ValueError(f"{path}: {array_key} header must have shape (frames, vertices, 3), got {shape}")
    return {"array_key": array_key, "member": matches[0], "shape": list(shape),
            "dtype": np.dtype(dtype).str, "fortran_order": bool(fortran_order),
            "frame_count": int(shape[0]), "vertex_count": int(shape[1])}


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(int(value) for value in values).items())}


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        raise ValueError("quantiles require at least one value")
    array = np.asarray(values, dtype=np.float64)
    levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {str(level): float(np.quantile(array, level)) for level in levels}


def _with_auxiliary_tokens(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the existing codec result under the capacity provenance name."""

    output = dict(row)
    output["auxiliary_tokens"] = _integer(
        output.get("auxiliary_token_count"), "auxiliary_token_count", 1
    )
    return output


def _annotate_capacity_selection(
    capacity_selection: dict[str, Any], measured_rows: list[dict[str, Any]],
    selected_rows: list[Mapping[str, Any]],
) -> None:
    """Keep required groups intact and add the optional auxiliary stress case."""

    for group, rows in list(capacity_selection["groups"].items()):
        capacity_selection["groups"][group] = [_with_auxiliary_tokens(row) for row in rows]
    capacity_selection["geometry_by_frames"] = {
        frames: _with_auxiliary_tokens(row)
        for frames, row in capacity_selection["geometry_by_frames"].items()
    }
    capacity_selection["longest_main_with_ties"] = [
        _with_auxiliary_tokens(row) for row in capacity_selection["longest_main_with_ties"]
    ]
    capacity_selection["max_joint_cases"] = [
        _with_auxiliary_tokens(row) for row in capacity_selection.get("max_joint_cases", [])
    ]
    subset_indices = {
        (str(row["asset_id"]), str(row["path"])): index
        for index, row in enumerate(selected_rows)
    }
    largest = min(
        measured_rows,
        key=lambda row: (-int(row["auxiliary_token_count"]), row["asset_id"], row["path"]),
    )
    identity = (str(largest["asset_id"]), str(largest["path"]))
    if identity not in subset_indices:
        raise AssertionError("largest auxiliary case is not in the selected subset")
    template = capacity_selection["groups"]["longest_main"][0]
    largest_row = {key: largest[key] for key in template if key in largest}
    largest_row["manifest_row_index"] = subset_indices[identity]
    capacity_selection["groups"]["largest_auxiliary"] = [_with_auxiliary_tokens(largest_row)]


def _validate_selected_row(
    record: dict[str, Any], *, validator: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
    codec: Any, parents_reader: Callable[[str | Path], Any],
    header_reader: Callable[[str | Path], Mapping[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    index, row = record["manifest_row_index"], record["row"]
    try:
        root_rt = validator(row)
    except Exception as exc:
        raise ValueError(f"train row {index}, asset {row.get('asset_id')}: invalid root_rt: {exc}") from exc
    if root_rt is None or "root_rt" not in row:
        raise ValueError(f"train row {index}, asset {row.get('asset_id')}: full root_rt descriptor is required")
    try:
        header = dict(header_reader(row["path"]))
        frame_count = _integer(header.get("frame_count"), "actual frame count", 1)
        if frame_count - HELDOUT_FRAMES < MIN_TRAIN_FRAMES:
            raise ValueError(
                f"actual frame count {frame_count} leaves {frame_count - HELDOUT_FRAMES} after reserving "
                f"{HELDOUT_FRAMES}; requires at least {MIN_TRAIN_FRAMES}"
            )
        if _integer(root_rt.get("frame_count"), "root_rt.frame_count", 1) != frame_count:
            raise ValueError(f"root_rt.frame_count does not match actual frame count {frame_count}")
        parents = np.asarray(parents_reader(row["path"]))
        metadata = capacity.row_metadata(row, index, validator)
        measured, stored = capacity.measure_parents(metadata, parents, codec)
    except Exception as exc:
        if str(exc).startswith("train row "):
            raise
        raise ValueError(f"train row {index}, asset {row.get('asset_id')}: {exc}") from exc
    measured = {
        **measured,
        "actual_frame_count": frame_count,
        "eligible_frame_count": frame_count - HELDOUT_FRAMES,
        "joint_count_bin": record["joint_count_bin"],
        "root_rt_present": True,
        "vertex_header": header,
        "auxiliary_tokens": _integer(measured["auxiliary_token_count"], "auxiliary_token_count", 1),
    }
    return measured, np.asarray(parents).copy()


def _validate_output_location(output_dir: Path, source_root: Path) -> None:
    output_dir = output_dir.resolve()
    source_root = source_root.resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("output must be outside the read-only source checkout")
    if output_dir.exists():
        raise FileExistsError(f"fresh exclusive output directory required; refusing overwrite: {output_dir}")


def _build(
    source_root: str | Path, config_path: str | Path, *, output_dir: str | Path | None = None,
    expected_counts: Mapping[str, int] | None = None,
    expected_manifest_sha256: Mapping[str, str] | None = None,
    expected_source_counts: Mapping[str, int] | None = None,
    target_count: int = TARGET_COUNT, seed: int = SELECTION_SEED,
    parents_reader: Callable[[str | Path], Any] | None = None,
    header_reader: Callable[[str | Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source_root, config_path = Path(source_root).resolve(), Path(config_path).resolve()
    output_path = None if output_dir is None else Path(output_dir).resolve()
    if output_path is not None:
        _validate_output_location(output_path, source_root)
    expected_counts = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    expected_manifest_sha256 = dict(EXPECTED_MANIFEST_SHA256 if expected_manifest_sha256 is None
                                    else expected_manifest_sha256)
    expected_source_counts = dict(SOURCE_COUNTS if expected_source_counts is None else expected_source_counts)
    if set(expected_counts) != {"train", "valid"} or set(expected_manifest_sha256) != {"train", "valid"}:
        raise ValueError("expected train/valid count and hash pins are required")
    config_bytes = config_path.read_bytes()
    config_hash = bytes_sha256(config_bytes)
    config = json.loads(config_bytes.decode("utf-8"))
    data = config.get("data", {})
    if data.get("global_rt_removal") is not True or data.get("coordinate_clipping") is not False:
        raise ValueError("requires retained-root RT removal and coordinate_clipping=false")
    if data.get("normalization", "input_sequence_mesh_bbox") != "input_sequence_mesh_bbox":
        raise ValueError("requires input_sequence_mesh_bbox normalization")
    if data.get("heldout_frames", HELDOUT_FRAMES) != HELDOUT_FRAMES:
        raise ValueError(f"heldout_frames must be {HELDOUT_FRAMES}")
    manifests = data.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {"train", "valid"}:
        raise ValueError("config data.manifests must contain train and valid")
    for split in ("train", "valid"):
        spec = manifests[split]
        if not isinstance(spec, Mapping) or spec.get("rows") != expected_counts[split] or spec.get("sha256") != expected_manifest_sha256[split]:
            raise ValueError(f"{split} manifest differs from the pinned count/SHA256")
    tracked = {str(config_path): config_hash}
    parents_reader = capacity.read_target_parents if parents_reader is None else parents_reader
    header_reader = read_npz_array_header if header_reader is None else header_reader
    with capacity.source_api(source_root) as (api, source_hashes):
        tracked.update(source_hashes)
        records = {
            split: read_manifest_bytes(manifests[split]["path"], expected_count=expected_counts[split],
                                       expected_sha256=expected_manifest_sha256[split])
            for split in ("train", "valid")
        }
        tracked.update({str(Path(manifests[split]["path"]).resolve()): expected_manifest_sha256[split]
                        for split in ("train", "valid")})
        disjointness = validate_train_valid_disjointness(records["train"], records["valid"])
        selected, quotas = select_train_records(records["train"], target_count=target_count, seed=seed,
                                                source_counts=expected_source_counts)
        codec = api["codec"].AuxSkeletonTokens(**config.get("auxiliary_codec", {}))
        measured_rows, parent_arrays = [], {}
        for record in selected:
            measured, parents = _validate_selected_row(
                record, validator=api["root_rt"].validate_descriptor, codec=codec,
                parents_reader=parents_reader, header_reader=header_reader,
            )
            measured_rows.append(measured)
            parent_arrays[record["row"]["path"]] = parents

        selected_rows = [record["row"] for record in selected]

        def cached_parents(path: str | Path) -> np.ndarray:
            key = str(path)
            if key not in parent_arrays:
                raise AssertionError(f"capacity selector requested an unvalidated parent array: {key}")
            return parent_arrays[key].copy()

        capacity_selection = capacity.select_capacity(
            selected_rows, codec=codec, validate_descriptor=api["root_rt"].validate_descriptor,
            parents_reader=cached_parents,
        )
        if len(capacity_selection["groups"].get("longest_main", [])) != 3:
            raise ValueError("capacity selection did not produce exactly three longest_main cases")
        max_joint_cases = sorted(
            measured_rows,
            key=lambda row: (-int(row["target_joint_count"]), -int(row["main_tokens"]), row["asset_id"], row["path"]),
        )[:3]
        capacity_selection["max_joint_cases"] = max_joint_cases
        _annotate_capacity_selection(capacity_selection, measured_rows, selected_rows)
        changed = [path for path, digest in tracked.items() if sha256(path) != digest]
        if changed:
            raise ValueError(f"source/config/manifest changed during selection: {changed}")

    subset_bytes = b"".join(record["raw"] for record in selected)
    subset_sha256 = bytes_sha256(subset_bytes)
    derived_path = (output_path / DERIVED_MANIFEST_NAME) if output_path is not None else Path(DERIVED_MANIFEST_NAME)
    capacity_path = (output_path / CAPACITY_NAME) if output_path is not None else Path(CAPACITY_NAME)
    subset_spec = {"path": str(derived_path), "rows": int(target_count), "sha256": subset_sha256}
    capacity_report = {
        "schema_version": 1,
        "status": "passed",
        "capacity_approved": False,
        "gpu_capacity_passed": False,
        "manifest": subset_spec,
        "source_manifest": dict(manifests["train"]),
        "selection_kind": "normal5000_cpu_input_preparation",
        "selection_seed": int(seed),
        "capacity_contract": "CPU parent/header stress selection only; no GPU capacity claim",
        "scope": {
            "npz_keys_read": ["target_parents", VERTEX_ARRAY_KEY + ":header_only"],
            "allow_pickle": False,
            "full_geometry_read": False,
            "learned_feature_cache": False,
        },
        **capacity_selection,
    }
    capacity_bytes = _json_bytes(capacity_report)
    capacity_sha256 = bytes_sha256(capacity_bytes)
    row_summaries = []
    for subset_index, (record, measured) in enumerate(zip(selected, measured_rows)):
        row = record["row"]
        row_summaries.append({
            "subset_manifest_row_index": subset_index,
            "source_manifest_row_index": record["manifest_row_index"],
            "asset_id": row["asset_id"],
            "path": row["path"],
            "integration_source": record["integration_source"],
            "joint_count": int(measured["target_joint_count"]),
            "joint_count_bin": record["joint_count_bin"],
            "original_xyz_token_length": int(measured["main_tokens"]),
            "branch_count": int(measured["branch_count"]),
            "auxiliary_token_count": int(measured["auxiliary_token_count"]),
            "auxiliary_tokens": int(measured["auxiliary_tokens"]),
            "actual_frame_count": int(measured["actual_frame_count"]),
            "eligible_frame_count_after_heldout": int(measured["eligible_frame_count"]),
            "vertex_count": int(measured["vertex_count"]),
            "face_count": int(measured["face_count"]),
            "source_line_sha256": bytes_sha256(record["raw"]),
        })
    source_line_bytes_sha256 = bytes_sha256(b"".join(record["raw"] for record in selected))
    summary = {
        "schema_version": 1,
        "status": "passed",
        "capacity_approved": False,
        "gpu_capacity_passed": False,
        "selection": {
            "name": "normal5000",
            "seed": int(seed),
            "target_count": int(target_count),
            "quotas": quotas,
            "selected_source_manifest_row_indices_sha256": json_digest(
                [record["manifest_row_index"] for record in selected]
            ),
        },
        "source": {
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "train_manifest": dict(manifests["train"]),
            "valid_manifest": dict(manifests["valid"]),
            "train_manifest_sha256_verified": expected_manifest_sha256["train"],
            "valid_manifest_sha256_verified": expected_manifest_sha256["valid"],
            "train_rows_verified": len(records["train"]),
            "valid_rows_verified": len(records["valid"]),
            "source_counts_verified": dict(expected_source_counts),
        },
        "derived_manifest": subset_spec,
        "actual_frame_count": {
            "array_key": VERTEX_ARRAY_KEY,
            "array_key_source": VERTEX_ARRAY_SOURCE,
            "header_only": True,
            "heldout_frames_reserved_per_asset": HELDOUT_FRAMES,
            "minimum_eligible_frames": MIN_TRAIN_FRAMES,
            "histogram": _histogram([int(row["actual_frame_count"]) for row in measured_rows]),
            "eligible_frame_histogram": _histogram([int(row["eligible_frame_count"]) for row in measured_rows]),
            "minimum_actual_frame_count": min(int(row["actual_frame_count"]) for row in measured_rows),
            "maximum_actual_frame_count": max(int(row["actual_frame_count"]) for row in measured_rows),
        },
        "joint_count_quantiles": _quantiles([int(row["target_joint_count"]) for row in measured_rows]),
        "original_xyz_token_length_quantiles": _quantiles([int(row["main_tokens"]) for row in measured_rows]),
        "train_valid_disjointness": disjointness,
        "rows_unchanged": {
            "selected_rows_in_original_manifest_order": True,
            "selected_source_line_bytes_sha256": source_line_bytes_sha256,
            "derived_manifest_sha256_matches_selected_source_lines": source_line_bytes_sha256 == subset_sha256,
            "full_json_rows_preserved": True,
            "root_rt_preserved_for_selected_rows": all("root_rt" in record["row"] for record in selected),
            "manifest_line_bytes_rewritten": False,
        },
        "selected_rows": row_summaries,
        "capacity_selection": {
            "path": str(capacity_path),
            "sha256": capacity_sha256,
            "status": capacity_report["status"],
            "capacity_approved": False,
            "groups": sorted(capacity_report["groups"]),
            "longest_main_count": len(capacity_report["groups"]["longest_main"]),
            "max_joint_case_count": len(max_joint_cases),
        },
        "normal_training_frame_contract": {
            "frame_count_min": 2,
            "frame_count_max": 24,
            "selection_pool": "all non-heldout NPZ frame IDs after deterministic per-asset reserve of 8",
            "sampling_without_replacement_within_exposure": True,
            "heldout_frames_used_for_training": False,
        },
        "quality_policy": {
            "drop_bad_rows": False,
            "resample_after_quality_error": False,
            "failure_mode": "fail with report before writing derived artifacts",
            "output_policy": "fresh exclusive directory; existing output is never overwritten",
        },
        "execution_scope": {
            "parent_arrays_read": len(selected),
            "vertex_array_bodies_read": False,
            "joint_coordinate_arrays_read": False,
            "mesh_faces_read": False,
            "learned_features_written": False,
            "gpu_used": False,
            "numpy_version": np.__version__,
        },
        "script_sha256": sha256(__file__),
        "capacity_helper_sha256": sha256(capacity.__file__),
        "source_helper_sha256": source_hashes,
        "source_config_manifest_hashes_rechecked": True,
    }
    return {"summary": summary, "subset_bytes": subset_bytes, "capacity_bytes": capacity_bytes}


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def prepare(source_root: str | Path, config_path: str | Path, *, output_dir: str | Path | None = None,
            expected_counts: Mapping[str, int] | None = None,
            expected_manifest_sha256: Mapping[str, str] | None = None,
            expected_source_counts: Mapping[str, int] | None = None,
            target_count: int = TARGET_COUNT, seed: int = SELECTION_SEED,
            parents_reader: Callable[[str | Path], Any] | None = None,
            header_reader: Callable[[str | Path], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build the selection and optionally commit all artifacts to a new directory."""

    if output_dir is None:
        return _build(
            source_root, config_path, expected_counts=expected_counts,
            expected_manifest_sha256=expected_manifest_sha256, expected_source_counts=expected_source_counts,
            target_count=target_count, seed=seed, parents_reader=parents_reader, header_reader=header_reader,
        )["summary"]
    output = Path(output_dir).resolve()
    source = Path(source_root).resolve()
    _validate_output_location(output, source)
    artifacts = _build(
        source, config_path, output_dir=output, expected_counts=expected_counts,
        expected_manifest_sha256=expected_manifest_sha256, expected_source_counts=expected_source_counts,
        target_count=target_count, seed=seed, parents_reader=parents_reader, header_reader=header_reader,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary").mkdir()
    _write_exclusive(output / DERIVED_MANIFEST_NAME, artifacts["subset_bytes"])
    _write_exclusive(output / CAPACITY_NAME, artifacts["capacity_bytes"])
    _write_exclusive(output / "summary" / SELECTION_NAME, _json_bytes(artifacts["summary"]))
    return artifacts["summary"]


def _write_failure_report(output: Path, source_root: Path, config_path: Path, exc: Exception) -> None:
    if output.exists():
        return
    if output == source_root or source_root in output.parents:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary").mkdir()
    report = {
        "schema_version": 1,
        "status": "failed",
        "capacity_approved": False,
        "gpu_capacity_passed": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "source_root": str(source_root),
        "config_path": str(config_path),
        "selection_seed": SELECTION_SEED,
        "target_count": TARGET_COUNT,
        "drop_bad_rows": False,
        "resample_after_quality_error": False,
        "derived_artifacts_written": False,
        "script_sha256": sha256(__file__),
    }
    _write_exclusive(output / "summary" / SELECTION_NAME, _json_bytes(report))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_root, config_path, output = args.source_root.resolve(), args.config.resolve(), args.output.resolve()
    try:
        summary = prepare(source_root, config_path, output_dir=output)
        print(json.dumps({"status": summary["status"], "output": str(output),
                          "manifest": summary["derived_manifest"],
                          "capacity_selection": summary["capacity_selection"]}, sort_keys=True))
        return 0
    except Exception as exc:
        _write_failure_report(output, source_root, config_path, exc)
        print(json.dumps({"status": "failed", "output": str(output),
                          "error": {"type": type(exc).__name__, "message": str(exc)}}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
