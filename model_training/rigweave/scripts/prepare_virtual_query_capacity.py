"""CPU capacity-input selection; reads only target_parents, never approves GPU use."""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from numbers import Integral
import os
from pathlib import Path
import sys
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np


TRAIN_COUNT = 18002
TRAIN_SHA256 = "653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27"
VENDOR_SHA256 = "39271ec8fc659e19461a7804a992b3796a9cca75c6073a065539dd4c030a8627"
PARENT_HASH_CONTRACT = "SHA256 of original-order parents cast to signed int64 little-endian (<i8), C-order bytes, no header"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def checked_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def main_token_bound(count):
    count = integer(count, "target_joint_count")
    # Root 0 is the only guaranteed non-branch. Valid trees need not be topological.
    return 3 * count + 4 * (count - 1) + 3


def geometry_proxy(metadata, frames):
    return 12 * frames * (2 * metadata["vertex_count"] + metadata["face_count"]) + 24 * metadata["face_count"]


@contextmanager
def source_api(source_root):
    """Reuse the comparison preparer's isolated NumPy-only source loading pattern."""
    rig = Path(source_root) / "model_training/rigweave"
    paths = {"codec": rig / "src/rigweave/dynamic_rig/aux_skeleton_tokens.py",
             "root_rt": rig / "src/rigweave/dynamic_rig/root_rt.py",
             "manifest": rig / "src/rigweave/dynamic_rig/virtual_frame_data.py"}
    hashes = {str(path): sha256(path) for path in paths.values()}
    loaded, previous = {}, {}
    try:
        for key in ("codec", "root_rt"):
            name = f"_capacity_source_{key}"
            previous[name] = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location(name, paths[key])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[key] = module
        tree = ast.parse(paths["manifest"].read_text(encoding="utf-8"), filename=str(paths["manifest"]))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "read_virtual_manifest")
        namespace = {"Path": Path, "Any": Any, "hashlib": hashlib, "json": json}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(paths["manifest"]), "exec"), namespace)
        loaded["read_manifest"] = namespace["read_virtual_manifest"]
        yield loaded, hashes
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def read_target_parents(path):
    path = Path(path)
    before = path.stat()
    with np.load(path, allow_pickle=False) as archive:
        parents = archive["target_parents"].copy()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if tuple(getattr(before, key) for key in fields) != tuple(getattr(after, key) for key in fields):
        raise ValueError(f"Asset changed during parent-array read: {path}")
    return parents


def row_metadata(row, index, validate_descriptor):
    if not isinstance(row, Mapping):
        raise ValueError(f"manifest row {index} must be an object")
    for field in ("asset_id", "path"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"manifest row {index}: missing {field}")
    if not Path(row["path"]).is_absolute():
        raise ValueError(f"manifest row {index}: absolute path required")
    for field in ("split", "source_split"):
        if field in row and row[field] != "train":
            raise ValueError(f"manifest row {index}: {field} must be train")
    if validate_descriptor(row) is None:
        raise ValueError(f"manifest row {index}: missing required root_rt descriptor")
    strict = row.get("rootless_target_strict_metrics")
    canonical = row.get("canonical_metrics", {})
    if not isinstance(strict, Mapping) or not isinstance(canonical, Mapping):
        raise ValueError(f"manifest row {index}: metrics must be objects; strict metrics are required")
    values, fields = {}, {}
    for key, aliases in (("target_joint_count", ("target_joint_count",)),
                         ("vertex_count", ("filtered_vertex_count", "vertex_count")),
                         ("face_count", ("filtered_face_count", "face_count"))):
        count = integer(strict.get(key), f"rootless_target_strict_metrics.{key}")
        present = [f"rootless_target_strict_metrics.{key}"]
        for alias in aliases:
            if alias in canonical:
                other = integer(canonical[alias], f"canonical_metrics.{alias}")
                if other != count:
                    raise ValueError(f"manifest row {index}: ambiguous count {key}: strict={count}, canonical.{alias}={other}")
                present.append(f"canonical_metrics.{alias}")
        values[key], fields[key] = count, present
    expected_hashes = []
    for name, item in (("row", row), ("canonical_metrics", canonical),
                       ("rootless_target_strict_metrics", strict)):
        if "target_parents_sha256" in item:
            expected_hashes.append(checked_hash(item["target_parents_sha256"], f"{name}.target_parents_sha256"))
    if len(set(expected_hashes)) > 1:
        raise ValueError(f"manifest row {index}: ambiguous parent-array hashes")
    return {"manifest_row_index": index, "asset_id": row["asset_id"], "path": row["path"], "split": "train",
            "integration_source": row.get("integration_source"), **values, "count_fields": fields,
            "main_tokens_upper_bound": main_token_bound(values["target_joint_count"]),
            "geometry_proxy": geometry_proxy(values, 24),
            "expected_target_parents_sha256": expected_hashes[0] if expected_hashes else None}


def measure_parents(metadata, parents, codec):
    parents = np.asarray(parents)
    count = metadata["target_joint_count"]
    if parents.ndim != 1 or parents.dtype.kind != "i" or len(parents) != count:
        raise ValueError("target_parents must be a signed 1D integer array matching metadata count")
    if parents[0] != -1 or np.count_nonzero(parents == -1) != 1:
        raise ValueError("target_parents must contain exactly one root, at index 0 with parent -1")
    encoded = codec.encode(parents)
    branches = int(np.count_nonzero(parents[1:] != np.arange(count - 1)))
    length = 3 * count + 4 * branches + 3
    digest = hashlib.sha256(parents.astype("<i8").tobytes(order="C")).hexdigest()
    expected = metadata["expected_target_parents_sha256"]
    if expected is not None and digest != expected:
        raise ValueError("target_parents SHA256 differs from manifest metadata")
    if length > metadata["main_tokens_upper_bound"]:
        raise AssertionError("measured main length exceeds the conservative bound")
    return {**metadata, "branch_count": branches, "main_tokens": length,
            "target_parents_sha256": digest, "auxiliary_token_count": len(encoded.tokens)}, parents.tolist()


def select_capacity(rows, *, codec, validate_descriptor, parents_reader=read_target_parents):
    if len(rows) < 3:
        raise ValueError("at least three training rows are required")
    metadata = [row_metadata(row, index, validate_descriptor) for index, row in enumerate(rows)]
    for field in ("asset_id", "path"):
        if len({row[field] for row in metadata}) != len(metadata):
            raise ValueError(f"duplicate manifest {field}")
    # Catch lexical aliases without opening assets or following remote symlinks.
    if len({os.path.normcase(os.path.normpath(row["path"])) for row in metadata}) != len(metadata):
        raise ValueError("duplicate normalized manifest path")
    identity = lambda row: (row["asset_id"], row["path"])
    candidates = sorted(metadata, key=lambda row: (-row["main_tokens_upper_bound"], *identity(row)))
    measured, arrays = {}, {}

    def measure(row):
        asset_id = row["asset_id"]
        if asset_id not in measured:
            try:
                measured[asset_id], arrays[asset_id] = measure_parents(row, parents_reader(row["path"]), codec)
            except Exception as exc:
                raise ValueError(f"train row {row['manifest_row_index']}, asset {asset_id}: {exc}") from exc
        return measured[asset_id]

    top_lengths = []
    census_rows = []
    next_bound = None
    for row in candidates:
        if len(top_lengths) == 3 and row["main_tokens_upper_bound"] < top_lengths[-1]:
            next_bound = row["main_tokens_upper_bound"]
            break
        item = measure(row)
        census_rows.append(item)
        top_lengths = sorted([*top_lengths, item["main_tokens"]], reverse=True)[:3]
    ranked = sorted(census_rows, key=lambda row: (-row["main_tokens"], *identity(row)))
    third = ranked[2]["main_tokens"]
    groups = {"longest_main": ranked[:3]}
    geometry_ties = {}
    for group, key in (("largest_vertices", "vertex_count"), ("largest_faces", "face_count"),
                       ("largest_geometry", "geometry_proxy")):
        maximum = max(row[key] for row in metadata)
        tied = [row for row in sorted(metadata, key=identity) if row[key] == maximum]
        geometry_ties[group] = len(tied)
        groups[group] = [measure(tied[0])]
    geometry_by_frames = {}
    for frames in range(2, 25):
        winner = min(metadata, key=lambda row: (-geometry_proxy(row, frames), *identity(row)))
        geometry_by_frames[str(frames)] = {**measure(winner), "geometry_frames": frames,
                                           "geometry_proxy": geometry_proxy(winner, frames)}
    selected_ids = {row["asset_id"] for group in groups.values() for row in group}
    selected_ids.update(row["asset_id"] for row in geometry_by_frames.values())
    selected = {asset_id: {"asset_id": asset_id, "path": measured[asset_id]["path"],
                          "manifest_row_index": measured[asset_id]["manifest_row_index"],
                          "target_parents": arrays[asset_id],
                          "target_parents_sha256": measured[asset_id]["target_parents_sha256"]}
                for asset_id in sorted(selected_ids)}
    return {"groups": groups, "geometry_by_frames": geometry_by_frames, "selected_parent_arrays": selected,
            "selected_parent_arrays_sha256": json_digest(selected),
            "longest_main_with_ties": [row for row in ranked if row["main_tokens"] >= third],
            "geometry_maximum_tie_counts": geometry_ties,
            "census_certificate": {"manifest_rows": len(rows), "length_search_parent_arrays_read": len(census_rows),
                "geometry_additional_parent_arrays_read": len(measured) - len(census_rows),
                "total_unique_parent_arrays_read": len(measured), "next_max_bound": next_bound,
                "third_measured_main_tokens": third, "all_rows_measured_for_length_search": len(census_rows) == len(rows),
                "stop_rule": "next_max_bound < third_measured_main_tokens; equality must be read",
                "bound_formula": "3*N + 4*(N-1) + 3 = 7*N-1, N>=1; no parent-order assumption",
                "bound_metadata_assumption": "Unread rows' joint counts and single-root-0 schema are trusted from the pinned canonical manifest; unread arrays are not validated by this selector.",
                "candidate_order_sha256": json_digest(candidates), "measured_rows_sha256": json_digest(census_rows)},
            "length_search_measured_rows": census_rows,
            "metadata_rejoin_contract": "Selection rows are metadata only. Rejoin the unchanged pinned whole manifest by index AND asset_id AND path; preserve root_rt and every other original field."}


def prepare(source_root, config_path, *, expected_config_sha256, parents_reader=read_target_parents,
            expected_train_count=TRAIN_COUNT, expected_manifest_sha256=TRAIN_SHA256):
    source_root, config_path = Path(source_root).resolve(), Path(config_path).resolve()
    checked_hash(expected_config_sha256, "expected config SHA256")
    checked_hash(expected_manifest_sha256, "expected train manifest SHA256")
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if config_hash != expected_config_sha256:
        raise ValueError("config SHA256 mismatch")
    config = json.loads(config_bytes)
    data = config["data"]
    if data.get("global_rt_removal") is not True or data.get("coordinate_clipping") is not False:
        raise ValueError("requires retained-root RT removal and coordinate_clipping=false")
    if data.get("normalization") != "input_sequence_mesh_bbox":
        raise ValueError("requires input_sequence_mesh_bbox normalization")
    spec = data["manifests"]["train"]
    if integer(spec["rows"], "train rows") != expected_train_count or spec["sha256"] != expected_manifest_sha256:
        raise ValueError("train manifest differs from the pinned count/SHA256")
    tracked = {str(config_path): config_hash, str(Path(spec["path"]).resolve()): expected_manifest_sha256}
    with source_api(source_root) as (api, source_hashes):
        tracked.update(source_hashes)
        rows = api["read_manifest"](spec["path"], expected_count=expected_train_count,
                                    expected_sha256=expected_manifest_sha256)
        codec = api["codec"].AuxSkeletonTokens(**config["auxiliary_codec"])
        selection = select_capacity(rows, codec=codec, validate_descriptor=api["root_rt"].validate_descriptor,
                                    parents_reader=parents_reader)
        changed = [path for path, digest in tracked.items() if sha256(path) != digest]
        if changed:
            raise ValueError(f"Source/config/manifest changed during selection: {changed}")
    return {"schema_version": 1, "status": "passed", "capacity_approved": False,
            "config_path": str(config_path), "config_sha256": config_hash, "manifest": spec,
            "source_root": str(source_root), "source_sha256": source_hashes,
            "script_sha256": sha256(__file__), "config_manifest_source_hashes_rechecked": True,
            "parent_array_hash_contract": PARENT_HASH_CONTRACT,
            "main_length_contract": {"formula": "3*N + 4*count(i!=0 and parents[i]!=i-1) + 3",
                "special_tokens": "BOS + class + EOS", "parts_bias_required": {},
                "status": "source-backed by primary report; selected-input tokenizer execution still required",
                "vendor_source": "src/tokenizer/tokenizer_part.py:186-220", "vendor_source_sha256": VENDOR_SHA256,
                "vendor_source_evidence": "Primary supplied pinned HGC source inspection; not independently read here",
                "vendor_tokenizer_executed": False, "parent_order": "original order, never DFS-reordered"},
            "geometry_proxy_contract": {"formula": "12*T*(2*V+F)+24*F", "groups_frames": 24,
                "geometry_by_frames": "One maximum row at every integer T=2..24; its geometry_proxy uses that T",
                "interpretation": "metadata stress-ranking proxy only; not measured GPU or total host memory"},
            "scope": {"npz_keys_read": ["target_parents"], "allow_pickle": False, "mesh_or_gt_coordinate_arrays_read": False,
                "root_rt": "descriptor validation only; no cache arrays read", "models_or_vendor_imported": False,
                "asset_integrity": "Parent content hashes plus NPZ stat before/after each read, not whole-NPZ hashes",
                "tie_break": "asset_id then path; manifest indices retain original row positions"},
            **selection}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, source = args.output.resolve(), args.source_root.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output == source or source in output.parents or output == args.config.resolve():
        parser.error("Output must be outside the read-only source checkout and not overwrite config")
    if output.exists() or temporary.exists():
        parser.error("Fresh output JSON required; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = prepare(source, args.config, expected_config_sha256=args.config_sha256)
        code = 0
    except Exception as exc:
        report = {"schema_version": 1, "status": "failed", "capacity_approved": False, "groups": {},
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
        code = 1
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "output": str(output), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
