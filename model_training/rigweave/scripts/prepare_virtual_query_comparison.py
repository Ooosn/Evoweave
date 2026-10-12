"""Predeclare G3 metadata on CPU; only target_parents is read from asset NPZs."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np

from virtual_query_comparison_selection import (
    SelectionError, json_digest, select_comparison, topology_key, validate_identities,
)


EXPECTED_COUNTS = {"train": 18002, "valid": 967}
TOKEN_BYTE_CONTRACT = "SHA256 of complete BOS-through-EOS tokens, signed int64 little-endian (<i8), C-order bytes, no prefix/header"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def source_api(source_root):
    """Import only three NumPy/stdlib modules; extract the real manifest function."""
    rig = source_root / "model_training/rigweave"
    paths = {"codec": rig / "src/rigweave/dynamic_rig/aux_skeleton_tokens.py",
             "root_rt": rig / "src/rigweave/dynamic_rig/root_rt.py",
             "fit": rig / "scripts/fit_cohort_selection.py",
             "manifest": rig / "src/rigweave/dynamic_rig/virtual_frame_data.py"}
    hashes = {str(path): sha256(path) for path in paths.values()}
    loaded, previous = {}, {}
    try:
        for key in ("codec", "root_rt", "fit"):
            name = f"_comparison_source_{key}"
            previous[name] = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location(name, paths[key])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[key] = module
        tree = ast.parse(paths["manifest"].read_text(encoding="utf-8"), filename=str(paths["manifest"]))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_virtual_manifest")
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
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Asset changed during parent-array read: {path}")
    return parents


def prepare(source_root, config_path, *, parents_reader=None, expected_counts=None):
    source_root, config_path = Path(source_root).resolve(), Path(config_path).resolve()
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    reader = read_target_parents if parents_reader is None else parents_reader
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    seed = config["probe"]["selection_seed"]
    if type(seed) is not int or seed != 20261013:
        raise ValueError("Pinned G3 selection requires probe.selection_seed=20261013")
    if config["data"].get("global_rt_removal") is not True or config["data"].get("coordinate_clipping") is not False:
        raise ValueError("Expected retained-root RT config with coordinate_clipping=false")
    manifests = config["data"]["manifests"]
    tracked = {str(config_path): config_hash}
    with source_api(source_root) as (api, source_hashes):
        tracked.update(source_hashes)
        codec = api["codec"].AuxSkeletonTokens(**config["auxiliary_codec"])
        rows_by_split = {}
        for split, expected in expected_counts.items():
            spec = manifests[split]
            if spec["rows"] != expected:
                raise ValueError(f"{split}: pinned row count must be {expected}, got {spec['rows']}")
            rows = api["read_manifest"](spec["path"], expected_count=expected, expected_sha256=spec["sha256"])
            tracked[str(Path(spec["path"]).resolve())] = spec["sha256"]
            for row in rows:
                if api["root_rt"].validate_descriptor(row) is None:
                    raise ValueError(f"{split}: missing required root_rt descriptor: {row['asset_id']}")
            rows_by_split[split] = rows
        validate_identities(rows_by_split["train"], rows_by_split["valid"])
        metadata = {}
        for split, rows in rows_by_split.items():
            items = []
            for index, row in enumerate(rows):
                try:
                    count, field = api["fit"]._joint_count(row)
                    parents = np.asarray(reader(row["path"]))
                    if parents.ndim != 1 or parents.dtype.kind != "i" or len(parents) != count:
                        raise ValueError("target_parents must be a signed integer vector matching metadata joint count")
                    encoded = codec.encode(parents)
                    items.append({"manifest_row_index": index, "asset_id": row["asset_id"], "path": row["path"],
                        "split": split, "integration_source": row["integration_source"], "target_joint_count": int(count),
                        "joint_count_field": field, "topology_sha256": topology_key(encoded.tokens),
                        "topology_token_count": int(len(encoded.tokens)), "target_parents_sha256": topology_key(parents)})
                except Exception as exc:
                    raise ValueError(f"{split} row {index}, asset {row['asset_id']}: {exc}") from exc
            metadata[split] = items
        changed = [path for path, digest in tracked.items() if sha256(path) != digest]
        if changed:
            raise ValueError(f"Source/config/manifest changed during extraction: {changed}")
        try:
            selection = select_comparison(metadata["train"], metadata["valid"],
                                          choose_fit_rows=api["fit"].choose_fit_rows, seed=seed)
        except SelectionError as exc:
            exc.provenance = {"manifests": manifests, "config_sha256": config_hash, "selection_seed": seed,
                "source_sha256": source_hashes, "config_manifest_source_hashes_rechecked": True,
                "parent_arrays_read": {split: len(rows) for split, rows in metadata.items()},
                "extracted_metadata_sha256": {split: json_digest(rows) for split, rows in metadata.items()}}
            raise
    return {"schema_version": 1, "status": "passed", "manifests": manifests, "config_sha256": config_hash,
        "selection_seed": seed, "config_path": str(config_path), "source_root": str(source_root),
        "source_sha256": source_hashes, "script_sha256": sha256(__file__),
        "selection_helper_sha256": sha256(Path(__file__).with_name("virtual_query_comparison_selection.py")),
        "topology_byte_contract": TOKEN_BYTE_CONTRACT, "numpy_version": np.__version__,
        "config_manifest_source_hashes_rechecked": True,
        "parent_arrays_read": {split: len(rows) for split, rows in metadata.items()},
        "extraction_scope": {"asset_array_key": "target_parents", "allow_pickle": False,
            "torch_model_vendor_imported": False, "mesh_arrays_read": False, "frames_or_surface_samples_selected": False,
            "root_rt_validation": "actual descriptor validation only; no RT-cache geometry read/recomputation",
            "asset_integrity": "parent-array contents fingerprinted; NPZ size/mtime/inode checked around each read, not whole-NPZ file hashing"},
        "frame_boundary": "train_heldout repeats train asset IDs by design; runner must use disjoint heldout raw frames. No raw frames or learned features are prepared here.",
        **selection, "selected_groups_sha256": json_digest(selection["groups"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, source = args.output.resolve(), args.source_root.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output == source or source in output.parents:
        parser.error("Output must be outside the read-only source checkout")
    if output.exists() or temporary.exists():
        parser.error("Fresh output JSON required; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = prepare(source, args.config)
        code = 0
    except Exception as exc:
        report = {"schema_version": 1, "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
            "counts": getattr(exc, "counts", {}), "groups": {},
            "config_sha256": sha256(args.config) if args.config.is_file() else None,
            "selection_seed": 20261013, **getattr(exc, "provenance", {})}
        code = 1
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "output": str(output), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
