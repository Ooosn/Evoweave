"""Select matched range-stress cases and package immutable NPZ inputs on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"manifest hash differs: {path}")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len({str(row["asset_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate asset IDs in manifest")
    return {str(row["asset_id"]): row for row in rows}, digest


def joint_count(row):
    count = row.get("rootless_target_strict_metrics", {}).get("target_joint_count")
    if count is None:
        count = row.get("canonical_metrics", {}).get("target_joint_count")
    if count is None or int(count) < 1:
        raise ValueError(f"missing positive target joint count: {row['asset_id']}")
    return int(count)


def range_max(row):
    return float(row["eligible_both_banks"]["maxabs"])


def frame_ids(row, count=8):
    witness = row["eligible_both_banks"]["worst_envelope"]
    requested = [witness["reference_raw_stored_frame"], witness["raw_stored_frame"],
                 *row["eligible_raw_stored_frames"]]
    selected = list(dict.fromkeys(int(x) for x in requested))[:count]
    if len(selected) != count or not set(selected).issubset(row["eligible_raw_stored_frames"]):
        raise ValueError("range witness cannot form a distinct eligible-frame input")
    return selected


def select_cases(audit_rows, manifests, old_train_ids):
    rows = []
    for original in audit_rows:
        if original["status"] != "ok":
            raise ValueError("source range audit contains a failed row")
        row = dict(original)
        accepted = manifests[row["split"]][row["asset_id"]]
        if str(accepted["path"]) != row["path"]:
            raise ValueError("range audit and accepted manifest paths differ")
        row["target_joint_count"] = joint_count(accepted)
        row["old_train_member"] = row["asset_id"] in old_train_ids
        rows.append(row)
    extremes = sorted((row for row in rows if range_max(row) > 4.),
                      key=lambda row: (-range_max(row), row["asset_id"]))
    moderate_valid = sorted((row for row in rows if row["split"] == "valid" and 2. < range_max(row) <= 4.),
                            key=lambda row: (-range_max(row), row["asset_id"]))[:8]
    if len(extremes) != 23 or len(moderate_valid) != 8:
        raise ValueError("expected 23 extreme and eight moderate validation cases in the pinned audit")
    available = [row for row in rows if range_max(row) <= 1.5]
    used, cases = set(), []
    for exposed, stratum in [(row, "extreme") for row in extremes] + [(row, "valid_moderate") for row in moderate_valid]:
        candidates = [row for row in available if row["asset_id"] not in used
                      and row["split"] == exposed["split"]
                      and row["integration_source"] == exposed["integration_source"]
                      and row["old_train_member"] == exposed["old_train_member"]]
        if not candidates:
            raise ValueError(f"no source/split/membership-matched control for {exposed['asset_id']}")
        def distance(row):
            return (abs(math.log(row["target_joint_count"] / exposed["target_joint_count"]))
                    + .25 * abs(math.log(row["vertex_count"] / exposed["vertex_count"])), row["asset_id"])
        control = min(candidates, key=distance)
        used.add(control["asset_id"])
        pair = len(cases) // 2
        for row, category in ((exposed, stratum), (control, stratum + "_control")):
            cases.append({key: row[key] for key in (
                "asset_id", "split", "path", "integration_source", "vertex_count", "frame_count",
                "target_joint_count", "old_train_member", "manifest_row_index")})
            cases[-1].update(case_index=len(cases) - 1, pair_index=pair, stratum=category,
                             audited_maxabs=range_max(row), selected_frames=frame_ids(row),
                             heldout_frames=row["heldout_raw_stored_frames"],
                             match_distance=distance(control)[0])
    if len({row["asset_id"] for row in cases}) != len(cases):
        raise ValueError("matched case selection reused an asset")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--old-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to replace an existing diagnostic corpus")
    audit = json.loads(args.audit.read_text())
    if not audit.get("complete") or audit.get("errors") or not audit.get("input_manifests_unchanged"):
        raise ValueError("range audit is incomplete or invalid")
    accepted, manifest_hashes = {}, {}
    for split in ("train", "valid"):
        accepted[split], manifest_hashes[split] = manifest(getattr(args, split), audit["manifests"][split]["sha256"])
    old_train, old_hash = manifest(args.old_train)
    cases = select_cases(audit["rows"], accepted, set(old_train))
    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / "inputs.tar"
    with tarfile.open(archive_path, "w") as archive:
        for row in cases:
            source = Path(row["path"])
            row["npz_sha256"] = sha256(source)
            row["npz_bytes"] = source.stat().st_size
            row["local_relative_path"] = f"npz/{row['case_index']:04d}.npz"
            archive.add(source, arcname=row["local_relative_path"], recursive=False)
            print(json.dumps({"packed": row["case_index"] + 1, "total": len(cases), "asset_id": row["asset_id"]}), flush=True)
    report = {
        "complete": True, "audit": str(args.audit), "audit_sha256": sha256(args.audit),
        "manifest_sha256": manifest_hashes, "old_train_manifest": str(args.old_train), "old_train_sha256": old_hash,
        "old_train_rows": len(old_train), "cases": cases,
        "selection": {"extreme_threshold": 4., "near_control_max": 1.5, "moderate_valid_min": 2.,
                      "frames": 8, "matching": "same split/source/old-train membership; nearest log joint count plus 0.25 log vertex count",
                      "interpretation": "targeted diagnostic subset, not a population quality estimate; extrema guaranteed in selected inputs"},
        "archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path)},
    }
    with tarfile.open(archive_path, "r") as archive:
        if len(archive.getmembers()) != len(cases):
            raise RuntimeError("archive member count differs")
        for row in cases:
            handle = archive.extractfile(row["local_relative_path"])
            digest = hashlib.sha256()
            with handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["npz_sha256"]:
                raise RuntimeError("archive member digest differs from original")
    (args.output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "cases": len(cases), "archive": report["archive"]}), flush=True)


if __name__ == "__main__":
    main()
