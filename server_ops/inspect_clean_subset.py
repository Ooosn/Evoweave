#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _scalar(v):
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return v.item()
        return v
    return v


def _decode(v):
    v = _scalar(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, np.str_):
        return str(v)
    return v


def _json_from_npz(z):
    for key in ("meta_json", "metadata_json", "meta", "metadata"):
        if key not in z.files:
            continue
        raw = _decode(z[key])
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"__unparsed__": raw[:1000]}
        if isinstance(raw, dict):
            return raw
    return {}


def _names(z):
    out = {}
    for key in z.files:
        lk = key.lower()
        if "name" in lk or "joint" in lk or "bone" in lk or "parent" in lk or "tail" in lk:
            arr = z[key]
            if arr.dtype.kind in {"U", "S", "O"}:
                try:
                    vals = [_decode(x) for x in arr.tolist()]
                    out[key] = vals[:30]
                except Exception as exc:
                    out[key] = f"<decode failed: {exc}>"
            else:
                out[key] = f"shape={arr.shape} dtype={arr.dtype}"
    return out


def _shape(z, key):
    return z[key].shape if key in z.files else None


def _first_existing(mapping, names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    compare_csv = root / "oracle_compare" / "pass1_oracle_compare.csv"
    rows = list(csv.DictReader(compare_csv.open(newline="")))
    print("COMPARE_ROWS", len(rows))
    print("FIELDS", rows[0].keys() if rows else [])

    frame_bad = [r for r in rows if r.get("frame_numbers_equal") == "False"]
    shape_bad = [
        r
        for r in rows
        if any(
            r.get(k) == "False"
            for k in (
                "skin_weights_same_shape",
                "parents_same_shape",
                "rest_joints_same_shape",
                "rest_tails_same_shape",
                "bone_transforms_same_shape",
            )
        )
    ]
    print("FRAME_MISMATCH_COUNT", len(frame_bad))
    for r in frame_bad:
        print(
            "FRAME_MISMATCH",
            r["source"],
            r["asset_id"],
            "new_frames=",
            r.get("new_frame_numbers"),
            "oracle_frames=",
            r.get("oracle_frame_numbers"),
            "candidate_count=",
            r.get("new_candidate_frame_count"),
            "new_policy=",
            r.get("new_frame_policy"),
            "oracle_policy=",
            r.get("oracle_frame_policy"),
        )

    print("SHAPE_MISMATCH_COUNT", len(shape_bad))
    print("SHAPE_MISMATCH_ASSETS")
    for r in shape_bad:
        print(
            r["source"],
            r["asset_id"],
            "new_joints=",
            r.get("new_joint_count"),
            "oracle_joints=",
            r.get("oracle_joint_count"),
            "delta=",
            int(r.get("oracle_joint_count") or 0) - int(r.get("new_joint_count") or 0),
            "policy=",
            r.get("new_row_contract_policy"),
        )

    if rows:
        sample = shape_bad[0] if shape_bad else rows[0]
        print("SAMPLE_NEW_KEYS", np.load(sample["new_path"], allow_pickle=True).files)
        print("SAMPLE_ORACLE_KEYS", np.load(sample["oracle_path"], allow_pickle=True).files)

    print("PER_SHAPE_MISMATCH_META")
    for r in shape_bad:
        with np.load(r["new_path"], allow_pickle=True) as new_z, np.load(r["oracle_path"], allow_pickle=True) as old_z:
            new_meta = _json_from_npz(new_z)
            old_meta = _json_from_npz(old_z)
            row_contract = new_meta.get("row_skeleton_contract") or new_meta.get("row_contract") or {}
            if not row_contract:
                row_contract = {
                    k: v
                    for k, v in new_meta.items()
                    if "row" in str(k).lower() or "tail" in str(k).lower() or "drop" in str(k).lower()
                }
            print(
                "META",
                r["source"],
                r["asset_id"],
                "new_shapes=",
                {k: _shape(new_z, k) for k in ("parents", "rest_joints", "rest_tails", "skin_weights", "bone_transforms")},
                "old_shapes=",
                {k: _shape(old_z, k) for k in ("parents", "rest_joints", "rest_tails", "skin_weights", "bone_transforms")},
                "row_contract=",
                json.dumps(row_contract, ensure_ascii=False, sort_keys=True)[:2000],
            )

    print("STRICT_REJECTS")
    for rel in (
        "texverse/strict/rejected.jsonl",
        "objaverse_xl/strict/rejected.jsonl",
        "objaverse_xl_more_anim/strict/rejected.jsonl",
    ):
        p = root / rel
        print("FILE", p)
        if not p.exists() or p.stat().st_size == 0:
            print("EMPTY")
            continue
        for line in p.open():
            print(line.rstrip())

    print("ROOTLESS_REJECTS")
    for p in sorted((root / "rootless_clean").glob("*.rootless_rejected.jsonl")):
        print("FILE", p)
        if p.stat().st_size == 0:
            print("EMPTY")
            continue
        for line in p.open():
            print(line.rstrip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
