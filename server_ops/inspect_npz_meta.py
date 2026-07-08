#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


IMPORTANT_META_KEYS = [
    "asset_id",
    "sequence_id",
    "source",
    "source_path",
    "archive_path",
    "clip_name",
    "action_name",
    "animation_name",
    "frame_sampling_policy",
    "frame_policy",
    "candidate_frame_count",
    "motion_fps_descriptor_vertices",
    "frame_start",
    "frame_end",
    "action_frame_start",
    "action_frame_end",
    "source_frame_start",
    "source_frame_end",
    "row_contract_policy",
    "row_contract_dropped_raw_count",
    "row_contract_dropped_tail_leaf_count",
    "row_contract_dropped_raw_names_first20",
]


def _decode(v):
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, np.str_):
        return str(v)
    return v


def _meta(z):
    for key in ("meta_json", "metadata_json", "meta", "metadata"):
        if key not in z.files:
            continue
        raw = _decode(z[key])
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    return {}


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        print("NPZ", p)
        with np.load(p, allow_pickle=True) as z:
            print("KEYS", z.files)
            for key in ("frame_numbers", "rest_vertices", "frame_vertices", "parents", "rest_joints", "rest_tails", "skin_weights", "bone_transforms"):
                if key in z.files:
                    arr = z[key]
                    if key == "frame_numbers":
                        print(key, arr.shape, arr.tolist())
                    else:
                        print(key, arr.shape, arr.dtype)
            meta = _meta(z)
            print("META_SELECTED")
            for key in IMPORTANT_META_KEYS:
                if key in meta:
                    print(f"{key}: {meta[key]}")
            print("META_KEYS", sorted(meta.keys()))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
