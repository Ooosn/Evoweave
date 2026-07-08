#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ASSET = "0005bab5f144465c9c92efaf13bf0f9d"


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def meta_json(z):
    if "meta_json" not in z.files:
        return {}
    raw = z["meta_json"]
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.bytes_):
        raw = bytes(raw).decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return json.loads(raw)
    return {}


def pose_spread(npz_path: str, sample_vertices: int = 4096):
    with np.load(npz_path, allow_pickle=True) as z:
        rest = z["rest_vertices"].astype(np.float32)
        frames = z["frame_vertices"].astype(np.float32)
        frame_numbers = z["frame_numbers"].astype(int).tolist()
        meta = meta_json(z)
    diag = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
    diag = max(diag, 1e-8)
    n = rest.shape[0]
    if n > sample_vertices:
        idx = np.linspace(0, n - 1, sample_vertices).round().astype(np.int64)
        rest_s = rest[idx]
        frames_s = frames[:, idx]
    else:
        rest_s = rest
        frames_s = frames
    disp = np.linalg.norm(frames_s - rest_s[None, :, :], axis=2) / diag
    centered = frames_s - frames_s.mean(axis=1, keepdims=True)
    desc = centered.reshape(centered.shape[0], -1)
    dists = []
    for i in range(desc.shape[0]):
        diff = desc[i + 1 :] - desc[i]
        if diff.size:
            dists.extend((np.sqrt(np.mean(diff * diff, axis=1)) / diag).tolist())
    dists = np.asarray(dists, dtype=np.float64)
    return {
        "path": npz_path,
        "source": meta.get("source"),
        "frame_sampling_policy": meta.get("frame_sampling_policy"),
        "candidate_frame_count": meta.get("candidate_frame_count"),
        "motion_fps_descriptor_vertices": meta.get("motion_fps_descriptor_vertices"),
        "frame_numbers": frame_numbers,
        "rms_disp_p50_bbox": float(np.percentile(np.sqrt(np.mean(disp * disp, axis=1)), 50)),
        "rms_disp_p95_bbox": float(np.percentile(np.sqrt(np.mean(disp * disp, axis=1)), 95)),
        "disp_p95_p50_frame_bbox": float(np.percentile(np.percentile(disp, 95, axis=1), 50)),
        "disp_p95_p95_frame_bbox": float(np.percentile(np.percentile(disp, 95, axis=1), 95)),
        "pairwise_rms_min_bbox": float(np.min(dists)) if dists.size else 0.0,
        "pairwise_rms_p50_bbox": float(np.percentile(dists, 50)) if dists.size else 0.0,
        "pairwise_rms_p95_bbox": float(np.percentile(dists, 95)) if dists.size else 0.0,
        "pairwise_rms_max_bbox": float(np.max(dists)) if dists.size else 0.0,
    }


def main() -> int:
    run = Path(sys.argv[1])
    oracle_root = Path(sys.argv[2])
    print("RUN", run)

    audit_rows = [r for r in read_jsonl(run / "texverse/pass0_audit/audit.jsonl") if r.get("asset_id") == ASSET]
    print("PASS0_CANDIDATES", len(audit_rows))
    for r in audit_rows:
        print(json.dumps({
            "candidate": r.get("candidate"),
            "usable": r.get("usable"),
            "reject_reasons": r.get("reject_reasons"),
            "max_action_frames": r.get("max_action_frames"),
            "max_bone_fcurves": r.get("max_bone_fcurves"),
            "skinned_vertices": r.get("skinned_vertices"),
            "skin_coverage": r.get("skin_coverage"),
            "max_bones": r.get("max_bones"),
            "actions": r.get("actions", [])[:4],
        }, ensure_ascii=False))

    pass1_rows = [r for r in read_jsonl(run / "texverse/pass1_motionfps_sequence40/manifest.jsonl") if r.get("asset_id") == ASSET]
    print("PASS1_SELECTED")
    for r in pass1_rows:
        print(json.dumps({
            "status": r.get("status"),
            "source_file": r.get("source_file"),
            "pass0_candidate": r.get("pass0_candidate"),
            "pass0_rel_source": r.get("pass0_rel_source"),
            "source_selection_policy": r.get("source_selection_policy"),
            "source_selection_score": r.get("source_selection_score"),
            "source_selection_candidate_count": r.get("source_selection_candidate_count"),
            "source_candidate_count": r.get("source_candidate_count"),
            "source_attempt_count": r.get("source_attempt_count"),
            "npz_path": r.get("npz_path"),
            "reject_reasons": r.get("reject_reasons"),
        }, ensure_ascii=False))

    compare_rows = [r for r in read_csv(run / "oracle_compare/pass1_oracle_compare.csv") if r.get("asset_id") == ASSET]
    print("ORACLE_COMPARE")
    for r in compare_rows:
        print(json.dumps({
            "status": r.get("status"),
            "new_frame_policy": r.get("new_frame_policy"),
            "oracle_frame_policy": r.get("oracle_frame_policy"),
            "new_candidate_frame_count": r.get("new_candidate_frame_count"),
            "new_frame_numbers": r.get("new_frame_numbers"),
            "oracle_frame_numbers": r.get("oracle_frame_numbers"),
            "frame_numbers_equal": r.get("frame_numbers_equal"),
            "new_joint_count": r.get("new_joint_count"),
            "oracle_joint_count": r.get("oracle_joint_count"),
        }, ensure_ascii=False))

    if pass1_rows:
        new_npz = pass1_rows[0].get("npz_path")
        old_npz = oracle_root / "texverse/pass1_motionfps_sequence40/npz" / f"{ASSET}_seq0.npz"
        print("POSE_SPREAD_NEW")
        print(json.dumps(pose_spread(str(new_npz)), ensure_ascii=False, indent=2))
        if old_npz.exists():
            print("POSE_SPREAD_OLD_OFFICIAL")
            print(json.dumps(pose_spread(str(old_npz)), ensure_ascii=False, indent=2))

    print("STRICT_REJECTS")
    for rel in [
        "texverse/strict/rejected.jsonl",
        "objaverse_xl/strict/rejected.jsonl",
        "objaverse_xl_more_anim/strict/rejected.jsonl",
    ]:
        rows = read_jsonl(run / rel)
        print(rel, len(rows), [r.get("asset_id") for r in rows[:10]], [r.get("reasons") for r in rows[:3]])

    print("ROOTLESS_REJECTS")
    for path in sorted((run / "rootless_clean").glob("*.rootless_rejected.jsonl")):
        rows = read_jsonl(path)
        print(path.name, len(rows), rows[:3])

    print("ORACLE_SUMMARY")
    p = run / "oracle_compare/summary.json"
    if p.exists():
        print(p.read_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
