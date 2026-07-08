#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import numpy as np


def meta_json(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        raw = z["meta_json"]
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, np.bytes_):
            raw = bytes(raw).decode("utf-8", errors="replace")
        return json.loads(raw)


def main() -> int:
    run = Path(sys.argv[1])
    csv_path = run / "oracle_compare/pass1_oracle_compare.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    checked = 0
    bad = []
    examples = []
    for row in rows:
        if row.get("rest_joints_same_shape") != "False":
            continue
        checked += 1
        delta = int(row["oracle_joint_count"]) - int(row["new_joint_count"])
        meta = meta_json(row["new_path"])
        dropped = int(meta.get("row_contract_dropped_tail_leaf_count", -1))
        names = meta.get("row_contract_dropped_raw_names_first20", [])
        ok = delta == dropped and meta.get("row_contract_policy") == "drop_unweighted_tail_end_leaf_rows"
        if not ok:
            bad.append({"asset_id": row["asset_id"], "delta": delta, "dropped": dropped, "policy": meta.get("row_contract_policy")})
        if len(examples) < 8:
            examples.append({"asset_id": row["asset_id"], "delta": delta, "dropped": dropped, "names": names[:8]})
    print(json.dumps({"checked": checked, "bad": bad, "examples": examples}, ensure_ascii=False, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
