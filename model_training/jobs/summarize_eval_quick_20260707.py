#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path


def load_json_lines(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rounded(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


def main() -> int:
    out = Path(sys.argv[1])
    print(f"out={out}")
    for name in [
        "ce_val.json",
        "ce_test.json",
        "eos_test.json",
        "generation_test.json",
        "generation_test.out",
        "diagnose_test.json",
        "diagnose_test.out",
    ]:
        path = out / name
        print(f"file {name} exists={path.exists()} size={path.stat().st_size if path.exists() else None}")

    for name in ["ce_val.json", "ce_test.json", "eos_test.json", "diagnose_test.json"]:
        path = out / name
        if path.exists():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"{name} parse_error={exc}")
                continue
            print(f"{name} json={json.dumps(obj, ensure_ascii=False)[:2000]}")

    rows = []
    gen_json = out / "generation_test.json"
    gen_obj = None
    if gen_json.exists():
        try:
            obj = json.loads(gen_json.read_text(encoding="utf-8"))
            gen_obj = obj
            if isinstance(obj, list):
                rows.extend(obj)
            elif isinstance(obj, dict):
                for key in ("rows", "results", "samples"):
                    if isinstance(obj.get(key), list):
                        rows.extend(obj[key])
                print(f"generation_test.json summary={json.dumps(obj.get('summary', obj), ensure_ascii=False)[:5000]}")
        except Exception as exc:
            print(f"generation_test.json parse_error={exc}")

    if not rows:
        rows = load_json_lines(out / "generation_test.out")

    print(f"generation_rows={len(rows)}")
    for model in sorted({row.get("model") for row in rows}):
        model_rows = [row for row in rows if row.get("model") == model]
        ok_rows = [row for row in model_rows if row.get("detokenize_ok")]
        f1 = [float(row["topology_f1"]) for row in ok_rows if row.get("topology_f1") is not None]
        chamfer = [
            float(row["joint_chamfer_mean"])
            for row in ok_rows
            if row.get("joint_chamfer_mean") is not None
        ]
        j2j = [float(row["j2j"]) for row in ok_rows if row.get("j2j") is not None]
        b2b = [float(row["b2b"]) for row in ok_rows if row.get("b2b") is not None]
        count_abs = [
            abs(float(row["joint_count_error"]))
            for row in ok_rows
            if row.get("joint_count_error") is not None
        ]
        count_signed = [
            float(row["joint_count_error"])
            for row in ok_rows
            if row.get("joint_count_error") is not None
        ]
        print(
            "MODEL_SUMMARY",
            json.dumps(
                {
                    "model": model,
                    "n": len(model_rows),
                    "ok": len(ok_rows),
                    "ok_rate": rounded(len(ok_rows) / len(model_rows), 4) if model_rows else None,
                    "topology_f1_mean": rounded(mean(f1)),
                    "topology_f1_median": rounded(statistics.median(f1) if f1 else None),
                    "topology_f1_min": rounded(min(f1) if f1 else None),
                    "joint_chamfer_mean": rounded(mean(chamfer)),
                    "j2j_mean": rounded(mean(j2j)),
                    "b2b_mean": rounded(mean(b2b)),
                    "joint_count_abs_mean": rounded(mean(count_abs), 4),
                    "joint_count_signed_mean": rounded(mean(count_signed), 4),
                },
                ensure_ascii=False,
            ),
        )
        for row in model_rows:
            if row.get("detokenize_ok") and row.get("topology_f1") is not None and row["topology_f1"] >= 0.95:
                continue
            print("BAD_ROW", json.dumps(row, ensure_ascii=False))

    if gen_obj and isinstance(gen_obj, dict):
        nested_rows = []
        for key in ("rows", "results", "samples"):
            if isinstance(gen_obj.get(key), list):
                nested_rows = gen_obj[key]
                break
        for model in ("dynamic", "static_base", "dynamic_explicit_tree"):
            flat = []
            for sample in nested_rows:
                item = sample.get(model)
                if not isinstance(item, dict):
                    continue
                metrics = item.get("metrics", {})
                topology = metrics.get("topology", {}) if isinstance(metrics, dict) else {}
                official = metrics.get("official", {}) if isinstance(metrics, dict) else {}
                flat.append(
                    {
                        "index": sample.get("index"),
                        "target_joint_count": sample.get("target_joint_count"),
                        "detokenize_ok": item.get("detokenize_ok", metrics.get("detokenize_ok")),
                        "has_eos": item.get("has_eos"),
                        "hit_max_without_eos": item.get("hit_max_without_eos"),
                        "pred_joint_count": metrics.get("pred_joint_count"),
                        "joint_count_error": metrics.get("joint_count_error"),
                        "joint_chamfer_mean": (metrics.get("joint_chamfer") or {}).get("mean")
                        if isinstance(metrics.get("joint_chamfer"), dict)
                        else None,
                        "j2j": official.get("j2j"),
                        "b2b": official.get("b2b"),
                        "topology_f1": topology.get("edge_f1"),
                        "error": item.get("error"),
                        "path": sample.get("path"),
                    }
                )
            if not flat:
                continue
            ok = [row for row in flat if row.get("detokenize_ok")]
            f1 = [float(row["topology_f1"]) for row in ok if row.get("topology_f1") is not None]
            chamfer = [
                float(row["joint_chamfer_mean"])
                for row in ok
                if row.get("joint_chamfer_mean") is not None
            ]
            j2j = [float(row["j2j"]) for row in ok if row.get("j2j") is not None]
            b2b = [float(row["b2b"]) for row in ok if row.get("b2b") is not None]
            count_abs = [
                abs(float(row["joint_count_error"]))
                for row in ok
                if row.get("joint_count_error") is not None
            ]
            hitmax = [row for row in flat if row.get("hit_max_without_eos")]
            print(
                "NESTED_MODEL_SUMMARY",
                json.dumps(
                    {
                        "model": model,
                        "n": len(flat),
                        "ok": len(ok),
                        "ok_rate": rounded(len(ok) / len(flat), 4),
                        "hit_max_without_eos": len(hitmax),
                        "topology_f1_mean": rounded(mean(f1)),
                        "topology_f1_median": rounded(statistics.median(f1) if f1 else None),
                        "topology_f1_min": rounded(min(f1) if f1 else None),
                        "joint_chamfer_mean": rounded(mean(chamfer)),
                        "j2j_mean": rounded(mean(j2j)),
                        "b2b_mean": rounded(mean(b2b)),
                        "joint_count_abs_mean": rounded(mean(count_abs)),
                    },
                    ensure_ascii=False,
                ),
            )
            for row in flat:
                if row.get("detokenize_ok") and row.get("topology_f1") is not None and row["topology_f1"] >= 0.9:
                    continue
                print("NESTED_BAD_ROW", json.dumps(row, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
