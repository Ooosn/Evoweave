# Evoweave Dataset Source of Truth

Status: current as of 2026-07-07.

This file is the workspace-level source of truth for the current training
dataset. Do not infer the current dataset from report folder names such as
`latest_*`; report folders are evidence snapshots only.

## Current Training Dataset

The current dataset is the 2026-07-06 rootless v3 rebuild after final
skeleton/mesh bbox-consistency screening.

Server root:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706
```

Final manifest directory:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests
```

Training manifest:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
```

Validation manifest:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

NPZ rows referenced by the manifests live under:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/rootless_clean/npz
```

## Final Screening

Final screening stage:

```text
quality_distributions/rootless_bbox_consistency
```

Final skeleton/mesh bbox consistency threshold:

```text
target_joint_mesh_bbox_diag_consistency >= 0.4
```

Known final-stage counts from the 2026-07-06 report:

```text
rootless strict rows before final bbox screening: 16997
accepted after final bbox screening: 16760
rejected by final bbox screening: 237
```

Final manifests are train/valid only. Original test rows are merged into train
at the final bbox-consistency screening stage.

## Historical Warning

`reports/latest_histograms_20260706/` is a historical snapshot and is not the
current training dataset pointer. Its README originally pointed to the older
2026-07-04 rootless rebuild:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_full_20260704/rootless_clean
```

That old path must not be used as the current training data unless this file is
explicitly updated to say so.

## Update Rule

When the production dataset changes, update all of these together:

- this file;
- `PROJECT_MAP.md`;
- `data_processing/docs/CURRENT_DATASET.md`;
- `server_ops/outputs/<stage>/README.md`;
- model-training job docs that point to final manifests.
