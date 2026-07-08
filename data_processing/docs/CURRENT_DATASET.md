# Current Dataset Pointer

Status: current as of 2026-07-07.

This file records the current production/training dataset pointer for the
data-processing module. The workspace-level source of truth is
`DATASET_SOURCE_OF_TRUTH.md`.

## Current Dataset

Current server root:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706
```

Final manifest directory:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests
```

Train:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
```

Valid:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

Referenced NPZ files live under:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/rootless_clean/npz
```

## Screening Contract

Final skeleton/mesh bbox-consistency cutoff:

```text
target_joint_mesh_bbox_diag_consistency >= 0.4
```

Known final-stage counts:

```text
rootless strict rows before final bbox screening: 16997
accepted after final bbox screening: 16760
rejected by final bbox screening: 237
```

Final manifests are train/valid only. Original test rows are merged into train
at the final bbox-consistency screening stage.

## Do Not Use As Current

Do not use `reports/latest_histograms_20260706/` as a current-data pointer.
That report folder is a historical snapshot and originally referenced the older
2026-07-04 rootless rebuild.
