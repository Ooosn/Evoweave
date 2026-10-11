# Current Dataset Pointer

Status: current as of 2026-10-12.

The workspace-level source of truth is `DATASET_SOURCE_OF_TRUTH.md`. The
current operating platform is HGC.

## Current Dataset

Shared root:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012
```

Final manifests:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

Counts:

```text
base:      16,697
UniMate:    2,290
train:     18,018
valid:        969
total:     18,987
quarantine:    81 (79 train, 2 valid)
```

## Storage Contract

This dataset uses the `manifest-paths` layout. This version creates no NPZ
copies: base rows reference the July HGC base, and accepted additions reference
the October 1 NPZ root. Those old roots remain live backing storage. Always
consume each accepted manifest row's `path`; never rebuild membership by
scanning NPZ directories. Original NPZ files and historical manifests are intact.

## Screening Contract

The final bbox threshold remains:

```text
target_joint_mesh_bbox_diag_consistency >= 0.4
```

The final stage now also checks every retained target joint in every stored
frame using that frame's mesh bbox center and maximum half-extent times 1.1.
Every normalized coordinate must be in inclusive `[-1,1]`. Any violating
coordinate rejects the whole asset, including inactive-skin joints and heldout
frames. The gate does not clip labels or change the NPZ schema/coordinates.

The 19,068-row input passed the earlier independent validator, hard audit,
and bbox recomputation. This new all-frame screen read every input with zero
errors and isolated exactly 81 assets. All 18,987 retained rows have zero
coordinate violations; maximum absolute values are 0.9996331334 (train) and
0.9956439734 (validation).

UniMate Pass1 requires Blender `action_frames >= 40`; it does not impose a
duration-in-seconds rule.

## Provenance

Current immutable receipt:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/provenance/coordinate_bounds_receipt.json
```

The `.READY` marker is verified. The receipt pins both accepted manifest hashes
and both quarantine indexes under `quarantine/normalized_joint_bounds/`.
Local copies and the 81-row summary are in
`server_ops/outputs/coordinate_bounds_hgc_20261012/`.

Original input-version summary and metadata archive:

```text
/home/wangyy/jobs/unimate_pilot_20261001/results
```

The 1,632 retained GitHub-origin additions with `mapped_missing` license
evidence require upstream license resolution before redistribution.

## Storage Cleanup

On 2026-10-02, rebuildable intermediates, superseded checkpoints, and the
verified node-local duplicate overlay were removed. The then-current manifests
and all 19,068 referenced NPZ files were revalidated before and after deletion.
Raw source roots, the current base/addition NPZ stores, provenance, metadata
archives, and the current flat checkpoint remain preserved. See:

```text
/home/wangyy/jobs/storage_cleanup_20261001/results/cleanup_receipt.json
```
