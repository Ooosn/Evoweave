# Current Dataset Pointer

Status: current as of 2026-10-13.

The workspace-level source of truth is `DATASET_SOURCE_OF_TRUTH.md`. The
current operating platform is HGC.

## Current Dataset

Shared root:

```text
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013
```

Final manifests:

```text
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

Counts:

```text
base:      16,683
UniMate:    2,286
train:     18,002
valid:        967
total:     18,969
new quarantine: 18 (16 train, 2 valid)
prior quarantine: 81 (79 train, 2 valid), still excluded
```

## Storage Contract

This dataset uses the `manifest-paths` layout. This version creates no NPZ
copies: base rows reference the July HGC base, and accepted additions reference
the October 1 NPZ root. Those old roots remain live backing storage. Always
consume each accepted manifest row's `path`; never rebuild membership by
scanning NPZ directories. Original NPZ files and historical manifests are intact.

Each row also carries `root_rt` cache metadata that must be consumed by the
RT-aware loader. Its 75 small cache shards total 60,761,423 bytes. The unchanged
NPZ coordinates are not already RT-aligned. Use consumer commit `6ea3be2` or
later; legacy readers that discard row metadata are not valid consumers.

## Screening Contract

The final bbox threshold remains:

```text
target_joint_mesh_bbox_diag_consistency >= 0.4
```

The RT overlay removes only the retained root's rigid motion using proper
polar rotation and a root-position pivot. It preserves scale/shear residuals,
all target joints, topology and weights. Seventy-two of the 73 nonrigid-root
assets pass this standard process; the remaining one is excluded for its
independent hidden-source-frame problem.

One center/scale comes from the selected aligned input meshes' union bbox,
with padding 1.1. All single/pair stored-frame windows were checked, bounding
any input subset before extra rotation augmentation. The accepted maximum
absolute joint coordinate is 0.9973959923. New rejection groups are 8 coordinate
bound failures, 9 numerical root consistency failures, and 1 hidden frame;
4 numerical failures also fail RT roundtrip checks. Whole assets are isolated,
never clipped, frame-pruned, or rescaled using GT. No source NPZ is rewritten.

The October 12 input version had already isolated 81 of the 19,068 merged
assets using per-frame source-space bounds. Those exclusions are unchanged.
The new revision scanned all remaining 18,987 assets without processing errors.

UniMate Pass1 requires Blender `action_frames >= 40`; it does not impose a
duration-in-seconds rule.

## Provenance

Current immutable receipt:

```text
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013/provenance/root_rt_receipt.json
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013/provenance/deployment_receipt.json
```

The `.READY` and `.CONSUMER_READY` markers are verified. Receipts pin accepted
manifest/cache hashes and both new quarantine indexes under
`quarantine/root_rt_quality/`. Acceptance includes 48 CPU tests and 80 real
loader checks on 24 assets. Local receipts and the 18-row indexes are in
`server_ops/outputs/root_rt_publish_20261013/`. The source of truth retains the
old 81-row coordinate-screening provenance as historical input evidence.

Original input-version summary and metadata archive:

```text
/home/wangyy/jobs/unimate_pilot_20261001/results
```

GitHub-origin additions with `mapped_missing` license evidence still require
upstream license resolution before redistribution; the old input snapshot's
license counts must not be assumed to be recounted current counts.

## Storage Cleanup

On 2026-10-02, rebuildable intermediates, superseded checkpoints, and the
verified node-local duplicate overlay were removed. The then-current manifests
and all 19,068 referenced NPZ files were revalidated before and after deletion.
Raw source roots, the current base/addition NPZ stores, provenance, metadata
archives, and the current flat checkpoint remain preserved. See:

```text
/home/wangyy/jobs/storage_cleanup_20261001/results/cleanup_receipt.json
```
