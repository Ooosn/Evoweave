# Evoweave Dataset Source of Truth

Status: current as of 2026-10-13.

This file is the workspace-level source of truth for the current HGC training
dataset. Do not infer the current dataset from report folders, NPZ directory
scans, or `latest_*` names outside the recorded results directory.

## Current Training Dataset

The current version is the retained-root rigid-motion cache overlay:

```text
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013
```

Its immutable manifests are:

```text
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

| Split | Base | UniMate | Total |
| --- | ---: | ---: | ---: |
| Train | 15,831 | 2,171 | 18,002 |
| Valid | 852 | 115 | 967 |
| Total | 16,683 | 2,286 | 18,969 |

This revision inspected all 18,987 rows of the October 12 input with zero
processing errors and isolated another 18 whole assets: 16 train and 2 valid.
Eight violate aligned input-window coordinate bounds, nine fail the numerical
root-position consistency gate, and one has a hidden source mesh at its first
sampled frame. Four of the nine numerical failures also fail float32 RT
roundtrip checks; they are not four additional exclusions. The previous 81
exclusions remain excluded. No joints/frames were pruned and no labels clipped.

Of the 73 assets with nonrigid relative root matrices, 72 pass the same proper
polar-rotation and root-pivot algorithm. Only the independently identified
hidden-frame asset is excluded. Scale and shear residuals are preserved.

Every accepted row preserves its original fields and backing NPZ `path`, and
adds a `root_rt` descriptor. **Do not discard that descriptor or interpret the
unchanged NPZ arrays as already RT-aligned.** The 75 cache shards total
60,761,423 bytes. Source NPZs and historical manifests were not modified.

The data loader applies cached inverse RT to mesh and every GT frame, selects
input frames in aligned space, and uses one selected-input mesh-union bbox
center/scale with padding 1.1. All single/pair frame windows were checked; this
bounds any subset of stored frames before additional rotation augmentation.
The accepted maximum normalized joint coordinate is 0.9973959923. Root drift
and float32 RT roundtrip limits are `1e-5` of the corresponding frame mesh bbox
diagonal. Additional global augmentation is checked again online, not covered
by the cached canonical-coordinate certificate.

The RT-aware flat and virtual-frame loaders require consumer commit
`6ea3be2f5eb94c342a9446eb6e43b755ac933b33` or later. The current implementation
checkout is `C:/Users/namew/.codex/worktrees/skeleton-throughput-20261011/evoweave`
on `codex/root-rt-alignment-20261013`. HGC has the verified checkout at
`/work/wangyy/root_rt_publish_20261013/consumer_code` and a durable Git bundle at
`/home/wangyy/jobs/root_rt_publish_20261013_v1/root_rt_consumer.bundle`.
Legacy Puppeteer and the old shared-structure wrapper reject this overlay.
The cache uses source skeleton metadata; it is not a GT-free inference method.
No new model training was launched with this revision.

Current manifest SHA-256:

```text
train: 653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27
valid: fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a
```

Publication evidence under the new root:

```text
provenance/root_rt_receipt.json
SHA-256: 169abe6f564f0e150d9c99fcdbf24aa3d584425cf55f4de26a36a0ffc4b46a9f
provenance/deployment_receipt.json
SHA-256: ca91f2085524e548292f7a1a2c729d24c4f12e45950f0a9bfe6c11f2a327a926
```

Both `.READY` and `.CONSUMER_READY` were verified. Acceptance includes 48 CPU
unit/regression tests, 80 real-loader checks on 24 boundary/nonrigid/random
assets, and a real hidden-asset rejection plus visible-asset export. Local
receipts and the complete 18-row exclusion indexes are under
`server_ops/outputs/root_rt_publish_20261013/`; remote indexes are under the
new root's `quarantine/root_rt_quality/` directory.

## October 12 Input Snapshot

Everything below records the unchanged input snapshot and its provenance.
These old manifests and counts are historical inputs, not the current RT
training membership or normalization contract.

The current dataset is the HGC rootless-v3/UniMate manifest overlay after the
all-frame normalized-joint coordinate screen. It excludes 81 assets from the
2026-10-01 version: 79 train and 2 validation assets. Original NPZ files and
historical manifests remain unchanged.

Canonical shared root:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012
```

Final manifest directory:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quality_distributions/rootless_bbox_consistency/final_manifests
```

Training manifest:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
```

Validation manifest:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

## Storage Layout

This is a `manifest-paths` overlay, not a self-contained NPZ directory:

- 16,697 base rows continue to reference the screened HGC rootless-v3 base;
- 2,290 UniMate rows reference unchanged NPZ files in the 2026-10-01 root;
- the current version contains manifests, screening evidence, and quarantine
  indexes only; no NPZ files were copied, moved, modified, or deleted;
- every consumer must read each manifest row's `path`;
- consumers must not rescan or reconstruct rows from `rootless_clean/npz`.

The final split is:

```text
train: 18,018 = 15,843 base + 2,175 UniMate
valid:    969 =    854 base +   115 UniMate
total: 18,987 = 16,697 base + 2,290 UniMate
```

The retained UniMate subset contains 658 Sketchfab-origin rows and 1,632
Objaverse-XL/GitHub-origin rows. The removed assets comprise 79 base rows and
2 UniMate rows. Both the July base NPZ root and the October 1 NPZ root remain
live backing storage for accepted rows; do not delete those roots merely
because their manifest versions are historical.

## Selection And Screening

The UniMate candidate contract was:

- high-confidence unseen relative to Evoweave's earlier screening universe;
- exclusions removed;
- one skeleton tree;
- at most 256 source joints and 300,000 vertices;
- at least one animation that drives the skeleton;
- one selected action per asset.

There was no duration-in-seconds filter. Pass1 uses Blender
`action_frames >= 40`; `1.625 s` is not a dataset rule.

All accepted additions passed sequence dedupe, Pass1 readability, rootless
rewrite, strict target checks, and the final skeleton/mesh bbox consistency
threshold:

```text
target_joint_mesh_bbox_diag_consistency >= 0.4
```

The final bbox-screening stage now also checks every retained target joint in
every stored frame, including heldout frames and inactive-skin joints:

```text
center = (mesh_min + mesh_max) / 2
scale = 1.1 * max((mesh_max - mesh_min) / 2)
max(abs((target_joints - center) / scale)) <= 1.0
```

Bounds are computed separately for each frame from retained mesh vertices only.
No label clipping, joint/frame pruning, or target-derived normalization is used.
Any violation isolates the whole asset. Invalid/nonfinite geometry fails closed.
This containment guarantee uses padding 1.1; consumers using a different
normalization must not assume the same bound without revalidation.

The full 2026-10-12 scan checked all 19,068 input assets with zero metric/read
errors. The exact 81-asset quarantine matched the prior independent inventory.
Accepted rows have zero coordinate violations; their largest absolute values
are 0.9996331334 (train) and 0.9956439734 (validation).

The prior 2026-10-01 combined build was checked after materialization:

```text
independent NPZ validation invalid: 0 / 19,068
final audit hard failures:          0 / 19,068
missing / duplicate / orphan paths: 0 / 0 / 0
combined bbox rejections:           0 / 19,068
```

That pre-coordinate-screening audit recorded 2,233 soft-flag rows. These are
historical input counts, not recounted counts for the current subset:
2,192 include duplicate joint coordinates, 1,267 include zero-length
parent-child edges, and 54 include target-alignment warnings. Categories can
overlap. This revision removes only the explicitly authorized coordinate-bound
violations; it introduces no other soft-flag removal rule.

## Quarantine

Rejected assets are excluded from both current accepted manifests and indexed
with original paths and rejection evidence under:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quarantine/normalized_joint_bounds/train_manifest.jsonl
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/quarantine/normalized_joint_bounds/valid_manifest.jsonl
```

The same rows are recorded in the final stage's `train_rejected.jsonl`,
`valid_rejected.jsonl`, and `rejected_manifest.jsonl`. Do not use rejected or
historical manifests as current training inputs.

## License Provenance

Accepted UniMate rows carry the following recorded license values:

```text
CC-BY:         638
CC-BY-NC:       14
CC-BY-NC-SA:     4
CC-BY-SA:        2
mapped_missing: 1,632
```

`mapped_missing` means the GitHub source mapping did not provide explicit
per-object license evidence. It does not mean public redistribution is
authorized. Resolve upstream repository/object licenses before redistribution
or commercial use, and preserve the per-row provenance fields in the manifests.

## Integrity And Recovery

Canonical coordinate-screening receipt:

```text
/home/wangyy/evorig/evoweave_rootless_v3_unimate_v1_coordbounded_hgc_20261012/provenance/coordinate_bounds_receipt.json
SHA-256: 31cfe31c49ca569c4651e24be7c218640ff7f34cb586df5e5ff28a255e54d311
```

Current manifest SHA-256 values:

```text
train: cc2d417bf3537df290ec489e9f3bc0fbcc5b2cfa9b6eb3f31706baac286689c3
valid: 545e3211e6680c9841cb5f895a0ab2ffb1748949d445c67803d45463b61df7ff
```

The new root has a verified `.READY` marker. The acceptance checks establish an
exact, disjoint accepted/quarantined partition and unchanged input hashes and
per-row source/license metadata. Local receipts and the compact quarantine
index are under `server_ops/outputs/coordinate_bounds_hgc_20261012/`.

Original UniMate expansion build summary:

```text
/home/wangyy/jobs/unimate_pilot_20261001/results/latest_unimate_dataset_build_summary.json
```

Metadata archive:

```text
/home/wangyy/jobs/unimate_pilot_20261001/results/unimate_v1_hgc_20261001_shared_manifests.tar.zst
```

The old build summary and metadata archive describe the 19,068-row input
version, not the current filtered manifests. The per-file SHA-256 manifest for
all 2,292 original addition NPZ files remains under
`evoweave_rootless_v3_unimate_v1_hgc_20261001/provenance/addition_npz_sha256.txt`.
The current accepted subset references 2,290 of those unchanged files.

The completed node-local hww03 overlay and integration scratch were retired on
2026-10-02 after the shared dataset was revalidated. Recovery now uses the
shared metadata archive above plus the preserved UniMate raw/provenance archive:

```text
/home/wangyy/jobs/storage_cleanup_20261001/results/unimate_expansion3706_raw_originals_20261001.tar.zst
```

The raw archive contains 3,706 downloaded expansion GLBs and has SHA-256:

```text
64747c403ee00c6f6b5d3ca6ad5a2908cbcd962efda8c13395770fc5f6e67528
```

The original 500 pilot GLBs were already absent from hww02 before cleanup. The
verified `full500_128933799_hww02_20261001T122957.tar.zst` processed archive is
retained as the closest surviving pilot recovery artifact; it must not be
described as a raw-original archive.

The verified cleanup receipt is:

```text
/home/wangyy/jobs/storage_cleanup_20261001/results/cleanup_receipt.json
```

## Historical Pointers

The former Westlake 2026-07-06 rootless-v3 root, the HGC 2026-07-14 base
manifests, and the 2026-10-01 combined manifests remain immutable history.
None is the current combined training pointer. The HGC base and October 1
NPZ stores still back the current accepted manifests and must remain intact.
`reports/latest_histograms_20260706/` is also historical.

## Update Rule

When the production dataset changes, update all of these together:

- this file;
- `PROJECT_MAP.md`;
- `data_processing/docs/CURRENT_DATASET.md`;
- the matching receipt under `server_ops/outputs/`;
- model-training job docs that point to final manifests.
