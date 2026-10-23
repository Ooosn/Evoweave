# Real Query, Single Target, Current Data Pipeline

## Authorization And Scope

On 2026-10-24 JST the user authorized a fresh two-H100 experiment with the
original real-query-first architecture, one randomly selected target pose per
asset visit, the existing fixed 5,000-asset training set, and 100,000 cumulative
asset exposures. This means 20 nominal passes, not 100,000 optimizer updates.
The user explicitly requested retaining the existing allocation and using SSH
for control; releasing that allocation has not been authorized.

This is not a continuation of the structure/XYZ-filling model and does not yet
implement the proposed second-stage cross-pose joint-position decoder.

## Preserved Data Contract

- Preserve the exact 5,000-row manifest (SHA-256
  `a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`).
- Preserve the disjoint 967-row validation manifest, heldout-frame partition,
  dynamic 2..24 input frames, and current surface sampling.
- Keep complete manifest rows and their retained-root RT descriptors.
- Apply the current RT alignment and one bbox normalization over the actual
  selected input meshes, with padding 1.1. Do not normalize each pose or the
  target pose independently. Changing the target within the same input set
  must not change normalization, geometry, or labels.
- Do not clip coordinates, alter GT, or inherit the structure-filling model's
  augmentation or loss settings without an explicit comparison decision.

## Model And Supervision

All selected real poses participate in the existing surface encoder and
cross-pose motion attention. The current loader already chooses a uniformly
random query from eligible raw frames, places it in slot0, and selects motion
FPS/random evidence relative to it. Use that query as the single training
target; do not draw a second target from the resulting FPS-biased input set.
Feed its final 1,024 condition tokens into the original complete UniRig XYZ
autoregressive decoder and supervise only that target's full sequence.

There are no virtual-frame slots, final virtual/real fusion, auxiliary pure-tree
loss, or XYZ-fill head in this experiment. Non-target poses still receive
gradients through cross-pose attention. Preserve official pretrained surface
and AR initialization and audit common initial motion tensors against B.

The older `ReferenceConditionRig` control iterates over every target pose; it
is not this single-target training recipe and must not be reused unchanged.

## Comparison And Budget

The principal quality reference is the strongest previously evaluated B
checkpoint, not A. Reuse the fixed 128-asset validation panel with eight
target poses per asset, original geometry-aware skeleton scoring, and failure
scores of zero. Report GT52 and non52 strata separately.

Record input-frame exposures separately from supervised target-frame exposures:
the latter must equal asset exposures in training. Also record optimizer
updates, elapsed time, target choices, raw frame identities, and visit seeds.
Single-target versus all-target supervision and B's auxiliary loss remain
declared differences; this comparison alone does not isolate virtual queries.

Audit the actual best-B learning-rate lineage before publishing the final
training configuration. B used a 20k OneCycle prefix, constant 1e-4 through
60k, then 3e-5; the separate 40k-to-50k low-rate branch is not its best-65k
lineage. Do not accidentally restart or stretch the old schedule.

## Prior Operation And Resource Boundary

The structure/XYZ-filling continuation completed 160,000 asset exposures,
2,068,624 supervised frames and 3,355 updates on both ranks at
2026-10-23T11:58:37.591084Z. Its result reports `complete=true`, `passed=true`,
and `quality_accepted=false`; both final model fingerprints agree. Preserve
its source and all outputs. Result SHA-256:
`d200706c995a1f3643c627e8902e7a74874d8cd15b35878ce46758066fee57e7`.

At 2026-10-23T17:08Z, allocation 130309880 still held pcg01 physical H100s 5/6,
both with zero allocated GPU memory. Its keeper is an in-job sleep loop without
a command interface. A fresh read-only `qrsh -inherit` attempt at 17:12Z was
rejected by execd. Do not release the keeper or copy environment variables to
pretend an ordinary SSH child belongs to this job. A training launcher must
verify actual scheduler membership and CPU/GPU limits before workload imports.

Preparation and CPU tests are allowed. Exact GPU preflight/training release,
source/config hashes, capacity receipts and a usable execution entry remain
to be published in `state/current.json`. No new training has started.

## Completed Implementation Review

The new standalone model, current-loader adapter, B-lineage learning-rate
schedule, original-metric evaluation adapter, guarded trainer and resource
supervisor are implemented. All22 focused CPU tests pass, including original
encoder/logit/gradient parity, exact common tensor preservation, random raw-query
selection, the unchanged B20k sampling prefix, separate input/target counts,
2100 exact-budget updates, and fail-closed stage/two-rank receipt checks.
Independent review also ran26 existing regression tests before integration.

The common encoder recipe remains B's12blocks/96registers/8head motion bias,
with no extra final condition norm. This restores the original query routing,
not every historical command-line default. There is no auxiliary structure
loss or training-time pruning. The final LR phase remains3e-5 through100k.

These are CPU implementation results only. Actual official-weight fingerprint
verification, two-rank CUDA execution, capacity measurement and training have
not run. See `analysis/legacy_single_target_preparation_review_v1.json` for
file hashes, tests and the remaining execution-entry blocker.
