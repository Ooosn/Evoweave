# Original Baseline Matched Evaluation

The human requested checking the surviving original single-real-query baseline
checkpoint and evaluating it on the current small panel, then selected local
execution. HGC ID/XYZ training continues unchanged. This is inference only,
not retraining or a switch to the historical A virtual-frame model.

## Bound Inputs

- July checkpoint: `checkpoint_sample_80000.pt`, 80,016 exposures, 1,667 steps;
  SHA256 `4cdde3cac8b8b83d690128785065bcde6b6186044bcbdeac6f701acff3b7bb55`.
- All 1,117 saved state tensors strictly restored on CPU with exact values.
  A CPU-only derived YAML selects SDPA for construction; the original YAML
  remains unchanged. Local Windows inference explicitly uses SDPA because
  FlashAttention2 is unavailable. No missing parameter is silently initialized.
- Current validation panel: 16 assets, eight input frames per asset, scored
  slots 0/2/4/6, 64 views. These are selected slots, not raw animation frame IDs.
  Panel SHA256: `da1b4eca4df249fa14f03af2b78d0b075b8fac23eec2b60141593be3f98fbe6f`.
- Same original continuous GT, root-RT alignment, selected-input mesh-union
  normalization, no clipping or pruning. Each scored query moves to slot0;
  all per-frame fields move together, with the original union center/scale.
- Source manifests are not rewritten. A hash-verified file resolver changes
  storage locations only; root-RT asset hashes use original canonical paths.
  Tests confirm exact geometry and frame selection after relocation.

The old checkpoint was trained with24 frames on the historical15920-row data;
this is its performance under the current matched8-frame protocol, not a replay
of its historical preprocessing or a matched-training-budget causal ablation.
Its original query-dependent surface sampling is retained and recorded.

## Scoring And Gates

Use the existing nearest-coordinate directed-edge F1 against original GT,
with failed/non-terminating generations scored zero. There is no new distance
threshold or one-to-one assignment. Report per-view values, equal-asset means,
52/non52 groups, EOS failures, and cross-frame equality of generated numbered
parent lists. That last metric is not a proof of semantic bone identity or
unordered-tree isomorphism. Exact parent-list agreement with source ordering
must not be substituted for spatial edge F1.

Order: real CPU strict restore and128-view GT/reference roundtrip audit;
primary acceptance; eight-view GPU smoke; primary acceptance; full64 views.
The greedy original grammar retains a2048-new-token limit and no forced EOS,
count guidance, snapping, or optimizer. The3076-position decoder fits the
1024-condition +2-prefix +2048-new-token budget.

Local artifacts are on verified D: NVMe SSD under
`server_ops/outputs/original_baseline_local_20261025/`. Dependencies are isolated;
the shared mygs environment and unrelated GPU tasks are not modified/stopped.
The local supervisor bounds CPU affinity,32GiB RAM,45% per-process CUDA memory
and85% aggregate VRAM, and waits for unrelated Python GPU jobs to finish.

The state component `active_operation.baseline_matched_evaluation` authorizes
exact stages. Separate operation-entrypoint bindings preserve the existing HGC
training entrypoint. Never pull local changes into live HGC `cpu4_code`.
