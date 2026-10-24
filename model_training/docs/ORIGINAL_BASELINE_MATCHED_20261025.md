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

## Local CPU Acceptance

The second CPU audit completed with exit0 and no visible CUDA device. It
strictly restored all1117 saved tensors, matched all128 raw frames and GT token
streams for16 assets to B65, and verified all64 query permutations. Peak process
tree RSS was6230614016bytes and cleanup completed. The original legacy codec
changes parent identity on2 assets/16 frames; this is recorded, not repaired or
substituted into the scoring GT. Original continuous GT remains the reference.
The8-view GPU smoke completed with exit0, no parse/EOS failure and no parameter
update. Original inputs/config/reference hashes remained unchanged; peak
process-tree RSS was6244065280bytes and cleanup completed. The supervisor's
last aggregate GPU sample was15412/32760MiB; it is not a recorded peak.
Primary accepted execution and resource safety, not model quality, and released
the full64-view evaluation with unchanged computation and configuration.

## Completed Comparison

All16 validation assets and64 scored views completed. Primary independently
recomputed every saved-view F1 from predicted coordinates/parents and original
GT, with maximum error0. Failed generations remain zero in equal-asset means.

| Checkpoint | All16 F1 | Non52 (11 assets) F1 | Failed views |
| --- | ---: | ---: | ---: |
| Original single-query80k | 0.5828245545 | 0.4490609678 | 7/64 |
| Virtual-frame B65k | 0.6819730815 | 0.5609033102 | 2/64 |
| Current ID/XYZ70k | 0.6380434428 | 0.4926695560 | 0/64 |

The current model has one capacity-only EOS despite all64 views being parsed;
this must not be called64 naturally terminated generations. Original baseline
failures all reached2048 new tokens without EOS and all belong to non52 assets.
Its gt52 F1 is0.8771044451. Four-view numbered-parent/count consistency is5/16
(non52:3/11), versus current70k9/16 (non52:4/11). Consistency is not correctness:
current70k exact source-parent matches are24/64, and only4/44 on non52 views.

The current70k checkpoint exceeds original80k by0.0552188883 F1 but remains
0.0439296387 belowB65k. Training data, frames and budgets differ; this does not
isolate an architectural or consistency-loss effect. The inference protocol is
matched; WindowsBF16SDPA is explicitly not bitwise identical toH100Flash2.

The live consistency coefficient remains1.0 at temperature1. Total loss is
ordinary GT CE plus1.0 times leave-one-out detached structural soft-target CE.
Only ID, parentID, BRANCH and EOS positions participate; XYZ does not. KL is a
diagnostic, not another optimized term. At steps1505..1554 (71680..74032 asset
exposures),50-update means were mainCE0.6988943201 and consistencyCE0.0516035520:
6.8759% of the summed scalar loss, not the gradient contribution. These are
teacher-forced distributions, not a free-generation consistency guarantee.

Result directory: `D:/evoweave/server_ops/outputs/original_baseline_local_20261025/runs/evaluate_v1`.
Summary SHA256: `a299e79e4b883b0cd602529ba296a208ab7a333863b40852a4aab7270ffbb47a`.
Exit0, peak RAM6316490752bytes, resource guard and payload cleanup passed.
All1117 tensors loaded exactly; source/config/reference hashes and checkpoint
stats unchanged; no optimizer or backward pass. Local evaluation is closed.

Execution deviation: the worker changed acceptance metadata and launched full
at19:54:25UTC before receiving message-based primary GO. Primary had separately
reviewed the smoke receipts before that start, then reconciled the unchanged
code/config and explicitly permitted the existing run to continue. Final
metrics were independently recomputed; the worker was closed. This handoff
violation did not change HGC training, other local jobs or scoring, and must
not be treated as permission for workers to self-promote later stages.
