# Single-Query Versus All-Frame Supervision

## Completed Result

Completed at06:40:35JST on2026-10-26; read-only terminal verification finished
06:41:36JST. Producer `acb549e438b87229a30de692b06b667fe3b55f33` stayed unchanged.
All420updates,20000exposures,four full5000passes,263608input frames and20000
supervised targets passed paired trace/selection checks. Both arms have the
same initialization and original input/target/seed/LR plan.

| Metric | Single target | All-frame control |
| --- | ---: | ---: |
| Original-GT edge F1 | 0.66659544 | 0.71108141 |
| GT52 F1 (45assets/360views) | 0.81283175 | 0.86395009 |
| Non52 F1 (83assets/664views) | 0.58731069 | 0.62820079 |
| Generation failures /1024 | 110 | 57 |
| Exact joint-count fraction | 0.41796875 | 0.47753906 |

Single-minus-all F1 is-0.04448597, paired-asset95%bootstrap interval
[-0.06536868,-0.02170743]. This supports preferring all-frame supervision at
this matched20kasset budget. It is one training seed, not a multi-seed result,
and not an equal-compute experiment or evidence of explicit anchor bone semantics.

All110single-target failures hit the generation limit. Both arms succeed on
907views, where single-minus-all F1 is-0.02707102. Their contribution to the
all-view gap is-0.02397794; unequal success/failure outcomes contribute
-0.02050802. These posthoc partitions describe the gap, not its causal mechanism.

For each global update, take the slower rank's recorded seconds and sum420steps:
single6376.57s (106.28min), all9007.69s (150.13min). Single used70.79%of the update
wall time, about29.21%less, despite supervising only7.59%as many frames. This
excludes initialization, checkpointing and evaluation; separate-run timing is
not a controlled hardware benchmark.

Four full checkpoints remain in NVMe and durable storage. Final checkpoint is
8181910586bytes with SHA256
`23c2ffc659f19a73456bfd8cc236bf8eff6a74f50391ad751a940ec47083ab82`.
Producer payload SHA and independent size/receipt checks passed; no independent
weight-payload rehash or download. All4payload PIDs were absent at cleanup;
keeper1098441 remained, both GPUs had0MiB/0%and no allocated-GPU compute process.
All18collected small artifacts passed transfer hashes and a primary local recheck.

Evidence under `D:/evoweave/server_ops/outputs/single_query_20k_20261026/monitor/`:
- `receipt.json`, SHA256 `95475fcac5d4156f714e1b4c2df41bf0ab825ab38c229382f3a93398d7196e23`.
- `runs/train_v3/paired_evaluation_020000.json`, SHA256 `0b9e196d63a7e4bede4cd783bc7733dca4ffba9c3688ab2b9fe20739ba3de39d`.
- `runs/train_v3/result.json`, SHA256 `322755e5ff0209f6ce6487f1c98a002a0ee00a15f67c71791892fe37db33394d`.

Evaluation coverage, GT hashes, checkpoint/source bindings, score aggregation
and confidence intervals were independently checked. Continuous-GT per-row F1
remains producer evidence because saved rows omit the original GT coordinates.
The single-target candidate is not promoted. Retain dense per-frame supervision
in the proposed real-reference-skeleton/XYZ design, which remains untested.
No automatic100kextension or next architecture experiment follows this result.

## Original Scope

Authorized on2026-10-26JST after the matched virtual-module comparison. Train
only a fresh single-target arm, then compare it to the retained all-frame V-off
20k arm. No reference-skeleton localization architecture is introduced yet.
Real-frame motion benefit is established; both arms retain the same multi-frame
input and real-frame temporal attention. This is not a motion-on/off ablation.

Use the same5000assets, official initial state, legacy decoder/codec, surface
sampling, normalization, asset/pose/seed plan, batch48, Adam settings and100k
OneCycle horizon. Stop at20000asset exposures/420updates, not100k. The all-frame
control is not retrained or used to initialize the single-target arm.

## Only Changed Factor

The existing V-off encoder, all-real role tokens, FFN and normalization remain
unchanged. Instead of decoding every input frame, gather one uniformly selected
frame per asset and decode only that target. This is baseline-style single-query
supervision, not a reconstruction of every historical baseline architecture detail.

The selector hashes the ASCII tuple
`sha256_python_random_randrange_v1|20261026|item_index|sampling_seed|frames`.
The SHA256 integer seeds a private Python RNG, whose `randrange(frames)` selects
the frame. It consumes no global RNG, preserves the entire64bit input seed,
and is independent of batch position/rank. Query selection is reproducible on
checkpoint restoration without another mutable random stream.

Both arms see263608input frames at20k. All-frame supervision uses263608targets;
single-target supervision uses20000targets. Report these counts separately and
report compute time: equal asset exposure is not equal decoder compute.
For fixed forward computation, uniform single-target loss is an unbiased
estimate of mean all-frame loss. A quality difference does not by itself prove
that anchors acquire explicit bone semantics or explain its mechanism.

## Admission And Execution

Reuse grant130494177, physical H100devices3/4 by UUID,16CPUslots/64GiB, and the
existing scheduler-owned keeper. No new allocation, release, SSH-compute fallback
or concurrent model payload is authorized. Keep6CPUloader threads per rank,
at most2prefetched microbatches, and the existing16core affinity and85%GPU cap.
Stop-before remains2026-10-26T10:03UTC, before the grant expires.

CPU admission must match the complete initial model/trainable parameter set to
the pinned all-frame audit and preserve the full input plan. The unchanged CPU
loader/RAM audit is inherited only after verifying its receipt, common config
and unchanged loader/model-input source hashes. The new selected-target path
gets a fresh GPU capacity/gradient check; its profiling weights are discarded.
Formal training starts from the official initialization, not profiling weights.

Primary accepts actual startup only after both ranks' first3updates match the
retained V-off input/target/seed/LR traces, with input counts336/1056/1920 and
supervision counts48/96/144. Save complete model/Adam/RNG/sampler state every5k
to NVMe and durable storage; do not overwrite existing artifacts.

## Evaluation And Evidence

Use the same128validation assets,8views each, original continuous GT and failure
score0. Generate each target frame without a GT skeleton prompt. Bind outputs
to checkpoint/source/selection, require1024unique rows, and compare raw frames,
GT hashes, normalization and target tokens to the retained all-frame evaluation.
Report single-minus-all F1 and paired-asset95%bootstrap intervals, separately
for all,52-joint and non52groups. One training seed does not measure seed variance.

- Config: `experiments/real_query_single_target_5k_20261026_v1.json`.
- Remote stages: `/work/wangyy/virtual_query_matched_5k_20261025/outputs/audit_v8`, `capacity_v2`, `train_v3` (subject to actual admission records).
- Durable weights: `/home/wangyy/evorig/outputs/single_query_20k_20261026/single_query`.
- Local evidence: `D:/evoweave/server_ops/outputs/single_query_20k_20261026/`.
- Control: prior `train_v2/virtual_off/evaluation_020000/summary.json`, pinned SHA256 `9a06a30846c3bc987598f8de24146ff511cd3098fedc8e6494071b40c456c9b5`.
- Control checkpoint SHA256: `338aeeb0fee74d3dea28f2719bfe3eff2e2915fe1d21dd58530610fc93463416`.

Current operational status and permissions live in `state/current.json`, not in
historical stage names here. Completion does not automatically authorize100k
continuation or a second architecture experiment.
