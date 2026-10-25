# Single-Query Versus All-Frame Supervision

## Scope

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
