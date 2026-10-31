# Frozen Full-Model Visual Inference

The user requested visualizations before experiment analysis on 2026-10-31.
Use the frozen 300k joint model for BOTH reference generation and target-pose
prediction. No training, optimizer loading, fitting, GT-reference repair,
static fallback or quality-based case selection is authorized.

Reuse the same twelve preselected panel indices as the old visual inspection:
5, 15, 23, 24, 30, 44, 54, 61, 70, 95, 105, 108. These cover source/joint-count
variation without looking at the new predictions. Every input comes from the
full run's retained immutable evaluation cache and must match its final
surface/GT/raw-frame/normalization identity. Do not regenerate a missing cache.

Generate one reference at slot 0 using the established dynamic UniRig official
15-beam sampling recipe, seed 20261029 derived per asset/raw frame, FP32 and
no TF32. Predict slots 1..7 using this model's reference-transport head.
Retain GT-head and GT-copy diagnostics and predicted-copy control. GT is used
for evaluation/display only, never to repair the generated primary reference.
Failures and wrong joint counts stay in the denominator and visual set.

Two native H100 ranks, 16 CPU slots and 64 GiB process-tree RAM reuse grant
130675914. Stop before 2026-10-31T10:55:00Z; preserve the allocation. CPU audit
v27 precedes GPU capacity_v27 admission. Recheck one 2048-token/15-beam capacity
decode per rank with the new checkpoint, then run only the twelve cases.

Display selected poses 0/2/4/7, not a purported consecutive animation. Deliver
prediction-only and GT-overlay contact sheets plus every individual case.
Inspect all sheets, expand ambiguous cases, and distinguish structural failure,
pose-transfer failure and annotation-granularity differences. Descriptive
12-case metrics are not a full validation estimate or proof of generalization.

The shared official inference path passed 16 predicted-reference tests and four
official-generation tests. The new binding does not alter the model, weights,
training implementation or historical inference contracts.

Current admission and completion evidence are recorded in state/current.json.
