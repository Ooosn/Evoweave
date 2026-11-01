# Paper Evidence Inference, 2026-11-02

The user authorizes autonomous baseline deployment and H100 inference for the
complete first manuscript. This is not training permission and does not reopen
checkpoint selection. Retain the selected 330k checkpoint and all prior results.

## Fixed First Cohort

Select 24 validation assets by ascending SHA256 of `20261102:<asset_id>`, after
excluding the original 128-asset development panel. Freeze the identities, eight
input poses, canonical points, original GT, and normalization before generation.
This remains validation data, not a never-observed test set: earlier controlled
experiments evaluated the full 967 validation assets. Released baseline training
overlap has not been eliminated.

All methods use greedy generation without GT skeleton or joint count. Evoweave
uses all eight corresponded poses; official UniRig and Puppeteer take the same
reference surface and normals in their own recorded normalization. Report this
input difference. UniRig consumes the retained dense surface and Puppeteer its
first 8192 points. Compare their reference skeletons only. Compare Evoweave's
predicted-reference dynamic head to copying that same reference across seven
target poses; do not misrepresent static methods as motion recovery baselines.

The Puppeteer result preserves raw joint tokens, without its geometric merging
postprocessor. The full released checkpoint strictly restores every parameter,
including the point encoder; its redundant auxiliary pretrained initialization
is omitted. UniRig's legacy source annotations are postponed in memory for
Python 3.10 compatibility. Neither pinned source checkout is edited.

Count failed generations as zero in edge F1 and exact-count rates. Report
symmetric nearest-joint geometry on successes with its explicit denominator;
never call it paired-ID MPJPE. Preserve raw token IDs and per-asset predictions.
Do not resample failed generations or choose a different checkpoint afterward.

Source root rigid transforms come from dataset skeleton metadata. This is not
raw-video evaluation and must not support a GT-free video-preprocessing claim.

## Execution

The exact CPU preparation is `audit_v36`. `capacity_v36` is the framework name
for the zero-update paper inference stage, not a repeated training stress test.
Record the CPU receipt and successful supervised cleanup before GPU admission.
Use `paper_eval_ops.py` plan, execute, and observe with the exact config path.
The native keeper 2268188 dispatches under job 131170166, physical GPU4 and its
eight-CPU subset of the retained two-H100, 16-CPU allocation. Preserve the 96h
expiry and 2026-11-04T11:40:04Z save-stop. Preserve other workloads and the keeper.
The stage is bounded to three hours and no optimizer updates. Shared wall times
are diagnostic timing, not an isolated speed comparison.

Finish source changes before dispatch; never synchronize a running checkout.
Deploy a fresh immutable staged checkout using `deploy_stage_checkout.py` with
verified SSD placement and durable archives of old and new source. Do not restore
deleted historical source files repeatedly. Inspect an existing stage before
doing anything else. Any failure must keep original evidence and cleanup status.

The case105 exact-coordinate token-replay discrepancy remains unresolved.
