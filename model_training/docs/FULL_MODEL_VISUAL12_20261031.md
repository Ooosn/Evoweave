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

Reuse grant 130675914, which still grants two H100s and 16 CPU slots. Anigroom
is now using GPU3 and CPUs6/7/9/10, so this visual task uses only GPU4 and CPUs
11/12/14/15/61/63/64/65: one native rank, eight execution CPUs, 64 GiB RAM cap.
The scheduler grant and real JOB_ID/NSLOTS remain unchanged. Validate kernel
membership, actual selected GPU availability and disjoint CPU affinity; do not
stop or modify the other workload. Stop before 2026-10-31T10:55:00Z and preserve
the allocation. Audit_v27 precedes capacity_v27. Recheck one2048-token/15-beam
capacity decode with the new checkpoint, then run only the twelve cases.

Display selected poses 0/2/4/7, not a purported consecutive animation. Deliver
prediction-only and GT-overlay contact sheets plus every individual case.
Inspect all sheets, expand ambiguous cases, and distinguish structural failure,
pose-transfer failure and annotation-granularity differences. Descriptive
12-case metrics are not a full validation estimate or proof of generalization.

The shared official inference path passed 17 predicted-reference tests, ten
launcher tests and four official-generation tests. The binding does not alter the model, weights,
training implementation or historical inference contracts.

Current admission and completion evidence are recorded in state/current.json.

## Deployment Hold

At 2026-10-31 14:23 JST, the pre-deployment clean-checkout gate found 421
tracked deletions in the existing remote checkout, including AGENTS.md and
DATASET_SOURCE_OF_TRUTH.md. These changes predate this deployment attempt;
no files were restored or overwritten. The 300k model and full checkpoint
still exist with expected sizes. No audit or GPU inference was started.
Human authorization to restore missing tracked code/docs has been requested.
The local collection, independent rescoring and rendering scripts are prepared
under D:/evoweave/server_ops/outputs/full_model_visual12_20261031/preparation.

The user explicitly authorized restoration. At 14:50:38 JST all 421 missing
tracked files were restored from the original HEAD; one missing Git blob was
recovered by refetching the project repository. The checkout is clean. No
data, weights or existing experiment outputs changed. Deployment may proceed
after a fresh native-allocation check; training remains prohibited.

Audit_v27 exited1 at14:58:03 JST before a payload/resource receipt was created:
verify_launch still required world_size2 after the native subset was validated.
The redundant check is corrected; non-scheduler paths still require two ranks,
and scheduler_runtime strictly binds the single-card visual subset. Preserve
the failure log. Replacement stages are audit_v28 and capacity_v28; no GPU
inference or model update occurred in v27. Seventeen protocol and ten launcher
tests passed after the correction.

Audit_v28 completed at15:01:09 JST, exit0 and clean native resources. All12
retained full-model canonical inputs and checkpoint identities passed; zero
optimizer updates and no CUDA initialization. Its result hash is
af6782b14d3fc609d8145199aad2ee1457885f376e5714c6620f1b4d63a08475.
Capacity_v28 is now admitted for the single-GPU visual run, with this primary
CPU result bound in the config. No source changes while the payload is active.

## Completed Visual Review

Capacity_v28 finished at 15:12:22 JST, exit 0, clean native resources, zero
optimizer updates. GPU4 was idle afterward; Anigroom PID236865 on GPU3 and
CPUs6/7/9/10 was unchanged. The full 15-beam/2048-token decode took 60.44 seconds
and reserved at most 30,480,007,168 bytes. No training was resumed.

All 12 fixed cases and 84 target slots per method are retained. Eleven references
are valid; seven have the GT joint count. Primary rows were independently
rescored locally. All 11 valid cases have lower normalized symmetric nearest-
joint distance with the head than with copied reference positions: asset-equal
means are 0.02075863179 versus 0.06756750377 across the same 11 assets/77 targets.
The failed asset remains in all-case quality denominators. These are descriptive
fixed-case results, not a validation-set generalization estimate.

Visual inspection covered all six contact sheets, the enlarged GT sheets and
individual cases 15/105/108. Thirty images are retained: prediction-only and
GT-overlay views for every case plus three contact sheets in each mode. Display
slots 0/2/4/7 are selected poses, not a consecutive animation. Camera/limits are
fixed within each asset, with no clipping observed or failure hidden.

- 5/15/24/44/54/61/95: matching joint counts, visible primary structure and pose
  following close to the surface/GT in displayed views; not exact-topology proof.
- 23: main body/limbs follow poses, but 52 predicted versus 22 GT joints include
  extra hand/detail subdivision; do not declare topology equivalence by appearance.
- 30: major lateral branches in GT are missing from the reference and outputs.
- 70: a 52-joint human-like reference misses multiple branches of the 122-joint GT.
- 108: 13 predicted versus 5 GT joints add branches and show substantial misalignment.
- 105: no valid decoded reference. The 828-token sequence includes one terminal EOS,
  but detokenization raised ValueError('all input arrays must have the same shape').
  Root cause is not diagnosed; no retry, GT repair or case replacement occurred.

Evidence: D:/evoweave/server_ops/outputs/full_model_visual12_20261031/
visual12_review.json and visual_inspection.json. Images are under
artifacts/capacity_v28. The frozen main model is unchanged. GPU authorization is
closed; awaiting the user's visual review before additional inference/analysis.
