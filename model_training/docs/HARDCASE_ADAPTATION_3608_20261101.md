# Hardcase3608 Joint Adaptation

User decision, 2026-11-01: freeze the3608candidate scope and per-asset sampling
weights3/2/1 and finish preparation. The initial04:00Asia/Tokyo start was blocked
by other GPU workloads; the user has rescheduled the next attempt to05:00.
This authorizes the stated CPU audit, native
capacity/resume check and training; it does not authorize an early GPU start,
another allocation, a larger run or automatic retries after a failed gate.

## Frozen Data And Sampling

The immutable October13 manifests remain the only membership authority:
18,002train and967validation assets, root-RT overlay and selected-input
sequence mesh-bbox normalization. No validation assets enter training. No NPZ
is copied out of the existing verified NVMe cache or modified.

Topology means a rooted unordered tree, ignoring enumeration and geometry.
Family frequency is computed only from training assets. Complexity uses an
OR over joint count, leaf count and maximum root-to-leaf edge depth.

| Pool | Rule | Assets | Per-asset weight | Visits in fixed30k plan |
|---|---|---:|---:|---:|
| Core | Frequency<=20 and joints>=64 OR leaves>=16 OR depth>=20 |1860|3|7088|
| Extension | Not core; frequency<=50 and joints>=48 OR leaves>=12 OR depth>=16 |1748|2|4474|
| Background | All remaining accepted training assets |14394|1|18438|

The frequency condition applies to the entire parenthesized complexity OR.
These are structural candidates, not established model failures or semantic
object classes. The5688asset dominant52joint family remains background.

One exact weighted cycle has23,470visits and visits every asset exactly its
declared weight. Within each tier, permutations finish before assets repeat;
the tier labels are shuffled deterministically. The30k plan is one complete
cycle plus6,530visits from the next cycle. All18,002assets are covered. Actual
candidate/background visit fractions are38.54%/61.46%; long-run expected
fractions are38.67%/61.33%. These are visit proportions, not asset fractions.
The seed is20261101. Every visit gets a different sampling seed. The two ranks
consume disjoint positions in one global plan with matching frame counts.
Consumed cursors, rank RNG and the complete plan fingerprint are checkpointed;
prefetch does not advance the consumed cursor.

Frozen topology profile SHA256:
`27666a73a6de586c23345b4d7a6d107eb8f29fb5fea3b30e9524a4e6ae4d6231`.
Frozen sampling plan SHA256:
`df290696c9fd58fac2fae521537a335b0b62b499f82eba6e03e54dc0cfb093be`.
Frozen generated JSON uses UTF-8 and LF on both Windows and Linux.
Remote data preparation:
`/work/wangyy/hardcase_adaptation_20261101/data/preparation.json`.
Local review artifacts:
`D:/evoweave/server_ops/outputs/hardcase_adaptation_20261101/`.

## Initialization And Optimization

Use the complete full300k recovery checkpoint, including all compatible Adam
moments and step counters, not its model-only export and not a capacity output:
`/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v26/checkpoint_300000.pt`.
Its8,278,702,930bytes must hash to
`a22824dd9134d1840ea60d75254589f1d95f248d495fe98381cc976dbc0fcaa7`.
Source counters are300,000asset visits and6,270optimizer updates. The adapter
checks parameter names, strict model tensors, both optimizer groups and the
optimizer fingerprint after import. New sampling/RNG and LR positions start
at zero; original optimizer moments and step counters continue.

| Setting | Frozen value |
|---|---|
| Additional asset visits |30,000; total model history330,000|
| Effective batch |48assets over2ranks|
| Physical per-rank batches |24for2..6frames,12for7..12,6for13..24|
| Input frames |Uniform2..24per global update, as in full300k|
| AR target frames |Existing random20%-40%, minimum2|
| Reference for XYZ supervision |Uniform complete GT tree from an input frame|
| Trainable parameters |Same backbone, surface/motion components, AR decoder and XYZ head|
| Backbone LR |2e-6at0visits; linear warmup to1e-5at1000; cosine to2e-6at30000|
| Head LR |3times backbone LR|
| AdamW |Retain moments; beta1=.95,beta2=.999,weight_decay=0|
| Loss |Existing asset-normalized AR +10*(XYZ +0.25*bone-vector loss)|
| XYZ warmup |Not restarted; weight10from the first adaptation update|
| Gradient clipping |Separate backbone/head norm1|
| Precision |Existing BF16 autocast training; FP32/noTF32 eager validation|
| Save/evaluate visits |5000,10000,20000,30000|
| Optimizer updates |628, including shorter updates at exact checkpoint boundaries|
| Recovery retention |All four full checkpoints and all four model exports, plus baseline unchanged|

No extra loss weights are applied:3/2/1changes sampling only. Architecture,
normalization, GT, tokenizer and production decoding are unchanged.
These are engineering choices for one bounded adaptation, not a proven optimal
recipe or a guarantee of recovering missing branches.

## Evaluation And Model Selection

Freeze before any gradient update. Validation has98core,96extension and773
background assets under the same training-frequency definition. Development
uses48core +16extension +48background =112assets. The locked remainder is
50core +80extension +725background =855assets. The12known diagnostic cases
are in development. "Locked" means not used to choose this adaptation's
checkpoint; it is not a claim of historically unseen validation data.

First evaluate the unchanged full300k source on the112development assets.
Then evaluate each of the four saved candidates on the exact same canonical
surface/GT/raw-frame/normalization identities. Stage1 uses8poses per asset,
greedy generation only for fast feedback, no GT reference or teacher prefix,
and failed generation scores zero. Stage2 uses GT reference slots0and4,
excludes each reference pose, and reports14target pairs per asset. Summaries
retain core/extension/background separately and preserve all failures.

Development shortlist targets are engineering criteria:

- Candidate-pool edgeF1 improves by at least0.03absolute (3percentage points).
- Background edgeF1 drops by at most0.005absolute (0.5percentage points).
- Parse failure count does not increase on the identical development views.
- Overall GT-reference normalized pose error is at most1.02times baseline.

The focus score averages assets equally across the64core/extension development
assets. Among eligible checkpoints choose the largest focus gain, breaking ties
in favor of fewer visits. If none qualifies, keep full300k; do not extend or
retune automatically. All checkpoints are retained regardless of shortlist.
Report paired per-asset differences and95%bootstrap intervals in the final
review, with failures retained. CE or training loss is not model selection.

After selection, compare only the selected checkpoint and full300k on the855
locked assets, using the same8input poses, canonical inputs and separate
stage1/GT-reference stage2 metrics. Do not choose another checkpoint using
locked results. A failed final comparison rejects the adaptation.

Also perform a production-policy comparison on a deterministic32asset locked
subset:16core,8extension,8background; lowest SHA256 of
`20261101:official-locked:<asset_id>` within each stratum. Use both predeclared
seeds20261101and20261102, one fixed reference slot0and all7other pose targets.
Use the unchanged official15beam sampling policy:top_k5,top_p.95,temp1.5,
repetition_penalty3; no repaired output or static-model fallback. Keep both
seeds and every failure rather than choosing a favorable seed. Inspect paired
improvements and regressions, especially actual recovered branches and excess
joints. Report predicted-reference downstream metrics separately from
GT-reference stage2. Greedy improvements alone are not production acceptance.
The locked/official comparison is a separately recorded post-training phase,
not an implicit resume of the stopped inference-diagnosis experiment.

The full105official replay changed10coordinate tokens at the same828token
length and same failed EOS status. Its cause remains unresolved. Preserve its
original outputs and failed exact-replay gate. This training is a new
user-approved hypothesis test, not an established repair of that failure.

## Resources And Startup

Reuse native HGC job131170166 onpcg01i, physical GPUs3/4 and16CPUslots,
64GiBhost-memory reservation; actual UUIDs are in the executable config.
Both cpuset and devices membership must match `/AGE/131170166.1/master`.
The keeper is2268188. Its inherited224CPU affinity is not an execution budget;
bind the approved16logical-CPU subset before native dispatch. Six data workers
per rank and single-thread BLAS remain inside the whole process-tree limit.
Keep GPU reserved memory below85%, no CUDA virtual-address-space limit.
Use the existing nonrotational `/work` cache and preserve200GiBfree SSD space.
Save and stop before2026-11-01T11:40:04Z (20:40:04JST), one hour before the
allocation's21:40:04JST expiry. Refresh the grant and occupancy before dispatch.

At01:36JST the native keeper was idle, but physicalGPU3had a separate process
usingabout3.8GiB. Do not stop it. Identify ownership and require both approved
GPUs to be idle before capacity/training. No GPU-count or allocation change is
authorized by this plan.

The re-created single-use heartbeat `evoweave-3608` wakes this chat at05:00JST.
Both capacity and training are guarded by `not_before_utc=2026-10-31T20:00:00Z`.
CPU-only preparation/audit may run earlier with CUDA invisible.

Executable config: `model_training/experiments/hardcase3608_20261101_v1.json`.
Entrypoint: `model_training/rigweave/scripts/train_hardcase_adaptation.py`.
The native keeper launches the existing bounded supervisor, which checks
cgroup membership, exact config/source/output, CPU tree, memory and cleanup.

From the isolated remote checkout, with the pinned Python and site environment:

1. Synchronize the clean committed source through Git. Run the agent guard
   `begin`, inspect live job/keeper/GPU state, and run
   `hardcase_adaptation_ops.py bind --source <HEAD>` while the keeper is idle.
2. Plan and execute `audit_v30` with `hardcase_adaptation_ops.py plan/execute
   --stage audit_v30 --source <HEAD>`. This CPU-only audit imports full300k Adam,
   verifies all dataset identities, retained cache, loader contracts, weighted
   coverage and exact rank-plan restoration; CUDA must remain uninitialized.
3. Review `outputs/audit_v30/result.json` and `audit_v30_resources.json`.
   Run `hardcase_adaptation_ops.py record --stage audit_v30 --source <HEAD>`.
   This records only verified evidence, updates config SHA and admits capacity.
   Commit and push the two changed config/state files before the next stage.
4. At the scheduled time, refresh the idle GPU grant. Plan/execute `capacity_v30`.
   Stress2..24frames at maximum allowed AR density and full-manifest extremes;
   compare source initialization with CPU audit; require exact model/Adam/RNG/
   weighted-sampler disk restoration. Capacity updates are disposable.
5. Review completed capacity and resource receipts, record capacity with the
   same tool, commit/push config/state, rerun `begin`, then freshly plan/execute
   `train_v30`. Start from the full300k source, never the capacity checkpoint.
6. Verify actual first updates, LR2e-6/head6e-6, XYZweight10, finite losses,
   gradients and weighted asset indices. Report the actual start result.
7. Wait on the supervised process, inspecting bounded progress at meaningful
   milestones. On completion/failure, record the same active operation and
   artifacts, commit/push, and preserve the scheduler allocation. Record the
   fixed post-training comparison separately before running it.

Work output is `/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v30`.
Durable checkpoint output is `/home/wangyy/evorig/outputs/hardcase3608_20261101/v1`.
Never reuse a stage id after an error or launch a duplicate. Do not edit the
live source while a stage payload is running. A gate failure is preserved and
diagnosed; the reminder does not permit bypassing it or silently retrying.

## Completed Preparation

CPU audit_v30 finished2026-11-01 02:02:07JST, exit0, native job131170166,
16CPU binding, peak process-tree RSS18,192,265,216bytes (16.94GiB), clean payload
exit and no CUDA initialized. The full300k checkpoint and all1079Adam state
entries imported with exact parameter/group compatibility. The weighted plan
has30,000visits,628updates,392,704input-frame visits and all18,002train assets;
both rank plans restore exactly. The112development assets and immutable967
validation membership are verified. Six loader contract cases passed. Existing
19,044file/419,101,379,008byte NVMe cache identity and sizes match accepted
full-data evidence; it was not recopied or fully rehashed again.

Primary CPU audit result SHA256:
`f7a99c2897a929e475309f62b031f96ba6296977606dbb48f90c063432120a4d`.
Native resource receipt SHA256:
`bb68d750383f810e22bafc4c66e9af8387701d4a5d7031919dd961440842fd4d`.
The passed audit is recorded in config/state and committed asf3fb20d.
Local main review:
`D:/evoweave/server_ops/outputs/hardcase_adaptation_20261101/preparation_review.json`.
Tests:38passed plus20subtests, covering weighted sampling, full Adam import,
scheduled GPU admission and existing full-data/input/replay behavior.
GPU capacity, teacher-prefix and disk model/Adam/RNG recovery checks remain
scheduled work. Formal training has not started. The heartbeat must reuse this
CPU acceptance rather than starting the audit again.

## Scheduled Start Check, 04:03JST

The04:00heartbeat ran and confirmed current CPU acceptance, source/config
identity, native keeper membership and16CPU binding. Both approved GPUs were
occupied by separate Anigroom training processes under the same account:

| GPU | PID | Used device memory | Separate run |
|---|---:|---:|---|
|3|51713|24312MiB|anigroom_original_weight_15k_20261101_r01|
|4|4158809|24606MiB|anigroom_panda_white_20261101_r01|

Observation time:2026-10-31T19:03:08.548950Z. The Evoweave native keeper had
no children; only the completed audit request exists. Neither capacity_v30 nor
train_v30 exists, and neither GPU stage was dispatched. All processes and the
native allocation are preserved. The earlier read-only readiness transport
timed out after printing its result; this independent short inspection
confirmed the actual resource conflict without launching anything.

Local/remote evidence:
`D:/evoweave/server_ops/outputs/hardcase_adaptation_20261101/scheduled_start_gpu_conflict_0400.json`
and `/work/wangyy/hardcase_adaptation_20261101/scheduled_start_gpu_conflict_0400.json`.
The one-shot heartbeat was deleted after it fired and recorded this blocker.
Current GPU admission is closed. CPU preparation and the original training
authorization remain valid; resolve the existing GPU occupancy before a fresh
admission. Do not stop the other jobs, change resources, repeat the CPU audit,
or automatically retry this blocked scheduled attempt.

## User Reschedule To05:00JST

After the blocked04:00check, the user explicitly requested waiting until05:00
to start. One new heartbeat was created for2026-11-01 05:00JST. The next native
capacity stage is approved again, with a new05:00not-before guard; formal
training still requires the real capacity result. Reuse accepted CPU_audit_v30
and the unchanged sampling/config/checkpoint identities. At05:00inspect both
GPUs and the original native grant afresh; do not infer that the Anigroom jobs
have ended. If idle, plan and execute capacity_v30, review/record and commit its
acceptance, then plan and execute train_v30. If still occupied, preserve all
workloads, report the specific blocker and do not automatically retry again.
