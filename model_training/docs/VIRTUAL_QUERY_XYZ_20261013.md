# Virtual Query XYZ: Approved Model And Experiment Contract

Status: implementation authorized, model acceptance pending. This document is
the current architecture contract. Historical 96-register/shared-parent/XYZ
regression trials are preserved evidence, NOT this model and NOT initialization.

## Approved Architecture

1. Load complete immutable manifest rows, including `root_rt`. Invert retained
   root rigid motion on mesh and GT. Normalize only the selected input mesh
   union with padding 1.1. Preserve scale/shear residuals, every target joint,
   and heldout-frame separation. No GT-derived scale, clipping, or extra rotation.
2. Sample 1024 trackable anchors per real frame. Anchor index q tracks the same
   sampled surface point across that asset's input frames.
3. Add a mesh-free virtual query with **1024 learnable anchor-aligned slots**.
   Every virtual slot q participates in temporal attention with real slot q.
   Role/register tokens are auxiliary internal slots, never substitutes for
   the 1024 virtual slots. Do not copy a real frame as the virtual query.
4. Run **12 spatial/temporal blocks**, followed by final normalization. Retain
   both final virtual tokens `[B,1024,D]` and all final real-frame tokens
   `[B,T,1024,D]`. Do not substitute initial/re-encoded/intermediate mesh tokens.
5. For each target frame, use final virtual tokens as attention queries and
   that frame's final mesh tokens as keys/values. Produce exactly 1024 **mesh
   conditioning tokens**, not predicted joint tokens, for that frame.
6. The shared original UniRig XYZ autoregressive decoder generates a complete
   skeleton token stream separately for every supervised frame. Preserve the
   original XYZ vocabulary, input embedding, output head, BRANCH semantics,
   BOS/class/EOS convention, and main tokenizer. Train against that frame's GT.
7. A separate small auxiliary AR decoder reads final virtual tokens and predicts
   a coordinate-free common-structure sequence. It contributes only a loss.
   Neither its tokens, hidden joint features, predicted length, parents, nor
   masks may condition the main XYZ branch. It is unnecessary at inference.

The virtual slots learn shared structure through both differentiable routes.
This does not guarantee identical freely generated topology across frames.
Common structure means the same asset across frames, not one skeleton template
for all assets. A coordinate reference is not a real model query frame.

## Auxiliary Sequence: Explicit Node IDs

Use fixed depth-first traversal of the target skeleton, with stable original
target-index sibling order, independent of posed XYZ. Assign IDs J0, J1, ...
in traversal order. An ordinary new ID extends the current chain. On a branch,
emit BRANCH, the existing parent ID, then the new child ID. For example:

```text
Main UniRig: XYZ0 XYZ1 XYZ2 BRANCH XYZ0 XYZ3 EOS
Auxiliary:     J0   J1   J2 BRANCH   J0   J3 EOS
```

BOS/PAD are sequence mechanics. IDs are local to each asset, not semantic names
or global joint IDs. Repeated IDs following BRANCH are references, not new joints.
Multiple returns to the same ancestor work identically. Encode/decode tests must
recover the source tree under the recorded DFS permutation. This is NOT the
earlier parent-ID-per-joint main decoder, P0/P1 depth tokens, or CLOSE grammar.
The exact auxiliary skeleton is not an inference prerequisite for the main task.

## Initialization And Loss

Initialize shared upstream surface encoder/projection and the complete XYZ AR
decoder from the pinned official UniRig checkpoint, never an old trained model
or any descendant of an incorrect-route trial. New motion/virtual/fusion/auxiliary
parameters have a recorded seed. Train end-to-end with the established BF16 path.

For each asset, average main CE over valid tokens within each supervised frame,
then average over its frames. Average auxiliary CE over its valid tokens once
per asset. Finally average assets:

```text
L_asset = mean_frame(main_XYZ_sequence_CE) + lambda * auxiliary_ID_sequence_CE
```

Padding is ignored; ordinary AR causal masking remains. There is no continuous
XYZ L1 head, generated-node matching, or auxiliary-generated joint mask. The
new protocol uses CE; any extra historical loss requires explicit review.

Lambda is experimental, not predetermined. Initial matched candidates are
0, 0.1, 0.3, with identical shared initial parameters, data/surface/frame streams,
optimization schedule, and asset exposures. Selection uses main heldout/free
generation quality, not the minimum auxiliary CE. Auxiliary influence on shared
gradients is measured; losses are not equated solely by their scalar values.

## Dataset And Publication

Authority: `DATASET_SOURCE_OF_TRUTH.md`. Required rows: train 18002 / valid 967.
Root: `/home/wangyy/evorig/evoweave_rootless_v3_rootrt_v1_hgc_20261013`.
Manifests: `quality_distributions/rootless_bbox_consistency/final_manifests/`.

```text
train SHA256 653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27
valid SHA256 fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a
```

Full data preprocessing and RT-loader acceptance are complete. The final extra
audit checked 44 assets / 232 loader cases and inspected 12 four-frame visuals.
This is data acceptance, not model acceptance. The source-skeleton-derived RT
cache is not a GT-free deployment RT estimator.

## Sequential Acceptance Gates

No gate passes just because a process exits successfully. The primary reviews
each result before authorizing the next stage; no script may automatically
continue from a failed/ambiguous fit into parameter sweeps or full training.

### G0: Representation And Architecture

- Codec roundtrips, non-DFS input permutations, branch references, malformed
  sequences, cycles, and padding pass CPU tests. Validate real dataset parents.
- Tests capture actual virtual/real/fused shapes and anchor-index routing.
- Auxiliary outputs/labels cannot alter main logits at fixed weights/inputs.
- Main and auxiliary losses independently reach virtual query/motion parameters;
  main loss reaches the original XYZ embeddings/head and the final frame route.
- Actual pinned upstream tokenizer/head sizes and sequence conventions agree.
- Teacher-forced logits and cached stepwise logits agree within numerical
  tolerance. Natural EOS and max-token failure are distinguished.

### G1: Full-Size Device Preflight

Use the real width, 1024 slots, 12 blocks, complete official initialization,
actual RT-aware inputs and BF16. Begin B=1/T=2, then inspect T=8 and T=24 with
resident optimizer state and backward/step. Record finite losses/gradients,
memory, throughput, parameter loading coverage, device/allocation, code/config
hashes. This validates execution only, not fitting or larger batch capacity.

### G2: Fixed-Sample Fitting Before Scale

First use one fixed training asset with two distinct eligible frames, then a
fixed eight-asset/four-frame set spanning source cohorts and joint-count ranges.
Keep surface/frame selections fixed for this diagnostic, but compute learned
features online and train end-to-end. No heldout frames or validation assets
enter optimization. Record selection before observing results.

Review at update 0/32/64/128. A declining CE alone is insufficient: inspect
natural generation, target-token accuracy, EOS, joint count, geometry/topology,
condition use, and gradient norms. If progress stalls or diverges, stop and
diagnose labels, AR alignment, conditioning, initialization and optimization.
Do not automatically add steps or replace the task to make the score improve.

Provisional fit acceptance: finite execution; main valid-token accuracy >=99%;
all fixed views end naturally; >=90% of fixed views have correct joint count
and edge F1 >=0.95 against the decoded quantized training target, together with
symmetric mean nearest-joint distance <=4/256 in normalized coordinates (two
coordinate bins across the [-1,1] range). Record original
continuous-GT metrics separately. These are diagnostic gates, not claims of
generalization. Main UniRig quantization can itself change parent identity, so
report encode/decode label ceilings rather than hiding those errors or modifying
the tokenizer. Any threshold revision requires an evidence-backed log entry.
If 128 updates are insufficient but progress is clear, decide and record a
bounded extension; do not silently run a long memorization experiment.

### G3: Matched Lambda And Generalization Checks

After fit acceptance, compare lambda 0/0.1/0.3 on a fixed bounded training budget
with fixed disjoint evaluation assets and heldout frames. Include low-joint and
unseen-topology slices; score failed generations as zero rather than dropping
them. Report paired main-task metrics and training curves. Fit success or low
auxiliary CE alone does not select a candidate. Do not inherit wrong-route weights.

### G4: Full-Data Training

Only after G0-G3 reviewed acceptance, select capacity/batch settings from measured
resident-state memory, refresh live resource state, and publish the exact formal
configuration. Start fresh from the common official initialization, not diagnostic
fit/sweep weights. Use all 18002 training assets and the untouched 967-asset
validation split. Preserve the previous comparison budget of 80016 actual asset
exposures unless the user changes it; distinguish assets from supervised frames.
Formal training remains blocked until this release decision is recorded.

## Required Records And Stop Rules

`state/current.json` is machine-readable operation authority. Every GPU operation
must name the exact stage, source commit, config/hash, manifests/hashes,
initialization/hash, selected assets/frames, resource allocation, output,
acceptance and stop criteria. Commit/publish before execution, then run the guard.
Bounded GPU operations also pin the entrypoint, production code ancestry/diff,
and original model/tokenizer configuration hashes. Historical launchers fail
the entrypoint gate even while a new-route probe is authorized.
Keep metrics, predictions, losses, gradients, throughput, memory and checkpoints
with exit status. Log failures and hypotheses, not just successful runs.

The following are forbidden silent substitutions: 96 registers as virtual query;
real-frame query instead of virtual slots; continuous H-to-XYZ localization;
auxiliary predictions on the main conditioning path; unrecorded tokenizer
changes; stale manifests; inherited trial weights; teacher forcing presented as
free-generation quality; forced EOS presented as success; or widening a run
without analyzing the previous stage.

## Decision Log

- 2026-10-13 JST: user approved 1024 virtual query / per-frame full XYZ AR with
  auxiliary-only shared-structure supervision.
- 2026-10-13 JST: user approved UniRig-like auxiliary BRANCH sequences with
  explicit joint IDs replacing XYZ. Earlier depth-token suggestion is superseded.
- 2026-10-13 JST: user delegated lambda selection to experiments and required
  written records, demonstrated fitting and analysis before full-data training.
- G0: current 18969-asset auxiliary codec round trips passed with no errors;
  CPU model/adapter tests passed. Original XYZ grammar equivalence was checked
  on all 202 real next-token prefixes of the fixed fitting case.
- G1: full-size B1, T2/T8/T24 forward/backward/optimizer passed. T24 peaked at
  14.70 GiB allocated and 17.30 GiB reserved. Both losses reached virtual slots.
- G2 initial attempt: execution failed before updates because an upstream
  grammar wrapper imported an unrelated invalid Python annotation. The wrapper
  was replaced by an exactly equivalent call to the unchanged original
  tokenizer grammar; the vendor source and tokenizer were not modified.
- G2 main-only fixed fit, 128 updates: NOT accepted. Teacher accuracy 99.5098%
  and CE 0.006759 masked identical free predictions for two different poses.
  Both outputs exactly matched frame 7; frame 9 had F1 0.88735 and J2J 0.04545.
  Targets differ at 45/103 positions, beginning at full-token index 8 (114/113).
  The unconditioned two-sequence teacher-forcing floor is ln(2)/102 = 0.006796,
  consistent with GT-prefix memorization, not proof of a specific root cause.
- Next: replay the same checkpoint/geometry without optimization; measure
  frame differences through encoder/fusion and correct/swapped/mean/zero
  condition logits at the first fork, including a decoder FP32 check. A
  successful diagnostic means valid measurement, NOT model acceptance.
  No longer fitting, lambda sweep or full-data training is authorized yet.
- Read-only diagnosis completed: exact free predictions/targets reproduced.
  Relative frame-feature L2 was 0.219107 at the surface encoder, 0.018726 at
  final real-frame tokens and 0.001415 at fused conditions. Across the 12 motion
  blocks, absolute frame delta RMS stayed near 0.38 while common feature RMS
  grew to 20.82. FP32 reference attention still selected token 113 for both
  conditions; this is not fixed by readout precision alone. Zero condition
  destroyed teacher fit: the common asset signal is used, but pose contrast is
  insufficient in this case. These measurements locate a bottleneck but do
  not alone establish the causal training remedy.
- Bounded intervention R1: set `frame_residual_scale=1` in
  `C_t = LN(Fusion(V_final, A_t_final) + scale * A_t_final)`. The residual
  preserves same-index final real-frame detail; virtual slots remain all 1024
  cross-attention queries. It uses neither raw initial surface features nor GT
  coordinates nor auxiliary predictions. Scale 0 is the exact original
  implementation, and remains the code default. Compare fresh official init,
  the same asset/frames/surface seed, lambda 0, LR 1e-4 and maximum 128 updates.
  No simultaneous optimizer or loss change. R1 remains experimental until its
  generated per-frame outputs pass the same predeclared fit gates.
- R1 result, 128 updates: NOT accepted. CE 0.00694108 and teacher accuracy
  99.5098%; both generated sequences again exactly match frame 7. Frame 9
  retains F1 0.88735 and J2J 0.04545. Official initialization, random seed,
  selected frames and parameter counts were verified matched. Adding this
  residual alone did not fix the failure and is not promoted to the default.
  Stop further training here until a discriminating experiment is selected
  from the two-run evidence. No unchanged extension, lambda sweep or full-data
  run is authorized. The proposed independent consultation has not been sent;
  its code/measurement transfer is awaiting explicit user approval.
- User subsequently authorized autonomous thorough diagnosis and repair,
  including implementation, supervision, module capacity, training duration and
  retained-root RT checks; formal training only after the problem is resolved.
  Priority is now regression against the established training path, not a new
  architecture proposal. No external consultation has been sent.
- Legacy-decoder CPU regression passed: actual old `_ar_losses` versus new
  per-frame code agrees on logits, CE, condition/decoder gradients and one
  AdamW update; variable padding, frame averaging and single-frame reduction
  are covered. This verifies decoder/loss plumbing, not full-model fitting.
- Next reference control: same fixed asset/two frames, RT/normalization/surface
  sampling, official initialization, LR and 128-update budget, but call the
  actual legacy query-first motion encoder once per target frame. This is an
  explicitly separate diagnostic baseline, NOT replacement of the approved
  1024-virtual-query architecture or a candidate for formal training. It tests
  whether the old path fits under this exact small-sample protocol before
  assigning the failure to new conditioning or training duration.
- Fixed a diagnostic-recording defect: per-frame rows previously repeated the
  asset's mean teacher CE/accuracy. New rows carry their actual frame metrics,
  with an explicit scope marker. The optimization objective, aggregate means,
  and all recorded free predictions were unaffected. Combined model, adapter
  and legacy-parity CPU suite now passes 22 tests.
- Legacy query-first reference passed at update 64 on the exact same two
  targets and RT-aligned data: both full free token sequences equal their own
  GT, both F1=1 and J2J=0, teacher CE=0.000450835, accuracy=1. This is a
  reference result only, not acceptance of the virtual-query candidate.
  It establishes learnability of this exact data/target task through the old
  path; the remaining failure is specific to the new path or its convergence.
- Next isolated test: unchanged approved virtual-query baseline (residual scale
  0), fresh official initialization, same data/seed/LR/lambda0; increase only
  the fixed maximum budget from128 to512 with intermediate per-frame/free
  evaluations. This explicitly tests slower convergence, not a new architecture
  or an inherited checkpoint. No automatic continuation beyond512.
- The unchanged approved baseline PASSED at update256, before its512 cap.
  Both free sequences exactly equal their own distinct targets, with natural
  EOS, F1=1, J2J=0 and teacher accuracy=1; mean CE=0.0000753484. This first
  accepted free-generation evaluation directly establishes sufficient capacity
  on this case and that128 updates were insufficient, not a required fusion
  architecture change. The old path converged earlier (64), but speed alone
  does not establish a bug. R1 remains unselected and default scale stays0.
  G2 single-case acceptance is not generalization or full-training approval.
- CPU-only real-case audit is now published to independently solve the raw
  cached column-vector RT equation, compare actual loader mesh AND skeleton,
  check a common mesh-only bbox, inverse consistency and bone preservation,
  and export the exact [9,7] normalized poses for visual inspection. Five
  local CLI synthetic tests passed, including four expected rejection cases;
  these synthetic results are not yet the real-asset audit result.
- Real-case audit subsequently PASSED on exact raw rows[9,7] (animation frame
  numbers[15,13]). Independent RT/bbox mesh AND GT coordinates equal loader
  outputs exactly; mesh inverse error <=4.58e-8 of original bbox diagonal,
  bone-length error after reversing normalization <=7.39e-8 of that diagonal.
  No NaN/Inf, out-of-bounds or clipping. Parent-inspected XY/XZ/YZ overlay shows
  mesh/GT alignment and the256-update predictions matching each own pose.
  This checks cache binding/application, not rederivation from source animation.
- Next G2 stage: eight fixed TRAIN assets, four distinct eligible frames each,
  spanning base/addition and4..10/11..60/61+ joint strata. Deterministic selection
  and seed are recorded before fitting; full RT descriptors remain attached.
  All16 focused selector tests passed, including original single-case parity.
  Fresh official initialization, original virtual architecture, lambda0 and
  the same AdamW1e-4 recipe; cap2048 optimizer updates =256 exposures per asset.
  Review at0/32/64/128/256/512/1024/1536/2048, with early stop only on the
  unchanged all-EOS/99%-accuracy/90%-geometry acceptance gate. No next-stage
  automatic continuation. Formal and lambda comparison remain blocked.
- Diverse8 lambda0 completed2048 updates but FAILED acceptance. All32 views
  ended naturally; mean main CE=0.0454355, teacher accuracy=0.979061, quantized
  F1=0.0737431 and correct-count fraction=0.0625. Only1/32 views passed the
  joint-count/F1/J2J gate. At512/1024/1536 all32 full predictions were identical;
  at2048 they comprised only two repeated sequences (9 and10 joints), although
  the eight true skeleton sizes are5/34/62/10/52/64/180/9. This is an observed
  cross-asset conditioning failure at this budget, not proof of insufficient
  parameter count or a newly established implementation bug. Single-case
  acceptance therefore does not release G2 or authorize full training.
- Next bounded G2 diagnostic enables the already-approved auxiliary ID loss
  at the predeclared weight0.1. Keep the exact same eight assets/four frames,
  official fresh seed, architecture (residual scale0), AdamW1e-4 and2048-update
  maximum. This tests whether shared structure supervision prevents the
  cross-asset collapse; it does not promote the weight on auxiliary CE alone.
  Apply the unchanged free-generation gate. No further lambda0 budget extension,
  no diagnostic checkpoint reuse, no automatic G3/formal continuation.
- Matched-comparison tooling is available but not GPU-authorized yet:64 fixed
  training assets,8 disjoint heldout-pose assets,32 validation assets with
  low-joint and full-train-unseen exact DFS-topology quotas. Preprocessing
  caches only raw sampled geometry/labels, never learned features. Eleven
  selection tests and eleven runner/evaluator tests passed; an independent
  mocked CPU control-flow audit checked all104 cache items, equal initial-state
  fingerprints/order/schedules and no early quality-based budget termination.
  The runner additionally rejects cross-split duplicate file paths, even when
  IDs differ. Real GPU comparison/smoke and exact candidate-config equality
  remain prerequisites, not claims implied by these CPU tests.
- Diverse8 lambda0.1 also completed2048 updates and FAILED: all32 natural EOS,
  one repeated full prediction, F1=0.1619892, accuracy=0.9767485, CE=0.0451565
  and only1/32 views passing the geometry/count gate. At1024 F1 temporarily
  reached0.3650033 and CE0.0368916, but this improvement regressed. Asset IDs,
  fixed-frame records and the complete update0 evaluation are byte-identical
  to lambda0; initialization records differ only in run config/code provenance.
  Auxiliary supervision alone did not produce stable fitting at this budget.
- Exact empirical prefix-only analysis gives a condition-free main CE minimum
  of0.04454644 on these32 targets, or0.02056472 when asset identity is known but
  pose is ignored. The final failed runs remain near the fully condition-free
  level. The analogous auxiliary-ID bound is0.06383669; its last8 training
  updates average0.07038794 (different successive weights, not a fixed-weight
  evaluation). These are counterfactual memorization bounds, not trained-model
  baselines or proof of a unique cause. The earlier successful single-case
  update256 did beat its own0.00679556 condition-free bound decisively.
- Next diagnostic: paired reference-only fits on the same8 assets/T4, original
  XYZ loss/head, fresh official seed and maximum2048 updates. One calls the
  actual legacy query-first motion encoder for each target frame. The other
  keeps the approved virtual-participating encoder but feeds its FINAL real
  frame tokens directly to XYZ, bypassing only final virtual-query fusion.
  Neither reference may replace the formal model. The second control must
  retain nonzero main gradients through virtual slots and leave fusion unused.
  Six focused reference/parity/authority tests pass, including exact frame
  routing, original-loss equivalence and individually pinned paired configs.
- Preparatory training code remains release-locked. Nine CPU tests cover exact
  asset accounting, formal48-assets/2-rank enforcement, all-T capacity bindings
  and resident-gradient accumulation. A real two-process HGC CPU/Gloo test
  executed the actual accumulation/checkpoint/evaluation flow on a tiny model;
  maximum gradient error versus the analytic global-asset mean was5.55e-17,
  AdamW state matched, and sampler checkpoint replay plus uneven validation
  shards passed. This does not certify NCCL/BF16/full-model GPU capacity.
- Capacity input selection now uses exact original-order token lengths and the
  conservative7*N-1 upper bound, not joint count alone. It also selects maximum
  geometry buffers at every T=2..24. Sixteen CPU tests pass; real selected-token
  crosschecks and GPU measurements are still required. Full-data execution
  requires capacity receipts bound to its model, coefficient and every T/B
  shape, including two-microbatch accumulation with resident gradients.
- CPU comparison predeclaration completed on all18969 real parent arrays:
  train has3424 distinct exact DFS-ID structures; validation has130 assets
  unseen to full train. The fixed64 training cohort is32 base/32 addition,
  with16 low/32 mid/16 high-joint assets. Validation32 includes10 low-joint
  assets and12 distinct globally unseen structures. This is intentionally
  stratified diagnostic membership, not a population-weighted quality estimate;
  G3 GPU work remains blocked until fitting is accepted.
- Paired reference controls completed2048 updates each (both exit0). Legacy
  query-first generated the exact GT token stream for32/32 views, teacher
  accuracy1.0, CE0.00019019 and J2J0. Virtual-final-real generated31/32 exact
  streams; its remaining9-joint view had J2J0.01098115 and raw F1=1, with
  overall teacher accuracy0.9989224 and CE0.00439181. Both full initial model
  fingerprints are a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635.
  IDs, frames and every target token match the failed original fit.
- The old geometric acceptance was not reflexive: nearest-neighbor argmin
  collapses coincident joint indices, so8/32 targets have self-edge F1<0.95.
  Legacy's exact32/32 output still averaged raw F1=0.95348109. Do not rewrite
  historical raw scores or silently replace the project's shared metric.
  Gate v2 accepts exact target token equality as stronger topology evidence,
  OR the original raw F1>=0.95; it STILL requires valid natural EOS, exact
  joint count, J2J<=4/256, >=90percent passing views and teacher accuracy>=0.99.
  Replay of immutable files (reference_gate_audit_v1.json,
  SHA a8467d9d43ed27fd86e6e7381d372d91951d75847b623f6243567fde12f3e99d)
  passes legacy at2048 and final-real at1536. Both failed original lambda0/0.1
  runs still pass only1/32 views at2048 and remain rejected.
- These controls localize a fitting bottleneck to the final fusion/condition
  interface under this recipe; the virtual-participating encoder and retained
  RT data CAN fit the cohort. They do not justify replacing virtual fusion
  with direct real tokens in the formal model. Next candidate is the existing
  aligned final-frame residual, C=LN(Fusion(V,A_t)+A_t), tested fresh on the
  single asset (max512) and identical diverse8 (max2048), lambda0 and unchanged
  constant AdamW1e-4. The original R1 single max128 failure was not conclusive.
  Full formal release remains blocked until candidate G2/G3/capacity review.
- Diagnostic-only optimization controls are now testable without changing
  defaults: optional mean-over-assets accumulation and the established
  exposure-based OneCycle schedule. Eighteen CPU tests passed for the updated
  runner/evaluator: default AdamW trajectory and moments exactly match, actual
  accumulation matches the direct mean, exposure/LR accounting and exact
  paired authority are checked. Neither option is selected for the residual
  comparison, which isolates the architecture change.
- Real capacity metadata is now certified:52 parent arrays inspected and15
  synthetic-XYZ original-tokenizer crosschecks passed. The longest main target
  is1362 tokens; largest geometry-buffer proxy atT24 is0.32782GiB. This is input
  selection evidence, not measured GPU capacity. CPU real8 replay also
  reproduced all32 target streams and found no literally identical mesh input
  with conflicting target tokens among48 within-asset pose pairs.
- Publication is currently blocked by automatic safety review pending explicit
  human authorization for pushing code/configuration/diagnostic summaries to
  the existing GitHub origin Ooosn/Evoweave. Local result commit fdffcf8 and
  tested runner/gate commit c1429e7 are preserved; no new GPU operation has
  been registered or launched. The remote/node remain at d540542. Do not use
  a bundle or another transport to bypass the denied publication requirement.
