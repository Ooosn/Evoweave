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
