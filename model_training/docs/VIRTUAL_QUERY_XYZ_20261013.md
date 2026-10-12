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
