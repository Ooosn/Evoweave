# Structure-conditioned XYZ token filling

This is the October22JST implementation direction authorized by the user.
The historical A/B run is stopped. Preserve its checkpoints, evaluations and
allocation; do not restart B80 or the former expanded-validation automation.
Exact GPU operations still require publication in `state/current.json`.

## Model Contract

- Keep the shared surface encoder and 1024-anchor virtual/query motion network.
  The structure and XYZ losses both train the shared features/attention through
  their own continuous input paths. Final mesh anchors are not detached.
- The existing two-layer explicit-ID decoder autoregressively predicts one
  `BOS J0 ... BRANCH parent Jnew ... EOS` structure, supervised by full GT
  structure CE. Inference respects the same active-ancestor grammar.
- XYZ receives discrete structure tokens only, not the structure decoder's
  hidden states or logits. It never predicts another skeleton.
- Insert three unknown coordinate tokens after each newly created joint.
  A BRANCH parent reference has no additional coordinate slots. Each XYZ
  position has an axis-mask embedding, its joint-ID embedding, and an expanded
  sequence position embedding. All are owned by the XYZ module.
- Four pre-LN residual blocks, width1024, eight heads, FFN4096, GELU, dropout0.
  Each block performs full bidirectional self-attention over structure plus
  coordinate slots, cross-attention to the current frame's final mesh anchors
  plus input-anchor 3D position encoding, then an FFN. No causal XYZ mask and
  no additional cross-frame or reverse XYZ-to-mesh attention are introduced.
- Internal structure and XYZ vectors update at every layer; discrete topology
  stays fixed. The same XYZ-module parameters process every frame.
- A shared 256-class head predicts X/Y/Z tokens at their slots in parallel.
  Use GT coordinates only as labels. This is not continuous XYZ regression,
  not autoregressive coordinate generation, and not iterative hard-token refill.
- Position capacity3076 is for the expanded structure/XYZ sequence. Mesh
  cross-attention memory does not consume these positions. Overflow is an
  error, never truncation. With the existing max256-joint grammar the worst
  expanded target is1534 tokens.
- There is no cross-branch alignment, consistency or representation loss.

## Coordinates And Initialization

Keep the exact official256-bin coordinate convention: first verify finite
coordinates in the declared continuous range; quantize with rounded scaled
values and legal-bin saturation; decode with bin-center offset+0.5. Do not
replace this with floor quantization or a different bin denominator.

Keep retained-root RT and the selected-input mesh-union bbox with padding1.1.
No GT-derived normalization, label clipping, mesh substitution or source-joint
deletion is permitted. The original complete GT remains the evaluation target.

The user explicitly requires NO inheritance from old A/B experiment weights.
The sole pretrained checkpoint is the pinned official UniRig release. Load
only its compatible mesh_encoder and output_proj groups with strict key/shape
and finite-value checks. Construct the virtual/query motion network and the
structure ID decoder afresh with seed20261013. Construct the entire XYZ fill
network with seed20261022. Do not construct an old full AR model to harvest
weights, load old experiment state, or restore an old optimizer. All counters
start at zero. The deleted B65 model is NOT a dependency; only its preserved
evaluation JSON is used as a matched historical reference.

Two v2 candidates differ ONLY in XYZ sequence position initialization. R uses
fresh positions; P independently copies the official OPT text-position rows
1026:3078 into its first2052 rows (OPT offset2 plus former mesh prefix1024).
Remaining position rows stay fresh. Both candidates use fresh structure-ID and
X/Y/Z-mask embeddings, attention, FFN, geometric-position MLP, norms and output
head. No trained ID embeddings exist in the official coordinate vocabulary.
Record exact source tensors and row offsets; copying positions is a hypothesis
to test, not an assertion that old AR pretraining transfers to the new task.
CPU initialization audits must show identical common parameters and identical
fresh attention/FFN/head parameters between candidates. GPU initialization
must exactly match its reviewed CPU parameter fingerprint.

Both arms run sequentially on the retained two-H100 allocation, not concurrently
with another route. Use the same seed, data order and effective48-asset batch;
initial bounded budget512 new asset exposures with128/256/512 checkpoints and
an initial evaluation. Preserve both optimizer-step and exposure counters.
Pretrained surface LR is3e-5; fresh motion, structure and fill LRs are1e-4.
Use fresh AdamW betas(.9,.999), no weight
decay, clip1, and a192-asset linear warmup from20% followed by constant LR.
Loss is mean coordinate CE plus mean structure CE, each normalized per asset.
This is an independent new-model comparison from a common official surface
initialization, not a continuation of old A/B training.

## Random Terminal Pruning

Draw a fresh batch-level gate:80% unchanged,20% prune. The gate is shared across
the optimizer update, independent of rank and physical microbatch partition.
On a pruned update each asset/visit independently draws a fraction in0..20%
of its nonroot joints. Delete K=floor(fraction*(N-1)) nodes by uniformly
sampling from the whole current nonroot-leaf pool after every deletion.
Use child counts plus swap-pop leaf maintenance, O(N+K), then compact once.

The full structure CE target is unchanged. Only XYZ's discrete input structure,
slot set and corresponding labels are pruned. Re-encode compact DFS IDs and
parent references, retaining the map back to original GT source indices.
All frames of an asset share the deletion set for that visit. No second
structure-decoder pass is necessary because XYZ does not consume its states.

## Data And Experiment Route

Use the existing immutable5000-row training subset, SHA256
`a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`, and the
967-row validation pool with SHA256
`fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a`.
Keep complete manifest rows/root_rt and the existing eight heldout frames.
Do not rescreen/rebuild accepted data or replace it with a fixed small cohort.

Implementation validation comes first: token/GT mapping, exact quantization,
bidirectional slot updates, shared-upstream gradients, absence of a hidden-state
bridge, dynamic pruning invariants, serialization and natural-EOS handling.
Then real-loader label/geometry checks, a bounded fit check and measured GPU
capacity. A failed data or mapping check blocks training, not just reporting.

Early matched trials use a fixed small train/heldout/validation panel and more
frequent evaluations. Measure GT-structure localization separately from free
structure-plus-XYZ generation; log coordinate CE/accuracy, geometry error,
partial structural quality, original-GT edge F1 and failure counts. Do not gate
learning solely on exact whole-tree equality. Failed generations count as zero
in aggregate quality, not disappear from the denominator.

The early panel is a fixed seeded subset of the already accepted B65 panel:
8 train assets, their8 heldout-pose cases and16 validation assets. Retain eight
input frames to preserve the original normalization/context; score positions
0,2,4,6, giving128 views rather than1536. Verify raw-frame identities and exact
old GT token sequences before reusing reference scores. This is not a new
full-validation pass and is not an estimate precise enough for small final
quality claims. The original spatial edge-F1 functions were extracted without
AST changes into a shared pure-NumPy module; old and new evaluation use them.

Compare the same initial shared tensors, asset order, frame choices, sampling
seeds, optimizer schedule and exposure checkpoints. Record both optimizer
updates and asset/frame exposures; the historical20k/40k numbers are asset
exposure milestones, not20000/40000 optimizer updates. Publish the concrete
short-run milestones/panel before launch. Only after finite execution, useful
GT-conditioned learning and nondegenerate free-generation trends may a selected
version extend to20k,40k and later budgets. No full validation at every early
checkpoint, no long-run automatic promotion on training-loss reduction alone.

Stop on nonfinite values, target/normalization/manifest drift, unexpected
truncation, wrong GT mapping, missed/duplicated evaluation rows, resource drift
or failed supervision. Diagnose the saved failing batch before resuming.

## Current Implementation Stage

Local implementation passed160 related CPU tests and54 subtests, including
legacy metric/data/sampling regressions and both saved visualization formats.
The CPU audit completed with exit0: all5000train/967valid parent-to-token
mappings,10 real-loader checks at2/24frames, exact original coordinate-token
parity, and128 matched old/new GT/frame identities passed. No data was reselected.
Maximum expanded fill length is1210(train)/1012(valid), below capacity3076.
The matched small-panel B65 reference F1 is0.873763(train seen),
0.845508(train heldout poses),0.681973(valid); these are NOT full-panel scores.

The bounded random-fill GPU preflight was published for two
boundary forward/backward cases, then two fixed non52-joint assets with four
frames each,32 optimizer updates, evaluated at0/8/16/32. Its fitted weights are
diagnostic only. Capacity profiling and normal training remain unpublished;
execution success alone does not establish learning or quality acceptance.

The superseded v1 preflight exited1 before model construction: the optional B65 checkpoint
had been deleted. The user confirmed deleting the old checkpoints. No model
computation or optimizer update occurred. The official UniRig checkpoint still
exists. B65 was a primary-selected warm start, not required by the architecture;
do not silently replace it. The user then explicitly rejected old experiment
inheritance and authorized completing validation through readiness for formal
5k training. The v2 initialization above supersedes v1. CPU initialization
checks precede the next exact bounded GPU stage; no automatic quality approval.

The v2 paired GPU fits subsequently passed real gradient checks and32-update
fits on the same21-joint and54-joint assets, four frames each. XYZ CE declined
77.8%--97.8%, L1 declined96.0%--99.2%, and both generated the exact target
structure by32updates. Plots confirm pose-specific geometry. The21-joint
quantized-GT reference F1 is0.70 under the unchanged coordinate protocol;
its original continuous GT self-F1 is1.0. The54-joint sample remains imperfect.
This demonstrates bounded trainability, not5k generalization. Full reviewed
figures and provenance are in `analysis/structure_xyz_fill_preflight_v2_review.json`.

Before longer trials, v3 revalidates failure-distance summaries and nonfinite
generation aborts, then measures the nine declared capacity cases. Initialization
receipts bind the official loading implementation/contract and the complete
actual parameter fingerprint; forward/evaluation source changes require fresh
GPU preflight evidence rather than pretending weights were reinitialized from
a different source. Old initialization tensors must remain byte-identical.

Checkpoint tests cover mid-budget restart, completed-budget extension, final
evaluation recovery, AdamW parameter identities/moments/step counts and RNG
replay. Reviewed artifacts are bound to their producer code/config identities
and current target/model source signatures, not only a generic passed flag.
