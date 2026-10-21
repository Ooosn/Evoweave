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

Initialization is selective, not all-or-nothing. Attention, FFN and output head
are fresh by default. Compatible structure/position embeddings are explicit
initialization candidates with recorded source rows and offsets, never silently
shared parameter objects. New joint/axis token meanings must not be confused
with known coordinate-value embeddings. Publish the exact common upstream
checkpoint and its fingerprint before GPU work; do not pretend a warm-start
comparison is a from-scratch comparison. Preserve the accepted old B65 model as
an independently measured reference rather than rerunning old A/B training.

The first paired trials warm-start the common surface/motion network and ID
decoder from accepted B65:65000 prior asset exposures,856432 supervised-frame
exposures,1365 optimizer updates; checkpoint SHA256
`1555619a677f13bcb2e044c2b7e72cf623483490ab0edc0eac24cb24b8ead7f9`.
Its shared/structure tensors passed the CPU finite/shape audit. Do not restore
its old optimizer or load its old coordinate backend into the fill network.
The random-fill control and selective-embedding candidate have the same fresh
attention/FFN/head tensors. The candidate independently copies the ID embedding
and original text-position rows after OPT's actual offset and1024 mesh-prefix
positions. Any remaining position rows and new X/Y/Z masks stay fresh.

Both arms run sequentially on the retained two-H100 allocation, not concurrently
with another route. Use the same seed, data order and effective48-asset batch;
initial bounded budget512 new asset exposures with128/256/512 checkpoints and
an initial evaluation. Preserve both optimizer-step and exposure counters.
Common/structure LR is3e-5, fill LR1e-4, fresh AdamW betas(.9,.999), no weight
decay, clip1, and a192-asset linear warmup from20% followed by constant LR.
Loss is mean coordinate CE plus mean structure CE, each normalized per asset.
This is a warm-start backend comparison, not a fresh full-model comparison.

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

The next published operation is the bounded random-fill GPU preflight: two
boundary forward/backward cases, then two fixed non52-joint assets with four
frames each,32 optimizer updates, evaluated at0/8/16/32. Its fitted weights are
diagnostic only. Capacity profiling and normal training remain unpublished;
execution success alone does not establish learning or quality acceptance.

Checkpoint tests cover mid-budget restart, completed-budget extension, final
evaluation recovery, AdamW parameter identities/moments/step counts and RNG
replay. Reviewed artifacts are bound to their producer code/config identities
and current target/model source signatures, not only a generic passed flag.
