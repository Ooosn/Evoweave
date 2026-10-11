# Reusable Skeleton Sequence Generation

## Scope

The user approved implementation and bounded feasibility testing of a new
generative decomposition. Previously trained weights are a temporary way to
test the method, not a constraint on its architecture or the final training
recipe. Formal full training belongs after method validation. This experiment
does not replace the preserved original checkpoint.

The primary product is an unknown, variable-length skeleton-token sequence
that can be reused across poses/motions of the same underlying skeleton.
Per-mesh localization is its supervision and deployment interface, not the
research objective. This is a hierarchy within our existing generative
pipeline, not a known-skeleton motion-reconstruction task. A token includes
continuous joint features and its generated parent relationship, not XYZ.

## Architecture

1. Preserve tracked surface anchors across frames. Their temporal attention
   retains the physically corresponding anchor index.
2. Let frame registers attend to all frame/register slots, not only the same
   register index. Reuse the existing temporal layer weights. The original
   aligned route remains the default and strictly checkpoint-compatible.
3. Pool all frames into 128 learned shared structure slots using two
   cross-attention blocks. These slots do not have fixed anatomical labels.
4. Autoregressively generate one parent token per joint, followed by EOS.
   Root 0 is unique and each subsequent parent precedes its child. Joint
   features are decoder hidden states after each joint's own token. No XYZ
   values enter the structure prefix.
5. Expose the skeleton sequence as an independent serializable result. A
   separate two-block position decoder binds it to a target mesh's independent
   surface features/coordinates. Rebinding does not rerun structure generation
   and does not require that mesh to have participated in motion encoding.
   The target head predicts three coordinate tokens per joint in parallel.

Shared structure is not a replacement for local spatial features. The position
decoder reads the target mesh's full independent anchors directly, not source
temporal states. Registers are not assumed to equal
joints or body parts. A shared output tree guarantees one tree within a single
inference call; it does not by itself prove correct joints or invariance when
the input frame subset changes.

## Targets and Coordinates

Rootless NPZs already store `(frames, joints, 3)` posed joint coordinates and
stable target parent/identity arrays. The new adapter exposes all selected GT
frames without altering the existing reader or its query-mesh normalization.
No rest-pose target, active-joint pruning, synthetic root, or tail target is
introduced. Structure order never depends on the coordinates of a new frame.

Important correction: v3 deletes auxiliary root/RT rows from the target tree;
it does not apply their inverse transforms to training geometry. The build
copies posed coordinates into the `*_rootspace` fields and writes identity
training root transforms. The actual deleted transforms are record-only.

A read-only audit of 48 assets/24 selected frames found common-query targets
in `[-3.0791483,3.9934535]`. In the largest case, mesh-center displacement
accounts for 3.66088 of the 3.99345 coordinate; the joint's value in its own
frame is only 0.35763. The proposed `[-4,4]` vocabulary was therefore withdrawn.

For localization, independently center each target mesh at its own bounding
box and divide mesh and GT by its maximum half-extent times a fixed 1.1 padding.
This transform is mesh-only and invertible, with its source-space affine
recorded. Source motion features retain their existing common-coordinate
chart; these two interfaces must not be confused or silently mixed. The
canonical NPZ and base loader are unchanged. There is no cross-frame XYZ
equality requirement and no shared target-frame coordinate in the skeleton
sequence.

The unpadded mesh-local audit range was `[-1.0053182,1.0139827]`, with 72 of
151,560 XYZ scalars just outside `[-1,1]`. Fixed 1.1 padding covers this bounded
audit. The new head keeps 256 bins over `[-1,1]`, but its grid spacing is now
in padded target-mesh units, not old common-query units. All new selections
must pass strict range checks, including the 32-frame probe pool; no clipping,
GT-derived scale adjustment, or joint removal is allowed. This is not a
full-dataset coverage claim.

Range evidence: external job `results/shared_structure_target_range_20261011.json`,
SHA256 `9a6c333c6c7140af125b9a7e4b4e0ddc88c0a27d5510083c71e6520d62c7fb57`.
Coordinate diagnosis: `results/shared_structure_coordinate_diagnosis_20261011.json`,
SHA256 `728a3fc6887ff3019e5a5cb908e5ca5ca52b294f903c348ee8a9d4f996b3a71f`.

## Bounded Validation

The exact configuration is `../experiments/shared_structure_probe_20261011.json`.
Use the immutable completed bias_h8 checkpoint as an initialization. Freeze its
geometry/motion feature encoder for this first cheap method test; train the
new shared memory, new parent-token embedding/head, inherited AR backbone,
and new frame position decoder. This is not the final from-initialization
full-training recipe. Disable unused original input embeddings and dropout.

Preflight uses minimum/maximum-joint assets from the selected train/validation
sets and T=2,8,24. It checks target identity, coordinate range, feature-cache
parity, structure causality and prefix parity, backward/update paths, sequence
serialization/rebinding, source immutability, and a T24/J512/32-target-frame
memory stress case. Only this exact stage may run
until its result is reviewed and recorded.

The subsequent small probe uses eight seeded training assets and sixteen
validation assets, dynamic T=2..24, 240 steps, and four one-asset microbatches
per update. These reused validation assets are exploratory, not a new held-out
quality benchmark. Quantitative comparison is initialized versus trained new
model on identical inputs, not an unfair claim against the old full model.

For bounded cost, use a fixed 32-frame pool per asset: 24 evidence candidates
and 8 target frames excluded from structure generation. Draw dynamic T=2..24
only from the evidence candidates; supervise those T frames plus the 8 reuse
frames, always with the full joint set. Cache only independent per-frame
surface features and raw source point tracks BEFORE motion encoding. Target
features use the independent mesh-local chart.
After selecting the actual T input frames, recompute motion evidence and the
temporal encoder. Never cache full-clip contextualized features and then slice
them into a supposedly smaller input. The fixed pool/reference is disclosed
as a diagnostic fixture; full training must restore fresh query/frame draws.

The loss averages XYZ supervision over the valid frames and joints of each
asset; structure is supervised once per asset. Accumulation averages actual
assets. A long clip does not silently increase its structure or location loss
weight. No blanket register-vector equality loss is introduced.

Evaluate true-structure localization separately from freely generated trees.
Report full-generation failure, joint counts, parent relationships, per-frame
joint geometry, and dependence on the input structure memory. The central
reuse check saves a sequence inferred from source evidence and applies it to
the excluded target meshes without regenerating or editing it. Also compare
sequences inferred from different evidence subsets on identical target meshes.
Report parent-sequence agreement and identity-preserving localization, not
just nearest-joint geometry. Do not assert that equal tensors or one shared
output inside one call establish pose/motion invariance. The present data
supplies one selected action per asset; disjoint poses from that action are
only a bounded reuse proxy, not proof across distinct actions. All frames share
one generated tree, but swapped joint identities or missing joints still count
as errors. A length cap is a failure boundary, not forced successful EOS.

## Operations

Keep the original checkpoint, caches, and prior results immutable. Use only
one already-granted H100 after live allocation/device/process checks. Maintain
a clean, committed server checkout while GPU work is active. Every stage has
its own prepared state, exact config, result artifact, and acceptance receipt.
Do not retry failed stages automatically, relax range checks, or bypass the
work guard. A completed diagnostic is not permission for full training.

The first launch stopped before the model child started because the common
runtime helper looked for `resolved_env.txt` next to the newer bias_h8
checkpoint. Its original environment was inherited from the historical flat
run, not copied to that directory. The failed launch log/result are retained.
Attempt `shared_structure_r2` explicitly pins that same environment file and
its checksum, and verifies it before preparing an operation. Model architecture,
data selections, and optimization settings are unchanged by this repair.

## Completed Feasibility Result

Attempt `shared_structure_r2` completed preflight, cache, and the 240-step probe
with all outer/controller/child exits zero. Training used 960 real asset
exposures over eight assets, with 12,531 evidence-frame and 20,211 target-frame
exposures. The geometry/motion feature encoder remained frozen, the original
checkpoint was unchanged, and the surface-cache checksum still matched after
training. No formal full retraining or model promotion was performed.

Preflight passed 12 real cases, two optimizer updates, sequence save/rebinding,
and localization-only gradients into the shared memory, joint embeddings, and
inherited decoder. Maximum BF16 prefix-logit discrepancy was 0.0078125 against
the declared 0.03 tolerance. The synthetic 24-source/32-target-frame, 512-joint
stress peak was 9.5284 GiB. The bounded training peak was 10.8434 GiB. These
figures apply to this frozen-feature probe, not a full-model training budget.

The main result is learnability of the generated sequence, not just reconstruction
under GT prefixes: all eight training assets freely generated the exact joint
count and parent sequence under each of T=2, T=8, and T=24. The sequence is saved
as joint features plus parents/mask, with no target-frame XYZ. Evaluation reloads
it and calls the mesh localizer without rerunning the structure encoder or AR.

For the T8 view (one result per asset):

| Measurement | Training, Before -> After | Validation, Before -> After |
| --- | --- | --- |
| Natural EOS | 3/8 -> 8/8 | 3/16 -> 16/16 |
| Exact joint count and parent sequence | 0/8 -> 8/8 | 0/16 -> 0/16 |
| Exact joint count alone | 0/8 -> 8/8 | 0/16 -> 1/16 |
| GT-prefix structure CE | 2.703208 -> 0.039194 | 2.799481 -> 1.871820 |
| GT-prefix target-frame MPJPE | 1.059743 -> 0.034388 | 1.230323 -> 0.560986 |
| Free-generation one-to-one joint coverage | 0 -> 0.908884 | 0 -> 0.054978 |

MPJPE is mean Euclidean joint error in each target mesh's padded half-extent
units. Coverage uses a 0.05 distance threshold and one-to-one matching, with
generation failures scored zero; both summaries weight assets equally. Natural
EOS means syntactic termination, not correct structure. Exact parent sequence
is the dataset's ordered identity criterion, not unordered-tree isomorphism.

The training T8 generated-sequence MPJPE is 0.034182, approximately 1.88% of
the mesh's largest bbox side. This average is not uniform success: asset 6189
has error 0.170148. A stricter fixed-GT-identity check finds 204/336 training
joints within 0.05 in **all eight** target frames. This checks persistent joint
identity instead of independently rematching each frame.

The structure memory is used on this small training set: with the correct GT
prefix but zeroed memory, structure CE rises from 0.039194 to 2.677090 and target
MPJPE from 0.034388 to 0.532470. This is a representation-dependence diagnostic,
not an ablation proving any particular motion-evidence channel is necessary.

Changing the seven non-reference evidence frames in the validation T8 view
preserves the exact generated parent stream for 11/16 assets. None of those
validation streams is completely GT-correct. Consistency of an incorrect
stream is not evidence that the desired skeleton representation is correct.

Crucially, the eight target frames are excluded from **structure inputs**, not
from training supervision on the eight training assets. Validation assets are
not used for optimization. The fixed pool also keeps one query/reference frame
in every source subset. The current evidence therefore establishes the separated
interfaces and small-set learnability, but neither unseen-asset quality nor
cross-action invariance. It must not be reported as a finished universal token
representation. Formal retraining remains a separate phase with fresh data and
resource validation; freezing the old encoder is only a feasibility-test choice.

The raw-mesh normalization API was subsequently made label-optional and tested
on CPU. Inference needs only mesh vertices/counts (plus faces/normals for the
surface tokenizer); absent input affines mean raw source coordinates. No GT
joints, masks, or parents are needed to prepare the target mesh. This API-only
extension does not change the supervised normalization math used above.

Artifacts under `/home/wangyy/evorig/outputs/shared_structure_probe_20261011_r2/`:

- `cache/surface_cache.pt`: SHA256 `b7ab23a15caef31b6c811f879a0212e76ab084a87566b63fe25e603d2fa0afa8`.
- `probe/checkpoint_probe.pt`: SHA256 `ac8cda59b6f055aa8a774251c78636d87df46c049ec5cc3df93e432c7a46f6d5`.
- `probe/sequences_trained.pt`: 88 generated sequence records, SHA256 `0bf57e438fa204023e16579c578a7133ab89c2626117bc82f64c5fe780937879`.
- Evaluation: job `results/shared_structure_r2_probe_evaluation.json`, SHA256 `685ddaca83921f38a5fe126e3f8b14ce6de5348983edab5d4cf97658660003af`.
