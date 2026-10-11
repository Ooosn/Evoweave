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
