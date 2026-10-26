# Reference Skeleton To Posed-Mesh XYZ

## Authorization And Question

On2026-10-26T19:42:50Z the user requested that, after the four-frame experiment,
research and experiments begin on transferring a skeleton from one mesh pose
to other observed poses. The prior agreed direction includes reference XYZ,
parent relationships and reference-mesh interaction, not a topology-only code.
Both preceding density runs are complete; their artifacts remain immutable.

Start with a separate task-specific readout on the shared surface/motion
backbone, not a duplicated full encoder. Frozen readout tests whether existing
features can be used. Joint XYZ supervision is a separate subsequent question;
failure of the frozen readout does not prove shared features cannot be learned.
Likewise, head-only success does not prove multi-task training will be harmless.

## First Implementation

- Reuse the no-virtual real-frame encoder from random20%-40% min2 at20k.
  This is a practical research starting checkpoint, not a proven winner.
- Branch before the AR-specific frame readout. Retain explicit anchor XYZ and
  normals so coordinates do not have to be reconstructed from latent features.
- Reference joint coordinates, parent offsets and root flags produce queries.
  Self/parent/child/sibling/other attention biases encode graph relationships.
  No learned semantic joint-ID vocabulary or joint-index positional embedding.
- Reference joint-to-mesh attention grounds these queries in the reference pose.
  Target memory includes shared target/reference features, both anchor positions,
  observed anchor displacements and normals. Anchor correspondence is inherited
  from the established trackable surface sampler.
- A256-wide,8-head readout has2reference and2target blocks. It outputs continuous
  XYZ offsets. The given reference pose is preserved exactly and excluded from
  losses and quality metrics. Joint count and parents are inherited, not regenerated.
- Attention distance is a soft reference-geometric prior, not an assertion that
  nearest surface points or attention weights are correct skinning weights.
- No virtual frame, diffusion prior, joint-time module, velocity smoothing or IK.
  Inputs are sampled pose sets, not assumed consecutive video frames.

This borrows the reference-coordinate/graph/mesh interface, not an entire model:
[MoCapAnything section3](https://arxiv.org/html/2512.10881v1#S3.SS3),
[AnyTop section3](https://arxiv.org/html/2502.17327v1#S3).
MoCapAnything starts from a known rig; natural predicted-reference error is an
additional issue here. The earlier pure-parent shared-structure failure is not
evidence that this coordinate-grounded interface has already been tested.

## Bounded Frozen Pilot

Exact recipe: reference_skeleton_xyz_frozen_pilot_20261027_v1.json.

-64train and32valid assets from the same immutable5000/967manifests. Four equal
 joint-count strata:1..20,21..51,52,53+. Deterministic SHA ordering with seed20261027;
 no asset replacement after seeing difficulty. This is a stratified diagnostic,
 not a population-representative sample or the earlier128-asset comparison panel.
-8complete observed poses per asset, using the existing retained-root RT,
 heldout-raw-frame partition, surface sampler and selected-input-union bbox.
 The eight reserved raw poses never enter this cache. Pilot validation measures
 heldout assets; it does not yet evaluate reserved raw poses of training assets.
- Freeze every donor parameter; cache the shared features on NVMe. Train only
 the new head for800updates,2assets/rank x2ranks:3200asset visits and22400other-pose
 target visits. Uniform random reference slot per visit; all seven other poses
 receive continuous L1, averaged coordinates, then joints/poses, then assets.
- AdamW peakLR3e-4,20update linear warmup, cosine to3e-5, weight_decay1e-4,
 gradient clip1. Save full head/optimizer/RNG/selection bindings at400and800.
- Evaluate fixed reference slots0and4 at0/400/800. For validation also generate
 those references naturally with the frozen legacy AR decoder, with no GT
 topology, joint-count oracle or GT-based correspondence injected into inference.
- Controls: unchanged reference coordinates, and weighted local Kabsch transfer
 using32observed corresponding anchors. Degenerate local fits use translation.
- First run a dual-GPU capacity/gradient check; only its accepted completion
 releases the800-update pilot. No100k run or automated architecture search.

## Metrics And Interpretation

For GT reference, report joint-ID-aligned mean Euclidean position error divided
by each target mesh's bounding-box diagonal (normalized MPJPE), and the fraction
of joints within5%of that diagonal (PCK@0.05). The latter is a declared diagnostic
threshold, not an established production acceptance criterion. Report both
per-view and per-asset evidence, and compare against copy/local-rigid controls.

Also preserve the project's nearest-joint edge F1. It has no spatial distance
cutoff and alone is insufficient to establish precise localization. Predicted
references may have different joint counts/order: use symmetric nearest-joint
distance and edge F1, not a misleading same-ID MPJPE. Failed references count
as zero edge F1 across their target views; successful-only distance metrics are
explicitly labeled. Topological consistency is by construction, not correctness.

GT-reference results are oracle diagnostics, never end-to-end claims. Natural
predicted references measure the extra error from stage one. Preserve both
old checkpoints and all pilot evidence before deciding the next jointly trained
or task-specific-layer experiment.

## Resources

Same original HGC job130494177, keeper1098441,2H100/16CPU/64GiB. Source and exact
stage configurations must be committed, admitted and immutable during compute.
The keeper's fixed launcher now dispatches this explicit new family to its own
trainer; all existing membership, UUID, affinity, RAM and GPU guards remain.
The48h allocation extension must be verified live before launch. No new GPU
allocation or release is requested. The donor remains train_v9; new outputs are
capacity_v11 and train_v11, never overwrites of earlier arms.
