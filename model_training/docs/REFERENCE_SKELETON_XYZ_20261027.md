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
capacity_v13 and train_v11, never overwrites of earlier arms. The initial
capacity_v11 attempt was blocked before computation because the lifetime guard
also matched historical submit arguments; its corrected parser reads only the
current hard_resource_list, with a regression test. The failed receipt is retained.

capacity_v12 completed its numerical checks but failed process-completion monitoring
during root-cgroup teardown. The supervisor now ignores this only after bounded,
demonstrated process exit; persistent live mismatches still fail. The repeated
capacity_v13 passed numerical checks and clean exit0 at2026-10-26T21:07:37Z.
It tested171/126joints,8poses,batch2on each rank, with all head gradient groups
present and the donor frozen. Peak allocated tensors were4.23GB/rank, reserved
4.45/4.43GB; observed total process RAM peaked6.18GB. The capacity head was discarded.
Result SHA256:6ac00c49117f97db44f1199e4acdd12f869a6427afa44901548cb15554108376.
## Completed Copy-Initialized Pilot

train_v11 stopped before head updates because of a natural-reference helper
import error. The corrected import and complete GT/predicted evaluation path
have CPU regression coverage. train_v12 completed800updates at2026-10-26T21:21:58Z,
with clean supervisor exit0 and both GPUs idle. Producer:f124566541a03c970a2f7ac9edd2063c1690fc99.
Result SHA256:b59a89eca06f5cec390d6848c693e1982bc1f94120d823b490d131c72f20c5b8.

GT-reference normalized MPJPE (fraction of target mesh bbox diagonal):

| Split | Copy | Local rigid | Learned |
| --- | ---: | ---: | ---: |
| Train64 |0.08036176|0.00475108|0.05789605|
| Valid32 |0.09860527|0.00523756|0.07820311|

Valid learned-minus-copy paired-asset difference:-0.02040215, stratified bootstrap
95%CI[-0.03129108,-0.01079889]. Learned-minus-rigid:+0.07296555,
CI[0.05453073,0.09353446]. The learned head is not a competitive tracker here;
this does not establish why or rule out jointly trained features.
Natural predicted-reference edge F1:copy0.36259814,rigid0.61139654,learned0.36433987.
Four of64references failed, contributing28of448target views with F1zero.
Successful-only normalized nearest-joint distance:0.08013879/0.02351093/0.06767666.
Do not compare this balanced32asset/oracle diagnostic directly with the earlier
128asset density benchmark. The key geometric assumption is known surface-anchor
correspondence between these posed meshes, not arbitrary unrelated meshes.

## Matched Rigid-Residual Pilot

Exact recipe:reference_skeleton_xyz_rigid_residual_20261027_v1.json.
Keep architecture, donor,64/32assets,8poses,head seed,800updates,optimizer,
per-rank asset/reference sequence and evaluation unchanged. Change only the
coordinate baseline:local-rigid XYZ plus the learned offset from the original
head. Zero output initialization is therefore exactly the rigid baseline on
every scored pose. The provided reference frame remains exact and excluded;
padding is zero; no gradient or GT-target input enters the geometric baseline.

Read the completed train_v12 caches in place on verified NVMe. Pin selection,
both rank manifests, every payload digest, producer/config/donor bindings,
rank-local order, shapes and disjoint reserved raw-frame IDs. Do not rebuild
the encoder or regenerate predicted references; do not copy small files to HDD.
Each output manifest retains the original producer plus an explicit reuse
binding. New head checkpoints use a separate frozen_rigid_residual_v1 root.

37 focused CPU tests pass, including zero-residual identity, reference/padding
gradients, actual cache loading/tamper rejection, both evaluation variants and
existing scheduler guards. capacity_v14 must pass3updates/rank on each rank's
largest cached skeleton and clean resource completion before train_v13 is
released. This is a bounded follow-up, not a full second encoder or100k run.

capacity_v14 passed and exited cleanly; observer at2026-10-26T21:40:07Z verified
both GPUs idle and keeper retained.126/171joints,8poses,batch2per rank, all four
gradient groups present; peak allocated tensors0.583/0.676GB. All reused payload
digests passed. Producer:efec0bd9e4f0e93a11938fc44e4a83d7a0629161.
Result SHA256:544b80e894573860bb75873ef9ad39d25185b14a4733c9d9460cf7d1cc42bde2.
The capacity head is discarded; train_v13 starts fresh with unchanged code/config.

## Completed Rigid-Residual Result

train_v13 completed800updates/3200asset visits/22400other-pose targets with
clean supervisor exit0. Observer at2026-10-26T21:42:59Z found both GPUs idle and
the original keeper retained. Producer:979acbe03a015fd16e4e3a31b9c9999e953a9645.
Result SHA256:33b8ba3b3c974161895bf3e5df4357acab7c9d8ea9d3229365f9ef61982fa9cb.
Peak process-tree RSS:5722169344bytes. Both60,338,618byte head checkpoints have
producer-verified SHA-identical NVMe/durable copies and observer-checked sizes:
400:e95533a4bdf1d54fa40fbdf590c256f305d60cb4452ab11c2c4f80845166ab85;
800:0f7997308a6484c967d2c6df164d9bdd9b2b77c61b7322c51a51aa654b55e4a6.

Independent aggregation rehashed all24small artifacts and verified unchanged
selection, original cache/payload bindings, all1600rank-update asset/reference/LR
sequences, and every copy/local-rigid metric row against train_v12. At step0,
the residual method exactly reproduces the scored local-rigid baseline.

| Split / Metric | Local rigid | Rigid + learned residual |
| --- | ---: | ---: |
| Train GT normalized MPJPE |0.0047510754|0.0047514243|
| Valid GT normalized MPJPE |0.0052375628|0.0052391126|
| Valid GT PCK@0.05 |0.9789816819|0.9789816819|
| Valid predicted-reference edge F1 |0.6113965357|0.6125576294|
| Valid predicted-reference successful-only normalized J2J |0.0235109305|0.0235108277|

Valid GT learned-minus-rigid error:+1.549735e-6, paired stratified asset95%CI
[7.731345e-7,2.402817e-6]. This is a very small deterioration, not an improvement.
Predicted-reference edge F1 difference:+0.00116109,95%CI[-0.00327716,0.00535316];
no reliable incremental benefit. Four failed references remain unchanged.
Step400 GT valid error0.00524544 also does not improve over the baseline.

Posthoc large-motion diagnostic:247of448GT validation target views have static
copy error above5%of the target bbox diagonal. Local rigid error0.00775287 versus
copy-initialized learned error0.12548115, so geometric success is not limited to
near-static poses. This subset is exploratory, not a separate held-out test.

Analysis: D:/evoweave/server_ops/outputs/reference_xyz_20261027/train_v13_analysis.json
SHA256:aa8806f26294cbdf7301e2fee9d7d2537589a7923ad13f523cb11ddc529df724.
Visual: D:/evoweave/server_ops/outputs/reference_xyz_20261027/reference_xyz_examples.png
shows the first validation asset on each rank, reference0 and the other pose with
largest GT static-copy error. Selection does not depend on predicted quality.

### Current Decision And Limits

The user explicitly corrected the direction after these pilots: use learned
generation, not an optimization/fitting solution. The prior recommendation to
promote local rigid tracking to the main method is withdrawn. Keep Kabsch and
rigid-residual measurements only as controls; do not continue that experiment
line or interpret its strong score as a reason to replace the requested method.

The main task remains: fully decode a real reference skeleton, then condition
a learned target-pose XYZ generator on that skeleton, reference mesh and other
observed meshes. Shared-backbone appended generator versus separate learned
model is still an experimental question. Do not silently select AR, diffusion
or direct continuous generation based on the word generation alone; separate
that architecture decision from the clear prohibition on a fitting-led route.
The short frozen direct-readout pilot is not proof that generative modeling or
jointly trained conditioning fails. Neither trained head is accepted as an
upgrade. The requested generative research objective is not yet complete.

This is one seed,64train/32valid balanced diagnostic, frozen donor and800head
updates. It neither proves learned corrections unnecessary in general nor
decides joint training, unmatched/remeshed input, arbitrary correspondence or
full population performance. The next reviewed experiment must address learned
conditional generation and conditioning capacity. No additional run is queued;
the completed diagnostic stage gates are closed.

## Conditional Autoregressive Learning Check

After the explicit generation-only correction, the user said to begin. The
first new implementation is an appended conditional AR coordinate generator;
this is an engineering starting choice, not a claim that AR must beat direct
continuous generation or diffusion. First-stage legacy AR and its complete
detokenization/branch merging remain unchanged. Given the resulting complete
reference joints/tree, the second stage generates exactly3J coordinate tokens
in the supplied parent-before-child joint order. It does not emit new IDs,
branches or topology; coincident generated coordinates do not merge identities.

The shared graph/mesh conditioner produces target-pose joint features. A fresh
four-layer,256-wide OPT decoder receives these features as a prefix plus the
current joint/axis condition and previous coordinate tokens. Causal next-token
cross-entropy trains it; greedy decoding with a KV cache uses only its own prior
tokens at inference. There is no Kabsch/registration/IK solution or fitted XYZ
baseline in generation. Geometric rows in evaluation remain controls only.

Second-stage coordinates use256uniform levels including endpoints[-1,1], nearest
level rounding and no clipping. This is a declared second-stage coordinate
format, not a change to the first-stage legacy tokenizer. Score generated XYZ
against original continuous GT, and separately record the quantized-GT error
floor so discretization is not confused with model error. Preserve the given
reference frame exactly and exclude it from loss/metrics; padding is masked.

Recipe:reference_skeleton_xyz_ar_fit_20261027_v1.json. Reuse the exact completed
frozen feature/natural-reference cache and donor. Choose the first2training
assets per joint-count stratum from the already SHA-ordered64asset panel:
8train total, independent of observed quality; retain all32valid assets. Use
800updates,2assets/rank x2ranks, all8poses, uniformly random reference per visit,
same3e-4 peak LR/20warmup/cosine/weight decay/clip as the previous bounded pilots.
This is a small-data learning diagnostic, not an equal-cohort comparison with
the preceding64training-asset runs or evidence of population generalization.

At0/400/800evaluate free generation with GT and natural references separately.
Also report teacher-forced coordinate-token accuracy, CE and position error
for GT references. Teacher forcing is diagnostic only, never free-generation
quality. These separate weak conditioning/learning from autoregressive error
accumulation. Save new checkpoints to conditional_ar_fit_v1, not prior roots.

45focusedCPU tests pass, including future-label causality, cached/full-prefix
generation equivalence, reference/padding preservation, conditioning gradients,
synthetic complete-token learning/free generation, complete metric path and
existing head/scheduler regressions. A new capacity_v15 must test3real updates
and complete free generation on the largest cached skeletons126/171joints;
natural predicted references in the retained panel have at most54joints. Only
clean capacity completion releases train_v14. No later run is automatically queued.
