# GT-Reference Joint-Anchor Binding On Full5000

Status: implementation and preflight, 2026-10-27. No new quality claim.
The previous shared-flow trial is retained, not resumed or overwritten.

## Question And Explicit Input Contract

The user clarified that stage2 receives the asset's complete GT skeleton at
one observed reference pose. It should exploit the already-trained mesh/anchor
representation to locate those same joints in the other observed poses.
Stage1 sampling, predicted/GT tree matching, joint names and predicted-reference
robustness are NOT part of this experiment. GT target coordinates enter only
the loss/scoring path. Neither head.forward nor inference accepts target GT.

Each pose's1024real anchor features come from the existing shared surface and
motion encoders. Their same-index correspondence is within one asset across
poses, never an assumed semantic correspondence between different assets.
All existing root-RT alignment, input-mesh-union normalization, complete trees,
2to24pose sampling,5000training assets and967validation assets remain unchanged.
No joint/frame pruning, coordinate clipping or geometric fitting is introduced.

## The Controlled Architectural Difference

Both arms have exactly the same parameter names, dimensions, initialization,
donor, optimizer states, data stream, reference sampling and objectives.

1. Encode reference GT XYZ, parent offset and root flag as joint tokens.
2. Two graph/mesh attention blocks read the reference frame's anchor tokens.
3. Project joint queries Q and anchor keys/values K,V. Association logits are
   scaled QK products plus a learned soft reference-coordinate distance prior.
4. reference_transport computes association A from reference anchor keys,
   then reads each pose's SAME-INDEX anchor values with A. target_content
   instead computes the association against that target pose's keys. This
   key/distance source selection is the only architectural difference.
5. Four identical graph/target-mesh/pose-set attention blocks refine the
   resulting joint features. A shared XYZ head predicts a learned displacement
   from the reference coordinates. This is a neural prediction, not a fitted
   rigid transform or an interpolated surface position. Reference XYZ is
   clamped exactly and is excluded from target loss/scoring.

The target memory contains that pose's feature, anchor position and normal.
It does NOT also receive an explicit paired-reference displacement channel,
which would hand the target-content control the same correspondence through
a second route. Both arms still inherit the existing backbone's multi-pose
information; this is not a comparison against a model with no correspondence
anywhere. Width256,8heads, reference depth2, target depth4, graph-hop cap8 and
maximum512complete joints are shared. No pose-index encoding or diffusion
noise is used. Head initialization is near the copy-reference control.

## Objective And Optimization

L = L_AR + lambda * (L_XYZ_L1 + 0.25 * L_target_bone_vector_L1).

The L1 terms average coordinate components, valid nonreference poses/joints
or edges, then equally over assets. Bone vectors are compared with TARGET GT,
not forced to equal reference bone length. Root-only trees have zero bone
loss. Lambda ramps to10over1000new asset visits. This scale is an engineering
choice, not an established optimum; both arms share it. Capacity checks must
measure actual finite head-only gradients to surface/motion encoders and
absence of that loss's gradient in the AR decoder.

The donor is fraction20_40_query checkpoint020000, SHA256
9704b23a194ca65836c6dfacd6637ca27542dc4b0836b1d3611185f0339cc888.
Its full model and backbone Adam state are retained. The original100kone-cycle
schedule resumes at20000; only new head parameters/Adam states are initialized.
Head LR is3times backbone LR. The original AR20%-40%, minimum2pose objective
remains, with one shared differentiable backbone encode per batch. Head-only
loss can update the shared encoder, not magically backpropagate through a
discrete sampled skeleton. The sampler/head seed is20261028for BOTH arms.

Global batch48, physical batches24/12/6per rank for2-6/7-12/13-24poses initially.
These sizes require fresh CPU-loader and real GPU acceptance. Any reduction
must preserve effective batch48and be recorded before formal launch.

## Training Horizon And Decision Rule

Train each arm on all5000assets for at least8complete passes (40000new visits,
840updates), not another420-update conclusion. Save complete model, Adam,
per-rank RNG and consumed sampler state every5000visits, with SHA-verified
durable mirrors. Arms alternate every10000visits on the same two GPUs, never
competing for resources. Full checkpoint/RNG/stream restores preserve each
arm's trajectory and expose matched intermediate results sooner. Match actual
asset/raw-pose/reference/AR-target traces afterward.

Only the fixed128validation panel controls extension. Compare each arm's30k
and40kpanel results. Extend BOTH to60k(12passes,1260updates) if either arm's
MPJPE improves at least1%relative, or PCK/F1 rises at least0.005absolute.
Otherwise stop both at40k. These are budget scheduling triggers, not quality
acceptance thresholds or proof of convergence. If still improving at60k,
explicitly report that the budget cap was reached without established convergence.
Finite/runtime failures stop for diagnosis; no silently changed loss or model
within an arm. No results from the remaining839assets influence this rule.

## Comparisons And Diagnostics

- At0and every5000visits: fixed128assets,8poses, GT references0and4, score
  the other7poses. Save all per-pose rows, not only a scalar loss.
- At the common final horizon: all967validation assets, plus separate839
  nonselection-asset results. These839were not used by THIS stopping rule;
  they are not claimed to be a never-before-inspected dataset.
- Report identity-paired MPJPE, PCK@5%, symmetric J2J, edge F1 and target bone
  vector error. Distances use each target mesh's bbox DIAGONAL. Aggregate
  per asset before averaging. Parent trees are supplied; position-induced
  nearest-joint correspondence errors can lower edge F1 without changing a tree.
- Matched reference-copy controls for every asset/reference. Report paired
  between-arm differences, paired-asset bootstrap intervals, per-asset wins
  and low/high reference-motion strata. One seed is not run-to-run uncertainty.
- On the128panel, permute target anchor tokens/positions/normals together at
  the stage2 interface while leaving reference tokens unchanged. This preserves
  each target's set of evidence but breaks reference-to-target indexing.
  Target-content output should be invariant up to floating-point error;
  transport sensitivity measures reliance on this explicit association route.
  Backbone features are not recomputed, so do not call this removal of ALL
  backbone correspondence or assume sensitivity itself proves usefulness.
- Compare predictions made using GT reference0versus4on their common targets.
  Lower reference dependence is useful only alongside accuracy, not by collapse.
- 32training assets: eligible raw poses and strictly heldout raw poses. Compare
  to actual training traces before calling any raw pose 'seen'. No blanket
  overfitting/underfitting diagnosis from these means alone.
- Fixed artifact-order visual examples with shared cameras/limits and no
  alignment; inspect failures and stretched bones, not cherry-picked successes.

A distance gain with worse local accuracy/edge matching is a tradeoff, not an
overall success. A null between-arm difference does not invalidate learned
mesh-to-joint decoding: it says the explicit transferred association has not
shown an increment over the matched target-content reader under this recipe.
There is no AR-only continuation arm, so no causal claim about improving the
original AR task. Historical flow results are context, not an isolated ablation
of the new binding because the head/objective/training horizon differ.

## Execution And Retention

The prior native allocation is no longer active. A new native HGC job may be
requested:2H100s,16CPU slots,64GiBwhole-tree RSS, GPU memory below85%,24hours,
on pcg01i to reuse verified SSD artifacts. The preparation observation showed
84free CPU slots and2available H100s there. Recheck before submission.
Bind exact grant, devices, CPU affinity and deadline only after allocation;
the checked-in initial runtime is deliberately unbound and cannot train.

Expected isolated stages: audit_v12, capacity_v17, train_v17. Source/state
must be committed and pushed before reservation and before each released
stage. Formal training remains blocked until primary acceptance of all20new
CPUtests, existing regressions, full-data loader audit and both-arm GPU stress.
No new raw dataset copy. Preserve all previous checkpoints and the new native
allocation after completion unless the user explicitly requests release.

Durable result root:
/home/wangyy/evorig/outputs/anchor_joint_binding_5k_20261027/v1
Local inspection/review root:
D:/evoweave/server_ops/outputs/anchor_joint_binding_5k_20261027

Reservation130674849(qsub) was not admitted: it had the job's devices cgroup
but cpuset:/, although affinity was set. No experiment payload ran. Its unused
reservation is canceled and replaced by native qrsh, retaining2H100/16slots;
the both-controller kernel check is unchanged. Initial bindinglinear:16was
also corrected while pending after the installed AGE manual established that
the amount is per PE task, not total. These are recorded provisioning errors,
not model experiments or a reason to weaken CPU accounting.

## Pre-Formal Optimization Correction

The old retained job130494177 ended at2026-10-27 19:33JST because the
scheduler enforced its48hour h_rt; qacct records deleted_byNONE. It was not
manually released. Replacement native job130675914 retains2H100/16CPU with
both kernel controllers verified; deadline is2026-10-28 19:55JST.

Full architecture capacity_v17 completed cleanly:36updates/rank, both modes,
nine stress cases each,24pose free inference, peak28.50GBtreeRSS and46.30GB
GPU reserved memory. This is capacity evidence, not quality. The coordinate-only
gradient check there covered target_content only; the followup covers both.

Measured full gradient norms12.55to76.15were dominated by the new head.
Global norm1 clipping therefore also scaled the pretrained backbone's
gradient. Before any formal training, split clipping into backbone norm1 and
head norm1. This preserves the combined objective's direction inside the
backbone while removing cross-branch clipping coupling. It is a design
correction, not evidence of improved predictive quality. Both arms use it.

Followup capacity_v18 verifies both modes on maximum-joint assets at6/12/24
poses, two optimizer steps each (12updates/rank), isolated coordinate-only
gradient flow for both modes, and24pose free inference. It requires the
SHA-verified36-case capacity_v17artifact and clean supervisor exit. Forward
code, parameters, input data, losses and shapes are unchanged, so that earlier
architecture/memory evidence remains applicable. The12-pass CPU data audit
also remains applicable; no loader or sampler changed. Runtime records
preclip norms and clip scales separately. Formal training remains blocked
until this followup passes. Examples are now retained at every evaluation,
including initialization, without changing evaluation inputs or predictions.

Capacity_v18 completed at2026-10-27 13:31UTC with clean supervisor exit,
12updates/rank, both coordinate-only gradient paths, separate clipping and
24pose free inference passed. Peak treeRSS24.99GB and GPU reserved45.67GB.
The original36-case capacity plus this12-case followup are accepted for launch;
neither is model-quality evidence. All24current anchor/model/protocol and
4driver CPU checks pass, with80unchanged earlier regressions retained.

At13:34:41UTC the same job130675914 was extended in place to h_rt345600
(96hours), with unchanged keeper1108368, start time,2GPUs and16CPU slots.
Expiry is2026-10-31 11:55UTC, training stop10:55UTC. The only change after
accepted capacity is runtime lifetime/deadline and a guarded resource-management
script; model/data/objective/scientific identity are unchanged. Formal train_v17
is released only for the exact new config hash in current.json.

## Interim Reference-Transport Learning Evidence

Formal source remains pinned at9db222d0070122b08c6d5dde3329f3be69d46c16.
At10k new visits (two complete5000passes,210updates), the reference-transport
arm improved on the fixed128selection panel. This is not yet a between-arm
result or final967/nonselection839 acceptance. Independent asset-macro
aggregation and2000paired-asset bootstrap checks passed locally.

| Metric | Copy reference | Transport5k | Transport10k |
| --- | ---: | ---: | ---: |
| Normalized known-ID MPJPE | 0.09266244 | 0.09229042 | 0.04472532 |
| PCK at5% target mesh diagonal | 0.53927101 | 0.54340580 | 0.70473408 |
| Spatial nearest-mapped edge F1 | 0.56117406 | 0.55272799 | 0.59815582 |
| Normalized symmetric nearest J2J | 0.06179935 | 0.06189210 | 0.03426620 |
| Normalized target bone-vector error | 0.02504090 | 0.02502127 | 0.02197576 |

The10k minus-copy95%paired intervals exclude zero in the favorable direction
for all five metrics. They describe asset sampling uncertainty for one training
seed, not repeat-run uncertainty. F1 uses nearest-joint mapped directed-edge
agreement with no distance threshold; since the GT tree is given, it is not
evidence that the model generated the topology. Do not equate it with PCK.

At5k, the saved8fixed assets (two GT references each, reference slot excluded)
had a median pose-specific correction energy fraction of0.04986. After removing
whole-skeleton translation, predicted articulation variation amplitude was only
0.00209times GT. At10k, these became0.56442and1.01534; articulation direction
cosine rose from0.19714to0.84067. These are subset motion diagnostics, not
all128estimates or accuracy percentages. Together with visible pose changes,
they reject the claim that10k improvements are only a reference-invariant
correction. Limb overshoot and local position errors remain visible.

Across all128assets, mean normalized disagreement between predictions made
using reference0versus4fell from0.08219at initialization to0.07980at5k and
0.04203at10k. Reference dependence is reduced, not solved. Splitting the128by
copy-reference motion into near-equal thirds shows10k position improvements in
all groups, with strongest gains on high-motion assets; low-motion F1's paired
interval still includes zero.

An anchored decoder difference (target readout minus reference readout) was
considered as a way to cancel pose-invariant corrections, but **not implemented**.
The10k evidence changes the diagnosis: the current architecture can learn
conditional articulation after the initially copy-like phase. Preserve the
running comparison and assess the matched content-reader arm before selecting
any architecture or loss change.

Local evidence: review_single_reference_transport_evaluation_005000.json,
review_single_reference_transport_evaluation_010000.json and corresponding
response_diagnostic_reference_transport files under the local inspection root.
10k full checkpoint SHA: fa3c55bf41ab7c3a268a41c3eaaf4d68366a9917f5a50b42e1d84bd9061f3d90;
8,278,607,378bytes, producer verified both SSD and durable copies. Local review
rehashes small artifacts and checks both full-checkpoint sizes, but does not
independently stream/hash the entire8GBcheckpoint.

### Visualization Contract

Reference GT must be shown on its OWN reference-frame point cloud, not silently
overlaid on a target pose. Show target GT and predictions on identical target
geometry with shared camera/axes. Use Chinese labels and explicit raw frame IDs.
The primary visual comparison omits the confusing copy-baseline panel; its
metrics remain in the quantitative tables. Bone lines remain visible over the
point cloud. The earlier ambiguous English copy-baseline figure is retained as
a historical artifact, not reused as the presentation template.
