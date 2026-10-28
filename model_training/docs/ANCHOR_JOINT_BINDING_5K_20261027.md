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

## Matched10k Interim Comparison (2026-10-28JST)

Both arms completed two complete5000passes/210updates. Independent bounded
trace-prefix inspection confirms identical actual microbatches, asset indices,
raw poses, references and AR-supervision slots; both consumed130,200input
frames. Actual initialization records match, including head hash93dfca71...
and retained backbone optimizer source. The checked training assets are disjoint
from the128selection assets. This prefix audit does NOT certify terminal
completion or heldout-raw-pose exclusion; those checks remain in final review.

| Metric | Reference transport | Target-content reader | Transport minus reader95%paired interval |
| --- | ---: | ---: | --- |
| Normalized known-ID MPJPE | 0.04472532 | 0.05824034 | [-0.01606724,-0.01094363] |
| PCK at5% diagonal | 0.70473408 | 0.62533986 | [0.06555747,0.09304293] |
| Nearest-mapped edge F1 | 0.59815582 | 0.55454824 | [0.03539978,0.05194265] |
| Normalized J2J | 0.03426620 | 0.04178895 | [-0.00881712,-0.00621135] |
| Normalized target bone-vector error | 0.02197576 | 0.02442684 | [-0.00303874,-0.00185068] |

Transport has23.2%lower position error,7.94percentage-point higher PCK and
0.04361higher F1 at this common training budget. This supports an early
learning advantage of the current transported-association recipe; it does
not establish final/converged superiority, seed robustness, or that every
possible target-content reader is worse. The five matched5kmetrics differed
only slightly, and PCK/F1 intervals there included zero.

Both arms now exhibit target-dependent motion. On the same8saved assets,
median articulation-variation amplitude relative to GT is1.015for transport
versus0.614for content reading; direction cosine is0.841versus0.745. These
subset amplitude ratios are NOT accuracies. Mean normalized cross-reference
disagreement on all128assets is0.04203versus0.04987; neither is reference
invariant. Fixed visual examples show remaining overshoot and local distortions.
In the highlighted arm case(raw25reference to raw4target), transport has lower
position error but worse bone-vector error/F1 than content reading, so aggregate
improvement is not a uniform win on every pose or metric.

Evidence under the local inspection root: review_evaluation_010000.json,
trace_prefix_audit_010000.json, response_diagnostic_target_content_010000.json,
pair_examples_evaluation_010000.png (visually verified), plus their exact
artifact/trace manifests. Content10kfull checkpoint SHA:
2379dd9a3a31b0f0cd14d5db1adf545ba33d7ce192622b513b9a49ebf0d31bc8.
No mid-run model/loss/data/schedule change was made. Continue both arms to the
common40k/60k horizon and retain separate final967/nonselection839checks.

## Matched20k Interim Comparison (2026-10-28JST)

Each arm completed four full5000passes/420updates and257,920input-frame
visits. The bounded actual-trace audit again verifies identical initialization,
microbatches, assets, raw poses, references and AR slots, with training disjoint
from the128selection assets. Terminal and heldout-pose checks remain pending.

| Metric | Reference transport | Target-content reader | Transport minus reader95%paired interval |
| --- | ---: | ---: | --- |
| Normalized known-ID MPJPE | 0.02305132 | 0.02781462 | [-0.00629610,-0.00336539] |
| PCK at5% diagonal | 0.88658784 | 0.83537068 | [0.03808838,0.06400164] |
| Nearest-mapped edge F1 | 0.71800090 | 0.71013218 | [0.00059081,0.01513009] |
| Normalized J2J | 0.01828115 | 0.02174776 | [-0.00433462,-0.00257082] |
| Normalized target bone-vector error | 0.01255571 | 0.01421156 | [-0.00222208,-0.00109822] |

Transport has17.1%lower MPJPE, with100/128assets better, and5.12percentage-point
higher PCK. The F1 advantage is now only0.00787. All intervals are2000paired
asset resamples from ONE matched training seed; they are not seed uncertainty
or multiplicity-adjusted evidence. The smaller differences than10k warrant
retaining the learning-speed hypothesis rather than asserting final superiority.

All128mean cross-reference disagreement is0.020760vs0.023906, ideal0.
On8saved assets/two references, median articulation-variation amplitude over
GT is1.106vs0.823 and direction cosine0.980vs0.973. These subset diagnostics
confirm motion following, NOT all-asset accuracy. They suggest remaining
over/under-response respectively, not a proven global error mechanism.
The fixed arm case visually improves; the fixed seated case retains foot
offsets. All panels retain identical camera/scale and explicit reference/target
surfaces. No post-hoc alignment or geometric fitting is used.

Evidence: review_evaluation_020000.json, trace_prefix_audit_020000.json,
response_diagnostic_{reference_transport,target_content}_020000.json and
pair_examples_evaluation_020000.png under the local inspection root. Small
artifacts were rehashed against the server. Each full checkpoint has
8,278,607,570bytes; both copies are producer-hash-verified and independently
size-checked, not independently rehashed during this local review.
Transport checkpoint SHA:8b7aa0c896c02d19fa1860ef8cb1580b0c65f7e4be64d09bacdb9d3f260173ae.
Content checkpoint SHA:3faaa4e72090ea13bdf530254016d6333879e427e02a17fee2c0e2116f3a87fc.
The live source remains9db222d; no training code or protocol was changed.

## Completed60k Result (2026-10-28JST)

Both arms completed60,000additional asset visits:12full5000passes,
1,260optimizer updates and787,504input-frame visits each. The30to40k change
passed all three preregistered improvement triggers for BOTH arms, extending
the common budget to60k. No source/config/loss/data/schedule change occurred.
The native16CPU/2H100 supervisor exited0at11:33JST with payload cleanup
complete and peak whole-tree RSS33,185,140,736bytes. At11:46JST, the same
keeper/job/cgroups/96hour grant remained, the supervisor was absent and both
allocated GPUs showed0MiB. The allocation was NOT released.

### Full Validation

Each967asset contributes14nonreference target predictions:8poses with GT
references0and4, excluding the reference itself. Aggregate within each asset,
then equally across assets; all13,538rows/method succeeded. Distances below
are divided by the TARGET mesh bounding-box diagonal, not world-unit errors.

| Metric | Copy reference | Reference transport | Target-content reader | Transport minus reader95%paired interval |
| --- | ---: | ---: | ---: | --- |
| Known-ID MPJPE | 0.09045948 | 0.00708115 | 0.00971083 | [-0.00308635,-0.00227372] |
| PCK@5% diagonal | 0.54526003 | 0.99189920 | 0.98263627 | [0.00725920,0.01167803] |
| Nearest-mapped edge F1 | 0.56402246 | 0.89008395 | 0.84601169 | [0.04072181,0.04713084] |
| Symmetric J2J | 0.05991595 | 0.00623647 | 0.00773343 | [-0.00166048,-0.00135753] |
| Target bone-vector error | 0.02508391 | 0.00607974 | 0.00810355 | [-0.00240713,-0.00174636] |

Transport has27.1%lower MPJPE, with898/967assets better;734have higher edge
F1 and913have lower bone-vector error. On the839assets NOT used by this
run's stopping rule, MPJPE is0.00713242vs0.00981637, PCK0.99151074vs0.98193841
and F10.88932394vs0.84509768; all five paired intervals favor transport.
The839are not claimed to be historically pristine. Intervals use2000paired
asset resamples from ONE training seed, not run-to-run uncertainty or a
multiplicity-adjusted collection of tests.

PCK is the fraction of known-ID joints within5%diagonal, not a claim that
99%of rigs are production-ready. Edge F1 uses bidirectional nearest-joint-mapped
directed-edge agreement without a distance threshold. The tree is supplied,
so this is NOT generated-topology accuracy. Bone-vector error compares the
target GT vectors, not a requirement to preserve reference bone lengths.

### Mechanism And Generalization Checks

On128selection assets, coherently permuting target features/points/normals
at the stage2 interface preserves their set but breaks reference-to-target
indexing. Transport MPJPE rises from0.00674510to0.17905045; content-reader
MPJPE changes from0.0090190166to0.0090189741, within floating-point variation.
Thresholded edge/PCK scores can vary slightly with these tiny numeric changes.
Together with the controlled accuracy gain, this supports use and usefulness
of the explicit transfer route under this recipe. Sensitivity alone would
not establish usefulness. Backbone features were NOT recomputed, so this is
not an ablation of every source of correspondence in the shared backbone.

Full traces verify all12pass boundaries, matched actual initialization and
microbatches/raw poses/reference/AR slots, training-vs967asset disjointness,
and exclusion of the heldout raw poses from training. Of the32asset eligible
diagnostic rows,99.330%of target frames and98.438%of reference frames actually
appeared in training; do not label every eligible row as seen.

| 32same training assets | Transport MPJPE | Content-reader MPJPE |
| --- | ---: | ---: |
| Training-eligible raw poses | 0.00584432 | 0.00756991 |
| Strictly heldout raw poses | 0.00568283 | 0.00746476 |

Each group has448target rows. These comparable means support the checked
heldout-pose behavior, but pose difficulty differs and they do not alone
rule out overfitting. On all967assets, changing the GT reference produces
mean prediction disagreement0.00762433vs0.00990077, ideal0. Accuracy improved
alongside lower disagreement, but reference dependence is not eliminated.

### Remaining Errors And Budget Limit

Transport/content asset-mean MPJPE medians are0.00583223/0.00749615;95th
percentiles0.01589519/0.02170462. There are23/62assets above2%diagonal and
3/11above5%;13/23assets have edge F1below0.5. These are descriptive diagnostic
thresholds, not preregistered quality cutoffs. The largest errors remain
0.05251293/0.14192521. Exact worst-asset IDs are retained in the additional
analysis for targeted follow-up; the tail cannot be declared solved from
the fixed saved visual examples.

Splitting967assets by copy-reference motion into near-equal thirds gives
transport/content MPJPE0.00373078/0.00459431(low),0.00653987/0.00863308(medium),
0.01098320/0.01592100(high). Thresholds are0.04925789and0.10957884in baseline
MPJPE. High-motion edge F1is0.79807vs0.71750: motion remains a useful failure
stratum despite the advantage. These are post-training descriptive groups.

At the same128selection panel,50to60k MPJPE still improves19.056%for transport
and12.283%for content. Thus this is a BUDGET-CAPPED result, not demonstrated
convergence or the final asymptotic ranking. The preferred current candidate
is reference_transport. Next work should inspect the residual23large-error
assets/reference dependence and choose a new declared common longer budget
or seed replication, not revert the architecture because its first pass
was nearly a copy. No further compute stage has been launched.

Primary acceptance is limited to this GT-reference stage2 comparison. There
is no predicted-reference test, stage1 free-AR retention test, AR-only
continuation control or random-backbone control. No isolated pretraining
benefit, improvement of original AR generation, seed-robust superiority or
all-case accuracy is claimed. The producer's immutable quality_accepted=false
means awaiting primary review; the separate scoped primary acceptance is
recorded in current.json without rewriting that producer artifact.

### Retained Evidence

Local inspection root: D:/evoweave/server_ops/outputs/anchor_joint_binding_5k_20261027.
Primary review:review_final_060000.json, SHA256
1af7513d14986a31ae6e8bac504174bf2a9903c56fe8d7c5e82f18fca72a91d3.
Independent train/heldout aggregates, tails and checked milestone curve:
final_additional_analysis_060000.json, SHA256
11374822c8f2610b3a8250a31ab27d8d29a677e69e773bba169e6ff9d088cd4a.
Formal result SHA256:a4876b7b89f97c482dc55cc1d178ef61f29a865a0de362efa24d1c000504ab3a.
Collection manifest SHA256:70efc5d63445dedacd62db17e4b0d546dfc34de4e7aca9153fdfb458c2528b31.

All small artifacts were independently rehashed. Full checkpoints were
producer-hash-verified in both copies and independently size-checked here,
not independently rehashed during local inspection. Transport60k checkpoint:
cfd446c1f6808304aa6a691884c2ed77d19db0f1850512dfd97b83e7eb0a0016,
8,278,607,634bytes. Content60k checkpoint:
82932dbda86f4b73e842c2379d205b130c0656e3e19dd8018a2db5704e1e9824,
8,278,607,570bytes. They remain under train_v17/{variant}/checkpoint_060000.pt
and /home/wangyy/evorig/outputs/anchor_joint_binding_5k_20261027/v1/{variant}.

Both pair_examples_evaluation_060000.png (same earlier128-panel cases) and
pair_examples_final_060000.png (predetermined full-panel artifact-order
cases) were visually checked. They use explicit reference-GT/reference
surface and target-GT/predictions on the SAME target surface, common camera
and scale, with no fitting or alignment. Their case sets differ by evaluation
panel, not by selecting favorable outcomes. Remote source stayed pinned to
9db222d0070122b08c6d5dde3329f3be69d46c16 through the entire run and inspection.
