# Inactive-Joint Capability of the Trained Motion Model

## Question

The model and its original u/O/M motion conditioning are already trained.
The question is whether this model has learned that absent local movement is
not absent skeleton, and whether it can use geometry and other supplied
context to predict those parts. Symmetry is merely one possible relationship
the network may learn, not a new hand-coded module or loss. This is not a
request for certain physical identifiability, another residual architecture,
or a new full training run.

Keep the completed bias_h8 sample80000 checkpoint unchanged. Experimental
relation adapters remain off. All targets remain complete. Motion strata are
offline evaluation labels only and never condition or prune the model.

## Existing-Output Audit

Sources are the hash-verified original relation_screen cache and its BASE
generations, not the subsequently trained experimental adapters. There are
16 validation assets, each with normal8, weak2 and static8 inputs. Query
geometry, target skeleton, surface references and normalization are matched.

The original model succeeds on 13/16, 12/16 and 14/16 inputs respectively.
All nine failures are nontermination at the 1400-token generation limit.

In weak2, the strict hidden-demonstrated subset has52 target joints across8
assets. The original model covers41/52 within0.05 query-normalized units.
All41 joints in successfully terminated cases are covered; the other11
belong to two failed generations. This demonstrates some inactive-part
prediction ability, not its general reliability or the cause of that ability.

Asset385 is a concrete positive case:52 target joints, only2 locally active
joints in weak2, and40 strict-quiet joints covered, including22 that move in
noninput poses. Asset128 shows a remaining problem:normal8 terminates but
only23/71 strict-quiet joints are covered. Other views also suffer full
generation failure. These failures must not be conflated.

Coverage is nearest-neighbor distance, not an anatomical one-to-one match.
It can overstate completion when several target joints share one predicted
neighbor. Report predicted counts and additional one-to-one/located-edge
checks rather than treating coverage alone as complete rig recovery.

## Matched-Joint Audit

`analysis/audit_inactive_joint_predictions.py` reads only saved input tensors
and token sequences on CPU. It detokenizes predictions without running a
model, replays all original stratum metrics exactly, and exports individual
target-joint matches plus query surface points. It uses the SAME quiet-joint
IDs in all three views, including the intersection of normal8/weak2 quiet
masks with other input joints active. Comparing aggregate quiet percentages
from different masks is not a matched-joint test.

Output: external job `results/inactive_base_saved_joint_audit_20261011.json`.
Its completed48-row export has SHA256
`6ecfb42bd1f4e2c4de97cf0ae15c7da959f1fc49982ae40dce1595bae9181fa5`.
No CUDA was initialized. The local transport timed out at300s, but the original
CPU child completed successfully; no duplicate job was started.

## Original-Evidence Intervention

Stage `inactive_motion_use` is one read-only evaluation of the ORIGINAL
trained model. Before intervention, all48 cached conditions must match the
saved base exactly, CE must differ by at most2e-5, and four64-token greedy
prefixes must match. Model weights, input geometry, poses, targets and
normalization remain fixed. No adapter is installed and no optimizer exists.

Two controls affect only the original explicit pair-state input:

- Allunknown: every pair is [1,0,0]. Allthree channels still exist.
- Permuted: the same deterministic permutation is applied to both anchor
  axes, preserving state distributions but breaking feature correspondence.

Run64 new natural generations on the32 nonstatic inputs. Static controls
need only CE/condition checks; reuse their base generations only after exact
condition identity. Report identical inactive-joint masks, generated joint
counts, distances and edges, separating nontermination from successful but
incorrect output. Old implicit multiframe processing always remains intact.

This distinguishes inactive-part prediction from dependence on explicit
u/O/M evidence. It does not claim that these two controls isolate every
possible use of global context, nor that the same16 reused assets establish
generalization. No new training or architecture change is authorized by a
successful execution check.

## Diagnostic Definitions

Strict quiet means rotation relative to the parent at most1degree AND joint
head displacement at most0.005 query units. Active means at least5degrees OR
0.02 query units. Hidden-demonstrated additionally moves in noninput poses
and has another active joint in the input. Roots and invalid local rigid
transforms are excluded ONLY from these diagnostics, never from the target.
Quiet articulation does not imply its surface is fixed in world coordinates.

The query normalization uses maximum half-extent; coverage tolerance0.05
therefore equals2.5percent of the query mesh's maximum extent. Thresholds are
diagnostic proxies, not labels that the network is asked to infer.

## Completed Results

The original-model intervention completed once from committed0ad5c0b, with
all three exit codes0. All48 CE/condition replays have exact zero difference;
all4 prefixes agree. It produced64 new natural generations plus32 verified
static reuses, with4.650649GiB peak allocation. Parameters, gradients and
cached inputs remain unchanged; all source hashes pass the final recheck.
This is a completed diagnostic, not a new model-quality approval.

The one-to-one audit confirms asset385's40 quiet joints and40 corresponding
edges in every view. Asset202 has the SAME17 quiet joints in normal8/weak2:
normal8 matches10 joints/6 edges, weak2 matches9/6, static matches1/1.
Asset128 still matches only21/71 quiet joints and17/71 edges in its successful
normal8 output; weak2/static do not terminate.

Across9 assets successful in all three original views, the SAME150 quiet
joints have84/83/71 one-to-one matches and66/57/40 located edges for
normal8/weak2/static8. Additional supplied poses can help inactive articulation
prediction, but this is a selected small cohort and does not isolate every
possible context pathway or prove the nearby surfaces remain fixed.

The new explicit-evidence controls show limited stable completion benefit:

- In paired-success cases with allunknown explicit evidence, quiet matches
  change179 to178 out of354 joint-condition instances; located edges remain
 134. The joint-coverage delta is-0.28percentage points, asset-bootstrap95%
  interval[-1.52,+0.79]points.
- With permuted evidence, quiet matches total167/300 for both base and control;
  located edges change123 to121. Equal totals do not mean identical individual
  predictions: asset296 gains2 quiet matches in normal8 and loses2 in weak2.
- Including full-generation failures, unknown matches change188 to183 out of
 556 instances; permutation changes188 to167. The21-match permutation loss
  comes from asset128 normal8 becoming nonterminating. Other termination
  changes can improve or worsen results. Do not equate this with consistent
  local inference benefit from correctly aligned motion relationships.
- Only14/32 unknown-control and12/32 permuted-control nonstatic token streams
  are identical to their originals. The cue is not numerically inert; changed
  sequences are not by themselves evidence of useful completion.

The556 instances are278 target joints evaluated under two input conditions,
not556 distinct joints. Success-conditioned cohorts differ between controls;
report their denominators and the all-case results together. All intervals
are exploratory whole-asset resampling of the same reused16 validation assets.

Conclusion: the completed model has some inactive-part prediction capability,
with genuine remaining location/connection and termination failures. In this
probe, correctly aligned explicit u/O/M states do not provide a stable large
additional completion benefit. The old geometric and implicit multiframe
paths remain active even in the controls, so this does NOT show that motion
in general is unused or that training with u/O/M had no effect on the backbone.
Keep the original checkpoint; do not promote the experimental relation adapter
or launch new training from this diagnostic alone.

Artifacts under the external job directory:
- `results/inactive_motion_use_evaluation.json`, SHA256
  `53cd8962709303773d0f8a5efc7179418bced43abd594ca60809c294e9e84e0b`.
- `results/inactive_motion_use_analysis_20261011.json`, SHA256
  `39b0726c8e9649c27a662617903b7f0e8fa1e03fae9c628803cb04c0133305bf`.
- `results/inactive_base_unique_audit_20261011.json`, SHA256
  `394c6a9d2f3afd0d570d7aa7ef487416213336bc54272455c9e7feeac6617179`.
- `inactive_base_matched_examples_20261011.png`, three matched-query examples.
