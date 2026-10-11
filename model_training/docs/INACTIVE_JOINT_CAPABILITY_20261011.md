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
