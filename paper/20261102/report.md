# Evoweave First Manuscript: Author Report

## Intake And Decisions

- Deliverable: complete English manuscript, readable PDF, editable source,
  Chinese narrative companion, real figures, verified references and evidence.
- Audience: graphics and vision researchers. No venue or page limit has been
  specified, so no venue acceptance claim or fictitious author block is used.
- User authorization: deploy released baselines and run bounded H100 inference
  autonomously, including GPU sharing; no routine questions. Do not train again
  or reopen checkpoint selection. Retain the selected330k model.
- Writing reference: Best-Paper-Writing-Skill, pinned at
  `b9eff99a2130bea960ef1050eacf9f40cb792dd4`; use its claim calibration, paragraph
  outline, functional exemplars, figure planning and mechanical draft checks.
- Route: L1 method improvement / insight, with video as secondary motivation.
- Tension: pose-dependent evidence is useful for structure, but independently
  reading each pose does not define persistent joint identity. Recover structure
  once and use reference-to-anchor association to transport learned evidence.
- Main handle: reference anchor transport. This transfers features, not surface
  coordinates, and does not perform geometric fitting or enforce rigid bones.
- Closest prior work: MoRig and SkeRig for motion-aware structural recovery;
  RigMo for learned joint rig-motion discovery; UniRig for inherited tree
  generation; Puppeteer and SkinTokens for modern learned rigging.

## Claim Calibration

| Claim | Evidence | Permitted wording | Boundary |
| --- | --- | --- | --- |
| Reference anchor transport improves joint recovery over a matched target-content reader | Two matched5000-asset arms,60000additional visits each,967validation assets,13538nonreference predictions per method | Known-ID MPJPE0.708% vs0.971% of target bbox diagonal,27.1% relative reduction | GT reference supplied; one training seed; budget capped, not convergence |
| The transferred association uses correspondence meaningfully |128asset interface permutation, target feature/position/normal sets unchanged | Transport MPJPE0.675% to17.905%; target-content reader unchanged numerically | Only the stage2 indexing route is disrupted; backbone correspondence remains |
| Joint identities persist across output poses | One generated reference tree is reused without changing parent indices or joint slots | A variable-size parented skeleton and pose-dependent XYZ with persistent IDs | Wrong/missing reference joints persist; no claim of recovering every hidden or inactive bone |
| Multi-pose AR supervision is useful | Matched5000assets,20000visits, same inputs and initialization,128assets x8views | Edge F1 rises from0.66660(one target) to0.71108(all targets) | One seed and development panel; no asymptotic or benchmark-wide SOTA claim |
| Random20%-40%, minimum2 is a practical training recipe | Same matched sweep;F10.71401 vsall0.71108;update125.0 vs150.1min | A competitive point estimate with fewer supervised targets and16.7% lower measured update time | Paired interval vsall includes zero; sequential timing is not isolated throughput |
| Pipeline can infer without a skeleton entering the predictor | Existing12case predicted-reference outputs; frozen24asset inference stage | Reference structure is generated before dynamic coordinate decoding | Dataset preprocessing uses skeleton-derived root transforms; raw-video GT-free preprocessing not established |
| Video matters for the paper | Video exposes motion but usually does not provide a target rig; MoCapAnythingV2 andTopoCap explicitly condition on a rig | Dynamic skeleton recovery complements rig-conditioned motion capture | Do not put raw-video accuracy or video-alignment claims in the abstract without a real video experiment |
|330k is the retained working checkpoint | Finite330k/380k/390k/400k diagnostic review | Working model remains330k; no additional training |12difficult cases were inspected/selection-informed, not an independent test set |

The headline numerical result belongs to the matched GT-reference experiment,
not to the selected330k end-to-end model. Never combine their scopes in one
sentence as if they came from the same test.

## Literature Verification

Full PDFs and page-indexed text were retrieved for all eight primary works;
URLs, SHA256 and page counts are in the local collection receipt. Read the
problem formulation, method and evaluation, not just the abstracts.

| Work | Actual input / output | Its role here | Comparison restriction |
| --- | --- | --- | --- |
| MoCapAnythingV2 | Video plus reference skeleton/asset; joint poses and rotations; rotation convention uses a reference pose-rotation pair | Importance of topology-general motion capture | Given-rig condition differs; XYZ-only output cannot resolve bone-axis twist |
| TopoCap | Video and target rig/rest transforms; graph-CVAE motion latent and conditional flow; rotations/elastic offsets | Rig-conditioned video motion | Does not discover the target hierarchy;Mobjaverse has5006topologies, not5006models |
| UniRig | Static surface; tree-token skeleton, separate skinning | Inherited decoder and static reference baseline | Credit original tree codec and initialization; no invented new tokenization |
| SkinTokens /TokenRig | Static surface; skeleton prefix followed by compressed skinning tokens;GRPO refinement | Modern joint skeleton/skinning baseline | Skeleton prefix alone is not full rigging reproduction or skinning comparison |
| Puppeteer | Static mesh rigging plus video-guided differentiable animation fitting | Static reference baseline and pipeline context | Current adapter scores raw joint tokens, without published geometric merging |
| MoRig | Static target mesh and moving point clouds; motion-aware rig and transferred animation | Closest learned motion-aware rigging route | Requires matching input generation and official weights; do not label unavailable runs reproduced |
| SkeRig | Corresponded mesh poses; motion clustering, skinning decomposition, MST and iterative rig optimization/pruning | Classical motion-driven structural baseline | Authors release an executable; DemBones is not automatically an equivalent implementation |
| RigMo | Mesh sequence; Gaussian deformation bones, weights and SE(3) motion | Novelty boundary for rig-motion discovery | Gaussian deformers are not automatically the annotated parented joint tree measured here |

Version-sensitive details are fixed to the PDFs in `literature_sources.json`.
The papers' headline benchmark numbers are not imported into our result tables
because data, normalization, outputs and splits differ.

## Paragraph Outline Before Prose

### Abstract

1. Recovering articulated motion needs both a structural representation and
   pose-dependent coordinates with stable identity.
2. State Evoweave's correspondence-based geometry input and generated-reference
   output, then name the feature transport mechanism.
3. Give the matched967asset result with the GT-reference condition explicit.
4. State the diagnostic value of the permutation test without claiming raw-video
   validation, skinning, or recovered rotations.

### Introduction

1. Explain why observing an object's movement in video is not the same as having
   its editable skeleton, and why the distinction matters for reuse and analysis.
2. Credit rig-conditioned motion capture and learned automatic rigging, then
   distinguish structural input from structural output.
3. Credit MoRig, SkeRig and RigMo; identify our concrete question about learned
   parented skeleton recovery and consistent joint coordinates rather than a
   false first-ever use of motion.
4. Diagnose pose-wise association ambiguity and motivate a reference-frame
   joint-to-anchor association evaluated through corresponding target values.
5. Explain how multi-pose AR structure learning and the dynamic coordinate head
   work together, including inherited UniRig components.
6. Summarize the controlled improvement and correspondence diagnostic at their
   correct scope, then list three specific contributions.

### Method

1. Define corresponded normalized pose surfaces, reference index, generated
   parent tree and target XYZ; distinguish training and inference references.
2. Define shared material anchors and alternating spatial/pose-set encoding.
3. Explain multi-pose AR supervision and generation of one reference tree.
4. Derive the joint queries, reference association and target-value transport.
5. Explain graph/mesh/pose-set refinement, residual coordinates and exact
   reference clamp.
6. Define AR, component-wise XYZ L1 and target bone-vector L1 with asset-equal
   averaging, masks and gradient paths.

### Experiments

1. State dataset, root-transform metadata, preprocessing and split/overlap limits.
2. Define known-ID and unknown-ID metrics, failure and aggregation policies.
3. Present matched GT-reference reader comparison as the strongest causal test.
4. Present permutation, reference sensitivity, motion strata and heldout poses.
5. Present AR target-density results without claiming statistical equivalence.
6. Present measured released-model reference predictions and head-versus-copy
   results separately from the controlled comparison.
7. Present selection-informed difficult-case diagnostics and failure categories.
8. Explain limitations and the exact requirements for a stronger video claim.

## Figure Plan

| Figure | Function | Source | Check |
| --- | --- | --- | --- |
|1|Explain input, generated structure and persistent joint trajectories|Actual canonical inputs and saved predictions, plus vector schematic|No fabricated geometry, no apparent raw-video success without video |
|2|Explain reference keys/target values and refinement|Reviewed implementation equations|No arrow implying target GT enters inference or explicit XYZ interpolation |
|3|Quantify matched reader comparison and permutation intervention|Completed60k aggregate/paired results|Units are percentage of target bbox diagonal;GT-reference label visible |
|4|Show target-density tradeoff|Seven-arm20k measured results|Sequential update timing labelled, no invented error bars |
|5|Show representative successes and failures on fixed examples|Frozen24asset or retained12case visual records|Shared axes/cameras, no skeleton alignment, explicit selection scope |

## Execution Record

- One fresh SSD source deployment preserved the previous clean checkout and
  verified durable source archives; no historical source files were restored.
- `audit_v36` rejected our CPU-import bytecode caches before preparing any inputs.
  Its original exit1 and confirmed native payload cleanup are preserved.
- Only four `__pycache__` directories containing `.pyc` files created by our own
  import check were removed. No official source files were edited.
- `audit_v37` uses the same fixed scientific protocol; it is not a repeated
  completed data audit. Its exact result is still pending at this report revision.
- Shared GPU/CPU admission is restricted to this inference-only operation.
  Retain native job131170166, keeper2268188, GPU4, eight CPU cores within the
 16core grant, own PyTorch allocation fraction0.30 and32GiB free-memory admission.
- Full training and checkpoint search remain closed. The previous case105 exact
  coordinate-token replay discrepancy remains unresolved.

## Open Evidence, Not Invented Results

1. Complete and inspect the frozen24asset inference, then fill measured tables.
2. Prepare SkinTokens' released weights and check the actual compatible runtime.
3. Resolve whether official MoRig weights and the SkeRig executable can be
   obtained and run with a fair input interface; report a concrete blocker if not.
4. A never-inspected, training-deduplicated end-to-end test and multiple training
   seeds are not currently available; do not describe validation as that test.
5. Raw-video reconstruction/alignment, skinning, FK-consistent rotations,
   robustness to correspondence errors and full production-rig acceptance are
   outside the completed evidence. Keep those claims out of the paper's results.

## Revision Status

- [x] User authorization, scope and retained model identified.
- [x] Writing-skill workflow and functional exemplars inspected.
- [x] Claim map and paragraph outline written before manuscript prose.
- [x] All eight primary paper PDFs retrieved; methods and protocols inspected.
- [x] Existing controlled numerical evidence located and scope checked.
- [ ] New inference results reviewed and manuscript tables frozen.
- [ ] Complete manuscript and Chinese narrative companion written.
- [ ] Scientific figures generated and visually inspected.
- [ ] Mechanical prose checks and cross-reference/citation checks passed.
- [ ] Readable PDF rendered and every page checked.
