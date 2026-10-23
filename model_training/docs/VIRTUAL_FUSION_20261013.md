# Virtual-Query Fusion: Scientific Experiment Report

## Scope And Outcome

This later all-target study is not the [original single-query baseline](BASELINE_FLAT_UNIRIG.md).
A/B training was stopped by the user on October 21, 2026; C completed its 20k comparison.
A completed 150k asset visits. B65k is the latest reviewed metric; a B70k save was recorded,
but its evaluation and full payload audit were not completed. B did not complete 80k.
Some old checkpoint weights were subsequently deleted; recorded paths/hashes do not establish
current availability. This report records results, not permission to resume training.

## Question And Controlled Architectures

The defining change from the original baseline is an added virtual query frame:
1,024 slots without an observed input mesh, interacting with the real poses.
The original baseline contains only real input frames and selects one as its query.
A/B/C all retain the added virtual frame and compare how its final features fuse
with real-frame features. All-target supervision is a separate change.

Within this virtual-frame architecture, does making each real pose the attention
query improve generation relative to using the virtual representation as query?
The A/B/C comparison alone does not isolate the benefit of adding a virtual frame.

All arms use 1,024 corresponding real/virtual anchor tokens, 96 internal registers,
12 spatial/temporal blocks, width 1,024, 8 heads, and one frame-fusion block.
The complete official UniRig XYZ AR decoder supervises every selected input pose.
A two-layer auxiliary DFS-ID decoder reads final virtual tokens with loss weight 0.1;
its generated structure, count and hidden states never condition the main XYZ branch.

| Arm / fusion mode | Queries | Keys/values | Residual and additional parameters |
| --- | --- | --- | --- |
| A / `virtual_queries_real` | Final virtual tokens | Final real-frame tokens | Virtual residual |
| B / `real_queries_virtual` | Final real-frame tokens | Final virtual tokens | Real residual; same initial parameter tensors as A |
| C / `relative_position` | As A | As A | As A, plus per-head input-anchor coordinate-delta bias, MLP 3->64->8 with zero-initialized output |

B still uses the virtual frame as keys/values. A real-frame attention query
does not make B a real-frame-only historical baseline.

C uses observed anchor coordinates, never GT joint coordinates. Shared modules, fresh official
initialization and seed 20261013 match; C's extra parameters have a separate full fingerprint.
The official checkpoint SHA is `d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4`.
Architecture and token contracts: [Virtual Query XYZ](VIRTUAL_QUERY_XYZ_20261013.md);
implementation: [virtual_query_xyz.py](../rigweave/src/rigweave/dynamic_rig/virtual_query_xyz.py).

## Fixed-5k Protocol

- Current screened data are 18,002 train / 967 valid assets; see [data source](../../DATASET_SOURCE_OF_TRUTH.md).
- Training uses one immutable 5,000-row subset: 4,397 base + 603 addition assets, no validation overlap;
  SHA-256 `a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`.
- Preserve manifest `root_rt`, inverse retained-root RT, selected-input mesh-union normalization
  with padding 1.1, complete GT, and trackable surface samples; no persistent learned-feature cache.
- Each visit samples T uniformly from 2..24 and fresh poses from non-heldout frames. Eight raw
  poses per asset are reserved. T is common to an optimizer update; exposure-index seeds align
  asset order and pose/surface draws across arms, independently of microbatch partition.
- Initial comparison: exactly 20,000 asset visits, four shuffled passes, 420 updates and
  263,608 supervised-frame visits per arm. Each 5k boundary uses 104 batches of 48 plus one of 8.
  Asset visits, supervised-frame visits and optimizer updates are different budgets.
- Effective batch is 48 on two GPUs. A/B use per-rank batches 24/12/6 for T=2..6/7..12/13..24
  with accumulation 1/2/4; C uses 12/6/3 with accumulation 2/4/8. This is not fixed frame-budget batching.
- BF16, AdamW, weight decay 0, gradient clip 1; initial OneCycle spans 20k visits, peak LR 1e-4,
  warmup fraction 0.1, division factors 5 and 10. Profile/fitting weights do not initialize this study.
- Evaluate at 0/5k/10k/15k/20k. The fixed panel has 32 training assets with eight non-heldout poses,
  the SAME 32 assets with all eight reserved poses, and 128 unseen validation assets with eight poses:
  256 + 256 + 1,024 = 1,536 main generations. Training-pool poses need not be exact previously seen inputs.
- Main score is original continuous-GT spatial edge F1, with every failed generation scored zero.
  Natural EOS/parse success and joint count are separate measures; generation is capped at 2,048 new tokens.
  Conditional J2J must state its successful-view denominator and GT convention; auxiliary scores are
  once per asset/group, not per pose. Spatial edge F1 is not exact tree or joint-identity equivalence.
- The 128 validation assets are a checkpoint-selection panel, not the complete 967-asset set.

## Matched 20k Results

| Arm | Train-pool F1 (256 views) | Heldout-pose F1 (256) | Validation F1 (1,024) |
| --- | ---: | ---: | ---: |
| A | 0.324185 | 0.304345 | 0.282056 |
| B | 0.766583 | 0.786313 | 0.729074 |
| C | 0.337525 | 0.295678 | 0.279036 |

B is substantially better under this matched protocol; C does not improve on A in validation.
The later query-first comparator was stopped at 10,864 visits; its reviewed 10k validation F1
was 0.690604 versus A's 0.192411. It has no 15k/20k comparator. It processes each target pose
separately for all-target supervision and is NOT a reproduction of the historical single-query run.

## Continuations And Learning-Rate Caveat

After the initial OneCycle tail (about 2e-6), both A and B returned to constant 1e-4 without
resetting Adam moments/betas. A used 1e-4 from 20k to 120k, then 3e-5 through 150k.
B used 1e-4 from 20k to 60k, then 3e-5 until the user stop. Saved model/optimizer/RNG/stream
state was restored; B's interrupted 60k stage resumed from its durable 50k boundary.
These changes confound extra exposure with LR policy. Unequal final budgets are not an
architecture-only superiority experiment or proof of an absolute quality ceiling.

| Arm / milestone | Cumulative asset visits | Validation F1 | Interpretation |
| --- | ---: | ---: | --- |
| A | 40,000 | 0.421903 | Improvement from 20k, concentrated in 35k->40k |
| A | 75,000 | 0.680518 | Best through 80k |
| A | 80,000 | 0.645741 | Final 80k below 75k |
| A | 120,000 | 0.713389 | Still below B20k |
| A | 140,000 | 0.7388475706 | Best reviewed A |
| A | 150,000 | 0.737160 | Completed final budget, 3,150 updates |
| B | 35,000 | 0.734596 | Best through 40k |
| B | 40,000 | 0.722256 | Final 40k below 35k |
| B | 55,000 | 0.765294 | Reviewed recovered-stage milestone |
| B | 60,000 | 0.750580 | Completed 60k, 1,260 updates |
| B | 65,000 | 0.7705062590 | Latest reviewed B metric |
| B | 70,000 | Not reviewed | Save recorded at 1,470 updates; no accepted evaluation/payload audit |

A140k generation success / exact count were 0.996094 / 0.484375; A150k was
0.989258 / 0.436523. Thus the final checkpoint is not automatically the best.
Full milestone curves, success/count denominators and per-view predictions remain in the evidence below.

## Earlier Bounded Studies

These studies used different cohorts and budgets; do not merge them with the 5k results.
- Fixed8/T4: A/B each completed 1,024 updates, 8,192 asset and 32,768 supervised-frame visits.
  First joint main+auxiliary passes were A at update 256, B at 128; both ended with 32/32
  exact main GT token sequences and 8/8 auxiliary parent trees. A can fit this task under the
  effective-batch/OneCycle recipe; this does not isolate batch versus schedule or prove generalization.
- Fixed64/T4: 512 updates, effective batch 48, 24,576 visits (384 per training asset).
  Only 8/64 training assets were monitored: 32 seen + 32 heldout views; validation was 32 other
  assets / 128 views. A/B heldout F1 was 0.475601/0.841470; validation F1 was 0.090075/0.361060
  with failures zero. Valid generations were 126/128 and 127/128; exact auxiliary validation
  trees were only 1/32 and 0/32. Both fit all 32 monitored seen-pose token streams; auxiliary
  training trees were 7/8 and 8/8. B passed relative main-task gates, not generalization acceptance.
  See [paired review](../experiments/fusion_ab_medium_v1_review.json); paired J2J used quantized GT
  on common-success poses, unlike original continuous-GT F1. An exact quantized-token fit is not F1=1 by definition.

## Conclusions And Limits

B's real-query/real-residual fusion is the stronger measured later all-target route, not the
original baseline. Neither fitting nor natural EOS establishes unseen-topology correctness.
Intermediate plateaus and late declines do not identify a unique optimization/overfitting mechanism.
The proposed evaluation on the remaining 839 validation assets was not completed; no result on
that cohort is claimed. Subsequent structure/XYZ work is a separate experiment, documented in
[Structure-XYZ](STRUCTURE_XYZ_FILL_20261022.md), not an accepted improvement established here.

## Configuration And Evidence Entrypoints

- Initial 5k configs: [A](../experiments/fusion_normal_a_v1.json), [B](../experiments/fusion_normal_b_v1.json), [C](../experiments/fusion_normal_c_v1.json).
- Continuation chains and LR overrides: [A150](../experiments/fusion_normal_a_lr3e5_150000_v1.json), [B60 recovery](../experiments/fusion_normal_b_recover60000_20261021_v1.json), [B80 target, stopped](../experiments/fusion_normal_b_lr3e5_80000_v1.json).
- Earlier paired configs: [A8](../experiments/fusion_a_fit8_v1.json), [B8](../experiments/fusion_b_fit8_v1.json), [A64](../experiments/fusion_a_medium_v1.json), [B64](../experiments/fusion_b_medium_v1.json).
- Runtime: [train_virtual_fusion_normal.py](../rigweave/scripts/train_virtual_fusion_normal.py); machine-readable status/provenance: [current.json](../state/current.json). Historical configs are not launch authority.
- Raw 5k outputs: `/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/`; earlier fixed-cohort outputs: `/home/wangyy/evorig/outputs/virtual_fusion_20261013/`.
- Local evidence root: `D:/evoweave/server_ops/outputs/virtual_fusion_20261013/`; the following paths are relative to it.
- 20k receipts: `normal_ops/reviewed_normal_a_completion_v1.json`, `normal_ops/reviewed_normal_b_completion_v1.json`, `normal_ops/reviewed_normal_c_completion_v1.json`.
- A terminal/curve index: `normal_ops/reviewed_normal_a_lr3e5_150000_completion_v1.json` and `normal_ops/sequence_20261018_execution_v1.json`.
- B65 evidence directory: `b80_20261021/primary_evaluation_065000_v1/`; files `reviewed_milestone65_v1.json`, `evaluation_selection.json`, `evaluation_065000/summary.json`, `evaluation_065000/rank_0.json`, `evaluation_065000/rank_1.json` bind scores to poses.
- Stop/70k status: `b80_20261021/user_stop_20261021_v1/stop_receipt_v1.json`.
- Recorded checkpoint hashes identify historical payloads; verify existence before any attempted reuse. This cleanup does not recreate, relocate or assert retention of deleted weights.
