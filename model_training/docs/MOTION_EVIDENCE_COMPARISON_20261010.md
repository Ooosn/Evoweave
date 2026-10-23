# Motion-Evidence Comparison

## Question And Outcome

Does explicit local-motion evidence improve the original single-query flat
UniRig model, and which fusion is worth carrying into a full run?

A paired short screen selected eight-head pair bias (bias_h8). Its fresh full
run completed 1,667 updates on October 10. Small-set diagnostics found a clear
contribution from the existing multi-frame pathway but little robust gain from
magnifying the added bias. Full-budget generation superiority over the
historical baseline remains unestablished.

## Experimental Design

This experiment uses frozen historical HGC manifests: 15,920 train and 856
validation assets, with disjoint identities and no UniMate additions. The
[exact configuration](../experiments/motion_evidence_base_compare_20261010.json)
pins all arguments and asset paths. It predates the current root-RT dataset.

The reference is July HGC flat UniRig sample80000. All new candidates start
from official UniRig weights and a new motion encoder, seed 20260529.
Zero-initialized adapters consume no extra RNG draws. July initialization RNG
was not saved, so bitwise reproduction of its motion weights is not established.

The matched recipe uses two H100s, batch 3 per GPU, accumulation 8, nominal
batch 48, BF16, motion checkpointing, AdamW weight decay 0.04 and clip 1.
OneCycle spans 1,667 updates, with peak LR 1e-4, warmup fraction 0.1 and
division factors 5/10. The full budget is 80,016 nominal exposures but 79,996
actual examples because the original loader retains partial minibatches.

### Variables

- Batch-level T is uniform over 2..24, including the query. Ranks share the T
  schedule; per-asset seeds are replayable. Non-query selection is approximately
  75% FPS / 25% random with at least one random frame; T2 uses one random frame.
- Assign 65,536 dense query samples to nearest fixed query anchors (1,024).
  Kabsch group motion provides observed activity and pairwise relative motion.
  Rank-deficient groups retain geometry/GT and mark their evidence unknown.
- Pair bias enters selected pose-attention heads. Token fusion is a zero-init
  projection of five pooled node features and loses pair identity. Hybrid uses
  both. Role/register attention is unbiased; all original u/c/d evidence states
  and learned coefficients are retained.
- Tokenizer, complete posed GT, decoder, original spatial/temporal layers and
  losses stay fixed. Static inputs are evaluation controls, not augmentation.

### Evaluation

Each screen runs 120 updates using the prefix of the full 1,667-step schedule.
The first 32 fixed validation assets are evaluated at T2/T8/T24 with paired
query/frame/surface seeds. The first eight also get a T8 static control and
greedy generation at T2/T24, cap 1,400 tokens. Generation therefore has 16
views but only eight independent assets. Invalid outputs score zero F1 against
complete continuous GT.

A relative CE difference below 1% is a practical tie, not a significance test.
Selection also considers generation, static behavior, time and memory. The
fresh full run restarts from initialization, not a screening checkpoint.

## Short-Screen Results

| Candidate | Mean CE | Valid views | All-view F1 | Median logged step, seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 1.520796 | 16/16 | 0.762480 | 18.914 | 68.417 |
| Bias, 2 heads | 1.531624 | 14/16 | 0.720449 | 20.480 | 69.489 |
| Bias, 4 heads | 1.535127 | 14/16 | 0.660961 | 20.868 | 70.355 |
| Bias, 8 heads | 1.516853 | 16/16 | 0.772189 | 22.986 | 71.625 |
| Token only | 1.525148 | 16/16 | 0.745807 | 20.290 | 68.453 |
| Hybrid, 8 heads | 1.511214 | 16/16 | 0.761046 | 23.665 | 71.627 |

Eight biased heads gave better generation suitability than two/four.
Hybrid reduced CE by 0.37% relative to bias_h8, within the practical tie,
while F1 was lower by 0.01114 and logged step time was 3% longer. Bias_h8 was
selected under the preference for pair bias unless pooled tokens clearly help.

The asset-bootstrap 95% CE interval for bias_h8 minus control is
[-0.01473, 0.00677]; hybrid minus control is [-0.01690, -0.00227].
These are conditional on 32 assets and one training seed, not evidence of
full-budget quality superiority. Times are medians of 12 logged steps after
startup, not end-to-end throughput; memory is rank-0 allocated peak.

## Completed Full Run

Runtime source `fb27ce9` completed at 2026-10-10 21:14:25 JST in 44,144.60
seconds. All 128 expected arguments matched. Checkpoint inspection recorded
step 1,667, nominal exposures 80,016, three optimizer groups with 888 populated
states and scheduler total/last epoch 1,667. All twelve bias tensors [8,3] were
finite and updated in each evidence channel.

The last inline validation, step 1,600, had CE 1.159892 and teacher-forced EOS
accuracy 1.0. Those measure prediction under a correct prefix, not reliable
free generation. The later small-set diagnostic below is not a full heldout
generation comparison. Historical completion is verified; current weight
availability is separately recorded in the [baseline](BASELINE_FLAT_UNIRIG.md).

## Bias-Scale And Motion Diagnostics

Full-forward repeats differed in CE by about 0.0014..0.0015 even with identical
surface features, query points and RNG. Variation localized to recomputed
motion evidence; the exact CUDA primitive was not isolated. Replaying captured
motion-boundary tensors produced exact condition and CE matches. The diagnostic
therefore uses cached evidence, with parameters and source checkpoint unchanged.

For 32 T8 assets, the gradient resultant ratio
`R = norm(sum(g_i)) / sum(norm(g_i))` was 0.190880. Its independent-direction
RMS reference was 0.196847 (ratio 0.969687), with mean pair cosine -0.001461
and 49.80% negative pairs. Gradients were dispersed but not unusually opposed
relative to that reference; this is not a hypothesis test or training replay.
The mean-gradient norm was 2.98712e-4 and median individual norm 1.40927e-3.
Repeated BF16 gradients differed by median 0.420% relative L2 per sample
(maximum 0.564%; mean gradient 0.397%), below the measured signal.

| Bias scale | Mean CE delta vs scale 1, T2/8/24 | F1, T2/24 | Valid views |
| --- | ---: | ---: | ---: |
| 0 | -0.0000970 | 0.889827 | 16/16 |
| 1 | 0 | 0.899344 | 16/16 |
| 3 | +0.0000299 | 0.892946 | 16/16 |
| 10 | -0.0001167 | 0.898829 | 16/16 |

CE uses 32 assets; generation uses eight assets at two T values.
Asset-clustered 95% intervals for CE deltas at scales 0/3/10 are respectively
[-0.0003025,0.0001268], [-0.0002003,0.0002759] and
[-0.0003324,0.0000902]. All cross zero. F1 delta intervals are approximately
[-0.020421,0.000007] for scale 0 and [-0.004520,0.002975] for scale 10.
These small-subset intervals do not cover training-seed variation.

On the matched eight T8 assets, replacing motion by repeated query frames
raised CE from 1.021711 to 1.074929 (+0.053218, 5.21%; paired 95% interval
[0.032047,0.073342]). Permuting pair-evidence anchor alignment increased CE by
0.001071 (interval [0.000355,0.001855]). The existing multi-frame features had
a much larger observed contribution than the added bias. Maximum coefficient
magnitude was 0.015117; maximum measured within-row bias span was 0.016970.

Decision: retain the learned scale and LR. Magnification showed no reliable
benefit. Higher-LR training was not tested; a small local derivative favoring
larger coefficients does not establish a useful training intervention.

## Frame-Budget Engineering Result

An opt-in sampler uses per-rank `B = min(cap, floor(F/T))`, with T uniform per
microbatch. The legacy fixed-batch path is unchanged. Variable batches
accumulate token-sum CE and normalize once by global valid target-token weight
after DDP averaging, before clipping. It saves the exact next epoch/batch
cursor and counts actual assets, input frames and valid target tokens.

For F=72, cap12 failed: T6/B12 exhausted the 93% memory cap, and T9/B8 plus
T8/B9 exceeded the 90% allocated-memory criterion. Cap6 passed all integer
T2..24 single-GPU cases, worst peak 70.153 GiB at T12/B6. The longest stress
target had 403 input tokens; this was not a full-dataset safety guarantee.

A two-rank save/resume test at source `85d19fc` processed 68/138/224 actual
assets and 1,004/1,900/2,838 input frames at steps 1/2/3. It resumed the step-1
checkpoint at epoch0/batch8 and stopped at 224 for max_samples223: one explicit
whole-step overshoot. DDP peak reached 72.680 GiB. This verifies accounting and
resume functionality, not dataset-wide memory safety or a throughput gain.
No full frame-budget training was completed by this test.

## Evidence

Configuration and historical recipe:
[motion_evidence_base_compare_20261010.json](../experiments/motion_evidence_base_compare_20261010.json).

Run root:
`/home/wangyy/evorig/outputs/motion_evidence_base_compare_20261010/full/bias_h8`.

Under `/home/wangyy/jobs/motion_compare_hgc_20261010/results/`:

- `complete_screen_summary.json`, `hybrid_layer_updates.json`.
- `full_bias_h8_completion.json`, `full_bias_h8_final_checkpoint_inspection.json`.
- `bias_replay_audit_evaluation.json`, `bias_diagnostic_cached_evaluation.json`,
  `bias_diagnostic_cached_analysis.json`.
- `frame_profile_evaluation.json`, `frame_confirm_evaluation.json`,
  `frame_budget_smoke_bias_h8_evaluation.json`.
