# Checkpoint-Matched Motion Evidence Comparison

## Authorization And Scope

The user authorized autonomous implementation, small-scale preflight, selection
of a suitable head count and fusion, and then full training on 2026-10-10 JST.
Report after full training starts and its initial stability is verified. This
supersedes the older July stack-close proposal as this branch's active work.
Do not change the flat UniRig tokenizer, complete posed GT, decoder, original
global/temporal layers, losses, trainable modules, or historical training budget.

This is an explicitly requested **historical-data comparison**, not an update
to the current production dataset pointer. Use the frozen HGC manifests in
`../experiments/motion_evidence_base_compare_20261010.json`: 15,920 train and 856
validation rows. All 16,776 paths were checked, with no cross-split identity
overlap or UniMate rows. The old branch-level Westlake dataset pointer is not
the manifest for this experiment.

## Matched Recipe

The reference is HGC flat UniRig sample80000 at step 1667. Preserve two H100s,
batch 3 per GPU, accumulation 8, nominal batch 48, BF16, motion checkpointing,
AdamW weight decay 0.04, clip 1, and the same 1667-step OneCycle schedule
(maximum LR 1e-4, warmup fraction 0.1, division factors 5 and 10).
The nominal budget is 80,016 exposures; the unchanged partial-minibatch loader
contributes 79,996 actual examples. Do not conceal that distinction.

Initialization is official UniRig weights plus a new motion encoder, not the
old final checkpoint or its crash-resume checkpoint. Historical initialization
RNG was not saved. New candidates share explicit seed 20260529, with zero-init
evidence adapters that consume no random draws. No claim of bitwise historical
motion-weight reproduction is justified.

## Approved Changes

- Uniform batch-level frame count T in [2,24], including the first query frame.
- Non-query quota approximately 75% FPS and 25% random, with at least one
  random non-query frame in every sample. At T=2 there is one random frame and
  no FPS frame. DDP ranks use the same T schedule; per-asset seeds are replayable.
- Dense query samples (65,536) are assigned to their nearest of 1,024 fixed
  query anchors. This is neither fixed groups of 64 nor grouping by GT bones.
- Kabsch group motion gives local observed activity O and pairwise relative
  motion M. Preserve all u/c/d states and learned coefficients, including wu.
  Rank-deficient groups have explicit unknown evidence, but their geometry
  tokens and complete skeleton GT remain present.
- Bias is added to selected pose-attention heads. Token fusion projects five
  pooled node features as a zero-initialized residual; it deliberately loses
  pair identity. Hybrid combines both. Role/register attention is unbiased.
- No static training augmentation, active-skin pruning, new supervision, or
  inference-only fallback. Static input is a validation control only.

## Preflight And Screening

CPU checks cover static/global/relative motion, order and duplication
invariance, invalid-group handling, split-head forward/backward equivalence,
checkpointed gradients, initialization, frame schedule, and random quota.
GPU preflight uses the accepted reference to verify zero-init condition/CE
parity, and raw logits from actual cached generation against teacher forcing
on the same prefix. T=24 and T=2 full-trainable B=3 Adam steps must be finite.
Test-only prefix forcing is never used for actual generation metrics.

Before new training, re-evaluate the accepted sample80000 checkpoint. Then
screen control, bias with 2/4/8 heads, token-only, and hybrid at the selected
bias head count. Each pilot has 120 optimizer steps with the **full 1667-step
LR schedule**, not a compressed schedule. All start fresh, use paired sampling,
and save model-only checkpoints for screening. The final run starts fresh
again and keeps optimizer/scheduler checkpoints.

Paired validation: first 32 frozen validation rows at T=2,8,24, with identical
query/frame/surface seeds; first eight also receive a T=8 static control.
Generation: first eight at T=2 and T=24, greedy, cap 1400, no forced EOS.
Topology metrics use complete continuous GT; invalid/non-terminating outputs
receive zero F1. Historical aggregate F1 used a different target/evaluation
protocol and is not directly comparable.

Selection considers CE across T, generation validity and F1, static control,
time and memory. A 1% relative CE difference is a practical tie threshold, not
a significance test. Prefer the cheaper suitable configuration in a tie;
retain pair bias unless pooled token fusion shows a clear benefit. Small
screening selects a practical candidate, not a globally optimal architecture.

### Initial Screen And Hybrid Head Choice

All five initial candidates completed 120 steps, matched all 128 expected
arguments, and passed paired evaluation-input and logged training-input checks.
Source, exit codes and full results are in the external job directory under
`results/initial_screen_summary.json` and `results/fixed_screen_execution.json`.

| Candidate | Mean CE over T=2,8,24 | Valid generations | All-row topology F1 | Median logged step (s) | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1.520796 | 16/16 | 0.762480 | 18.914 | 68.417 |
| bias_h2 | 1.531624 | 14/16 | 0.720449 | 20.480 | 69.489 |
| bias_h4 | 1.535127 | 14/16 | 0.660961 | 20.868 | 70.355 |
| bias_h8 | 1.516853 | 16/16 | 0.772189 | 22.986 | 71.625 |
| token | 1.525148 | 16/16 | 0.745807 | 20.290 | 68.453 |

Use eight biased heads for the planned hybrid screen. Its CE is lowest among
bias candidates; all generations terminate, unlike two/four heads, and its
all-row F1 is higher. Accept 12.2% longer logged steps than two heads. The 1% CE
tie rule does not override the observed generation degradation of fewer heads.
Times are medians of 12 logged steps after excluding startup, not whole-run
throughput; memory is the rank-0 allocated-memory peak, not nvidia-smi usage.

This is a suitability decision, not proof of architectural superiority. The
whole-asset bootstrap 95% interval for bias_h8 minus control CE is
[-0.01473, 0.00677], spanning zero. Generation covers only eight assets at two
frame counts, and no repeated initialization was performed. Token-only does
not show a clear quality benefit. Compare hybrid_h8 before choosing the final
fusion; no formal run is authorized by this head-choice result alone.

### Final Fusion Selection

The hybrid_h8 screen also completed 120 steps and all paired-input and
argument checks. Mean CE was 1.511214 (T2 1.514128, T8 1.510173, T24 1.509341),
static CE 1.409289, generation success 16/16, and all-row topology F1 0.761046.
Median logged step was 23.665 seconds and peak allocated memory 71.627 GiB.
All 12 bias tensors and 12 token tensors were finite and updated from zero.
The bias-only and token-only candidates likewise updated every adapter layer.

Select **bias_h8** for the fresh full run. Hybrid's 0.37% lower CE relative to
bias_h8 lies within the predeclared 1% practical tie, while generation F1 is
lower by 0.01114 and measured step time is 3.0% higher. Both have 16/16 valid
generations. Under the recorded preference for pair bias unless token fusion
has clear benefit, retain bias-only and do not add the pooled-token path.
The no-evidence control is a diagnostic comparator, not evidence that motion
fusion must improve at full budget. Its short-run CE is practically tied with
the selected configuration; full-budget superiority remains unverified.

Evidence: `results/complete_screen_summary.json` and
`results/hybrid_layer_updates.json` under the external job root. The
asset-bootstrap CE interval for hybrid minus control is [-0.01690, -0.00227],
but this is conditional on one short training seed and the selected 32 assets;
it does not override the practical tie rule or prove final generation quality.
The full run uses the original 1667 steps and full optimizer/scheduler saves,
official UniRig initialization, and a fresh motion encoder with seed 20260529.
Neither a screening checkpoint nor the historical final checkpoint is resumed.

## Execution And Acceptance

Use the existing allocation 129547842 on pcg01i, physical H100 devices 5 and 7.
Refresh queue, PID and GPU state before launch. Preserve its keepalive and all
unrelated jobs. Virtual memory must be unlimited; never request s_vmem.
Source is synchronized via Git into a separate clean HGC worktree; no copying
uncommitted core code. Runtime versions and third-party assets stay unchanged.

`record_motion_operation.py` records exactly one stage/candidate in current
state. Commit and push each prepare/result before the next GPU operation.
`run_motion_experiment.py` checks that state and refuses accidental overwrites.
Results and full logs live outside the source tree. Do not promote to full
training until preflight, reference evaluation and candidate results are read.

Initial full-run acceptance requires both GPUs on the intended job, finite
loss and gradients, varied T, nonzero adapter updates, no OOM/traceback, and
the first 5000-nominal-exposure checkpoint with optimizer and metadata.
These establish execution stability, not final model quality.

### Full Run Initial Acceptance

The fresh bias_h8 full run started from committed source/state `fb27ce9` on
the two intended H100 devices. All 128 expected arguments matched, including
1667 total steps, no early stop, no init/resume override and optimizer saving.
`checkpoint_sample_5000.pt` was saved at step 105, nominal exposure 5040, and
training continued through logged step 120 (loss 1.479439, pre-clip gradient
norm 1.018278). Logged frame counts covered all integers from 2 through 24.

CPU-only mmap inspection verified the 7,995,136,886-byte checkpoint, three
optimizer groups with 888 populated parameter states, scheduler total 1667
and last epoch 105, and all twelve bias tensors of shape [8,3]. Every one of
the 288 coefficients was finite and nonzero, including all u/c/d columns.
No full-model or optimizer-tensor finite scan was performed. Both live ranks
used physical GPUs 5 and 7, each at 100% utilization at 09:58:22 JST.

The primary accepted initial stability, not completed training or final
quality. Evidence is `results/full_bias_h8_stability.json` and
`results/full_bias_h8_checkpoint_inspection.json` under the external job root.
At initial acceptance, training remained running in the existing allocation
and the runtime checkout stayed at its launch commit while local state
recorded `stable_running`.

### Full Run Completion

The full run completed normally at 2026-10-10 21:14:25 JST, after 44,144.60
seconds. Controller, child and outer exit checks passed. Both the final and
sample80000 checkpoints contain step 1667, nominal exposure 80016, all 128
matched arguments, three optimizer groups with 888 populated states, and
scheduler total/last epoch 1667. All twelve final bias tensors are finite
and updated, including the u/c/d columns. Last logged training step 1660
is expected under the ten-step log interval; checkpoints establish completion.

The final inline validation is step 1600: CE 1.159892 and teacher-forced EOS
accuracy 1.0. No additional GPU generation evaluation was performed, so
full-budget generation quality and superiority to the historical baseline
remain unverified. Completion is execution acceptance, not quality acceptance.

Evidence: `results/full_bias_h8_completion.json`,
`results/full_bias_h8_final_checkpoint_inspection.json` and
`results/full_bias_h8_artifact_inventory.json` under the external job root.
A fresh 22:20 JST read-only inspection confirmed both GPUs idle and qlogin
129547842 retained. Source ran unchanged at `fb27ce9`; completion is recorded
locally after inspection. The state permits inspect/report only; a new GPU
evaluation or training operation needs its own committed preparation.

### Frame-Budget Follow-Up Contract

The user chose a fixed per-rank microbatch frame budget, allowing more asset
exposures for short clips. The opt-in sampler uses B=min(cap, floor(F/T));
T remains uniform per microbatch, not per asset. The default legacy path is
unchanged. An explicit calibrated cap is required because decoder memory also
scales with assets and complete target lengths, not only B*T.

Frame-budget training accumulates token-sum CE gradients and normalizes once
by global valid target-token weight after DDP averaging, before clipping.
This mode currently accepts only the recorded CE-only BF16 recipe; auxiliary
losses need an explicitly designed normalization before they can be enabled.
Samples, input frames, token weights and the per-T exposure histogram are
counted from actual microbatches. Checkpoints save the next epoch/batch cursor
because variable batch counts invalidate fixed-length-epoch resume arithmetic.
An optional actual-sample stop is checked after a complete optimizer step.
Scheduler steps still count optimizer updates; fixed frame budget does not
imply the old effective batch of 48 or an unchanged asset exposure budget.

The first calibration uses a disposable copy of the completed checkpoint,
at most 32 fixed validation assets and two retained padding-stress samples.
It includes existing Adam states and gradients resident before the second
forward. Allocation is limited to 93% of one H100; approval requires measured
allocated peaks no greater than 90% and finite gradients/updates. The first
OOM ends the ascending candidate sweep. Unsafe or unmeasured cases are never
approved by a complete=true profiling report. Two-rank execution and resume
still require a separate recorded smoke test before formal training.

First calibration completed at 23:21 JST with controller/child exit zero.
F72/cap12 is rejected: T6/B12 OOM at the 93% allocation cap; T9/B8 and T8/B9
are above the 90% allocated-memory ceiling. Measured T24/B3, T18/B4, T14/B5,
T12/B6 and T10/B7 peaks are 68.442, 69.012, 68.010, 70.153 and 69.150 GiB.
No all-T cap is approved by this sparse sweep. F72/cap6 will be independently
confirmed for every integer T2..24 before a two-rank save/resume smoke.
Evidence: external results/frame_profile_evaluation.json; GPU5/7 idle at
23:22 JST, allocation129547842 preserved. No checkpoint was written.

### Bias Replay Localization

The two-asset T8 boundary audit completed at 23:29 JST with all exits zero.
Surface features, query points, validity masks and CPU/CUDA RNG states are
identical across full repeats; recomputed evidence states differ by up to
4.72e-6. Full-repeat CE differences are 0.001418 and 0.001515; restoring RNG
does not remove them. Every module is in eval mode and parameter versions
are unchanged. The exact CUDA primitive is not isolated by this experiment.
Replaying the captured motion-boundary tensors yields bitwise-identical
condition and exactly equal CE to the original full forward, on both assets
and both replays. This validates a cached-evidence intervention protocol,
not a modification to the trained model or evidence mathematics.
Repeated BF16 backward bias gradients still differ by about 0.42-0.46% in
relative L2; the complete gradient study must record this numerical floor.
Evidence: external results/bias_replay_audit_evaluation.json.

All-T F72/cap6 confirmation completed at 23:38:52 JST on source8e00ab6.
All 23 exact schedules passed finite-update and 90% memory acceptance.
The worst allocated peak is 70.153 GiB at T12/B6; T24/B3 is 68.442 GiB,
T2/B6 is 26.517 GiB. All three exits are zero; GPUs idle at23:39.
The stress subset's longest complete target has403 input tokens. This is
not a guarantee for every training asset, long-target mixture, or DDP bucket
allocation. The next execution gate is a separate two-rank save/resume smoke.
Evidence: external results/frame_confirm_evaluation.json.

### Completed Cached Bias Diagnostic

Source95424a4 completed at23:47:14 JST with controller/child/outer exits0.
All408 CE comparisons,32 sample gradients plus32 repeated gradients, and64
natural greedy generations completed. All104 normal/static cache contracts
and32 CE replays matched exactly (zero CE difference), with exact coefficient
restoration. No optimizer or checkpoint writes. GPUs idle at23:47.

For the32 T8 assets, R=||sum g_i||/sum||g_i|| is0.190880; the independent-
direction RMS reference sqrt(sum||g_i||^2)/sum||g_i|| is0.196847. Their ratio
is0.969687. Mean pair cosine is-0.001461, with49.80% negative pairs. Gradients
are dispersed, not unusually mutually opposed relative to this geometric
reference. This is not a hypothesis test or a reconstruction of training.
The mean gradient norm is2.98712e-4; median individual norm is1.40927e-3.
Repeated BF16 backward differs by0.420% median per-sample relative L2,
0.564% maximum, and0.397% for the mean gradient, below the measured signal.
Channel R / independent-reference values are u:0.32037/0.30971,
c:0.19881/0.22137, d:0.19194/0.19733. No channel shows a large opposing trend.
The common-scale derivative at1 is-2.486e-6 CE per unit scale;19/32 assets
locally favor increasing it, but this does not establish a useful LR change.

| Bias scale | Mean paired CE delta vs1, T2/8/24 | Topology F1, T2/24 | Natural success |
| --- | ---: | ---: | ---: |
| 0 | -0.0000970 | 0.889827 | 16/16 |
| 1 | 0 | 0.899344 | 16/16 |
| 3 | +0.0000299 | 0.892946 | 16/16 |
| 10 | -0.0001167 | 0.898829 | 16/16 |

CE uses32 unique assets; generation uses8 unique assets at two frame counts,
not16 independent assets. Bootstrap after averaging frame counts within
each asset gives CE-delta95% intervals [-0.0003025,0.0001268] for0,
[-0.0002003,0.0002759] for3, [-0.0003324,0.0000902] for10. All cross zero.
Scale10 F1-delta interval is[-0.004520,0.002975]. Scale0 is approximately
[-0.020421,0.000007]. These small fixed-subset intervals exclude neither
null effects nor all practical alternatives; they do not cover seed variation.

On the matched first8 T8 assets, true-motion scale1 CE is1.021711;
repeating the query frame raises it to1.074929 (+0.053218, about5.21%;
paired95% interval[0.032047,0.073342]). Permuting the pair-evidence anchor
alignment raises CE by0.001071 (interval[0.000355,0.001855]). Real multi-frame
features have a much larger observed contribution than this extra bias.
Largest coefficient magnitude is0.015117; maximum measured within-row
anchor-bias span is0.016970, consistent with a small correction.

Decision: preserve current learned scale and LR. There is no demonstrated
benefit from simply magnifying the coefficients, and no evidence here for
an abnormally severe gradient conflict. A higher-LR training intervention is
not tested; full held-out generation superiority remains unverified.
Evidence: external results/bias_diagnostic_cached_evaluation.json and derived
results/bias_diagnostic_cached_analysis.json (asset-clustered bootstrap).
