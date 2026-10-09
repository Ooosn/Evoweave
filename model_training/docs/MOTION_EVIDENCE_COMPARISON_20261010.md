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
