# Small-Sample Pair-Aware Value Probe

## Objective and Scope

The full query-pose skeleton remains the target even when supplied motion
reveals only part of it. Motion can resolve geometric ambiguity; low observed
motion means unknown, not absent joint or proved rigidity. The accepted base
already contains a learned geometry/structure prior. This probe tests a new
interface to that prior, not the assertion that the base has none.

User authorization on Oct11 is for continued small-sample experiments, not
another full training. The historical HGC 15,920/856 manifest override and
query-mesh bbox normalization remain unchanged. No target pruning, bone-name
templates, motion-derived GT labels, or noninput poses enter conditioning.

## Explicit Architecture

After pose-inner attention and before same-anchor temporal attention, each of
the twelve blocks updates only its anchor tokens. Role/register prefix tokens
are excluded from this update. Original attention and learned pair bias stay.

For x of shape B,T,Q,D, set h = Linear_in(LN(x)), with width64. For each of the
three unchanged pair-state channels k in {u,c,d}, compute m_k = E_k h / Q.
Then add Linear_out(GELU(Linear_mix([h,m_u,m_c,m_d]))) to x. Only Linear_out
starts at exactly zero; the internal projections have ordinary initialization.
The Q denominator preserves channel evidence mass; tiny c/d values are not
renormalized to full-strength messages. Multiplication uses B,Q,T*64 values,
without materializing a Q-by-Q-by-D edge-feature tensor or repeating E over T.

Under allunknown evidence, m_u summarizes geometry while each node's own h
remains available. A learned update can therefore differ across inactive
regions, without inventing measured motion. Whether this improves skeleton
completion is an empirical question. Adding this capacity alone is not proof.

Adapters are enabled explicitly after strict base-checkpoint loading. Default
construction has no adapter parameters and preserves existing checkpoint keys.
This experimental opt-in is not implicitly enabled in the formal trainer.

## Matched Protocol

- Freeze the completed bias_h8 sample80000 base, including its old pair bias.
- Select32 training and16 validation asset IDs deterministically, round-robin
  across source and target-count bands <=16,17-40,41-64,>64. Record all indices.
  This is a deliberately stratified diagnostic, not a population estimate.
- Each asset has normal8, weak2, and query-repeated static8 inputs. Query pose,
  complete target and normalization must match exactly across these views.
- Capture actual surface features and all evidence once per view. Every cache
  must replay full-forward CE within2e-5. Both arms reuse the same tensors.
- Train only adapters for120 AdamW steps, LR1e-4, weight decay0.04, clip1,
  microbatch1, accumulation4. Weight CE by actual valid target tokens within
  each step. Each arm gets480 asset-view exposures with identical shuffled order.
- Actual arm supplies all real u/c/d to the new adapter. Capacity control
  supplies allunknown only to the new adapter. Both old backbones and pair
  biases still receive real motion. The control is not geometry-only.
- Evaluate frozen base, both trained arms, and the actual arm with new-adapter
  evidence ablated. Use natural greedy generation up to1400 tokens, without
  GT-prefix rescue, count forcing, or excluding failures from topology F1.
- Save only adapter/optimizer states, with immutable base identity and input
  cache hash. Existing full checkpoints are never rewritten.

Model decoder stays in eval mode. Frozen motion blocks use their existing
activation checkpointing with zero dropout. Input activation gradients are
enabled in baseline and candidate forwards to align computation paths; this
does not unfreeze the base. Transformer fastpath is disabled consistently.

## Offline Missing-Structure Diagnostics

For retained joint j with retained parent p, form M_j(t)=inv(A_p(t))*A_j(t)
from stored bone transforms using target_raw_indices. Measure rotation of
M_j(t)*inv(M_j(query)) and displacement of the joint head in the parent's
reference space, in query normalization units. This cancels parent-driven
motion. Roots and invalid/nonrigid transforms (singular values >1pct from1)
are unknown and excluded, with coverage recorded.
Local-transform activity is a proxy, not proof that the available mesh surface
reveals that joint: skin support, visibility and ambiguous deformation can
still limit what the model can observe. All retained joints stay in the target.

Quiet strict: <=1degree and <=0.005 query units; loose sensitivity: <=3degrees
and <=0.01. Active: >=5degrees or >=0.02. These are transparent diagnostic
thresholds, not calibrated anatomical truth. Keep the underlying continuous
measurements. Hidden-demonstrated means quiet in supplied frames, active in
other stored frames, while another joint is active in the supplied frames.
Noninput frames are not unseen assets or a strict held-out animation split:
the existing FPS frame selector can inspect the40-frame sequence.

Record GT-to-predicted joint distance, coverage within0.05 query units, and
nearest-joint mapped directed-edge recall by stratum. This correspondence is
a geometric heuristic, not established joint identity. Also record whole-tree
F1, count error, natural EOS/hitmax, cycles and near-zero edges. A near-zero
edge can also arise from legitimate quantization and is only a collapse proxy.

## Gates and Interpretation

First run the separately recorded relation_preflight: selected min/max target
counts from each split, all three views, exact zero-init condition/prefix
parity, <=2e-5 CE replay, and two finite optimizer steps per arm. Verify only
adapter updates and the90pct single-GPU memory cap. Review and record its
result before preparing relation_screen. No silent retries or full training.

Accepting execution means the comparison ran as recorded, not that the method
won. Improvements must be judged against both the frozen base and capacity
control, with paired asset-level uncertainty and low/no-motion generation.
The offline analysis resamples whole assets (all three views together),4000
times. Intervals are exploratory and not multiplicity-corrected; fewer than
three contributing assets in a stratum cannot support an interval. Failed
generations count as zero coverage/edge recall and zero whole-tree F1.
This narrow frozen-base experiment cannot establish the best end-to-end
architecture, long-run behavior, or generalization from32 training assets.

## Current Result

CPU checks passed:19 residual tests,3 encoder-integration tests,8 probe/metric
tests,6 unchanged motion-evidence tests,3 paired-analysis tests. Controller
tests:4 passed,1 skipped on Windows because it requires the Linux launcher.

GPU preflight from53209ff completed Oct11 01:04 JST, all three exits0. Selected
training targets have4/102 joints; validation targets7/147. All12 full/cache
CE and condition comparisons are exactly equal. Both arms'12 zero-init
condition/CE comparisons are exact and all64-token greedy prefixes match.
Each arm completes2 steps (8 samples,46 input frames,1438 target tokens), with
nonzero output-projection gradients at step1 and internal-projection gradients
at step2. All96 adapter parameter keys update; only1,807,872 parameters train.
The loaded base hidden width is1024. Base parameter versions/gradients and
source checkpoint size/mtime are unchanged. Peak allocated memory4.710GiB.
No checkpoint or persistent input cache was saved by this preflight.

The plain paired screen then completed from d2f88f1 at01:41 JST, all exits0,
144 exact cached-boundary comparisons,192 generation rows, and4.724GiB peak.
Each arm used480 exposures/2880 frames/81045 target tokens. The source cache
and adapter-only checkpoints remain remote. Base parameters did not change.

| Model | Validation CE | All-row topology F1 | Hitmax /48 |
| --- | ---: | ---: | ---: |
| Frozen base | 1.217828 | 0.590208 | 9 |
| Plain actual-E residual | 1.243144 | 0.522948 | 12 |
| Same-capacity unknown-E residual | 1.246196 | 0.548763 | 9 |
| Actual arm, new-branch E ablated | 1.243077 | 0.544602 | 10 |

The plain candidate is REJECTED for quality. Its case-normalized training CE
fell7.37pct in the last20 steps, but validation CE rose2.08pct. The unknown
arm similarly improved training and worsened validation. Static F1 fell from
0.611542 to0.487161; static quiet-joint coverage fell0.6658 to0.4518.
Actual versus unknown changed CE by-0.245pct but topology F1 by-0.02582.
Ablating new-branch E after training changes CE by only-0.0054pct, with an
asset bootstrap interval spanning zero. This is consistent with an unwanted
explicit-E-independent feature shift and overfitting, not evidence of a useful new
motion correction. It does not disprove motion-based structure completion.
On the strict hidden-demonstrated weak2 stratum (8 assets,52 joints), actual
and unknown have identical coverage/edge recall, not an extra motion benefit.

## Targeted Anchored Follow-Up

Change only the residual parameterization to F(x,E)-F(x,U), where U supplies
allunknown [1,0,0] on every pair. Share every projection between both terms.
This removes the learned explicit-E-independent offset: for E=U the correction is
exactly zero even after training. The old backbone still processes geometry
and motion and predicts the complete skeleton. In partially observed inputs,
new motion corrections can still propagate through the original global and
temporal attention; inactive joints are not removed or assigned negative labels.
Here x already contains the base's implicit motion features. F(x,U) is not a
pure static encoder; only the NEW module's explicit pair-state evidence is
replaced in the reference term.

This guarantees a neutral NEW correction for allunknown states, not that the
base's static predictions are perfect. Numerical evidence from truly static
meshes may be near rather than exactly U, so actual static behavior is also
measured. No confidence threshold, evidence channel deletion or GT pruning is
introduced. Shared output bias cancels and can legitimately remain unchanged.

Reuse the hash-verified original inputs, full targets, selected assets, seeds,
initial projections,120-step/480-exposure schedule and optimizer. New weights
start fresh; do not continue the rejected plain adapter. Test only one learned
anchored arm: an allunknown branch is identically the frozen base by design.
Require pre-attachment cache parity, zero-init parity, and post-update allunknown
condition/prefix parity on min/max2 train and2 valid assets before the short run.

After training, generate all48 actual-input cases, check all48 unknown controls
against the frozen-base conditions, and measure48 CE interventions that permute
E's two anchor axes only in the new branch. The latter preserves E distributions
but breaks correspondence; it tests aligned relation use separately from a
generic motion-presence switch. The original pair bias always receives true E.
Reuse previous frozen-base generations only after exact condition parity, and
label them as reused. This second look at the same16 validation assets is
exploratory, not an independent confirmation or authorization for full training.

Anchored preflight from98e52c0 passed all exits at02:14 JST. Six base-cache,
six zero-init and six post-training unknown CE/condition comparisons are exact;
all zero and trained-unknown64-token prefixes match. Two finite updates change
84 parameter keys;12 shared output biases cancel and remain unchanged. Old
parameters and full hashes of source checkpoint/cache/report remain unchanged.
Peak allocated memory4.724GiB. This verifies the trained neutrality contract,
not generation improvement. The anchored short screen is separately gated.
