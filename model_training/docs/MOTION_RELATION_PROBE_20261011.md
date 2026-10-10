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

This accepts execution correctness only. The paired120-step small-set screen
still needs its own committed operation; no new full training is authorized.
