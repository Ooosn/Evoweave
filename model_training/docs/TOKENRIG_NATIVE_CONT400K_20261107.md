# Native Qwen 300k to 400k Continuation

User November 7: restore the complete native 300k checkpoint including Adam,
continue to at most 400k, save every 10k, screen on the same valid panel, then
run the original 12 hard cases at 15 beams for exactly two new candidates.
Neither the final checkpoint nor any new model is automatically promoted.

Restore native source 2f50732ee08e98d609c38bc2f73ec3657e5c7022, the full parent
checkpoint SHA256 743f55cc65f833c874c6c0e1ba87181f85fbe04d053d63f7be6047f15161e844,
6270 updates and both rank RNG/sampler states. Keep the original full dataset,
asset order prefix, batch48, frame-dependent microbatches, model, losses and
precision. No hard-case weighting or optimizer reset. Hold the completed
OneCycle endpoint: backbone 2e-6, coordinate head 6e-6, beta1 .95. Extend the
same seeded stream by exactly 100000 visits / 2090 updates to 8360 total.

Preserve all ten full checkpoints from 310k to 400k in SSD and durable storage.
Do not re-export every checkpoint. At each node, evaluate the identical 128
assets and 1024 canonical views under FP32/noTF32 greedy, and the original
1792 GT-reference transport pairs. Keep the original input tensors, raw frames,
normalization and GT fingerprints; no resampling or baseline regeneration.
Do not rerun an unrelated 967-asset endpoint test to choose these candidates.

Freeze the two new candidates after all ten nodes: rank by asset-balanced
greedy original-GT edge F1 (failures zero), then lower GT-reference MPJPE, then
earlier checkpoint. Do not use formal hard-case results to change this shortlist.
Replay the exact 12 inputs and the fixed 20261029-derived seeds, original
15-beam sampling policy, native vocabulary and prefix, with each checkpoint
used for both stages. Reuse the completed OPT300k/330k and native250k/300k
evidence. Compare all-case edge F1 and J2J on identical common-success views,
and inspect missing/extra branches so a geometric mean alone cannot select a
broken tree. Retain OPT330k as the working model unless separately promoted;
the preferred new candidate is a descriptive choice, not a population test.
Keep the prior case105 exact-token replay discrepancy unresolved.

Native job 131942123, pcg01i GPU 3/4, 16 CPU slots, h_rt=345600, with whole-card
78 GiB and process-tree 64 GiB limits. The grant expires Nov11 14:54 JST, save
margin starts Nov11 13:54 JST. Because this phase reuses verified Nov4 scratch
dependencies, its earlier save-stop is Nov9 03:20 JST, before their 5-day age
boundary; the expected run is well inside this margin. Do not touch or refresh
old paths to evade cleanup. Use fresh pinned SSD source and a separate output.
Inherit measured capacity of the unchanged completed dual-GPU training, not
another stress run. A native CPU restoration audit and separate exact admission
precede formal training. Check the two first real updates against the audited
indices, IDs and sampling seeds. No automatic retry, cleanup of prior results,
quota release, new GPU request, or unrelated task interruption.

Native CPU audit finished Nov7 15:16 JST (exit 0, no CUDA, payload cleaned).
It checked the full model and 916 Adam entries, both saved RNG states, the
unchanged 300k sampling prefix and the retained validation inputs. The exact
plan adds 1295824 input-frame visits (5215808 total). The GPU launch will
separately verify live rank RNG restoration and its first two updates.
The accepted result and resource hashes are pinned in the training config;
the same receipts are retained in the durable output's `receipts` directory.
The queue wrapper checks idle status on the two granted UUIDs only. Occupancy
on other devices is not a conflict; scheduler membership and the allocated
UUIDs remain independently checked by the supervisor.

The native training payload started Nov7 15:23:37 JST at f1817c9bde503b1c9ed1cf934e8b19db95311468.
Both ranks restored the 300k model, 916 Adam entries and their saved RNGs.
Actual updates 6271 and 6272 matched all 96 planned visits, sampling seeds,
terminal learning rates and finite loss/gradient checks. At 15:28 JST, both
ranks reported 300192 visits / 6274 updates, and the sampled peaks were 72825
MiB on each GPU. No new checkpoint was due yet. Inspect live receipts for
subsequent progress. Do not sync the running checkout.
