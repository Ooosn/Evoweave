# Standalone Stage 2 Schedule Comparison

Compare 12 spatial + 12 same-anchor temporal layers (ST repeated 12 times)
with 16 spatial + 8 same-anchor temporal layers (SST repeated 8 times).
Keep the surface query count, width, registers, learned motion bias, target
loss and reference-anchor transport readout. Do not instantiate a skeleton
decoder or construct a skeleton token target. The CPU audit extracts only the
released UniRig mesh encoder and its projection into a verified mesh-only pack;
both runs use that pack, the same common block initialization and the same
readout initialization. No joint-trained donor or capacity weights initialize
formal training.

Use the retained immutable 5,000-asset manifest and verified SSD cache. Each
arm runs 100,000 asset visits over 20 passes with effective batch 48 on two
H100s. Compare the exact asset, frame, sampling seed and reference streams.
At 12 and 24 frames use the largest-joint and largest-geometry stress cohorts
with forward, backward, AdamW updates and a second step per cohort. Attempt
24 assets per rank at 24 frames independently for both arms. If either lacks
capacity, retest both with 12 per rank and accumulation 2 for 13-24 frames.
Short frames stay at 24 per rank. The whole-device limit remains 78 GiB;
capacity acceptance requires a 3 GiB margin. No fallback on non-memory errors.

The unchunked first attempt could not keep the 24-asset high profile in memory.
In particular, the SST schedule exhausted the cap while recomputing its
two-spatial group even at the 12-frame stress case. Preserve those receipts.
For the new isolated attempt, checkpoint a group on independent slices of at
most four assets; this does not split an asset or change its temporal
attention, precision, data, objective or optimizer update. Test the 24-frame
case first in each arm.

Save and verify full recovery states at the first update and every 10,000
visits. Evaluate the fixed 128-asset panel at initialization and every 10,000
visits using eight input poses and GT references at slots 0 and 4. Encode each
mesh clip once and read both references. Retain paired geometry and input
fingerprints, MPJPE, PCK, bone-vector error and measured evaluation time.
This measures coordinate prediction given a reference skeleton, not an
end-to-end claim about a predicted reference. The working model stays unchanged.

Use native job 131942123 on pcg01i, GPU3/4 and 16 CPUs, with save-stop at
2026-11-11 13:50 JST. Stages are admitted and recorded on a private branch
without changing the checkout during a payload. If any stage fails, retain
the resource and rank evidence, preserve the grant and do not retry unless
the explicit 24-to-12 capacity fallback applies.

## Fixed evaluation inputs and continuation

Full-data, anchor readout, and SkinTokens fine-tuning use the same frozen
evaluation record. It stores dense points and normals, tracked anchor indices,
positions and normals, GT joints and parents, raw frame IDs, and the
normalization fingerprint and per-frame mesh diagonals. Each panel can select
its own assets and frames. An existing panel is read from disk, not resampled;
different metrics, decoding rules, or checkpoint weights do not change it.
Full hashes and source/GT checks belong to materialization or an offline audit.
Checkpoint evaluation reads the frozen tensors and checks only basic identity
and coverage. It does not rerun a surface sampler or hash its output.

The original Stage 2 run stopped after saving ST12 at 10,000 visits because
the legacy evaluator resampled the same mesh on CUDA and got different face
and anchor draws. Keep those historical metrics separate from the frozen
panel. The recovery configuration binds both arms to the retained full-run
128-asset archive, after verifying all source frames, normalization and GT.
Re-evaluate ST12 at initialization and at the saved 10k point, restore the
model, Adam, both ranks' RNG and sampler cursors, and continue from update 210.
SST8 starts from its audited initialization. The existing capacity evidence,
24 assets per card, 100k target and 10k interval are unchanged.
