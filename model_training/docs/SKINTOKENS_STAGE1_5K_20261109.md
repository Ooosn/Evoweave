# SkinTokens Stage 1 Controls

After the native single-pose baseline finishes, review and record its fixed
128-asset learning curve. Then run, in order, the frame-local, temporal
multi-target, and temporal single-query arms on the same retained allocation.
No failed or incomplete stage releases another arm, and no working checkpoint
is automatically replaced.

All three start from the released SkinTokens mesh encoder, projection and
skeleton Qwen, with the same identity-initialized dynamic stack. No project
checkpoint, optimizer, Stage 2 head, or Stage 2 loss is loaded. Keep the same
5,000 assets, six passes, 30,000 asset visits and 627 AdamW updates with the
same asset order and 30k OneCycle as the static baseline. The dynamic arms use
the same 2-24-frame clips and batch-48 exposure stream. Use the conservative
base recipe's physical batches per rank: 24 for 2-6 frames, 12 for 7-12, and 6
for 13-24, with accumulation to the same update size. Use no padding of the
last global-16 update at each exact 10k boundary.

* Frame-local and temporal multi-target use the same deterministic 20-40%
  target selection (minimum two). Frame-local keeps the same parameters,
  restricts temporal blocks to groups within one frame, and supplies only
  unknown rather than observed motion to their biases.
* Temporal single-query retains the same inputs and weights, with an explicit
  query role at the sampled first real frame. Its anchors track into the
  context frames, but only the query readout enters the condition tail and
  decoder; it has exactly one supervised tree per visit. The other frames
  remain available to the shared interaction stack. This changes the readout
  and supervision density, not the input visit plan.

At 0/10k/20k/30k decode only slot 0 on the same archived 128 clips with the
same FP32 greedy grammar. Only the dynamic arms receive the other seven
frames. Paired metrics check raw frame, reference geometry, normalization and
original GT fingerprints. A generation that does not terminate contributes
zero to Edge F1; geometry is reported on scored views and compared on the
common successful subset when useful. Do not mix these results with all-frame
or 15-beam scores.

Commit and privately publish every exact admission, and record each complete
arm before admitting the next. Use fresh stage-specific source on verified SSD
and retain both copies of each full checkpoint. CPU audits prove common
initial tensors and counts; native no-update 6-, 12- and 24-frame stress covers
the largest skeleton and geometry profiles before training.
