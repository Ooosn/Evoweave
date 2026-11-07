# Native TokenRig Candidate Comparison

User authorization, November 7: compare the completed native Qwen candidate to
the previous models, inspect the existing metric matrix, and infer the original
hard cases. The possible extra 30k is a discussion, not authorization to train.

Predeclare native 250k (best development stage1) and 300k (planned endpoint and
best development stage2). Each pipeline uses one checkpoint for BOTH stages.
Retained OPT 300k and working OPT 330k are reused, never regenerated or replaced.
Do not reselect cases, tune sampling or mix checkpoints based on these results.

CPU audit reaggregates existing stage1 greedy and stage2 GT-reference rows.
Report each full-panel mean separately, and strictly paired submatrices only
where all eight raw frames, normalization, surface and original-GT fingerprints
agree. A 112-asset adaptation panel cannot be directly compared to a 128-asset
full-data panel. GT-reference scores are not end-to-end pipeline quality.

Formal descriptive inference uses cases 5/15/23/24/30/44/54/61/70/95/105/108,
the exact retained OPT canonical tensor bytes, original per-case seed derived
from base 20261029, 15 beams, top-k 5, top-p .95, temperature 1.5, repetition 3,
2048 maximum new tokens, FP32/no TF32/eager. Native Qwen uses its actual
articulation vocabulary and trained 512-token projection/prefix, not OPT's
articulationxl or an unprojected 1024-channel prompt. No GT prefix, fallback,
repair, fitting, retry, training, or forced-length stress. Failed generations
remain zero in quality denominators. Geometry comparisons name identical
common-success sets; distinguish predicted_head from GT_head and copy controls.

Use the current native job 131351736 only, GPU 3/4 and its 16-CPU affinity.
Record fresh kernel cpuset/devices membership and exact config before GPU work.
Keep the original 78 GiB whole-card bound, 64 GiB process-tree RAM bound and
2026-11-07T16:51:44Z save-stop. The capacity stage name is supervisor plumbing
for this zero-update inference, not a capacity stress. CPU audit must complete,
be reviewed, committed and privately synchronized before that stage is released.
Use a new fixed-commit SSD checkout; old training source/results stay untouched.
Any failure preserves evidence, cleans only owned payloads and stops without
automatic retry or resource release. No new allocation or other-task stop.

Render all 12 matched-camera comparisons, including reference and target poses.
Independently rescore saved joints and verify canonical identities. These are
fixed, already inspected diagnostics, not untouched population acceptance.
Keep the old case105 exact-token replay discrepancy explicitly unresolved.
