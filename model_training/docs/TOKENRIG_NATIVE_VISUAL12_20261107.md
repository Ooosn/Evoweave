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

## CPU Audit Accepted

Native audit source 4001d32b0aff6e87b42d1dcd987d2cddf0d1eb98 exited 0, with
actual cpuset/devices membership under 131351736, no CUDA initialization, and
complete owned-payload cleanup. Both candidate weights and all 12 retained
input/GT/seed identities match. Audit result SHA256:
`6c9cbd2fdd50cf08a5e28918e819e6f5233baa1571cb26d07686da9d5dcd5117`;
resource receipt SHA256:
`62c3fd9ce77daa9276885d6b305b22c420259faa8cb13fe16899e0846ee2b185`.

Strict 114-asset matrix: OPT300k versus native300k has greedy edge F1
80.1999% versus 84.2217%, GT-reference normalized MPJPE 0.290019% versus
0.257518%, and GT-reference edge F1 96.6471% versus 97.0914%. Native250k has
84.5156%, 0.299510% and 96.7986%, respectively. Including OPT330k leaves only
19 strictly matched development assets, not a directly comparable full panel.
The fixed 12-case formal comparison is therefore still required.

## Completed Formal Comparison

The exact native source 6f88425752b63d419592422afdb8a87507dcb25e completed at
2026-11-07T05:21:04Z (14:21 JST), with zero optimizer updates and about 356
payload seconds. Config SHA256:
`8c419a833fe184163f17e7670afe77dbd93d9b38741a59f34b41beb7f34f755f`.
Native supervisor exit 0, actual job membership and owned-payload cleanup all
passed. GPU 3/4 peaks were 13007/12543 MiB. Both GPUs are idle afterward;
the original grant is retained and the completed training checkout is unchanged.

Both new candidates parse all 12 references. All 48 reference skeletons and
1344 target-method rows across four models were independently rescored from
saved joints. Exact input tensors, raw frames, normalization, original GT,
seed and sampling policy agree with retained baselines. All 12 matched-camera
case figures (reference 0, targets 2/4/7) and two overview figures were inspected.

| Model | All12 predicted-reference target F1 | Same11/77-view J2J |
|---|---:|---:|
| Retained OPT300k | 73.0250% | 2.075863% |
| Working OPT330k | 75.6011% | 1.839580% |
| Native250k | 74.5002% | 1.445202% |
| Native300k | 72.0464% | 1.592410% |

The common geometric set excludes case105 only because the retained OPT300k
reference did not parse; it remains zero in the all12 quality denominator.
For the three models that succeed on all12/84 target views, J2J is 1.952829%,
1.643408% and 1.810137% for OPT330k/native250k/native300k. J2J is normalized by
the target mesh bounding-box diagonal; no alignment or fitting is applied.
GT-reference MPJPE on these12 is respectively 0.206151%, 0.251036%, 0.198406%
and is a separate control, not the primary predicted-reference pipeline score.

Case30 improves clearly at native250k: mean target F1 43.645% to 65.457% and
J2J 2.1801% to 1.0715% versus OPT330k; major lateral branches return. At
native300k the reference collapses and overgenerates 175 joints versus 91GT.
Case70 still lacks major branches; 120 predicted joints at native300k versus
122GT does not imply correct topology (target F1 31.905% versus 55.444%).
Case105 improves edge matching but worsens positions. Case108 severely
overgenerates 102/131 joints versus 5GT, with zero edge F1; OPT330k has14 joints
and 43.636% target F1. Other cases are mixed: 44 improves, simple chains and
several articulated examples retain similar structure, and some positions regress.

Post-hoc explanation, NOT a changed primary denominator: case108 contributes
93.916% of native250k's all12 mean J2J reduction versus OPT330k. Across the other
11 cases, J2J is 1.334436% versus 1.313900%, only 1.539% relatively lower.
Native300k is 1.647561% on that same explanatory subset. Thus the lower main
J2J cannot by itself establish a better skeleton or justify model promotion.
Keep native250k as a candidate, working OPT330k unchanged, and do not start
extra30k or assume that extending the 300k endpoint fixes these regressions.
This comparison does not settle population quality. The previous case105 exact
coordinate-token replay discrepancy is still unresolved.

Review SHA256 `fc910e5578c0cac06fc2f0a385677b82f1a4becc6da7fd296ac3dace374b654d`.
Result SHA256 `929288030609ae9ccaad8d5ac36c4c36226438355d3c32f38898481cd12653ec`;
resources SHA256 `134c854b7a52d81bab912b44a928630823cd6c25e7acde459305072cb1d43f30`.
Durable root:
`/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_visual12_20261107`.
The 24,029,113-byte verified comparison archive has SHA256
`f87bec192b1e12e7273a6ec4c7138b22a8b891908ad380d86e764a9047e8d28e`.
The 5,537,827-byte visual archive has SHA256
`2e09cf38e4e28d48e220eeb8b810d0bc02b9a462b689080076b244af27a255f0`;
all16 member bytes were verified. Local report:
`server_ops/outputs/tokenrig_native_visual12_20261107/report_v2/comparison_report.html`.
