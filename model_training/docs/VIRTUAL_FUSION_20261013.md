# Isolated virtual-frame fusion experiments

This is the active experimental contract on branch
`codex/virtual-fusion-20261013`, based on committed source `19d81ba`.
The older checkout, unpublished working changes and training outputs are
read-only references, not execution destinations. GPU authority remains in
`model_training/state/current.json`; this document is not launch permission.

## Current direction: normal 5000-asset trend comparison

2026-10-16 update: A completed all20000 exposures,420 updates and263608
supervised-frame exposures with exit0. Both complete traces are finite,
initialization/exposure identities match, and all five evaluations are saved.
Its original-GT F1 at20000 is0.324185 train-pool /0.304345 heldout poses /
0.282056 unseen validation. These are the same32/32/128-asset panels, eight
frames each, with generation failures scored zero. Execution integrity is
accepted, not architecture quality. Legacy has no15000/20000 comparator;
at the matched10000 node A validation F1 was0.192411 versus legacy0.690604.
Preserve A's final checkpoint and complete terminal receipt. B is next, using
real queries / virtual memory / real residual with otherwise identical runtime.
Only its independent two-rank9-case capacity profile is released now; review
that profile and remaining grant time before publishing fresh B training.
Do not apply the discussed residual removal, fusion-depth changes or new
frame-budget pairing to this comparison. Report original-GT F1 as the primary
metric; label experiment size, exposure milestone and panel on every table.

2026-10-15 update: the user stopped the legacy run after10864 exposures
(228 updates); its complete10000 checkpoint and evaluation are preserved.
The latest instruction explicitly defers new frame-budget/batch-pairing
features to a separate later study. This comparison keeps random2..24 inputs
and effective48 assets, not a fixed1152-frame update budget. A starts fresh
from the same official tensors, with the same exposure plan as legacy.
Its completed two-rank9-case capacity confirmation passed with exit0 and
peak allocated memory60.57GiB. Per-rank microbatches are24/12/6 for
T2-6/T7-12/T13-24, accumulated1/2/4 times to keep global effective48.
The reviewed receipt is `reviewed_capacity_a_v1.json` in the normal job
directory. Later B/C/D still need route-specific capacity approval.

The user's2026-10-14 correction supersedes the fixed-small-cohort promotion
sequence below. Fixed8/fixed64 runs are diagnostic history, not a substitute
for normal online training and not an additional prerequisite for this stage.
The legacy64/T4 reference completed all512 updates and its final evaluation
at2026-10-14T12:46:58Z with exit0. Preserve its outputs and allocation129839822.
The user explicitly approved starting the normal sequence at2026-10-14T13:16Z:
legacy -> A -> B -> C -> D, using both physical GPUs5/7 for ONE route at a time.

Use the ALREADY PREPARED immutable5000-row subset, SHA256
`a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`.
Its original rows and root_rt metadata are preserved, with4397 base/603 addition
assets and no validation overlap. Data selection/screening is complete: do not
repeat NPZ censuses, rebuild RT, or introduce another data adapter. The focused
RT consumer review and12 existing CPU tests passed on2026-10-14.

Every training visit selects a fresh input count T uniformly from2..24 and
resamples poses from ALL non-heldout frames. T is shared within each global
optimizer update so both DDP ranks and every route use the same shape schedule.
The existing mesh-only motion-FPS/random pose selector, root-RT removal,
selected-input mesh-union normalization, trackable surface sampling and
all-input-frame main supervision remain unchanged. There is no fixed-pose or
learned-feature cache. Exposure-index seeds make asset order and pose/surface
draws independent of the physical microbatch policy.

Compare legacy query-first, A, B, C and D from the same official initial shared
tensors. Preserve effective batch48, BF16, OneCycle peak1e-4/warmup.1/div5/
final10, weight_decay0, clip1. A/B/C/D retain auxiliary weight.1; legacy has no
auxiliary task. Every route receives EXACTLY20000 asset exposures: four
shuffled visits to each of5000 assets. Evaluate at0/5000/10000/15000/20000.
Each5000 segment is one visit per asset:104 updates of48 plus one update of8,
for420 updates total. The old80016 full-dataset budget does
not apply to this trend comparison. No fit-to-perfect stopping criterion.

Use a deterministic random panel of32 training assets and128 independent
validation assets. Training-pool evaluation uses eight non-heldout poses per
asset; these are not necessarily the exact input sets seen during optimization.
Heldout evaluation uses ALL eight reserved poses for the SAME32 assets. Each
validation asset contributes eight poses. Thus every checkpoint has1536 main
generations (256/256/1024); auxiliary evaluation is once per asset/group only
for A/B/C/D. Failure-as-zero F1 includes every generation; conditional J2J reports
its denominator. This panel monitors trends; it is not the full967 validation
set or a statistical generalization guarantee.

The completed A single-GPU profile measured T24/B6 at55.48GiB and T24/B8
at70.03GiB (above the85-percent limit). It is not a legacy/DDP capacity receipt.
Before training, require two-rank coverage against long-main-token and geometry
extremes, plus longest auxiliary targets for A/B/C/D; include resident gradients
and optimizer state. Use a few physical-batch buckets: profile the maximum T
in each bucket at its actual B. Shorter T is covered only at the SAME B and for
identical stress-asset membership from the existing pinned selection. This
avoids retesting every shorter tensor shape without changing any training input.
Keep at least15 percent measured allocated-memory headroom.
Report physical batch, accumulation, memory and time, not just effective48.
Run each route on physical5/7 with DDP. Profiles never
supply trained weights. Preserve allocation129839822 and check full runtime
plus safety margin before every launch.

New source: `/work/wangyy/virtual_fusion_20261013/normal_code`.
New jobs: `/home/wangyy/jobs/virtual_fusion_normal_20261014_v1`.
New outputs: `/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014`.
The legacy64 source remains frozen at2658d6c in its existing `code` directory.
`train_virtual_fusion_normal.py` reuses the established training/loss loop;
its bounded normal5000 authority is separate from full-data formal release.
Do not launch another full-data80016 run or use historical small-fit gates.
Only fusion-specific C/D parameter additions are excluded from the common
initialization fingerprint; record the actual full fingerprint separately.

## Preserved model and data

Keep the retained-root RT manifests (18002 train / 967 valid), selected-input
mesh-union normalization, 1024 virtual/real anchor tokens, 96 internal
registers and 12 spatial/temporal layers. Keep the complete official UniRig
XYZ autoregressive decoder, original vocabulary and all-frame supervision.
The auxiliary DFS-ID decoder consumes final virtual tokens and contributes
only a loss; its generated structure, length or hidden state never enters
the main prediction path. Fresh trials load official checkpoint SHA
`d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4`.

## Isolation and authorization

- Local code: `C:/Users/namew/.codex/worktrees/virtual-fusion-20261013/evoweave`.
- Local operational artifacts: `D:/evoweave/server_ops/outputs/virtual_fusion_20261013`.
- HGC source/cache/temp: `/work/wangyy/virtual_fusion_20261013/` on verified SSD.
- HGC jobs: `/home/wangyy/jobs/virtual_fusion_20261013_v1`.
- HGC outputs: `/home/wangyy/evorig/outputs/virtual_fusion_20261013/`.
- Initial qlogin129547842 (physicalH100s5/7) completed both fitting runs,
  then expired at the scheduler96h hard limit on2026-10-13 18:37JST.
  Replacement job129839822 was granted on pcg02i at2026-10-14 06:52:02.966JST,
  exactly twoH100s/mem64G, physical5/7 verified by UUID on this NEW node.
  The new /work is verified NVMe RAID0/XFS. Preserve its keepalive; the96h
  limit expires at2026-10-18 06:52:02.966JST. Before every long launch,
  refresh live remaining walltime and require full duration plus safety margin.
  The expired pcg01i grant is never an execution destination.
- User authorized GitHub publication of code/config/result summaries only.
  No datasets, weights or credentials. Every output is exclusive-create.

## Comparison sequence

First audit the existing auxiliary-weight .1 step2048 checkpoint read-only:
greedy auxiliary generation from BOS, natural EOS, exact count and DFS-parent
tree, plus deterministic cross-asset virtual-condition swap. Verify the
checkpoint SHA and unchanged parameter state; perform zero optimizer steps.

Round one is A (`virtual_queries_real`) versus B (`real_queries_virtual`).
A preserves original virtual Q, real K/V and virtual residual; B uses real
Q, virtual K/V and real residual. Other modules and initial tensor bytes match.
Only if neither resolves the problem, round two compares C
(`relative_position`) and D (`token_film`). C adds zero-initialized per-head
anchor-coordinate delta bias, MLP 3->64->8, to A's attention logits. D uses a
same-index virtual MLP 1024->256->2048 to modulate normalized real tokens,
preserving the real residual. Neither consumes GT positions in fusion.

## Training and acceptance

All candidates: seed20261013, auxiliary weight .1, fresh official weights,
same fixed diverse8/T4 inputs, effective batch8 via asset gradient accumulation,
1024 updates = 8192 actual asset / 32768 supervised frame exposures. OneCycle
peak1e-4, warmup.1, div5/final10, weight_decay0, clip1, established BF16 path.
Pair-ready fingerprints gate the first optimizer update. Evaluate updates
0/32/64/128/256/512/1024. Preserve the first passing checkpoint but complete
the same training budget on both cards; do not selectively shorten baseline.

Main acceptance: all32 valid natural EOS, teacher accuracy >=.99, >=29 views
with exact count, normalized J2J <=4/256 and (raw edge F1 >=.95 OR exact
canonical GT tokens). Auxiliary acceptance: all8 valid natural EOS with exact
DFS parent trees. Both must pass at the same checkpoint; CE alone is insufficient.

Passing candidates enter fixed64 train / heldout poses / fixed32 valid,
effective batch48, 512 updates =24576 actual asset exposures, same schedule
and data across comparisons. Evaluate all four input poses per asset at
0/128/256/512, including the eight training/heldout-pose assets and32validation
assets (192main generations per checkpoint). Initial generation is separate
for each fusion mode even when initial parameter tensors are identical.
Success rate and failure-as-zero F1 against continuous originalGT must not
decrease versus A; paired common-success mean J2J may worsen by at most1/256.
J2J uses the quantizedGT, with exactly the same common-success pose set on
both sides; missing common-success evidence does not automatically pass.
This is one-seed bounded validation, not a statistical/generalization claim.

Prefer A when sufficient, otherwise B,C,D in order; stop expanding once a
complete candidate passes. If all fail, diagnose auxiliary structure versus
frame-information loss before any new hypothesis. Full training stays blocked
until primary review, hash-bound real capacity checks across T2..24 and the
existing formal release gates. Formal training uses fresh official weights,
not these fitting checkpoints, and retains the prior80016-exposure budget.

## Completed first round

A and B both completed1024updates/8192actualassetexposures. First complete
main+aux passing checks were A256 andB128; final both reproduce32/32main GT
token sequences and8/8auxiliary parent trees. A already passed at2048actual
exposures, the total exposure count of the older failedB1 constantLR run.
This establishes that the original fusion can learn this fixed task under
the new effectivebatch/OneCycle recipe; it does not isolate the contributions
of batch averaging and schedule, or establish heldout generalization.
Do not run C/D. A remains the preferred architecture pending medium review;
B is the already-tested matched comparator, not a commitment to replacement.

## Completed medium comparison and new reference request

Both A and B completed512 updates, effectivebatch48, or24576 asset exposures
on the same64 training assets and four fixed poses. Each asset therefore had
384 visits. The monitored training-generation subset is8 of64 assets, not
the complete training cohort. Validation uses32 different assets/four poses.
The complete matched receipt is
`../experiments/fusion_ab_medium_v1_review.json`.

At512, both reproduce all32 monitored training-pose GT token streams. Exact
auxiliary training trees are A7/8 andB8/8. Heldout-pose main F1 is
A0.475601/B0.841470; unseen-asset validation F1 is A0.090075/B0.361060,
with failed generation counted zero against continuous originalGT. Valid
generation counts are126/128 and127/128. Exact auxiliary validation trees
remain A1/32 andB0/32. B passes the predeclared relative main-task gates,
but this is not evidence that unseen-asset structure generalization is solved.

On2026-10-14 the user requested an actual old-training comparison before
formal release, to investigate the shared data/training pipeline and small
cohort coverage or overfitting. Next run the proven legacy query-first route
on the SAME raw cache, official initial shared weights,64/32 split,T4,
effective48,512updates and0/128/256/512 full main generation checkpoints.
Each target pose is processed separately by the original query-first encoder.
This matches target exposure across architectures; it is not a historical
single-target sampling-run reproduction. The old route has no virtual/auxiliary
structure branch, so its auxiliary metrics are not applicable rather than zero.
Inspect complete learning curves, not only the final loss. A poor old-reference
result alone does not uniquely identify data, optimization or cohort size.

Formal release and further variants wait for this requested reference.
Only A/B were trained in this fusion round, each at8 and64 assets; C/D have
implementation/tests but no training runs. Earlier8-asset legacy/final-real
controls do not substitute for this matched64-asset reference.
