# Isolated virtual-frame fusion experiments

This is the active experimental contract on branch
`codex/virtual-fusion-20261013`, based on committed source `19d81ba`.
The older checkout, unpublished working changes and training outputs are
read-only references, not execution destinations. GPU authority remains in
`model_training/state/current.json`; this document is not launch permission.

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
- Preserve qlogin 129547842, physical H100 GPUs 5 and 7. No additional GPUs.
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
