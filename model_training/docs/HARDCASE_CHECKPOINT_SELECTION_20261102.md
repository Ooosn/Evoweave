# Finite330kto400k Working Model Selection

Status2026-11-02: COMPLETE. Final working model330k; no more training or search.
Use`state/working_model.json`for new user-authorized inference. The protocol
below records this completed operation, not permission to repeat its stages.

The user explicitly supersedes the earlier fixed400k-only comparison: inspect
existing matched validation, review two intermediate checkpoints on the same12
official cases, choose a clearly better working model or settle on330k, and
stop. No training, extra checkpoints, new seeds, repeated baselines or stress.

All eight checkpoints have112assets/896greedy stage1views on identical canonical
inputs and GT. Rank the six intermediate points by failure-zero overall stage1
F1, then coreF1, failures, earlier checkpoint. The preselected pair is380kand390k;
330kand400kalready have formal results and are reused, not recomputed.
The frozen manifest is`selections/hardcase_330k400k_20261102.json`, SHA256
34653d80c4e0bd138b72bebf2f01df8d2018afb8bd26198589a9d23a821de151.
Greedy validation and GT-reference head scores are not end-to-end15beam results.

Use the fixed original cases5/15/23/24/30/44/54/61/70/95/105/108and original
canonical inputs, GT, normalization and seed20261029. Dynamic15beam, topk5,
topp.95, temperature1.5, repetition3,2048tokens, FP32/noTF32/eager unchanged.
Reference0, targets1..7; each candidate12references and336method-target rows.
No GT prefix/repair/fitting/fallback/retry or altered sampling/weights.

Only audit_v34/capacity_v34for380kand audit_v35/capacity_v35for390kare admitted.
New audits check the actual new weights and retained input/seed identity; prior
2048token capacity proof fromnativev31is inherited, never rerun. Each completed
stage must be recorded before the next; config/state commits precede GPU work.
Keep native131170166/keeper2268188, GPU4/8CPUsubset of existing2H100/16CPU96hgrant,
host64GiB, GPU85percent,1800second payload limit, stop2026-11-04T11:40:04Z.
Fresh fixed-commit SSD deployment preserves all old directories/results and
verified durable archives. Never restore individual old files or sync during payloads.

Working-choice rule is frozen before either new12case result: no new reference
or target failures, no single-case J2Jloss greater than0.01of mesh diagonal,
and background validF1loss no greater than0.005. Require either target edgeF1
gain at least0.01with J2Jno worse than1percent relative, or J2Jimprovement at
least5percent relative with targetF1loss no greater than0.002. Smaller mixed
changes mean330k. Among qualifying candidates use higher targetF1, lowerJ2J,
then earlier checkpoint. Inspect all cases, especially30/70/105/108; never infer
correctness from joint count alone. No further search after this pair.

Record one exact working-model checkpoint for both pipeline stages. This is
user-authorized practical selection, not independent locked855/32population
acceptance. Retainfull300kas historical baseline and all training/checkpoint
results. Preserve the old105exact-coordinate-token replay cause as unresolved.

## Completed Result

All four native stages completed with exit0, kernel provenance and cleanup.
380kinference source8c7320e06310a7b82f0663bee98fa89b3af0b5c8;
390ksource864e2745767bfa59f8c31644895dcb7a65997b0c.
Result SHA256respectively48ead4b9b7c62b873127d861ed74e682d6ff0ed5b706f3def518847cc52be526
and94c08ee436af48f0976a53e5b48d3df2d8cbb5cdf614cddb807de0ced3470064.
No baseline regeneration, capacity stress, parameter updates or new GPU grant.

All672new target scores and24reference scores were independently recomputed;
all27matched-camera comparison images were inspected. All four models parse
12/12references, which does not mean their skeletons are correct. Same12assets
and84target views, predicted-reference head only:

| Model | Failure-zero target edgeF1 | J2J / target mesh diagonal |
|---|---:|---:|
|330k|0.7560105580|1.952829%|
|380k|0.7292356574|1.978787%|
|390k|0.7540208321|2.134423%|
|400k, reused|0.7351737799|2.248990%|

Both intermediate candidates are worse in both aggregate metrics than330k,
so neither qualifies even without relying on borderline benefit thresholds.
380khas4geometry gains/8regressions,390k2/10.380kpartly improves30but strongly
regresses105;390kregresses30and105.70keeps52versus122GTjoints;108is not fixed.
GT-reference head results remain separate controls and do not overturn this.

Final durable checkpoint for BOTH stages:
`/home/wangyy/evorig/outputs/hardcase3608_20261101/v1/weights_030000.pt`,
SHA256abaa87aa7b377f17093850981024a4d63529449f68f082e7c6502ea7a5571018.
Actual working copy and durable copy were freshly rehashed and match. This is
the300kbase plus30kadaptation, not400kweights or capacity-test weights.
No further checkpoint search or training is admitted. Keep all newer results
and the native96hgrant; no allocation release or other-task stop occurred.

Local report`D:/evoweave/server_ops/outputs/hardcase_checkpoint_selection_20261102/comparison_report.html`.
Work evidence`outputs/checkpoint_selection_20261102`; durable evidence
`/home/wangyy/evorig/outputs/hardcase3608_20261101/selection_review_v1`.
Verified86member comparison archive SHA256
c8b6910c9f5f667959544b0002dbff48e3314b2e826a173fa0c9e6fb2a4968cb.
Working choice is explicit user authorization, not untouched population-test
acceptance. Prior105exact replay cause remains unresolved.
