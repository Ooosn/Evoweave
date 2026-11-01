# Finite330kto400k Working Model Selection

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
