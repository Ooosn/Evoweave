# Continue330k To400k

User2026-11-01 explicitly requests total400,000asset visits: the original300k
plus30k completed adaptation plus70kadditional visits. The earlier proposed
20kextension cap and quality-based early-stop suggestion are superseded. Keep
3/2/1weights, effective batch48, backbone LR2e-6constant and XYZ head3times
(6e-6). Do not rewarm to1e-5, reset Adam or select another source checkpoint.

Restore the complete final330k checkpoint:
`/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v30/checkpoint_030000.pt`,
SHA256 `a62dd25991f948298889a68525ee2b0babb9cceb10cfd435e6cd0668422134f7`.
It contains628adaptation updates,392,704input-frame visits and both rank RNG
and weighted-stream cursors. Preserve the original300k optimizer ancestry and
all1079Adam state entries; never initialize from a model-only export.

The fixed adaptation plan extends30,000to100,000visits. All consumed30kasset
indices, unique sampling seeds,628frame-count/update-boundary entries and rank
cursors must match exactly. Future positions continue at30000, not zero. The
entire adaptation then has2091updates, with1463new updates for this70k suffix.
New visits are16675core,10400extension and42925background; all18002train assets
remain covered, validation membership and root-RT/normalization are unchanged.
Extended sampling SHA256:
`43d6c1d6c1c40113a36b8916ccb56fad67f232d4e1c9f21c7b49b2954488c144`.

Keep the original5k/10k/20k/30k update-boundary prefix; save and evaluate at
adaptation40k/50k/60k/70k/80k/90k/100k, i.e.total340k..400k. Retain all seven
full recovery points and model-only exports on the existing verified NVMe
work volume and durable backup. Never prune the parent run's four checkpoints.
Reuse the fixed112asset canonical development inputs and retained330k baseline;
do not repeat its baseline generation. Greedy remains fast validation only.
No3percentage-point shortlist criterion stops this newly authorized70k run.
No automatic model promotion is authorized; after completion compare the
requested400k endpoint with retained baselines, including the original12case
official15beam diagnostics, before making broader quality claims.

The previous105exact coordinate-token replay discrepancy remains unresolved.
Its now-parseable30k output is not a diagnosis of that prior replay difference.

Resources: existing native HGC131170166, keeper2268188, physical GPUs3/4,
16CPUslots and64GiB host cap. Actual qalter96h extension passed on2026-11-01;
expiry2026-11-04T12:40:04.726Z, save-stop2026-11-04T11:40:04Z (20:40:04JST).
Future HGC requests default96hours. No new allocation, GPU-count change,
release of this grant or interference with another task is needed or permitted.

Use new audit_v32 and train_v32 stages. The CPU audit verifies the new complete
checkpoint restoration, both weighted-prefix contracts and future plan, while
reusing the accepted immutable cache/loader evidence. The original native
two-rank capacity and disk model/Adam/RNG restore proof are explicitly inherited
because model math, physical batches, data shapes and hardware are unchanged.
Only the declared configuration/restore/plan plumbing may differ from that
proof; all original model-computation source hashes must match. Do not repeat
the old loader audit or GPU stress training. After reviewing the actual CPU
receipt, record inherited capacity and train admission, commit/push/synchronize,
then freshly plan and launch once. Verify both real GPU restoration receipts
and the first two actual weighted updates before reporting successful startup.

Configuration: `model_training/experiments/hardcase400k_20261101.json`.
Operations: `hardcase_adaptation_ops.py --continuation`, including plan/execute
and audit record. Output `outputs/train_v32`; durable root
`/home/wangyy/evorig/outputs/hardcase3608_20261101/continue400k_v1`.
Do not edit or synchronize the live checkout while a stage runs. Nonfinite
loss, failed restoration, resource violation or early exit preserves evidence
and stops this attempt; no automatic retry or GPU replacement. The native
deadline saves a complete recovery point rather than silently losing progress.
Monitor actual progress quietly; report meaningful completion, failure or
required user action, not ordinary checkpoints. Completion means100k cumulative
adaptation visits/2091updates, exactly70k/1463new, plus the planned verification;
training completion alone is not production-quality acceptance.
