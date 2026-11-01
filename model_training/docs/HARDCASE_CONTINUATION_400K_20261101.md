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

## Verified Completion2026-11-02

train_v32 finished2026-11-02at00:52:17JST, source14695057f5d3a477c30c535cd3eaa743b94ba6bb.
Actual new70,000visits/1463updates, cumulative adaptation100,000visits/2091updates,
model400,000visits and1,283,280adaptation input-frame visits all match. Both rank
logs are finite and every actual weighted index/ID/seed/LR/frame count matches.
All seven full checkpoints/model exports and work/durable metadata and sizes
were verified. Saving code had verified both SHA256copies and model tensor
roundtrips; this completion review did not rehash unchanged multi-GB files.
All112asset/896stage1-view and1568GT-reference stage2-row identities match the
retained300k/330k baselines. Native payload cleanup passed; grant131170166and
keeper2268188 remain reserved, with no selected-GPU processes at final review.

Final greedy stage1 F1=.6228072403(core=.5910150735, extension=.7016990413,
background=.6283021400), success871/896 and25failures. This is development
evidence, not the pending official15beam case comparison or quality acceptance.
The400k endpoint was predeclared; do not switch to another milestone using
locked or qualitative results. full300k stays the production reference.

The server checkout has29uncommitted historical-file deletions, none among
the66hashed training sources. All66source hashes still match the training
receipt. Their origin is not established; do not restore or commit them without
user approval. The missing paths are retained in the primary review/current
state. This blocks clean-source admission for subsequent formal inference,
not acceptance of the already completed training execution. User permission
to restore exactly these29files was requested. Follow-up evoweave-40was deleted
while user action is needed; do not restart training or bypass the blocker.

Byte-identical primary review(work, durable and local), SHA256:
`d781acf2c0412f3034904e07c994d5fc959ca113909bc0003ae6da4eeb400faf`.
Local: `D:/evoweave/server_ops/outputs/hardcase400k_20261101/primary_completion_review.json`.
Remote: `outputs/train_v32/primary_completion_review.json`.
Fixed400k model export SHA256:
`b82aca74b6837437e39f4803025bf996b7cab2425bc681dd49b236f7ff7b98e8`.
Full100kadaptation checkpoint SHA256:
`d7776b6cd249b2fff204ec4d6e6e01a76350c1d43e72675d7db802661d3a449c`.
The105prior exact coordinate-token replay discrepancy remains unresolved.

## Repeated Source Loss Diagnosed

2026-11-02inspection confirms `/etc/tmpfiles.d/tmp.conf:26` on pcg01i contains
`v /work 1777 root root 5d`; systemd-tmpfiles-clean runs daily. The long-lived
Git checkout is inside that scratch tree. The prior18restored paths and current
29missing paths have zero overlap; restored prior18sample files still exist
with their2026-10-31restoration timestamps. This supports age cleanup of unused
historical paths rather than a training requirement to keep restoring files.
The daily mechanism and its scope are confirmed; per-file deletion journal
access is unavailable, so no exact unlink time/process trace is claimed.

The preceding request to restore29files is superseded. Preserve the existing
checkout, retain Git/verified packed source durably, and use clean fixed-commit
stage deployments on verified SSD. Do not alter administrator cleanup, touch
files to evade age policy, hide Git deletions or weaken source guards. No files,
models or data were restored/moved by this diagnosis. Formal fixed400k inference
remains pending a corrected deployment; no new training is needed.
Evidence: `D:/evoweave/server_ops/outputs/hardcase400k_20261101/source_retention_diagnosis.json`.
