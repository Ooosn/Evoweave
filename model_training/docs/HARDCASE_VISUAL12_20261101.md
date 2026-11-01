# Final-30k Fixed-Case Inference

User2026-11-01 requests actual inference on the earlier difficult cases after
the hardcase adaptation, rather than treating the3percentage-point development
shortlist target as a reason not to inspect predictions. This independently
authorizes zero-update case diagnosis, not retraining or automatic promotion.

Freeze the final30k endpoint before seeing predictions. Use its model-only
export `outputs/train_v30/weights_030000.pt`, SHA256
`abaa87aa7b377f17093850981024a4d63529449f68f082e7c6502ea7a5571018`, for both
reference generation and target-pose prediction. Do not try other checkpoints
or seeds based on these results. Preserve the rejected development shortlist
and unchanged full300k reference; this is not the conditional855asset locked
evaluation or a population-quality acceptance test.

Use the same12preselected cases5/15/23/24/30/44/54/61/70/95/105/108 and their
retained full300k canonical inputs. The main old problems were missing lateral
branches30, missing multiple branches70, invalid reference105 and excess or
misplaced structure108. Retain all12including controls and every failure.
Reuse the original completed capacity_v28 predictions as the frozen baseline
only after exact input, GT, frame, normalization, policy and seed checks.
No fresh full300k replay is claimed. The previous105coordinate-token exact
replay discrepancy and its failed gate remain unresolved, not bypassed or fixed.

Keep official dynamic15beam sampling with top_k5, top_p.95, temperature1.5,
repetition_penalty3 and max2048tokens, FP32/noTF32/eager, base_seed20261029
derived per asset/raw reference frame, reference slot0, targets1..7. No GT
reference or prefix for primary generation, repair, fitting or static fallback.
The GT-head and copy controls are separately named diagnostics. Show selected
poses0/2/4/7, not a consecutive animation, with matched-camera before/after/GT
views. Inspect recovered and excess branches and pose following, not counts alone.

Use existing native job131170166/keeper2268188, GPU4 only and the established
8CPU subset11/12/14/15/61/63/64/65 within its16CPU grant. Keep64GiB host and85%
GPU-memory limits. Recheck both cgroups, scheduler grant, keeper idleness,
selected-GPU occupancy and remaining time. Do not modify other workloads,
request resources, release the allocation, retry or change GPUs automatically.
Stop before2026-11-01T11:40:04Z (20:40:04JST), with a1800second payload bound.

Use new native stages audit_v31 and capacity_v31. CPU audit checks the frozen
source/baseline evidence, new weight hash and all12canonical identities without
CUDA. After recording its actual passed result and resources, commit/sync the
exact config and state before admitting GPU inference. Recheck full2048token
15beam capacity with this checkpoint once, then generate the12references and
their target predictions. The shared official generation implementation and
model architecture are unchanged. Do not edit the live checkout during payloads.

Outputs: `outputs/capacity_v31`; durable copies:
`/home/wangyy/evorig/outputs/hardcase3608_20261101/visual12_final30k_v1`.
On exit, review native cleanup, zero optimizer updates, denominators, input
identity, independent scores and all comparison images; record and synchronize
the result in the same operation. Report actual improvements and regressions
without changing the earlier shortlist result or claiming generalization.

## Completed Case Review

capacity_v31 ran13:31:30..13:37:48JST, exit0, zero optimizer updates, passed
native resources and payload cleanup. The full2048token/15beam check passed
in59.46seconds with30,480,007,168bytes peak reserved memory. Job131170166and
its16CPU grant remain reserved; the keeper has no children and both GPUs are
idle. No additional allocation, alternative checkpoint, seed retry or training
was used. Runtime source is318a1fae3937c513c9ff2c662e412e40f6c7930a.

Independent rescoring passed for all336rows:12assets times7targets times4
separate methods. Canonical surface, GT, raw frames, normalization, official
policy and seeds match the frozen baseline. References valid11/12before and
12/12after; predicted-reference target failures7/84before and0/84after.
All cases remain in quality denominators. The baseline is the original retained
capacity_v28output, not a newly successful exact replay.

| Case | Predicted joints before -> after | GT joints | Visual interpretation |
|---|---:|---:|---|
|30|50 ->75|91|Major lateral spans partly recovered and follow target poses; terminal branch subdivisions remain missing.|
|70|52 ->52|122|Same simplified structure; multiple major outer branches remain missing.|
|105|invalid ->52|136|Now parseable with finite target predictions, but heavily simplified and misplaced; not correct reconstruction.|
|108|13 ->14|5|Excess branches persist; slightly closer positions do not mean better topology.|

Case30reference edgeF1 changes0.311212to0.440054; its mean predicted-reference
target J2J changes0.040792to0.021801. Case70is essentially unchanged,
0.032176to0.032243. Case108geometry improves0.096232to0.087552but reference
edgeF1 worsens0.461538to0.436364. Case105reference edgeF1 is only0.194320,
despite becoming parseable. The old105exact-replay discrepancy is not diagnosed
or resolved by this new model's successful parse.

For the identical11assets valid in both arms, mean predicted-reference J2J is
0.020758632before and0.018395804after;8improve and3regress (5/15/70). J2J is
the symmetric nearest-joint distance divided by the target mesh bbox diagonal,
averaged within asset then equally across these same assets; lower is better.
Do not compare the old11asset geometry mean directly with the new12asset mean.
All12asset predicted-target edgeF1 changes0.730250to0.756011with failures zero.
GT-reference stage2 MPJPE changes0.002163646to0.002061511on these12cases only;
it is not predicted-reference end-to-end error or the112asset development mean.

All27comparison images are retained, including prediction-only and GT-overlay
views for every case and three contact sheets. All12cases were inspected in
the contact sheets;30/70/105/108were expanded in both modes. Camera and scale
match within each asset, and labels/layout were checked after rendering.
The result supports partial case-level gains, especially30, not a blanket
repair or independent population-quality acceptance. No model was promoted.

Review: `D:/evoweave/server_ops/outputs/hardcase_visual12_20261101/comparison_review.json`
(SHA256 `fe79a845bcc7ab40f2c394a825b019a11f101bbfc0a046d7c7aa707b8d1fa51f`).
Visual findings: `visual_inspection.json`; figures: `comparison_images/` in
that same local output directory. Raw generated references, examples, rows,
native receipts and durable remote outputs remain intact. This operation's
compute admission is closed; no automatic follow-up, retry or promotion.
