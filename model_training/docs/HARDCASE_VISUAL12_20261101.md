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
