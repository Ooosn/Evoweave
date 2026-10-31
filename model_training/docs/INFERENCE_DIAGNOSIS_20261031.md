# Paired Inference Diagnosis

The user authorized finding the cause of the new visual failures and testing
better inference on 2026-10-31, with targeted fine-tuning as a possible later
option. This operation is inference-only. Preserve all frozen checkpoints,
past predictions, native GPU allocation and the other task on GPU3.

## Questions And Fixed Comparison

1. Do the two frozen pipelines reproduce all 24 saved official reference token
   sequences exactly in the current runtime and the same canonical inputs?
2. Does requiring complete branch endpoints improve validity without hiding
   structural errors?
3. Does deterministic beam search with that constraint improve the fixed-case
   structure and target-pose results compared with the sampled policy?

Use the same 12 preselected cases 5/15/23/24/30/44/54/61/70/95/105/108, reference
slot0, and seven target slots. Use both old joint_080000 + pretrained_stage2_080000
and full300k + full300k pipelines. Their architectures are identical. Both use
the same full-model canonical input cache; every old saved reference input
identity must match before admission. Baseline generation calls the unchanged
stage1_official_inference.generate_condition directly. Require all 24 token
streams and statuses to match before trying any new policy. A mismatch stops
the operation and retains partial results for investigation.

## Policies

- official: unchanged established15beam sampling, temperature1.5, top_k5,
  top_p0.95, repetition penalty3,2048new-token limit, asset-specific saved seed.
- strict_sampling: identical sampling options, plus a syntax-only restriction:
  no empty skeleton and no special token until all six branch-endpoint
  coordinate tokens are complete. No token removal or post-hoc repair.
- strict_beam: the same syntax restriction,15beams and repetition penalty3,
  with sampling disabled. Sampling temperature/top-k/top-p are not applied.

The latter two are experimental policies, not the unchanged official policy.
No retries, choosing a seed by its score, GT-based candidate selection, static
fallback, geometric fitting, checkpoint update or training is allowed. Failures
remain in the12case denominator. A valid parse alone is not a quality success.

Each model's existing verified15beam/2048token capacity receipt is inherited;
the upper memory bound is unchanged. The CPU audit reruns the new policy tests
in the actual server Python environment, verifies immutable artifacts, and
records package versions. Admission requires clean pinned source and native
membership. The operation uses GPU4 and CPUs11/12/14/15/61/63/64/65 under the
existing2H100/16slot grant, never GPU3. Deadline remains2026-10-31T10:55:00Z,
payload limit1800seconds, memory cap64GiB and GPU fraction0.85.

## Evidence And Interpretation

Record every generated reference and its condition hash, every target-pose row,
all fixed cases, environment versions and exact baseline replay results. Export
each model/policy's examples, independent summaries and prediction/GT views.
Compare structure F1 with failures scored zero, parse success, joint-count
agreement, and geometry on explicitly matched successful cases. Report both
improvements and regressions; do not claim validation-wide superiority from12.

Existing greedy evaluation is analyzed separately:114assets/912poses have
identical canonical identities. Old/new edge F1 means0.776068593/0.799781449,
paired asset-bootstrap95% delta[0.005893556,0.041707523];68assets improve,
38regress,8tie. Fourteen input-mismatched assets are excluded from paired
claims. Joint-count strata are only complexity proxies, not human/nonhuman
labels. This evidence does not establish official-policy end-to-end quality.

No fine-tuning is admitted in this operation. A later proposal must select
hard examples from training assets only, reserve independent validation,
mix in ordinary-skeleton replay, define a small update budget and preservation
criteria, and keep the base checkpoint unchanged. Model/data strategy changes
must be motivated by the completed diagnosis rather than selected good images.

Current admission and completion evidence are recorded in state/current.json.
Audit_v29 completed at16:41:43JST with native exit0 and clean resources. All24
saved reference identities, frozen checkpoints and inherited full-beam capacity
matched; the six diagnostic-policy CPU tests passed in the actual job runtime.
CUDA was not initialized. The job loads project-specific Transformers4.46.1
and Tokenizers0.20.3, not the bare interpreter's global5.4.0/0.22.2 packages.
This is a current execution snapshot, not a recovered historical lockfile.
Capacity_v29 is now admitted with the primary CPU result bound; no source
changes are permitted while the payload is active. Training remains prohibited.
Local primary comparison and reviews live under
D:/evoweave/server_ops/outputs/inference_diagnosis_20261031/.
