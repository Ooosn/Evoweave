# EvoWeave Paper Draft V1

## Authorization and Deliverables

The user authorizes autonomous literature review, writing, baseline deployment,
and bounded H100 inference for a complete first paper draft. Routine writing and
experimental choices do not require another confirmation. Training and
checkpoint search remain closed. The fixed working model is 330k.

Deliver an English manuscript, editable sources, figures from real evidence,
references, a Chinese narrative explanation, and an author report separating
established results from missing experiments. The Introduction is the priority,
but the deliverable is a complete draft, not an Introduction-only outline.

## Scientific Position

Recover a parent-child skeleton and its pose-dependent joint locations from
corresponded dynamic surface observations without supplying a target skeleton
to the predictor. Video motivates the application, and dynamic geometry is an
intermediate representation. Do not claim a validated video-to-skeleton system
until the complete video path is actually tested without ground-truth skeleton
conditioning or ground-truth-dependent preprocessing.

Credit motion-aware rigging, static autoregressive rigging, deformation-based
skeleton discovery, and reference-conditioned motion capture separately. Do
not describe motion-aware rigging itself as new. Distinguish the skeleton we
predict from a complete skinning/rotation/FK-compatible production rig.

## Required Reading

- MoCapAnything V2 and TopoCap: input skeleton conditions and output motion.
- UniRig, SkinTokens, Puppeteer: geometry-conditioned skeleton/rig prediction.
- MoRig and SkeRig: motion-aware inference and per-instance skeleton discovery.
- RigMo and other closely relevant work if needed to avoid a false gap.
- Best-Paper-Writing-Skill, pinned at
  b9eff99a2130bea960ef1050eacf9f40cb792dd4.

The writing skill is a writing reference, not authority to invent evidence,
copy exemplar sentences, execute unreviewed scripts, or ask questions after
the user has explicitly requested autonomous work.

## Evidence Rules

- Reuse existing, traceable experiments before running anything new.
- Existing 112-asset development evaluation is not an independent test set.
- The 12 difficult cases informed checkpoint selection and are diagnostic,
  not an unbiased held-out quality benchmark.
- Keep ground-truth-reference controls separate from predicted-reference
  end-to-end results. Count generation failures, and do not drop hard cases.
- Root-RT caches use source skeleton metadata. Audit this assumption and
  never silently equate canonicalized mesh evaluation with raw-video recovery.
- Match observations, normalization, coordinate transforms, and scoring for
  baseline comparisons; disclose different pretrained data and extra inputs.
- Keep the unresolved old case-105 exact-token replay discrepancy visible in
  the author report. Do not turn it into an unsupported failure-cause claim.

## Resource and Preservation Rules

Check the actual scheduler grant, process ownership, GPU memory, source
revision, and CPU budget before compute. The user permits sharing an H100
with other training. Preserve other workloads, the existing 96-hour grant,
all old code directories, and all training results. Record any shared run as
shared; do not present its wall time as an isolated throughput benchmark.

Use stage-scoped SSD source and dependency deployments. The HGC /work tree is
scratch with a five-day retention policy, not durable source storage. Keep
verified packed backups and Git provenance. No further training or unrecorded
generation, no silent retries or fallback models.

## Work Status

- Literature and writing-skill review: in progress.
- Method/code and existing-evidence audit: in progress.
- Live HGC and baseline readiness inspection: in progress.
- Frozen inference plan: pending the above inspections.
- Manuscript and figures: pending claim/evidence mapping.
- Independent revision and artifact validation: pending.
