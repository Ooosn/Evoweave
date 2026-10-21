# Selected8192 Structure-Condition Audit

This is a read-only follow-up to `STRUCTURE_XYZ_FILL_20261022.md`, not a new
architecture, optimization recipe, data selection, or training authorization.

## Reviewed Evidence

The selected random-initialization run completed8192 asset exposures,108048
supervised frames and175 optimizer updates. Actual request traces cover all5000
training assets;1808 were visited once and3192 twice. The fixed panel contains
32 asset/frame contexts but only24 distinct assets, because training-eligible
and heldout-pose contexts share8 training assets.

Validation-panel original spatial edge F1 improved from0.10714 at4096 to0.15203
at6144 and0.17719 at8192. GT-structure XYZ L1 fell to0.17346,26.10% below the
constant-center-bin baseline. These are small-panel early-learning observations,
not a full-validation estimate or model-quality acceptance.

Free structures remain two humanoid-like templates:28 outputs with52 joints and
4 with28 joints. The21-joint fixed nonhumanoid example still gets an inappropriate
humanoid structure, and its GT-conditioned coordinates are poorly separated.
This warrants checking condition use before treating the second template as a
formal-training readiness gate. No more training is automatically authorized.

## Exact Experiment

- Load only the completed own-new-model8192 checkpoint, SHA256
  `8b91087fb52cd282daa0e66e2589d78da4d41ab11658ab650eaa1a04a406d77d`.
- Verify the exact producer configuration, family, architecture, training
  contract, source commit, model-tensor fingerprint and original panel files.
- Reuse all32 fixed contexts with the same eight input frames and scored raw
  frame identities. Recheck original continuous GT and original reference
  tokens. Correct-condition free generation must replay all32 saved results.
- Encode each context once. At an identical teacher-forced GT prefix, compare
  the correct virtual-query memory with a different asset's memory and zeros.
- Repeat those comparisons at the same generated self-prefix. This does not
  replace GT or turn self-prefix agreement into GT accuracy.
- Select the first cyclic partner with a different exact GT parent array in
  the same group and rank. Pairing is deterministic, not necessarily bijective;
  it is a counterfactual diagnostic, not a new inference pipeline.
- Treat zero memory as an out-of-distribution control, not the principal proof
  of meaningful condition use. Real-asset swaps are the principal comparison.

## Measurements

Separate new joint IDs, BRANCH, parent references and EOS. Report raw vocabulary
CE/accuracy and grammar-renormalized CE/accuracy. Also report choice positions
only, excluding prefixes whose next token is forced by the grammar. Report
counter-minus-correct paired CE, probability-distribution JS divergence and
top-choice changes. All logarithms are natural. Aggregate equally per case,
with token counts and52/non52 breakdowns retained.

Save paired logits and per-position evidence, including self-prefix contrasts
before and after the first GT mismatch. Memory cosine and relative L2 are
descriptive only; neither establishes conditional information use on its own.

## Safety and Acceptance

No optimizer is constructed, no backward pass occurs, and no checkpoint is
written. The complete model-state fingerprint must be unchanged before/after
the audit and equal to the completed training fingerprint on both ranks.
Run only after training has exited, on the retained allocation130309880,
pcg01i physical GPUs5/6, under the existing exclusive work lock. Each rank uses
one eight-frame context at a time; peak allocated memory must remain below85%.
Preserve the allocation and all8192 artifacts. Stop on changed provenance,
GT/frame mismatch, failed baseline replay, nonfinite values or resource drift.

Completing this audit means the diagnostic is execution-valid, not that the
model is good or that20k is approved. The primary interprets paired evidence
together with the bounded-fit and full5k learning trajectory, then publishes
the next exact operation. Old A/B experiment weights remain prohibited.
