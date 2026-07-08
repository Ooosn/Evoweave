# Puppeteer Decoder Backbone Feasibility

This note records the current model-side interpretation. It is not part of the
flat UniRig baseline.

## Local Reference

Official Puppeteer code was cloned as a read-only reference:

```text
model_training/third_party_references/Puppeteer
```

Pinned commit:

```text
1c0f9fc6ad209667a0ec5ceac9b59964938a8b51
```

Key files:

```text
model_training/third_party_references/Puppeteer/skeleton/skeleton_models/skeletongen.py
model_training/third_party_references/Puppeteer/skeleton/skeleton_models/skeleton_opt.py
```

## What Puppeteer Gives Us

Puppeteer has a skeleton-specific autoregressive decoder route:

- `SkeletonGPT` builds a `SkeletonOPT` causal decoder.
- With `joint_token`, each joint is tokenized as four tokens:
  `x, y, z, parent_index`.
- Coordinates are discretized in `[-0.5, 0.5]`.
- Parent token `0` means no parent/root; positive parent token `p` maps to
  parent index `p - 1`.
- The decoder supports `generate(inputs_embeds=processed_point_feature)`, which
  means condition tokens can be provided directly as embeddings.

This makes it structurally compatible with the Evoweave pattern:

```text
Evoweave motion/mesh condition tokens
-> projected condition embeddings
-> causal skeleton decoder
-> autoregressive skeleton tokens
```

## Correct Integration Interpretation

The useful experiment is not to force Puppeteer weights into the existing
UniRig object by matching UniRig layer shapes. The cleaner experiment is:

```text
Puppeteer skeleton decoder backbone
+ Evoweave dynamic condition prefix
+ Evoweave rootless skeleton targets
```

This does not require Puppeteer and UniRig to have matching layer counts or
hidden sizes. The Evoweave condition projection only needs to map into the
Puppeteer decoder hidden size.

This also means Puppeteer training must use a Puppeteer decoder profile. Do not
reuse `rigweave/scripts/run_dynamic_ar_train.sh`, UniRig tokenization, or UniRig
LR/scheduler defaults for this variant.

## Interface Details That Matter

`SkeletonOPTDecoder.forward` has three routes:

- `input_ids` only: inference step; it embeds skeleton tokens internally and
  adds bone-position and token-type embeddings.
- `inputs_embeds` only: first generation call; it treats the whole embedding
  sequence as condition tokens.
- `input_ids` plus `inputs_embeds`: training route; the code assumes
  `inputs_embeds` is already assembled by the caller.

For Evoweave training, that means a wrapper should assemble the full embedding
sequence explicitly:

```text
[projected Evoweave condition embeddings,
 Puppeteer skeleton token embeddings + bone-position embeddings + token-type embeddings]
```

Then it can call the Puppeteer decoder with `inputs_embeds`, `attention_mask`,
and teacher-forcing labels. This is a clean wrapper problem, not a fallback.

## Main Design Choices

1. Condition length

   Puppeteer uses `cond_length=257`. Evoweave currently has more condition
   tokens. The most compatible first version should pool/project Evoweave
   condition tokens to 257 tokens before feeding Puppeteer. Resizing
   Puppeteer's learned position space is possible but less clean for an initial
   checkpoint-transfer experiment.

2. Tokenizer

   To use Puppeteer's skeleton prior, use Puppeteer-style joint tokens:

   ```text
   x, y, z, parent_index
   ```

   Keeping the old flat UniRig tokenizer while using Puppeteer weights would
   throw away the main reason to use Puppeteer.

3. Output head

   If the token space stays Puppeteer-compatible, the released LM head may be
   partly useful. If Evoweave changes coordinate bins, max-joint range, or
   output semantics, the LM head should be reinitialized while keeping the
   decoder backbone.

4. Baseline separation

   This is a future model variant. It should not be mixed into the immediate
   flat UniRig baseline.

5. Optimizer/profile

   The local official reference contains skeleton inference/evaluation code and
   released-weight loading, but no complete skeleton training launcher with
   optimizer defaults. Until that profile is recovered or deliberately defined,
   no Puppeteer training job should be created. The skinning training defaults
   in the repository are unrelated to skeleton decoder training and should not
   be reused here.

## Audit Script

After the HuggingFace weights are available locally, run:

```bash
python rigweave/scripts/audit_puppeteer_decoder_checkpoint.py \
  --puppeteer-root /path/to/Puppeteer \
  --checkpoint /path/to/puppeteer_checkpoint.pt
```

The script checks for `transformer.model.decoder.layers.*` keys and reports the
presence of embeddings, bone-position embeddings, condition projections, and LM
head keys. It does not train and does not load Evoweave data.

## Current Feasibility Judgment

Feasible as a separate architecture line:

```text
Puppeteer initialized AR skeleton decoder
+ Evoweave motion-conditioned prefix
+ explicit parent-index joint tokenization
```

Not part of the immediate baseline:

```text
flat UniRig motion baseline
```
