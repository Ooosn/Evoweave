# Evidence-Aware Motion-Conditioned Skeleton Generation

This is a future-direction note, not the immediate baseline plan.

## Problem

The current task is not a standard "condition fully specifies target" problem.
The condition and target have different semantics:

```text
motion condition: partial, action-dependent, noisy, insufficient
GT skeleton: complete, canonical, action-invariant, sufficient
```

Directly training:

```text
mesh + motion -> full skeleton
```

can work because the GT is always complete. But it is not conceptually clean:
the model is not forced to distinguish structural motion evidence from action
style, missing observations, or noisy deformation artifacts.

## Key Principle

Motion should be treated as evidence, not as the skeleton itself.

More precisely:

- low motion should mean unknown or weak evidence, not a negative structural
  label;
- abnormal motion should mean low-confidence evidence, not an instruction to
  create abnormal topology;
- complete skeleton structure should be supported by mesh and learned rig prior;
- reliable motion should refine or disambiguate local structure.

## Desired Factorization

A more principled model would separate:

```text
mesh / shape prior        -> complete skeleton prior
motion observation        -> partial structural evidence
action / pose / noise     -> nuisance variables to explain away
```

The model should not simply concatenate mesh and motion tokens and let a decoder
discover this separation implicitly. The research opportunity is to make the
separation explicit enough to improve robustness and explainability.

## Possible Architecture Direction

One clean direction is:

```text
mesh encoder       -> complete skeleton prior tokens
motion encoder     -> evidence tokens + confidence tokens
fusion/decoder     -> confidence-weighted skeleton generation
```

The motion branch should answer questions such as:

- where do local rigid groups change relative motion;
- where are likely rotation centers;
- which regions have reliable articulated motion;
- which regions are static, under-observed, or unstable.

The decoder should always attend to mesh/shape prior, while motion evidence can
modulate or refine predictions according to confidence.

## Relation To Rigging

The forward process is:

```text
skeleton + skinning + action + mesh -> observed motion
```

The inverse problem is:

```text
mesh + observed motion -> skeleton
```

Since action and skinning are nuisance variables in the inverse problem, a model
that treats observed motion as a complete condition is under-specified. A better
model should learn which parts of observed motion are explainable by a plausible
skeleton and which parts should be ignored or down-weighted.

## Why This Is Not Just Dropout

Random motion dropout is only a weak approximation. The real issue is not that
we need more corrupted inputs. The real issue is that motion condition is
naturally insufficient and action-dependent.

A strong method should improve the semantics of conditioning:

- represent motion reliability;
- separate evidence from action nuisance;
- preserve complete skeleton prediction even when motion evidence is partial;
- use motion when it is informative instead of learning to ignore it globally.

## How This Could Become A Second Innovation

The first innovation can remain:

```text
VGGT-style motion-aware skeleton generation
```

The second innovation can be framed as:

```text
complete canonical skeleton generation from insufficient and unreliable motion evidence
```

or:

```text
evidence-aware motion-conditioned skeleton generation
```

This direction should be evaluated only after a clean baseline is established.
The baseline must first tell us whether the current architecture genuinely uses
motion or mostly relies on static mesh prior and decoder shortcuts.
