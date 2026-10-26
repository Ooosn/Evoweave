# Three-Frame and Random-Fraction Supervision

User authorized both follow-up arms on2026-10-26. Implementation and focused
CPU tests are in progress; formal training has not started.

## Fixed Comparison

Both arms start independently from the same official initialization, preserve
all2..24 input frames, use the same5000 training assets, input/pose/RNG stream,
legacy decoder, no virtual branch, batch48 and original100k learning-rate horizon.
Stop at20000 asset exposures (four passes,420 updates), then evaluate the same
128 assets x8 views against original continuous GT; failed views score zero.
Retained single/two/all controls are reused, not retrained.

- Three-frame: uniformly sample min(3,T) distinct targets per asset visit.
- Random-fraction: independently draw U uniformly in[0,1), p=0.1+0.2*U,
  K=max(1,floor(p*T)); uniformly sample K distinct targets without replacement.
  Integer rounding means the realized fraction need not lie in10%-30%,
  especially for short sequences where the minimum-one rule dominates.
- Each asset visit has its own private SHA-seeded RNG. Input sampling is unchanged.
- Average loss over selected frames per asset, then over assets. Padded slots
  never enter the decoder and contribute neither loss nor accuracy.

## Execution

Reuse the previously accepted all-frame initialization/data/loader evidence
and measured all-frame capacity envelope. Test only new selectors, variable-K
loss/accounting, inherited-evidence binding and the affected runtime paths.
First-update checks are part of formal training, not a separate audit job.

The old pcg01 allocation is preserved. It has too little remaining time for
both runs. Request one new2-H100/16-CPU/64-GiB scheduler allocation on pcg02,
then execute the arms sequentially. Code and checkpoint output use verified
NVMe /work; durable full checkpoints are mirrored and hash-verified.
Do not modify the old pcg01 checkout or any retained control artifacts.

Keep complete model/Adam/RNG/sampler checkpoints at5k/10k/15k/20k for both arms.
Report F1, failures, non52 subgroup, exact counts, supervised-frame counts,
matched per-update time, and paired asset-bootstrap intervals against retained
controls. One training seed and the20k point do not establish final convergence
or statistical equivalence. No additional architecture or100k run is authorized.
