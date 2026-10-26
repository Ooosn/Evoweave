# Three-Frame and Random-Fraction Supervision

User authorized both follow-up arms on2026-10-26. Implementation and66 focused
CPU tests passed; formal train_v7/train_v8 are approved, actual startup pending.
The earlier train_v5/train_v6 launchers rejected missing code_commit metadata
before creating any training outputs or parameter updates. Their failure logs
remain retained. The metadata is corrected; the monitor now reports pre-worker
launcher failures immediately instead of treating absent training files as pending.

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

The existing pcg01 allocation130494177 is preserved and extended in place.
Native qalter -when now changed h_rt from24h to36h, with the same two H100s,
16CPU slots,64GiB and keeper1098441. Expiry is2026-10-26T22:33:29.625Z;
training stops before22:03UTC. The unused pcg02 request130589725 was canceled
while still pending after the user asked why the original cards were not reused.
No allocation was released. Do not overwrite any retained control artifacts.
The two new arms run sequentially under the same resource supervisor.
Code/checkpoints use verified NVMe /work; full checkpoints also have durable mirrors.

The exact input plan gives263608 input frames for either arm at20k.
Three-frame supervision selects59416 targets (584 visits have only2 inputs).
The fractional rule selects45936 targets, averaging2.2968 per asset visit.
First3 updates contain144 assets/1920 inputs, with432/323 targets respectively.
Accepted full-frame capacity SHA is0a5aa224435f8496d43e14a3d0bfc19b55957412b0e4e6561b19d8311ade475c;
accepted CPU/data/initialization SHA is92706351041d631fed21b07757d7776f944dc0a5bef135bf11816ee0fdb7419a.
Both bound configurations passed the actual inherited-receipt check locally.
No new CPU weight audit or GPU capacity stage is required.

Keep complete model/Adam/RNG/sampler checkpoints at5k/10k/15k/20k for both arms.
Report F1, failures, non52 subgroup, exact counts, supervised-frame counts,
matched per-update time, and paired asset-bootstrap intervals against retained
controls. One training seed and the20k point do not establish final convergence
or statistical equivalence. No additional architecture or100k run is authorized.
