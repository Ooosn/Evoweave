# Three-Frame and Random-Fraction Supervision

Both fresh20k follow-up arms completed on2026-10-26, with cleanup verified at
14:15UTC (23:15JST). The66 focused CPU tests passed. Execution gates are now
closed; no100k or additional training is authorized.
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

## Completed Result

Both new arms used the same official initialization, data/input/LR plan and
evaluation as the retained controls. Each completed420updates/four full5kpasses.
Original continuous-GT evaluation:128assets x8views; failed generations score0.

| Supervision | F1 | Non52 F1 | Failures/1024 | Exact Joint Count | Update Minutes | Supervised Frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_query | 0.66660 | 0.58731 | 110 | 41.80% | 106.3 | 20000 |
| two_query | 0.69958 | 0.61720 | 71 | 46.39% | 115.9 | 40000 |
| three_query | 0.70624 | 0.62229 | 68 | 45.70% | 116.7 | 59416 |
| fraction_query | 0.69944 | 0.61869 | 62 | 39.65% | 113.9 | 45936 |
| all_frames | 0.71108 | 0.62820 | 57 | 47.75% | 150.1 | 263608 |

Three-minus-two F1 is+0.006654,95%CI[-0.006960,+0.021542]. This is a higher
point estimate, not an established improvement across validation assets or
training seeds. Three is0.004846 below all-frame and uses22.27%less update time;
its interval versus all also includes0, which does not establish equivalence.
The three-frame update total is only0.71%above two in these sequential runs.

Fraction-minus-two F1 is-0.000146,95%CI[-0.018682,+0.019211]. Its total update
time is1.72%below two and24.15%below all. Fraction has62failed views versus71
for two, but exact joint-count accuracy is39.65%versus46.39%. Fewer parse/
generation failures did not translate into better overall F1 or count accuracy.
These point estimates do not support choosing fractional supervision over fixed
two. Fixed three is a useful quality/cost candidate, not a promoted global baseline.

Both new arms improve single-frame F1; their all/non52/gt52 asset-bootstrap
intervals versus single are positive. The1-to2 improvement is larger than the
2-to3 point-estimate increment. Dense/all-frame remains the highest F1 point estimate.

| Difference | F1 Delta | Paired Asset Bootstrap95%CI |
| --- | ---: | --- |
| three_query_minus_all_frames | -0.004846 | [-0.023123, 0.014447] |
| three_query_minus_two_query | 0.006654 | [-0.006960, 0.021542] |
| three_query_minus_single_query | 0.039640 | [0.020968, 0.058703] |
| fraction_query_minus_all_frames | -0.011645 | [-0.032920, 0.009995] |
| fraction_query_minus_two_query | -0.000146 | [-0.018682, 0.019211] |
| fraction_query_minus_single_query | 0.032841 | [0.012556, 0.052814] |
| three_query_minus_fraction_query | 0.006799 | [-0.010274, 0.022472] |

The bootstrap resamples128validation assets (2000replicates,seed20261025);
non52 has83assets and gt52 has45. These exploratory intervals are not adjusted
for multiple comparisons and do not measure training-seed variability.
There is only one training seed and the20k point on an unchanged100k LR schedule.
No equivalence margin was specified. Update time sums the slower rank per step
and excludes initialization, checkpoint copies and evaluation; it is not a
controlled hardware speed benchmark. Per-view continuous-GT F1 remains
producer-computed; aggregation and paired intervals were independently recomputed.

### Retained Artifacts

- Scientific source:6b55970e4d34abc376e6b0d18163b5937c2f377a; immutable formal
  producer:2279b2157fbbf559f0f980dd9cc617f823253c7b.
- three_query/train_v7 ended11:31:03.8519UTC; fraction_query/train_v8
  ended14:13:43.795712UTC. Both supervisors exited0 with payload cleanup.
- All840updates matched retained input/GT/LR records and independently replayed
  target selection. First3 startup receipts passed on both ranks.
- Both arms retain full5k/10k/15k/20k model/Adam/RNG/sampler checkpoints on NVMe
  and durable mirrors. Producer SHA and independent size/receipt checks passed;
  payload bytes were not independently rehashed a second time.
- Final three checkpoint SHA:9ef70f5716de5c6e1f274303f510bd35508db2895d876dc87353a32b872d9d65.
- Final fraction checkpoint SHA:fc348a54857fd55dfeb0f1452e5f4cd45ff7c160dc5d9f089b614cd31bbd38e6.
- Primary rechecked36collected small artifacts, with zero digest mismatches.
- Report:`D:/evoweave/server_ops/outputs/query_density_20k_20261026/five_arm_comparison.json`,
  SHA:8e2085cc59d69b34b8043c50edcf73402a11d0c5dfa250f9b092793399c3f0ef.
- Terminal receipt:`D:/evoweave/server_ops/outputs/query_density_20k_20261026/monitor/receipt.json`,
  SHA:cff6490aee5c34a025cdce7a1249f2bda7434f51175c30138920b44da62fde05.
- Both GPUs were idle at14:15:15UTC, all observed payload PIDs gone, keeper1098441
  retained. Allocation130494177 was extended in place, not replaced or released.
