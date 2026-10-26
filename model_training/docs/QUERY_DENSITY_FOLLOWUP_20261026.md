# Fraction20-40 Minimum2, Then Fixed4 Supervision

User requested these two fresh20k experiments in this order on2026-10-26.
Both runs completed on2026-10-27JST. See the measured result below.

## Protocol

1. train_v9 / fraction20_40_query: per asset visit draw U uniformly in[0,1),
   p=0.2+0.2*U, K=max(2,floor(T*p)), then sample K distinct input-frame slots.
2. Only after train_v9 succeeds, start train_v10 / four_query with K=min(4,T)
   distinct uniformly sampled input-frame slots.

Both retain the full2..24-frame encoder input, identical official initialization,
legacy decoder, no virtual route, loss mean over frames within each asset then
over assets, batch48, input/pose/RNG stream, data/normalization and100k LR horizon.
Each starts fresh, visits the same5000assets four times (20kexposures/420updates),
and evaluates the same128assets x8views with original continuous GT and failure0.
Existing one/two/three/fraction10-30/all-frame controls are not retrained.

Exact deterministic plans have263608input frames each. New fractional supervision
selects75065targets, averaging3.75325per visit, with no one-target visits.
Its K histogram is2:7533,3:3108,4:2952,5:2665,6:1707,7:1178,8:659,9:198.
Fixed four selects77920targets:584visits have2inputs,912have3,18504haveatleast4.
The first3updates contain144assets/1920inputs and510/576targets respectively.
The selector draws do not alter global or input RNG state.

## Execution

Reuse the original scheduler allocation130494177 onpcg01i, GPUs3/4 by UUID,
16CPUs/64GiB. Native h_rt was extended in place to36h in the previous phase;
expiry2026-10-26T22:33:29.625Z, conservative stop22:03UTC. Do not release or
replace the allocation. No new allocation is requested.

Only selector and runtime variant mappings changed. Five focused CPU tests
passed, including independent recipes, minimum2, full integer seeds, old-rule
regression, full20k target accounting and unchanged inputs. The already proven
variable-K decoder and loss path is unchanged. Reuse pinned all-frame CPU/
initialization/data and larger all-frame capacity receipts; no repeat audit/
capacity job. First3 checks occur inside formal training.

Retain full5k/10k/15k/20k checkpoints on NVMe and SHA-verified durable mirrors
under /home/wangyy/evorig/outputs/query_density_followup_20k_20261026.
The observer reports launcher failures even before workers exist.
Queue only train_v9 initially: the keeper sorts request filenames lexically,
so queueing train_v10 at the same time would violate the requested order.

Source and configurations are immutable while either payload runs.
Final reporting covers F1,non52,F1 differences with paired asset-bootstrap
intervals, failures, exact joint-count accuracy, realized K distribution and
update times. One seed/20k and sequential timings retain the previous limitations.
This density experiment does not authorize100k or further density arms.
The user's later authorization separately starts reference-skeleton XYZ research
after both runs complete; it does not reopen these completed training stages.

## Completed Result

The requested order was respected: fraction20-40 finished2026-10-26T17:39:47Z;
four-frame was admitted17:42:33Z and finished20:31:29Z. Both completed420updates,
20000asset exposures and263608input frames, with75065/77920supervised targets.
All840updates passed independent input/GT/LR/selector accounting. The first
three updates also verified actual scheduler cgroups and the common initialization.

Same128validation assets x8views, continuous original GT, failed generation=0:

| Targets per visit | Edge F1 | Non52 F1 | Failures /1024 | Exact count | Update minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| One | 0.66660 | 0.58731 | 110 | 41.80% | 106.3 |
| Two | 0.69958 | 0.61720 | 71 | 46.39% | 115.9 |
| Three | 0.70624 | 0.62229 | 68 | 45.70% | 116.7 |
| Four | 0.70424 | 0.62366 | 78 | 42.97% | 118.6 |
| Random10%-30%, min1 | 0.69944 | 0.61869 | 62 | 39.65% | 113.9 |
| Random20%-40%, min2 | 0.71401 | 0.63079 | 65 | 45.61% | 125.0 |
| All | 0.71108 | 0.62820 | 57 | 47.75% | 150.1 |

The new random rule has the highest overall and non52 point estimates, and
uses16.7%less update time than all-frame. It is5.4%slower than four-frame in
this sequential measurement despite slightly fewer targets; this is not a
controlled hardware benchmark or evidence for a particular overhead mechanism.

Paired asset-bootstrap95% intervals (2000replicates, seed20261025):

| Difference | F1 difference | 95% interval |
| --- | ---: | --- |
| Random20-40 minus all | +0.00292 | [-0.01153,+0.01778] |
| Random20-40 minus three | +0.00777 | [-0.00668,+0.02256] |
| Random20-40 minus two | +0.01442 | [+0.00187,+0.02739] |
| Random20-40 minus random10-30 | +0.01457 | [-0.00110,+0.03075] |
| Four minus random20-40 | -0.00976 | [-0.02585,+0.00520] |
| Four minus three | -0.00199 | [-0.01514,+0.01122] |

The random20-40 rule is a practical starting candidate, not a proven universal
winner. Its differences from four, three and all include zero. This is one
training seed at20k within the unchanged100k LR horizon; asset intervals do
not measure training-seed variability, and exploratory comparisons are not
corrected for multiplicity. Edge F1 uses nearest-joint edge mapping, without
a coordinate-distance cutoff; subsequent XYZ research must also report actual
position errors, not rely on this metric alone.

Eight full checkpoints have verified durable mirrors. At20:33UTC the observer
confirmed payload cleanup, both GPUs at0MiB, and the original keeper retained.
The primary rehashed all36collected small artifacts; large checkpoint payload
hashing/mirroring was verified by the producer and observer metadata/stat checks,
not repeated independently by downloading the payloads.

Evidence:
- Producer: a0bda62bb8fba70320310b063da8c7898ddaf441; selector core382fbfb1652fbb667e504c3e63ccef9eb3ee8be0.
- Local comparison: D:/evoweave/server_ops/outputs/query_density_followup_20k_20261026/seven_arm_comparison.json;
  SHA25632529e04a241af22044567bf6720c5af4afa8cbc2876e68373f93d69dd92cf46.
- Local receipts and36artifacts: same root/monitor, including monitor/runs/train_v9 and train_v10.
- Random20-40 final checkpoint SHA2569704b23a194ca65836c6dfacd6637ca27542dc4b0836b1d3611185f0339cc888.
- Four final checkpoint SHA256f64ba563ae5df03e58b763e2592d495ff59e8e3003e6ae5a1ecc61031cfad07a.
- Remote outputs: /work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v9 and train_v10.
- Durable mirrors: /home/wangyy/evorig/outputs/query_density_followup_20k_20261026.
