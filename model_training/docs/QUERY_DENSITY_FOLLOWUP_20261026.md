# Fraction20-40 Minimum2, Then Fixed4 Supervision

User requested these two fresh20k experiments in this order on2026-10-26.
Implementation is complete; actual startup is pending.

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
No further experiment or100k run is authorized.
