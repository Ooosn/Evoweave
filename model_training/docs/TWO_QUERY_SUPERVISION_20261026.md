# Fixed Two-Target Supervision

## Approved Question

The user approved this next point after the completed one-target/all-target
comparison: uniformly choose two distinct real input-frame slots per asset
exposure, without changing multi-frame encoding. Real-frame motion benefit is
established and is not being tested again. No reference-skeleton/XYZ architecture
change is included. No automatic four-target or100k continuation is authorized.

## Matched Contract

- Family `real_query_two_target_5k_20261026_v1`, arm `two_query`.
- Same official initialization, fixed5000assets,20kasset exposures,420updates,
  263608input frames, batch48,100kOneCycle horizon, zero auxiliary/consistency
  weights, no virtual frame, legacy codec, all-real roles, shared FFN/norm.
- Each exposure uses two distinct uniform slots, sampled without replacement;
  mean loss over selected frames, then mean over assets.40000supervised frames.
- Selector `sha256_python_random_sample2_v1`: hash ASCII
  `algorithm|20261026|item_index|sampling_seed|frames`, preserving full unsigned
  decimal integers, no whitespace/newline; digest big-endian integer seeds a
  private `random.Random`. Call `sample(range(T),2)` once and sort its result.
  The independent RNG does not consume input/augmentation/model randomness.
- Fresh initialization, never the trained single/all-frame20k weights or
  capacity weights. Internal5ksegments resume only this arm's own full state.
- Same128assets x8views, original continuous GT, failed views score zero.
  Retained all-frame reference is pinned by the config. Retained single-frame
  reference is the completed `train_v3/single_query/evaluation_020000` result;
  compare its exact GT/frame/normalization/token identities before reporting.
- Report two-minus-all and two-minus-single F1 with paired asset-bootstrap95%
  intervals (2000resamples,seed20261025), failure counts, count accuracy,
  non52/52groups, actual targets and slower-rank update time. Equal asset budget
  is not equal compute. An interval containing zero is not proof of equivalence.

## Admission And Retention

Primary accepts focused CPU tests, fresh no-CUDA initialization/selector audit,
then fresh two-GPU capacity evidence before formal training. The unchanged
loader audit is inherited only under strict source/config matching. Resources:
existing scheduler job130494177,pcg01,2H100,16CPUslots,64GiBhost memory,
85%per-GPU memory ceiling; actual cpuset/devices membership is required.
Stage names are audit_v10,capacity_v3,train_v4 under
`/work/wangyy/virtual_query_matched_5k_20261025/outputs`. Verify live idle state
and the exact retained grant before every compute admission. Do not modify the
remote producer checkout while a stage is running. Stop before10:03UTC;
allocation expires10:33:29UTC on2026-10-26. Do not release it automatically.

Preserve full5k/10k/15k/20kcheckpoints on verified NVMe and SHA-verified durable
mirror `/home/wangyy/evorig/outputs/two_query_20k_20261026`. Historical outputs
must not be overwritten. Local small evidence:
`D:/evoweave/server_ops/outputs/two_query_20k_20261026`.

## References

- [Completed single/all comparison](SINGLE_QUERY_SUPERVISION_20261026.md)
- [Original matched V-on/off contract](VIRTUAL_QUERY_MATCHED_5K_20261025.md)

## Status

Implementation reviewed;109CPU regression tests passed, including full20k
input/selector replay, exact loss/gradient equivalence and checkpoint accounting.
Evidence: local `preparation/tests_v2_result.json`. Fresh CPUaudit_v10 completed
04:07UTC, no CUDA/optimizer, exact initialization versus both controls, unchanged
input RNG and independently verified40kselected targets on the20kinput plan.
Receipt SHA `6cdc598b88089ba034c8e7c05239556edba854ad3f5949c485490e78b7283082`.
Capacity_v3 completed04:29UTC, six cases/twelve optimizer updates per rank, finite
positive active gradients, no virtual gradients, correct two-target counts.
Maximum reserved GPU memory42.1016GiB; peak host-tree RSS11603755008bytes.
Receipt SHA `ca58f108f9be724314b720ab09e117e687126b9499fff78e4d64aca315295dcf`.
Profile weights are discarded. Fresh train_v4 startup was independently verified
04:47:31UTC: first3updates/144assets/1920inputs/288targets, same retained inputs,
GT hashes, LR/beta and initial state; two-target selection independently replayed.
Actual producer `9df5d29eb96cf02658184069f4d7e5293ec5fed5` remained immutable
through completion. The read-only monitor exited successfully after terminal
verification. Runtime code remains pinned to
`f877619c2f9a01d2bd458f3fcbbed4f3288bdb0a`.

## Completed Result

Formal training/evaluation ended2026-10-26T07:22:00.643871UTC; independent
terminal verification completed07:22:57UTC. Both ranks completed420updates,
four full5000asset passes,263608input frames and40000supervised frames. The
100kLR horizon was preserved, not completed. All22collected small files passed
transfer SHA verification and primary rechecking. Execution and comparison are
accepted; no additional run or reference-model promotion is authorized.

| Metric | One target | Two targets | All targets |
| --- | ---: | ---: | ---: |
| Original-GT edge F1 | 0.66659544 | 0.69958198 | 0.71108141 |
| Non52 F1 | 0.58731069 | 0.61720479 | 0.62820079 |
| GT52 F1 | 0.81283175 | 0.85152212 | 0.86395009 |
| Failed views /1024 | 110 | 71 | 57 |
| Exact joint-count fraction | 0.41796875 | 0.46386719 | 0.47753906 |
| Supervised frames | 20000 | 40000 | 263608 |
| Update seconds | 6376.567991 | 6952.551192 | 9007.689587 |
| Update minutes | 106.2761 | 115.8759 | 150.1282 |

| Group | Two minus one F1 |95%paired asset interval| Two minus all F1 |95%paired asset interval|
| --- | ---: | --- | ---: | --- |
| All128assets | +0.03298654 | [+0.01360304,+0.05076125] | -0.01149943 | [-0.03152234,+0.00831841] |
| Non52,83assets | +0.02989409 | [+0.00337048,+0.05613875] | -0.01099601 | [-0.04046881,+0.01855740] |
| GT52,45assets | +0.03869038 | [+0.01674572,+0.05916814] | -0.01242796 | [-0.02874466,+0.00554633] |

Two targets improve one-target quality under this matched20krecipe, with9.03%
more recorded update time. They cost22.82%less update time than all targets,
while their F1 point estimate is lower by0.01150. Treat two-target supervision
as a low-cost candidate, not proven equivalent to all-frame supervision:
the interval contains zero but permits a meaningful deficit; no equivalence
margin was predeclared. No four-target follow-up was started automatically.

Intervals use2000paired asset resamples with seed20261025; all eight views of
an asset stay together. This is validation-asset uncertainty for one training
seed, not training-seed variability or a100kconvergence claim. Each update's
time is the slower rank's value; totals exclude initialization, checkpoint
and evaluation and are not controlled hardware benchmarks. Common multi-frame
encoding remains, so target-count reductions do not translate linearly into
overall speedup. This study does not identify anchor bone semantics.

Per-view continuous-GT F1 is producer-computed. The independent observer and
primary posthoc calculation checked immutable row hashes, complete coverage,
GT/pose/normalization/token identity, aggregate scores, failures and intervals;
continuous GT geometry was not recomputed from saved rows. The selected-target
token hashes are producer evidence, with gather semantics separately unit-tested.

### Completion Evidence

- Local comparison: `server_ops/outputs/two_query_20k_20261026/three_arm_comparison.json`,
  SHA`875c5df10f47ab9fc1199b037ed95a96b77bc75f6d2c2ca7b108aec8e8114e32`.
- Terminal monitor receipt SHA`0f7583ff46df0e219baac3236672c4148727108a6333854f5ea7bf27fdfe8649`.
- Native result SHA`ac180e5f13db63df123d8cbfdcc904ba7ac14e7d45d49131808bd6640af5c411`.
- Native paired report SHA`0dcb19d2809161074c79e8979f30be16b493e508ad75a6b6dcb9966a90b4b8af`.
- Resources SHA`a1f13cb0bde676ccb26bed42dbb46350c92da9234dabde02ba44dec5f043e0f8`;
  exit0, peak host-tree RSS24.7521GiB, cleanup confirmed.
- Final full checkpoint: `outputs/train_v4/two_query/checkpoint_020000.pt`
  under the remote run root,8181910586bytes,
  SHA`5a7b432ab6013e5fd9a4260e7878b9a4d038662cd10e39c04f7bd8a45b6ce8dc`.
  Its durable mirror and all earlier5k/10k/15kfull checkpoints are retained.
  Producer hashes plus independent file-size/receipt checks passed; weights
  were not downloaded or independently rehashed.
- At07:22:57UTC all four payload PIDs were absent, both allocated GPUs had0MiB
  and no compute process, and keeper1098441/job130494177 were retained.

Current permissions are inspect/report only. No additional experiment,
100kextension, architecture change, allocation or release is authorized.
