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
Capacity_v3 is admitted; formal training is still blocked. Profile weights are
discarded. Runtime code remains pinned to `f877619c2f9a01d2bd458f3fcbbed4f3288bdb0a`.
