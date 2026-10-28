# GT-Reference Stage2: Value Of The Skeleton-Generation Framework

## Question And Authority

On2026-10-28 the user requested a comparison outside the already-trained
skeleton-generation system, followed by purposeful analysis/optimization and,
if appropriate, a longer run such as80k. This follows the completed
[60k association comparison](ANCHOR_JOINT_BINDING_5K_20261027.md#completed60k-result-2026-10-28jst).
GT reference joints/tree remain inputs. Do not introduce predicted-reference
matching, geometric fitting, or another motion-input-benefit ablation.

The old reference_transport/target_content pair established a benefit of the
explicit association route under that recipe. Both used the same pretrained
and jointly trained backbone, so it did NOT isolate that framework's value.

## Three Regimes

All use the SAME reference_transport head, complete GT reference, multi-frame
surface/motion architecture, immutable5000training assets, data/ref selection,
coordinate/bone loss, head initialization and learning-rate trajectory.

| Regime | Backbone start | Backbone Adam start | AR objective |
| --- | --- | --- | --- |
| joint_pretrained | Existing skeleton-generation donor | Retained donor state | On |
| pretrained_stage2 | Same donor weights | Same retained donor state | Off |
| scratch_stage2 | Constructor random, including surface/motion | Fresh | Off |

The first regime already completed60k. Preserve its exact full model/Adam/RNG
checkpoint and consumed sampler, then extend its verified prefix for80k.
Do not retrain it or silently reset its optimizer. The two new controls start
from zero stage2visits, not from the learned60k joint head. Their random head
seed is the same20261028as the completed experiment. The scratch backbone uses
seed20261028and loads neither official nor project-trained weights. Dormant
AR modules remain serialized for checkpoint/parameter-name compatibility,
but are neither executed nor differentiated in the stage2-only regimes.

Joint versus pretrained_stage2 isolates continuing the AR objective under
the same warm start. Pretrained_stage2 versus scratch_stage2 compares the
PRACTICAL WARM-START PACKAGE: learned weights AND retained Adam state versus
random weights/fresh Adam. It does not isolate weight pretraining from
optimizer history. Donor RNG state is shared for matched stochastic execution
even in scratch; no donor learned tensors/Adam state are applied there.

The scratch control is the same network and matched recipe, not a tuned best
standalone system or equal-total-lifetime-compute comparison. The warm start
includes20kproject AR visits and official upstream pretraining. If scratch
underperforms, inspect its optimization and learning curve before attributing
the whole gap to integration. If pretrained_stage2matches or exceeds joint,
continued AR training has not shown an incremental stage2benefit here.

## Budgets And Phases

- Preserve2..24input poses, batch48, physical24/12/6per rank, independent
  backbone/head norm1clipping, lambda0to10over the first1000stage2visits,
  target bone-vector weight0.25and head LRmultiplier3.
- The LRposition is20000+stage2visits in ALL regimes, completing the existing
 100k LRcycle at80k. Scratch has no20kprevious learned exposure; its metadata
  distinguishes true backbone visits from the shared LRposition.
- First run pretrained_stage2/scratch_stage2alternating each10k to60k. Save
  full checkpoints and fixed128evaluations every5k. This phase ends for primary
  comparison against the retained joint60k, without automatic further compute.
- After that review, separately admit common80k if appropriate. Resume the
  joint60k parent and both60k control checkpoints, alternating70k then80k.
  These are80,000asset visits,16complete5000passes, NOT80,000updates.
- The80k data plan must preserve every generated request in the old60kprefix;
  use the existing checked completed-stream extension, never relabel a new
  sampler cursor as a resume.
- No model/loss/data/LR changes during a phase. An optimization change needs
  a separately recorded comparison, not a silent repair of one running arm.

## Gates And Evaluation

Local tests cover loss-path isolation, unchanged joint formula/gradients,
no-checkpoint random construction, strict parent migration, full stream-prefix
parity, phase admission and orchestrator behavior. Regressions cover the
existing XYZ/real-frame routes, data/stream restoration and native scheduler
guards. Before formal training, run the complete16pass CPU/loader audit and
native two-rank6/12/24pose largest-joint capacity cases in all3regimes, two
updates/case. Verify positive encoder/head learning signal, absent AR-branch
gradients for stage2-only, identical pretrained encoders, different random
encoders, identical initial heads, finite free inference/exact reference
clamps and declared parent model/Adam/RNG/sampler restoration. Discard profile
weights. Preserve16CPU/64GiBtree/85%GPU enforcement and both native cgroups.

Use the unchanged128panel for interim comparisons. At the common final
budget, evaluate all967and separately the839not used for this stopping
decision, plus32training assets' eligible/strictly-heldout raw poses. Check
actual full traces before calling poses seen or heldout. Preserve normalized
known-ID MPJPE, PCK@5%, nearest-mapped edge F1, symmetric J2J, target bone-vector
error, cross-reference consistency and coherent target-anchor permutation.
Use paired asset intervals and fixed artifact-order visuals; one seed does
not establish seed robustness. The839are not historically pristine. GT trees
are supplied; edge F1is not generated topology accuracy. Stage1 free-AR
retention remains a separate required check before broad shared-model claims.

## Execution

Use the retained native job130675914onpcg01i, physicalGPU3/4,16CPU, keeper1108368.
The grant expires2026-10-31 20:55JST; stop-before is19:55JST. Do not release it
without a human request. CPU audit, capacity and each formal phase require
their own exact state/config/output admission. No SSH compute exception.
The server code checkout may change only while the previous payload is
confirmed finished. Pin the source for each whole phase.

Recipe:model_training/experiments/anchor_integration_5k_20261028_v1.json.
Entrypoint:model_training/rigweave/scripts/train_anchor_integration.py.
Durable outputs:/home/wangyy/evorig/outputs/anchor_integration_5k_20261028/v1.
Local inspection:D:/evoweave/server_ops/outputs/anchor_integration_5k_20261028.
All datasets/weights/artifacts remain outside Git. Code/config/diagnostic
summaries may be published to the already-authorized branch.

## Admission Progress

Local108tests passed:38current anchor/integration tests,70regressions. Native
CPU audit_v13finished2026-10-28 12:42JST with clean exit0, no CUDA initialized,
16complete5000passes and1,043,152planned input-frame visits. The60kprefix and
completed-stream extension match exactly. All6loader cases have bitwise equal
CPU fields, exercised both rank prefetch queues and preserve global RNG.
Maximum training joints234; peak process-tree RSS17,276,526,592bytes.
Result SHA256:4494b15f5afc1dd922cfba60a5db411fc1f0abe996139ac8a53fb1b63624de56.
Scientific SHA256:8e0467089d886766aa94b1ae6b6002f7ea03f1d2bff27ce161acef1c1b752da1.
Capacity_v19is admitted next; formal training is not yet released.

Capacity_v19subsequently completed2026-10-28 13:02JST, exit0and clean payload
cleanup. All18updates/rank completed across3regimes x6/12/24poses x2steps.
Active encoder/head gradients were positive/finite; stage2-only AR gradients
were absent. Pretrained encoders match exactly, random encoders differ and
all initial heads have SHA93dfca718d9c2130a390a172fa391433652c41c4001d36bdad680cb6da8066e9.
Each regime passed free24pose inference at234/230joints. Parent60k restoration
preserved model/Adam/RNG and counters60000/787504/1260with a verified sampler
extension; both ranks' parent head SHAis68d30f1e7562b12574307e6b3dd6e8da791edc85a97f9ef6f7225178fe9f21aa.
Maximum reserved GPU bytes45,797,605,376/45,147,488,256; peak tree RSS25,253,093,376.
Result SHA256:be38458a228d1e89611b94db8bcd46c65347b259372cf2f8df860e9db3c91179.
Profile weights are discarded; this is capacity evidence, not quality.
Formal train_v18controls_to60k is now admitted. No80kphase is yet admitted.

Formal train_v18started2026-10-28 13:06:47JST, native supervisor3629610,
source8b5cb9afa415c19781fb348fa1cdfcded2c96a1e. At13:19JST the pretrained-stage2
arm had1,920visits/40updates with finite losses and AR supervision disabled.
Configuration SHA419b381d6755b19f8880b418c9ae1bce1da4c99b34a13f7c37258595f0e2d3e4.
Pin this deployed source for the whole phase. Later local state/documentation
commits must not be checked out remotely while its payload is active.

## Matched10k Interim Evidence

The new controls completed10k each. Independent actual-prefix inspection
matches all3arms' initialization head, consumed assets/raw poses/references,
planned AR slots and LR trajectory through2fullpasses/210updates/130,200input
frames. Only the joint arm applies AR supervision. The new pretrained donor
matches the old joint source; scratch applies neither pretrained weights nor
Adam state. Training is disjoint from128selection assets; GT-copy controls
are identical and row-to-asset aggregation was independently recomputed.

| Metric | Joint pretrained | Pretrained stage2 only | Scratch stage2 only |
| --- | ---: | ---: | ---: |
| Normalized MPJPE | 0.04472532 | 0.04913710 | 0.09238944 |
| PCK@5% | 0.70473408 | 0.67950518 | 0.53762169 |
| Edge F1 | 0.59815582 | 0.60357186 | 0.54887951 |
| Symmetric J2J | 0.03426620 | 0.03766356 | 0.06185848 |
| Target bone-vector error | 0.02197576 | 0.02086930 | 0.02501458 |

Pretrained-stage2 minus scratch MPJPE is-0.04325235, paired95%interval
[-0.04990214,-0.03660175]; all5metric intervals favor the warm start. Joint
minus pretrained-stage2 MPJPE is-0.00441177, interval[-0.00566570,-0.00320955],
but bone-vector error is+0.00110646, interval[0.00052018,0.00172046]. Edge F1
difference-0.00541604 has interval[-0.01425795,0.00379260]. Thus joint has an
early position advantage with a local-bone tradeoff, not a comprehensive win.
Scratch remains near the copy baseline; this early point does not establish
its eventual ability or a final integration benefit. Keep the run unchanged.

The same previous arm/leg cases were rendered and inspected, with input GT
on its own surface, target GT separately, and all3predictions on the same
target surface. Both fixed examples currently favor pretrained-stage2 over
joint despite the latter's better aggregate position score. Their displayed
numbers are per-target-frame errors, not the128means. Use the readable_v2
figures, not the retained first layout with cramped title spacing.
Evidence:review_controls_010000.json, trace_prefixes/010000, and
fixed_case_{2,3}_010000_v2.png under the local inspection root.

## Matched30k Interim Evidence And Follow-Up Recovery

At2026-10-28 18:47JST, fresh HGC evidence confirmed job130675914and the original
train_v18supervisor3629610still running in the native16CPU allocation. The
pretrained-stage2arm reached32,352visits; scratch reached30,000. Both30k
checkpoints have verified scratch/durable copies. No training stop, restart,
duplicate submission, remote source update or GPU release was performed.
The conversation interruption had interrupted follow-up, not server training.
The unfinished resource receipt's passed=false is provisional, not a terminal
failure: complete=false, exit_code=null and native membership remain intact.

The independently collected30k artifacts and consumed trace prefixes pass
all matching checks: six complete5000passes,630updates and394,280input-frame
visits per arm; identical actual assets/raw poses/references/planned AR slots
and LR trajectory; matched head initialization, correct warm starts and no
AR supervision in the two controls. All128selection assets remain disjoint
from training; GT-copy rows and independently recomputed aggregates agree.

| Metric | Joint pretrained | Pretrained stage2 only | Scratch stage2 only |
| --- | ---: | ---: | ---: |
| Normalized MPJPE | 0.01213245 | 0.01146367 | 0.09155911 |
| PCK@5% | 0.97704979 | 0.98053687 | 0.51809039 |
| Edge F1 | 0.81258312 | 0.82593031 | 0.53556209 |
| Symmetric J2J | 0.01004071 | 0.00957377 | 0.06178304 |
| Target bone-vector error | 0.00882835 | 0.00825481 | 0.02576529 |

Joint minus pretrained-stage2 MPJPE is+0.00066878, paired95%interval
[0.00038494,0.00100746]; edge F1difference is-0.01334720, interval
[-0.01862209,-0.00831084]. All5paired intervals favor pretrained-stage2.
This reverses the10k position ordering: continued AR has NOT demonstrated
a stage2benefit at this matched point. The warm-start package still beats
scratch on all5intervals; MPJPE difference is-0.08009544, interval
[-0.08942357,-0.07099276]. Scratch remains close to the copy solution, but
this one matched recipe/seed is not a tuned best standalone system.

The same fixed arm/leg figures were rendered and visually checked. Both
pretrained routes now follow the target pose; scratch remains close to the
reference. These two fixed examples favor joint, contrary to the128mean
ordering. Retain them and distinguish per-frame results from aggregate
claims; do not cherry-pick new examples to fit the mean result.

Evidence:review_controls_030000.json, trace_prefixes/030000, and
fixed_case_{2,3}_030000_v2.png under the local inspection root. Continue the
existing phase unchanged, compare future common40k/50k/60k nodes, then review
at60k before separately admitting80k under the existing human authorization.
Retained10k/30k reviews must not be overwritten or rerun. This remains an
interim single-seed selection128result, not final quality acceptance.

## Optimizer Continuity Check

On2026-10-28 19:29JST, read-only CPU inspection of the actual10k/20k/30k
checkpoints confirmed complete model/Adam/rank-state saving and cumulative
optimizer counters. Both control heads' representative Adam steps are
210/420/630; pretrained surface/motion representatives are630/840/1050,
retaining420prior updates, while scratch surface/motion are210/420/630.
All existing optimizer states in these modules contain exp_avg and exp_avg_sq;
the inspected head-output moments are nonzero at every checkpoint. These
are representative per-parameter counters, not an assertion that every
conditional parameter has the same step count.

The inactive pretrained AR decoder retains its420step state; scratch has no
AR decoder optimizer states, as expected because that branch never executes.
Both ranks retain torch/CUDA/NumPy/Python RNG and sampler state. Per-rank
stream cursors are5000/10000/15000, matching global10k/20k/30k visits. Backbone
LRs at those checkpoints are8.85449718e-5/7.55118492e-5/5.95222359e-5, with
head LR3x; the schedule continues from accumulated visits rather than
restarting warmup. All six checkpoint metadata records confirm two verified
copies. Diagnostic CUDA initialization was false; no training was modified.

At segment switches, the optimizer Python object is reconstructed and then
its full saved state is loaded BEFORE any update, along with model/RNG/stream
and counters. Each regime uses its own last checkpoint, not the other arm's
optimizer. Only initial construction creates a fresh head state and, for
scratch, a fresh entire optimizer. This check is not a new bitwise comparison
of uninterrupted versus resumed training. Evidence:
optimizer_continuity_20261028.json under the local inspection root.

## Matched40k Interim Evidence

The common40k review completed2026-10-28 20:54JST. All3arms' actual traces,
references, initialization and LR match through8full5000passes,840updates
and527,856input-frame visits. The controls apply no AR supervision. Producer
metrics were independently reaggregated from rows;128selection assets are
disjoint from training. Checkpoint mirrors and artifact identity checks pass.

| Metric | Joint pretrained | Pretrained stage2 only | Scratch stage2 only |
| --- | ---: | ---: | ---: |
| Normalized MPJPE | 0.00959594 | 0.00959109 | 0.04389107 |
| PCK@5% | 0.98777276 | 0.98743794 | 0.71056187 |
| Edge F1 | 0.84749777 | 0.84828114 | 0.59497495 |
| Symmetric J2J | 0.00819387 | 0.00812393 | 0.03343135 |
| Target bone-vector error | 0.00725581 | 0.00703994 | 0.02103794 |

Joint minus pretrained-stage2 MPJPE is+0.0000048461, paired95%interval
[-0.00025630,0.00026315]; edge F1difference is-0.00078336, interval
[-0.00567861,0.00407063]. PCK and J2J intervals also include zero. Only
bone-vector error still favors pretrained-stage2: difference+0.00021587,
interval[0.00006302,0.00037529]. Do not retain the30k all5metric superiority
claim as the current result, and do not mistake no detected difference for
a formal equivalence finding. Ongoing AR has no demonstrated incremental
stage2benefit at this node; first-stage preservation remains untested.

Scratch MPJPE improves from0.09155911at30k to0.04389107at40k, a52.1%relative
reduction, with PCK rising51.81%to71.06%and edge F1from0.53556to0.59497.
Warm-start benefits persist on all5paired intervals: pretrained-stage2 minus
scratch MPJPE is-0.03429998, interval[-0.03727274,-0.03141853], with lower
MPJPE on all128assets. But scratch is now visibly learning, not simply stuck
at copying. Preserve the unchanged recipe to60k and assess its later slope
before deciding about80k or any separately controlled optimization changes.

The same arm/leg cases were rendered and checked. Scratch now lowers the
raised arm and bends the leg toward the target, but still under-responds;
both pretrained routes closely follow the target. The two fixed cases still
slightly favor joint despite the aggregate near-tie. No cases were replaced.
Evidence:review_controls_040000.json, trace_prefixes/040000 and
fixed_case_{2,3}_040000_v2.png under the local inspection root.

Fresh20:53JST evidence showed the original train_v18supervisor and native
job130675914still running: pretrained-stage2 at45,816visits, scratch40,000.
Latest checkpoint copies are verified at45k/40k respectively. No training
restart, source deployment, configuration change or allocation release.
Next common review is50k, then terminal60k; all interim single-seed/GT-reference
and selection128limitations continue to apply.

## Matched50k Interim Evidence

The common50k review completed2026-10-28 23:00JST. Actual data/reference/AR-slot
and LR traces, initializations, no-AR controls, checkpoint identity/mirrors
and independent row aggregation all pass. Each arm has10complete5000passes,
1,050updates and657,368input-frame visits; selection128remains train-disjoint.

| Metric | Joint pretrained | Pretrained stage2 only | Scratch stage2 only |
| --- | ---: | ---: | ---: |
| Normalized MPJPE | 0.00833082 | 0.00784938 | 0.02178393 |
| PCK@5% | 0.99085268 | 0.99275764 | 0.89604011 |
| Edge F1 | 0.86648285 | 0.87630217 | 0.73392139 |
| Symmetric J2J | 0.00727791 | 0.00688910 | 0.01687654 |
| Target bone-vector error | 0.00641413 | 0.00608844 | 0.01406114 |

Joint minus pretrained-stage2 MPJPE is+0.00048143, paired95%interval
[0.00023450,0.00075344]; edge F1difference is-0.00981932, interval
[-0.01466965,-0.00511780]. J2J and bone-vector intervals also favor
pretrained-stage2, but PCK difference-0.00190497 has interval
[-0.00415636,0.00005304] and does not exclude zero. The current result is
modest superiority on4metrics, not every metric or final-budget evidence.

Scratch MPJPE falls another50.4%relative to40k, to0.02178393; PCK rises
71.06%to89.60%and edge F1from0.59497to0.73392. Pretrained-stage2 remains
better on all5paired intervals; its MPJPE difference versus scratch is
-0.01393455, interval[-0.01557306,-0.01232112], with127/128assets better.
The persistent warm-start benefit and rapidly improving scratch curve
must both be reported. Do not extrapolate eventual equality or failure.

The unchanged arm/leg cases were rendered and checked. Scratch now follows
the target substantially, with per-frame errors1.52%and1.60%, although it
still underperforms both pretrained routes. The arm case favors pretrained
stage2 while the leg case favors joint. Evidence:review_controls_050000.json,
trace_prefixes/050000 and fixed_case_{2,3}_050000_v2.png under the local root.

Fresh22:58JST evidence showed train_v18and native job130675914still running:
pretrained-stage2 at59,368visits, scratch50,000; latest verified checkpoint
copies at55k/50k. Continue unchanged until both60k controls finish, then
verify terminal exit/resources/cleanup/counters and the matched primary
comparison before separately admitting any80kcontinuation. The existing
human authorization does not replace exact per-phase admission. No source
deployment, restart, optimizer reset, recipe change or GPU release occurred.

## Completed60k Primary Comparison And80k Decision

Controls train_v18finished2026-10-28 23:57:07JST with exit0, native scheduler
membership, passed resource receipt and complete payload cleanup. Each arm
ended at60,000visits/1,260updates/787,504input-frame visits. Both checkpoint
copies are verified; actual prefixes/init/reference/LR/no-AR controls and
independent row aggregation all pass. The retained joint arm is matched at
the same counters. Terminal result SHA256:
8dbd444c39bedf8e501105bb1581fb61c13cac78857a6706f3264be8f2788534.

| Metric | Joint pretrained | Pretrained stage2 only | Scratch stage2 only |
| --- | ---: | ---: | ---: |
| Normalized MPJPE | 0.00674328 | 0.00653322 | 0.01648303 |
| PCK@5% | 0.99430256 | 0.99434555 | 0.93769481 |
| Edge F1 | 0.89501710 | 0.90136931 | 0.78042397 |
| Symmetric J2J | 0.00600785 | 0.00579236 | 0.01291642 |
| Target bone-vector error | 0.00574855 | 0.00547103 | 0.01174753 |

Joint minus pretrained-stage2 MPJPE is+0.00021006, paired95%interval
[0.00003600,0.00038602]; edge F1difference is-0.00635221, interval
[-0.01090435,-0.00220918]. J2J and bone-vector intervals also favor the
stage2-only warm-start arm, but PCK difference-0.00004299 has interval
[-0.00133002,0.00123971]. Continued AR does not demonstrate incremental
stage2benefit at60k. Warm-start package benefits persist across all5intervals:
MPJPE pretrained-stage2 minus scratch is-0.00994980, interval
[-0.01124970,-0.00869978]. These are single-seed selection128/GT-reference
results, not enlarged-set or first-stage-retention acceptance.

All3arms still improve from50k to60k by approximately19.1%/16.8%/24.3%in
MPJPE. This sustained decline, including continuing scratch improvement,
justifies the already-authorized common80k continuation WITHOUT a model,
loss, data, LRcycle, initialization or optimizer change. Fixed cases2/3were
rendered and checked; pretrained-stage2 wins both at this node, but scratch's
arm example is slightly worse than50k despite aggregate improvement. Preserve
all fixed cases and do not equate the aggregate curve with every pose.

Evidence:review_controls_060000.json, trace_prefixes/060000,
artifacts/controls_{launch,resources,result}.json and
fixed_case_{2,3}_060000_v2.png. The completed phase state and its prior interim
reviews are archived at model_training/state/archive/
anchor_integration_controls60k_20261029.json, with execution authority off.

### Exact80k Continuation Admission

Candidate stage train_v19uses
model_training/experiments/anchor_integration_common80k_20261028_v1.json,
SHA256 af4c48f43254658ecd635f0f815aa248edae4ae66f1b035ea9d3e67ab9656cfb.
Only execution.phase and the two exact control resumes differ from the
original configuration. Scientific SHA256 remains
8e0467089d886766aa94b1ae6b6002f7ea03f1d2bff27ce161acef1c1b752da1;
model/jobs/tools code is unchanged. Reuse the verified16pass audit and
three-regime native capacity checks, including declared joint60k restoration,
rather than rerunning unchanged gates. Final startup plan must still verify
the actual completed control states, native keeper/grant/GPU identities,
deadline, disk space, clean committed source and exact unused output path.

Joint resumes its retained train_v17/reference_transport60k checkpoint,
SHA cfd446c1f6808304aa6a691884c2ed77d19db0f1850512dfd97b83e7eb0a0016.
Pretrained-stage2 resumes train_v18/pretrained_stage2/checkpoint_060000.pt, SHA
3624030132d0ae5b907a0d0641601604e220f3b28f8fd201bb063a03a8e8df3a;
scratch resumes train_v18/scratch_stage2/checkpoint_060000.pt, SHA
be8eb4d12917d52d8761c9c1223384c4441deea0dd1037bc88d6df5aefb1a50b.
Model/Adam/RNG/counters and consumed sampler prefix must be retained. The
shared global80k target is1,680updates and1,043,152input-frame visits per arm,
adding420updates/20kvisits per arm. Three regimes alternate to70k then80k;
the LRposition reaches the existing100kcycle end without restarting it.

At00:04JST onOctober29, the native keeper had no children, no granted-GPU
compute processes were present, all earlier requests were complete, the old
source was clean and train_v19paths were unused. Free workspace was about
2.97TB, with ample durable space. Current admission is pending the final
committed-source plan and actual launch confirmation, not evidence of a
running80k payload. Expanded evaluation and stage1retention remain required.

The final plan subsequently passed at2026-10-29 00:15:37JST. It directly
inspected all3actual60k checkpoint containers, finding head Adam step1260,
pretrained encoder step1680and scratch encoder step1260. Both rank RNG
records exist; the declared parent stream extends correctly, both control
streams restore normally, and all3arms' next microbatches match per rank.
All6samplers have plan SHA
92078ff94a9c88ff665a3b458e2f3c95c7ec5747d9c6e5cf4d919d73f5452556.
The plan initializes no CUDA and accurately records its short CPU inspection
outside the scheduler job; compute still starts only as a native keeper child.
Remaining time before stop was67.66hours, exceeding the conservative12hour
continuation/evaluation planning window. Plan SHA256:
8bde861a58eb9aec45ab50e99683ca35291d142cbe2b1e68e48f40da166582a4;
retained locally as artifacts/common80_admission.json. A diagnostic-only
NamedTuple formatting error was corrected before this successful plan;
no training code, checkpoint or scientific setting changed.

Exactly one train_v19request was then accepted. The original native keeper
started supervisor1206189at00:16:03JST, pinned source
7dd6f3cb2a276bf40928d49f1e96a417f2fdec26. At00:21JST the joint arm's first
real update was1261/60,048visits/787,792input-frame visits on BOTH ranks.
AR supervision is active; AR/XYZ/bone losses are finite. LRposition is80,000,
backbone LR1.34638223e-5and head LR4.03914669e-5, preserving the existing
schedule. Native membership and16CPU enforcement remain verified; observed
GPU use was42,341/42,361MiB, peak process-tree RSS26,002,509,824bytes.

The70k/80k local reviewer must concatenate the verified60k prefixes from
train_v17joint/train_v18controls with train_v19suffixes, respecting old and
new source identities and cumulative counters. The existing controls reviewer
is60k-only; do not blindly pass70k or overwrite retained earlier reviews.
Do not deploy later local status/documentation commits during this payload.

## First-Stage Feedback Question Clarified By The User

On2026-10-29JST the user explicitly asked whether stage2 gradients through
the shared encoders/attention IMPROVE the first-stage skeleton output. The
question is not whether stage1 predictions feed stage2, and not merely
whether stage2 accuracy improves. It is currently UNMEASURED. Teacher-forced
AR loss and GT-reference stage2 scores cannot answer it.

Use three matched evaluation conditions: the original pretrained donor,
pretrained-stage2-only after training, and joint-pretrained at the SAME
endpoint. Stage2-only versus donor directly measures the effect of learned
shared features on the unchanged AR branch, conditional on an actual tensor
identity check of the nonshared generation branch. Joint versus donor is the
overall first-stage change; it cannot alone attribute improvement to stage2
because additional AR training also occurred. Joint versus stage2-only tests
the added continued AR objective under matched stage2 exposure. Do not
silently add a new AR-only training experiment to answer this question.

Reuse the existing free-generation evaluator, fixed128asset membership and
identical raw-frame/normalization/tokenizer/greedy settings. Generation must
not take a GT reference skeleton or GT token prefixes beyond the ordinary
BOS/class prompt. Measure original-GT edgeF1, generation success, joint-count
accuracy and matched geometry metrics, with failures retained in denominators,
paired asset intervals and the same examples. Donor results may be reused
only when checkpoint, input, target and decoding identities match exactly.
The old backbone-only checkpoint schema and the full shared-head checkpoint
schema need explicit strict extraction/loading; never silently skip keys.

Prepare this evaluation locally without touching the active train_v19source
or GPU workload. Execute only after its payload finishes and after separate
exact evaluation admission. Report improvement, unchanged performance or
degradation based on actual generated skeletons; no claim is available yet.

## Newly Authorized AR-Only Continuation

On2026-10-29JST, the user explicitly requested an AR-only randomized-frame
continuation AFTER the active experiment finishes, to compare first-stage
generation with the retained joint checkpoints. The user clarified that
30k/40k/50k/60k means TOTAL visits including the original20k, not additional
stage2visits. Therefore add only40kvisits/8fullpasses/840updates, matching
527,856new input-frame visits under the existing joint stream.

| Cumulative AR backbone visits | New comparison-phase visits | Retained joint checkpoint |
| ---: | ---: | --- |
| 30000 | 10000 | train_v17/reference_transport/checkpoint_010000.pt |
| 40000 | 20000 | train_v17/reference_transport/checkpoint_020000.pt |
| 50000 | 30000 | train_v17/reference_transport/checkpoint_030000.pt |
| 60000 | 40000 | train_v17/reference_transport/checkpoint_040000.pt |

Randomized means the established20%-40%/minimum2AR target-frame policy, NOT
random model initialization. Start from the same train_v9/fraction20_40_query
checkpoint_020000.pt, SHA
9704b23a194ca65836c6dfacd6637ca27542dc4b0836b1d3611185f0339cc888.

The user specifically conditioned optimizer inheritance on what the ORIGINAL
20k-to-joint transition actually did, not the later60k-to80k resume. Exact
source9db222d0070122b08c6d5dde3329f3be69d46c16loads the donor model AND full
optimizer state before adding the new head and before any update. The actual
train_v17initialization receipt records backbone_optimizer_retained=true and
head_optimizer_new=true. The latter only concerns the newly created stage2
head. Independently inspected joint60k Adam counters are1680for the encoder
and1260for the new head, consistent with420inherited backbone updates.
Accordingly the new AR-only control MUST retain the same backbone Adam
moments/counters/groups and starting per-rank RNG; it must not reset Adam.

Match the joint phase's new stream seed20261028and actual data/raw-frame
requests, randomized AR targets,2..24input poses, normalization, effective48
batch, physical24/12/6and backbone norm1clipping. Keep LRposition
20000+additional_visits on the original100kcycle. Do not continue the donor's
different old data cursor, restart warmup, or shorten the LRcycle to end at
cumulative60k. The treatment is removal of stage2forward/loss/optimization,
not another architecture or initialization change. Independently audit actual
consumed prefixes against the retained joint runs.

At each requested node, compare first-stage FREE generation on identical
assets/raw frames/normalization/tokenizer/greedy decoding, with original-GT
edgeF1, success, joint-count accuracy, clearly scoped geometry metrics and
paired-asset intervals/fixed examples. Failed generation is zero for
F1/success/count; never report zero geometric distance for failed cases.
This added matched AR-only control directly tests the incremental contribution
of stage2training to stage1, unlike a simple joint-before/after comparison.

This is explicit human authorization of a queued experiment, NOT exact
execution admission. Do not interrupt or overlap train_v19or deploy code
while it is active. After its complete built-in evaluations and clean exit,
prepare/test/profile a separately guarded exact AR-only configuration, then
admit and launch through the retained native grant. No new GPU allocation,
GPU release or AR-only training beyond cumulative60k is authorized here.
The follow-up is not complete until this new comparison is analyzed.

## Local70k/80k Reviewer Ready

On2026-10-29 02:07JST, preparation/review_controls.py was extended locally
to support70k and80k using the same invoke_review.ps1. It verifies the retained
60kprefix bytes/hash against the accepted60kmanifest, reads a stable resumed
suffix, and concatenates only the exact210/420updates required. Source changes
at update1261are explicit. Every step, asset/input-frame counter, effective
batch and resumed LRposition is checked; actual data/reference/LR/initialization
and metrics are still compared across all3arms by the existing analysis.

Ten targeted tests pass, covering all3regimes at70k/80k, legacy nodes, changed
prefix hashes/lengths, truncated records, reset/gapped counters, wrong source,
wrong AR switch/LRposition and incomplete terminal receipts. All6retained real
60k traces and the clean controls terminal receipts pass compatibility checks.
No historical comparison report was regenerated. Terminal readiness is checked
BEFORE retaining receipts, preventing incomplete mutable files from blocking
a later retry; reports are created exclusively, never overwritten.

A live read-only joint70k probe verifies both ranks at1,470updates/70,000visits/
915,016input frames, with first resumed update1261/60,048visits/LRposition80k.
Its preserved combined prefixes can be reused by the eventual full comparison.
This probe is not a3arm quality result. Evidence:continuation_live_probe.json,
preparation/continuation_tests_result.json and the two retained joint70k traces.
No GPU calculation, training change or remote source deployment occurred.

When all3arms reach70k, run invoke_review.ps1 -Samples70000 -Examples with
normal separate PowerShell argv tokens. At80k use -Samples80000 -Examples
-Terminal only after all built-in expanded evaluations and clean payload exit;
the script refuses a nonterminal80k review. The subsequent independent expanded
evaluation analysis and first-stage/AR-only comparisons are still required.

## Verified Common70k Comparison

At2026-10-29 04:16JST the independent local review completed and retained
review_controls_070000.json. Every arm has1,470updates,70,000asset visits
(14complete passes over the same5,000assets) and915,016input-frame visits.
The hashed60k prefixes plus210resumed updates are continuous on both ranks.
Actual assets, raw poses, reference poses, planned AR slots and LR trajectories
match; standalone controls apply no AR loss. All128selection assets are
disjoint from the consumed training identities. Both checkpoint copies have
producer hash verification and independently checked sizes; small artifacts
are independently hashed, not another full-weight rehash.

| Regime | Normalized MPJPE | PCK@0.05 | Edge F1 | Normalized J2J | Bone-vector error |
|---|---:|---:|---:|---:|---:|
| Joint pretrained | 0.00613435 | 0.99520149 | 0.90501139 | 0.00554239 | 0.00538451 |
| Pretrained stage2-only | 0.00576649 | 0.99543027 | 0.91415867 | 0.00518700 | 0.00514534 |
| Scratch stage2-only | 0.01435333 | 0.95532654 | 0.79730635 | 0.01138198 | 0.01080300 |

Each arm has1,792reference-target rows across128assets; metrics average within
each asset first, then weight assets equally. Normalized distances use the
target mesh bounding-box diagonal. Relative MPJPE reductions from60k are
9.03%/11.74%/12.92%, respectively. Pretrained stage2-only is about6.00%lower
than joint in MPJPE; joint-minus-pretrained paired95%interval is
[0.00021104,0.00051721]. Its edge F1 difference interval is
[-0.01262240,-0.00599688]. J2J and bone-vector intervals also favor stage2-only;
PCK includes zero. Intervals use2,000paired asset bootstrap resamples, NOT
multiple training seeds. Both warm-start arms still beat scratch across all
five intervals; the warm-start package includes weights AND Adam history.

The original fixed cases2and3 were rendered with identical input/raw frames,
view and scale, then visually inspected for readable labels and geometry.
Files:fixed_case_2_070000_v2.png, fixed_case_3_070000_v2.png and
visual_selection_070000_v2.json. Case2 uses reference25/target4; MPJPE is
0.00634978/0.00530701/0.01409289. Case3 uses reference39/target3; MPJPE is
0.00509075/0.00353924/0.00920715. These are fixed examples, not population
evidence or outcome-selected cases.

These scores concern stage2with a supplied GT reference skeleton. They do NOT
answer whether shared-encoder updates improve first-stage free generation.
The running common80k payload and remote source were unchanged. Do not repeat
this review or notification. Next inspect the full80k terminal state including
built-in enlarged evaluations, then complete the separately admitted stage1
feedback and AR-only work. No additional GPU load was launched by this review.

## Local Stage1 Feedback Components

On2026-10-29 05:18JST, added scripts/stage1_feedback.py and15focused CPU tests
in tests/test_stage1_feedback.py. This is a local library, not a launcher or
new execution permission. No running remote source or GPU workload changed.

The helper extracts the complete V-minus backbone from either the original
density checkpoint or an anchor checkpoint with an explicitly declared layout.
All keys/shapes/dtypes must match the target model. Weight checks compare every
parameter and buffer bytewise, separating surface/motion encoders from ALL
nonshared tensors. Requiring unchanged nonshared tensors will reject a changed
decoder, condition branch or auxiliary buffer; synthetic success does not
establish that the actual trained checkpoint satisfies this check.

Free generation takes only the canonical mesh-surface fields, rejects target
or reference fields, and uses the existing greedy decoder with BOS/class and
the existing vocabulary processor. It never invokes teacher forcing. Scoring
then uses original GT, the established nearest-joint edge F1 and symmetric
nearest-joint distance divided by the target mesh bbox diagonal. Distance is
not paired-ID MPJPE and no extra geometric alignment is performed. Failed or
invalid generations have zero F1/success/count accuracy and null geometry.
Aggregates weight assets equally and expose success-only geometry denominators.

Tests include actual tiny V-minus architecture/decoder execution on CPU with
no teacher forward, model-state/RNG mutation or CUDA initialization. The test
record/log are preparation/stage1_tests_result.json and stage1_tests.log.
Full evaluation still needs exact checkpoint/input/raw-frame/normalization
bindings, baseline-reuse identity checks, guarded distributed orchestration,
resource/capacity/deadline validation, paired intervals and fixed visuals.
The AR-only training implementation/configuration is also still outstanding.
Do not describe these component tests as a first-stage quality measurement.

## Common80k Completion And Replay Review

train_v19 finished at2026-10-29 06:23:20JST with exit0, passed native16CPU
resources and payload cleanup. The peak process-tree RSS was39,492,255,744bytes.
The original dual-H100 reservation remains; a06:27JST check verified idle keeper,
no approved-GPU compute processes, original cpuset/devices membership and clean
source7dd6f3cb2a276bf40928d49f1e96a417f2fdec26. Its old train permission is closed.
Terminal result SHA256:52b45b6108fb415fa972695d62dcd8360ff13e03abe54f5fce4470ad58eff4cb.

review_controls_080000.json independently verifies1,680updates,80,000asset
visits and1,043,152input-frame visits per arm, the full16passes over5,000assets,
actual matched input/reference/LR traces, initialization and AR switches,
checkpoint mirror metadata/sizes, and complete terminal resource receipts.
Fixed cases2/3were retained at80k and visually checked. Individual case errors
need not improve monotonically: case3pretrained and both scratch examples are
slightly worse than their70k examples despite improved aggregate MPJPE.

The new local read-only expanded reviewer passes9focused tests. It independently
checks all967validation assets,839outside the panel,32train assets in eligible
and reserved modes, all30,660rows/arm, aggregation, GT copy controls and reference
consistency. It reads NPZ array headers for the32train assets and reconstructs
their reserved8frame pools; all heldout inputs exactly match those pools and
never appeared in the actual80k training traces. Eligible targets were actually
consumed in99.7768%of evaluation pairs, references98.4375%; do not call every
eligible frame previously seen. The source read was header-only, not a dataset
copy. Producer per-regime science hashes are checked against regime_config,
separately from the overall experiment hash.

Observed full967metrics (NOT final acceptance while replay is unresolved):

| Regime | Normalized MPJPE | PCK@0.05 | Edge F1 |
|---|---:|---:|---:|
| Joint pretrained | 0.00620920 | 0.99290604 | 0.90529186 |
| Pretrained stage2-only | 0.00577176 | 0.99373965 | 0.91265314 |
| Scratch stage2-only | 0.01414770 | 0.95677643 | 0.80086366 |

review_expanded_080000.json is complete as a diagnostic record but passed=false:
the final128panel has the SAME raw-frame metadata and EXACT GT-copy metrics,
yet head predictions differ from the inline80k evaluation on1,792/1,792/1,680
rows, respectively. MPJPE asset-mean deltas are-2.1786e-7/+1.9483e-6/-1.1311e-6;
max per-row edge-F1changes are0.17143/0.09148/0.09524. Do not silently replace
an exact replay check with a tolerance or call this a training failure. Surface
sampling, checkpoint reconstruction, and numerical precision remain hypotheses.
All raw evidence and partial audits are retained; no inference was rerun yet.

A separately guarded capacity-profile diagnostic is prepared locally in
anchor_checkpoint_replay_20261029_v1.json: the SAME pretrained-stage2 80k
checkpoint, original fixed cases2/3, two fresh model loads, two independently
prepared inputs per load, BF16 and FP32(no TF32), at most900seconds, ZERO
optimizer updates. It compares complete backbone/head hashes, input/feature/
prediction hashes, original query points/GT/raw frames, repeat/reload differences
and GPU RNG. Its configuration cannot enter train/audit mode. Thirty-four CPU
tests including integration regression and stage1 components pass. Execute only
after committed exact state, fresh native resource checks and a recorded dry-run.
No AR-only launch or acceptance of first-stage quality is implied.

At06:52JST the exact live plan passed on idle native130675914, both cgroups,
16CPU slots and the original GPU UUIDs. Read-only checkpoint inspection ran
separately under a four-CPU affinity/one Torch thread with CUDA hidden; it was
not represented as scheduler membership. Actual donor versus stage2-only80k
backbone comparison found500changed tensors among508shared, and ZERO changed
tensors among447nonshared parameters/buffers. The latter establishes the
unchanged generation-branch weights needed for the feedback interpretation,
but says nothing yet about generated-output improvement.

The plan is preserved as artifacts/replay_v20_admission.json, SHA256
01bf31610af1ea498cc37b2fe82e7056d1370ec92de36083c202a089d91264be.
Exactly one capacity_v20 request was queued at06:53:06JST using source
4574c35251df17e7ed6f00178eb743e91ad09cbf and config SHA256
37276f649cf196ccb1e56f9440b628db34f1a431f5d539ba63c42253ee64b074.
The diagnostic has a separate22:30UTC payload cutoff, without shortening the
retained allocation. Use preparation/observe_replay_v20.sh, not the old training
observer (whose experiment-name filter excludes this new diagnostic config).
No subsequent local edits may be deployed while this payload is active.

## Replay Result And Scoped80k Review Resolution

capacity_v20 completed at2026-10-29 06:58:02JST with exit0, passed resources/
cleanup,16native CPU slots, and ZERO optimizer updates. Peak process-tree RSS
was25,032,437,760bytes and peak reserved GPU memory was16,378,757,120bytes/rank.
All receipts and predictions are hashed and retained by replay_v20_collection.json;
result SHA256:3d1abc14b271124cf958fc992fb76e21572aa05562d38aad23ce7756a18ea6bb.

For both original fixed cases, all complete surface-input hashes are identical
across independent preparation and two model loads. Backbone/head hashes are
identical; GT/raw frames and retained query points match; CUDA RNG does not
change during inference. Nevertheless SAME-LOAD BF16 replay has maximum XYZ
change0.00390625 in selected-input-normalized model coordinates, before dividing
by any target bbox diagonal. FP32/no-TF32 maxima are8.94e-8/1.19e-7for same-load
repeats and1.79e-7for reloads. Feature hashes also vary in both precisions.
Thus numerical computation is a demonstrated precision-sensitive source of
variation; checkpoint reload is not necessary to trigger it. This does not
identify the individual CUDA kernel or prove a universal bound over967assets.
FP32 is NOT bytewise exact, and BF16-versus-FP32 differences are not model gains.

The original expanded report with passed=false is preserved, not weakened or
rewritten. review_common80_resolution.json records the resolution: execution,
data, paired inputs, independent aggregates and strict reserved-pose checks are
accepted; existing uniform-BF16 aggregate stage2comparisons are reportable with
the numerical-repeatability caveat. Perfect per-frame score equality is not
claimed. This is not overall model-quality acceptance or stage1 evidence.

On967assets, pretrained-stage2 MPJPE is7.0451%lower than joint. The joint-minus-
pretrained95%paired-asset interval is[0.00037771,0.00049738]; edge F1difference
is-0.00736129with interval[-0.00871651,-0.00602390]. All5metrics favor pretrained
stage2on967and839, including PCK; the latter was inconclusive on the128panel.
839are outside the selection panel, not historically untouched test assets.
On32training assets, heldout-pose MPJPE is0.00496330/0.00461038/0.01086734;
these are new poses of known assets, not unseen-asset generalization.
Permuting anchor correspondence at the head interface increases selection128
MPJPE to0.177990/0.161139/0.241177from0.005849/0.005467/0.013556. This verifies
dependence at that interface, not a whole-model causal or topology-generation
claim. All metrics use supplied GT reference skeletons.

Both finished execution gates are closed; do not rerun or restart them. The
required next work is the already human-authorized AR-only continuation, SAME
20kdonor+Adam/RNG, only+40knew visits, cumulative30/40/50/60k compared with joint
additional10/20/30/40k. Its implementation and exact admission remain pending.
Keep training BF16 and the original100kLR horizon. Prepare uniform FP32/no-TF32
stage1 free evaluation across donor/joint/stage2-only/AR-only, with a full AR
generation capacity check before admission. Existing BF16 generation artifacts
are not automatically identity-matched baselines. generate_asset now exposes
an explicit precision argument, preserves the prior default, and passes17CPU
component tests(36including integration/replay regression). No actual stage1
free-output quality comparison or new AR-only training has been executed.
# AR-only Execution Followup, 2026-10-29

Implementation source: 810366fb3bf75d1e8a533fc8355c76f126dc859d.
Local validation: 62 CPU tests passed, including complete donor Adam restoration,
the two-rank40k prefix against the original60k plan, exact840updates/527856input-frame
visits, actual trace mismatch rejection, and paired failure/geometry denominators.
Only the separate CPU audit_v21 is admitted initially; no GPU run is implied.

The original OPT uses FlashAttention2 for BF16 training. All11 first-stage comparison
checkpoints instead use the same explicit FP32/noTF32 eager-attention evaluation
constructor with identical parameter keys and architecture. The training backend
is unchanged. Profile the complete2048-token decoder separately before admission;
the capacity-only forced non-EOS output is never scored as quality. Normal scoring
uses unmodified greedy EOS behavior, mesh-only inputs and BOS/articulationxl.
Compare donor, joint+10/20/30/40/80k, stage2-only+80k, and AR-only+10/20/30/40k.
No historical BF16 generation result is reused. Match input/GT/normalization hashes
per asset/frame and report paired-asset intervals with common-success geometry.
