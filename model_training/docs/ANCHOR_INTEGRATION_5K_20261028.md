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
