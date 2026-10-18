# Isolated virtual-frame fusion experiments

This is the active experimental contract on branch
`codex/virtual-fusion-20261013`, based on committed source `19d81ba`.
The older checkout, unpublished working changes and training outputs are
read-only references, not execution destinations. GPU authority remains in
`model_training/state/current.json`; this document is not launch permission.

## Current direction: normal 5000-asset trend comparison

### Revised approved sequence: A150000, B80000

2026-10-19 01:11JST user explicitly confirmed revised budgets and lifted the
temporary promotion hold. This supersedes the earlier unlaunched B50k plan:
1. Finish the already running A120000-to130000 stage at3e-5 and review it.
2. Resume FINAL A130000 to150000, keeping3e-5 and full optimizer/RNG state.
3. Resume FINAL B40000 to60000 at1e-4, then FINAL B60000 to80000 at3e-5.
4. Freeze each route's best checkpoint selected on the unchanged128-asset
   validation panel, then compare them on the remaining validation assets
   not used to select checkpoints (expected839 of967), eight views each.
5. Preserve qlogin130258195 after training and evaluation.

Preserve AdamW moments/betas and rank RNG/sampling state at every transition.
No model, loss, tokenizer, training membership, RT/normalization, decoding,
warmup or scheduler restart changes are authorized. A/B cumulative budgets
remain different; this is not a matched-budget architecture-superiority proof.
Future stages are authorized but require exact prepared and reviewed configs.
B40000 and A130000 have completed and been accepted; only A130000-to150000
at3e-5 is executable in the current state. Preserve both final and best checkpoints.
The immutable revised plan is `normal_ops/sequence_20261019_plan_v2.json`,
SHA256 `c74106ad0e6146febb228ff84e87de6716e60f818fde77f5820e421e06fc07fc`.
New A150/B60/B80 stages need exact pinned configs/checkpoints and tests; do not
extend a running process in place or deploy local changes while it runs.

A130 terminal01:36JST2026-10-19: exit0,2730updates/130000assets/1685592frames,
210suffixrows eachrank. Primary verified16 terminalhashes and both1536-view
125/130k panels, same originalAselection. Valid F1 at130k0.733665703 is A's
best so far, success0.989257813, exactcount0.452148438. Final fullcheckpoint
8438115760bytes, SHA256
`d1c041e055e8be507137d0851fb137b29244a5f0bf604bc0bddc2f4f309b8676`,
sourcea189013. Completion review SHA256
`49dd74b60ea429bd090e4a2b243982a660e6492291a4b42fbff369f134440bbd`.

A130-to150 configuration SHA256
`17b99060f97158009091f1c47f447c6af5ac5fee14a62241e64c74d92e7ac1a3`,
core5e38742,53 CPU tests passed. It preserves source AND continuation LR3e-5,
full optimizer and40/80/120/130 history; adds420updates/264272frames, ending
3150updates/1949864frames. Evaluate135/140/145/150k. New plan SHA256
`70ac110617d1d84b935c6aeaaa4ed95776533159638591a673c5f048aa0021e1`.

A150 launched02:05:04JST2026-10-19 with frozen runtimec5f0216. First8
primaryaccepted02:14JST:2731..2738/130384assets/1690344frames. Both ranks
restored947AdamW states/betas and sourceLR3e-5; actual A130 tensor fingerprint
`0c0acb49fcecc8e95b2b5ec10c792aaf7094f1cbc72f2172489e56b4f2eac305`.
Primary rehashed10startup artifacts and accepted28strict checks. Startupreview
SHA256 `3f2c19fec1fada464a58abba6c1afee78fcf88715ecd685e856b359c91e51ba0`.
The successful first8 runner is v2. V1 used an obsolete in-memory1e-4 restore
expectation; a monitor-only correction and41CPUtests passed before v2.
No GPU restart or remote code mutation occurred. Full supervisor PID119932 is
active: at02:16:47JST bothranks2745/130720assets/1694328frames, noerrors.
Startup acceptance does not imply training completion or model-quality acceptance.

B40-to60 at1e-4 is CPU-prepared in localcode50df272,55tests passed, configSHA
`92e59e3b73a0637785928f18b59f3bda42c86ef144d5fd832933ec5b84ce93a3`.
It preserves FINAL40k optimizer/RNG and the full sampling prefix, then adds
420updates/260008frames to1260updates/788304frames; evaluate45/50/55/60k.
Preparation receipt SHA256
`381d3904b949bdb6147d3f50b9b2c035cff189772560a869339ba0a39471b352`.
This is local preparation only: do not deploy while A150 runs. B60 requires
A150 terminal review, fresh idle/resource checks and a pinned stage launcher.
B60-to80 and expanded-validation execution remain to be implemented.

A120k acceptance:840 suffix updates,22 terminal hashes and all eight1536-view
evaluations verified, with per-view F1/success recomputed against original A's
unchanged panel. Final120k is also best: train0.856699,heldout0.822426,
valid0.713389, still below B20k0.729074. Final full checkpoint8438113456bytes,
SHA256 `e3e8a8569641005c5798d9974ecf011f388987c0bfb72d47b2f6ebd736437628`.
Primary acceptance receipt:
`D:/evoweave/server_ops/outputs/virtual_fusion_20261013/normal_ops/reviewed_normal_a_extension120000_completion_v1.json`
SHA256 `2193f3dc8fdb693db994a5b0cff5e0146717ac3ba002726df1872b22f0b06d66`.

B40k preparation passed36 focused CPU tests. B's original OneCycle tail was
about2e-6; the explicitly selected continuation protocol raises LR to1e-4,
as A20k-to40k did, while preserving optimizer moments/betas. This intentional
LR change is not an optimizer reset. All later low-LR stages use3e-5.

B40 launched16:52JST on2026-10-18 with frozen remote source5703f2c. At17:01JST,
both ranks passed the first8 updates421..428, reaching20384 assets/267592
supervised frames. Actual restored model fingerprint is
`ecaf772173cd255e7e56011967915fd76baa95a3bd27deee8ba4818b13014c76`,
identical to incoming B20 tensors on both ranks;947 AdamW state entries and
betas(0.9499999512612207,0.999) were retained. Full B40 supervision is approved;
this accepts startup integrity, not completed training or model quality.
Primary review `normal_ops/reviewed_normal_b_extension40000_startup_v1.json`
has SHA256 `f1ad62b786f291bc440cd9def3c47307830f19677ad91514a45dde0caf3b10cf`.

A120-to130 at3e-5 is prepared in codee1d2f9a, with49 CPU tests passed,
config SHA256 `584f9b709f7c9b993a5f78f38a1318bca5871883dc16add911073aa26dcdd325`.
It preserves the full120k sampling prefix and adds210updates/130088frames,
ending2730updates/1685592frames; evaluate125k and130k. It launched23:00:36JST,
with frozen remote sourcea189013. Primary accepted first8 at23:09JST:
2521..2528/120384assets/1561264frames,947AdamW states/betas restored, LR3e-5,
actual A120 tensor fingerprint
`f5314b59215aca744803f1046302fab4323041815c3f4e6130d3928b733247b1`.
Startup review SHA256
`93b257f34a8764673e2bcfef9adec342eb8cc8463d281dc129c5d7be721427ba`.
The successful local monitor runner is first8_v2, not failed preflight_v1;
no GPU training restart occurred. A130 full supervision is authorized.

B40-to50 at3e-5 was CPU-prepared in localcode2888186, with51tests passed and
config SHA256 `acc9b1ccdb85d47ef380bc092e07e903189d6fcf8f7f65324025acd085d0c92b`.
It adds210updates/128760frames, ending1050updates/657056frames; evaluate45/50k.
It was never launched and is now SUPERSEDED by the B40-to60-to80 sequence.
Do not activate its config. Expanded-validation execution remains to be implemented.
The30-minute heartbeat `a-b` follows the approved sequence, stays quiet on
unchanged healthy progress, and must never duplicate the active supervisor.
Current execution tracker is
`D:/evoweave/server_ops/outputs/virtual_fusion_20261013/normal_ops/sequence_20261018_execution_v1.json`.

B40 terminal22:52JST on2026-10-18: exit0, both ranks840updates/40000assets/
528296frames,420suffix trace rows each,18 local terminal artifact hashes and
all four1536-view evaluations verified. Primary recomputed per-view F1/success
and confirmed original B panel equality. Valid F1 at25/30/35/40k is
0.712425/0.728128/0.734596/0.722256. Best35k improves over20k0.729074 by0.005522;
final40k is lower, with success0.919922 and exactcount0.500977. Continue from
FINAL40k after A150 under the revised sequence, not best35k. Final full checkpoint8438108720bytes,
SHA256 `4e48925a44b0ec78d0b88a3a4169823d70d6bce9068e0385cef9e4d3447c0928`,
source5703f2c. Completion review SHA256
`2e852bcbf18ca37981567dd52015565ea447ebf91009388ea11dbd9404c0f09e`.

### Historical A80000-to120000 authorization

2026-10-18 03:22JST user explicitly requested release of the old two-H100
qlogin, a replacement two-H100 qlogin, automatic continuation of A toTOTAL120000,
and retention of the replacement allocation after completion. Old129839822
was verified idle and released. New130258195 was granted03:26:02JST on the same
pcg02i physical GPUs5/7, with a fresh96h limit. Only the allocation identity
changes; the source is final80000, not best75000. Preserve full model/AdamW,
RNG and sampling state, constant1e-4, fixed5000-asset pool and1536-view panel.
Prepare and validate the exact continuation before training; do not restart,
reset heads or change the learning-rate schedule. Retain the keepalive after
the120000 checkpoint and final evaluation, within the scheduler's96h limit.

The pinned continuation adds840 updates and513672 supervised-frame exposures,
ending at2520 updates/1555504 frames. New plan SHA256:
`26ff999633011ec96e21b0015e671e18c654a92c9e0b5a7bfe1cc7cc18df4a70`.
Its full80000 prefix is unchanged; first new update is1681/80048assets.
Code129f713 passed34 continuation/gate/stream/training CPU tests. Reuse the
accepted A capacity receipt because architecture, batch/precision policy,
node and physical devices are unchanged. Require at least24h remaining at
launch; the fresh allocation provides96h. Independently verify first8 updates
and restored80k model/AdamW state before full supervision. This does not
authorize automatic B continuation if A later exceeds B; report that result.

### Historical A80000 completion

2026-10-18 completion: A reached80000 exposures/1680 global updates/1041832
supervised frames and terminal exit0, verified at01:13JST. All840 additional
updates retained1e-4. Both full traces,22 terminal artifact hashes and all eight
new1536-view evaluations passed review. Per-view F1/success were recomputed;
the evaluation selection exactly matches original A. This accepts execution
integrity, not formal model quality or a matched-budget comparison with B20k.

| Asset exposures | Train-pool F1 | Heldout-pose F1 | Validation F1 |
| --- | --- | --- | --- |
| 40000 | 0.482980 | 0.481394 | 0.421903 |
| 45000 | 0.561562 | 0.557844 | 0.498740 |
| 50000 | 0.656981 | 0.626412 | 0.588187 |
| 55000 | 0.683880 | 0.665054 | 0.625416 |
| 60000 | 0.704813 | 0.673164 | 0.644832 |
| 65000 | 0.720171 | 0.680320 | 0.641813 |
| 70000 | 0.758991 | 0.749522 | 0.661098 |
| 75000 | 0.751346 | 0.742180 | 0.680518 |
| 80000 | 0.725442 | 0.733758 | 0.645741 |

Scores are original continuous-GT edge F1, failed generations scored zero,
on the fixed1536-view panel, not the full validation dataset. Best validation
is75k,0.048556 below B20k0.729074; final80k is0.083333 below B. Validation
generation success/count exactness decline from96.78/40.43percent at75k to
93.95/36.62percent at80k. This does not identify an overfitting mechanism or
an absolute architecture ceiling. The user's conditional B extension required
A to surpass B; that condition was not met, so no B continuation was launched.

Both75k and80k checkpoints are8438111024bytes and remain preserved. Full SHA256:
- 75k: `7aac9118eef587e4aad313bb35d69319977fe6d98973c748c8f70fab4c61ac12`
- 80k: `e3ec08bd177b583874c85b33f8d921ef1e4fbd3902533320b88bae56571bb600`

Primary completion review:
`D:/evoweave/server_ops/outputs/virtual_fusion_20261013/normal_ops/reviewed_normal_a_extension80000_completion_v1.json`
SHA256 `a0ad0947eac52533f68e661308ae607efb129feb5e56477c6909696e509aafd8`.
At02:25JST fresh checks confirmed clean frozen runtime21f3a2e, launcher and both
ranks gone, GPUs5/7 idle, and allocation129839822 preserved. This local/GitHub
completion metadata is not deployed over the completed runtime. Only inspect
and report remain allowed; no new GPU work, architecture/head initialization
changes, or allocation release without a new user decision.

### Historical A40000-to80000 authorization

2026-10-17 13:36JST user authorization: continue A's own40000 full checkpoint
to TOTAL80000, adding40000 exposures at the existing constant1e-4. Restore
AdamW moments/betas and rank sampler/RNG; no new warmup, decay, model/data
change or initial evaluation replay. Plan SHA256
`f561a7d8432792920292cc1cf06f74858872b06173187d9ac3f132efdf72b7f2`
has1680 total updates and1041832 supervised frames (840 updates/513536 frames
additional). Save/evaluate at45000 through80000 every5000, using the same
fixed1536-view panel. Prepared code2c0e182 passed42 CPU tests, including two
successive checkpoint restores, plus6 exact-plan continuation checks.

The user explicitly accepted possible allocation expiry before completion.
At13:37JST the existing twoH100 grant had62091.5seconds remaining. Observed
timings imply51552.9seconds at average speed, versus68289.3seconds under a
conservative per-update-tail/evaluation allowance; the latter is not a95percent
confidence bound for total runtime. Launch requires freshremaining>=51553seconds,
idle GPUs5/7 and all pinned source/checkpoint gates. This admission uses the
point estimate only under the explicit user risk decision; completion is not
guaranteed. Preserve every checkpoint and qlogin. A same-target partial restart
after expiry needs a separately verified saved-checkpoint binding and is not
automatic. No replacement allocation is requested now.

At launch this operation superseded the completed/report-only40k operation below,
which is retained as historical evidence. Actual40k restored tensor identity,
optimizer state and dedicated first8 checks must pass before full supervision.
No D/E/F, dataset expansion or formal80016 promotion is authorized.

### Completed A40000 evidence

2026-10-17 completion: A continuation finished with exit0 at06:09JST, exactly
40000 asset exposures,840 global updates and528296 supervised-frame exposures.
The420 additional updates all used1e-4; the original model, AdamW moments/betas,
sampler/RNG state and20k prefix were restored and verified. No initial evaluation
was replayed. All four new fixed1536-view evaluations were preserved and
recomputed from per-view records; panel selection exactly matches original A.

| Asset exposures | Global updates | Train-pool F1 | Heldout-pose F1 | Validation F1 |
| --- | --- | --- | --- | --- |
| 20000 | 420 | 0.324185 | 0.304345 | 0.282056 |
| 25000 | 525 | 0.304480 | 0.309754 | 0.282285 |
| 30000 | 630 | 0.333847 | 0.319775 | 0.285100 |
| 35000 | 735 | 0.343242 | 0.348920 | 0.299811 |
| 40000 | 840 | 0.482980 | 0.481394 | 0.421903 |

These are original continuous-GT topology F1 scores with failed generations
scored zero. Training uses the same5000 assets; the fixed evaluation panel has
32/32/128 assets with8 views each, not the entire validation dataset. Final
generation success is238/256,244/256,955/1024. Validation improved by0.139847
from20k (49.58percent relative), mostly in35k->40k. The25k near-plateau did not
establish saturation. This continuation changes both training exposure and the
learning-rate policy, so it does not isolate either cause. Execution integrity
is accepted; formal model quality and a matched-budget A40k-vs-B/C20k claim are not.

Final checkpoint:8438108528bytes, SHA256
`8725e36d17733d56dbb76242f5a5ab5a657ee51a4f4be1ed8392e190579d6f13`.
Final evaluation summary SHA256:
`be535d882c0b84a2d7caf6802a923f4153e5b91bc6e926bd861b571a7e3bcf24`.
Primary completion review is
`D:/evoweave/server_ops/outputs/virtual_fusion_20261013/normal_ops/reviewed_normal_a_continuation_completion_v1.json`
(SHA256 `4c454f1da947b5e94129aa1974fa94d8d9cfc5210b506961d5e3821a85adb7c5`).
At completion the HGC runtime was frozen at6715a2b. Fresh checks confirmed both
training ranks and launcher gone, GPUs5/7 idle and allocation129839822 retained.
That continuation is finished. Its completed/report-only operation is preserved;
the separate80000 operation above is the only new training authorization.

### Historical continuation authorization

2026-10-16 user revision: finish the current C20000 run, then SKIP D.
Resume the preserved A20000 full checkpoint to TOTAL40000 exposures, adding
20000 with constant learning rate1e-4, no new warmup or decay. Restore AdamW
moments/betas and rank sampling/RNG state; do not initialize from B/C/profile
weights or restart A from official weights. The model/data/physical batch
policy are unchanged. Keep the original20000 exposure prefix exactly and add
evaluation/checkpoints at25000/30000/35000/40000 on the same1536-view panel.
The resulting plan has840updates and528296supervised-frame exposures, with
264688 new frame exposures. This is a continuation with an explicitly changed
learning rate, not a matched-budget comparison against B/C20000.
C completed on frozen cf1c1ae with exit0,20000exposures/420updates/263608frames.
All five1536-view evaluations, full two-rank traces, actual source identity and
final checkpoint SHA were reviewed. Final original-GT F1 is0.337525 train-pool /
0.295678 heldout poses /0.279036 validation. This accepts execution integrity,
not model quality. The final C checkpoint SHA is
c502729b683e5696d4f64f5be3d464a5d2c899a3afef357538af45c5b42e810c.
Prepared continuation commit455793a was merged into this authorized
branch. At launch, state released only the A continuation entrypoint, bound to C's exit0
result, the original A full checkpoint and the accepted A capacity receipt.
Admission requires at least96978seconds remaining: accepted A worst-case
microbatch timings over the exact new420-update plan,25percent training margin
and10h for four evaluations/checkpoints/full-state restore. This26.94h threshold
is conservative admission accounting, not an ETA. Both granted GPUs must be idle.
The first new update must be421/20048exposures at1e-4. Verify actual model and
optimizer/RNG restoration plus the dedicated continuation first8 monitor before
routine supervision; the historical official-initialization hash is not the
resumed model's weight hash. No D/E/F or additional experiment is authorized.
The checkpoint was saved before its20k evaluation; restore that saved RNG
state without replaying the old evaluation. No bitwise GPU equivalence to a
hypothetical uninterrupted40k run after that evaluation is claimed.

2026-10-16 C transition: B completed20000 exposures/420 updates/263608
supervised-frame exposures with exit0 and five complete1536-view evaluations.
Its final original-GT F1 is0.766583 train-pool /0.786313 heldout poses /
0.729074 unseen validation, with generation failures scored zero. B's full
terminal evidence and checkpoint are preserved; execution integrity is accepted,
not formal quality. Legacy has no15000/20000 node for a matched comparison.
C's independent dual-rank capacity profile completed with all9cases accepted,
exit0, finite gradients/losses and a maximum46.5353GiB allocated,58.77percent
of device memory below the85percent cap. Primary review is SHA-bound in
reviewed_capacity_c_v1.json (5e31b951...); training is now released only after
a fresh remaining-grant check against113348seconds. This conservative allowance
uses worst stress timing plus25percent and8h evaluation/checkpoint overhead,
not an observed normal-data ETA. C uses the existing
relative_position mode: A's virtual-query fusion plus per-head relative input
anchor-coordinate bias, not a new residual/depth intervention. Its explicit
pairwise bias intermediates motivate conservative per-rank12/6/3 batches for
T2-6/T7-12/T13-24, with2/4/8 accumulation to retain effective48.
The full exposure/frame plan, supervision and evaluation remain identical.
The nine capacity cases still cover all three longest-main assets, the pinned
geometry maximum and longest auxiliary target at each bucket maximum.
C's full initial fingerprint includes its added bias parameters; only the
shared fingerprint must equal A/B, and full fingerprints must match across ranks.
Fresh C training must match full initial fingerprint e1bf55a0... on both ranks
and shared a89b29c9..., without loading profile or prior-route trained weights.
The initial1536-view evaluation precedes all updates; audit the first8updates
against the same asset/seed/frame plan before routine milestone supervision.

2026-10-16 update: A completed all20000 exposures,420 updates and263608
supervised-frame exposures with exit0. Both complete traces are finite,
initialization/exposure identities match, and all five evaluations are saved.
Its original-GT F1 at20000 is0.324185 train-pool /0.304345 heldout poses /
0.282056 unseen validation. These are the same32/32/128-asset panels, eight
frames each, with generation failures scored zero. Execution integrity is
accepted, not architecture quality. Legacy has no15000/20000 comparator;
at the matched10000 node A validation F1 was0.192411 versus legacy0.690604.
Preserve A's final checkpoint and complete terminal receipt. B is next, using
real queries / virtual memory / real residual with otherwise identical runtime.
Its independent two-rank9-case capacity profile completed with exit0; both
ranks passed finite execution and the85-percent allocated-memory limit.
Maximum allocated memory was60.5701GiB. Primary review accepts the capacity
receipt `reviewed_capacity_b_v1.json`; B training is now released with the
same24/12/6 per-rank microbatch buckets and1/2/4 accumulation counts as A.
The minimum remaining-time check is89890seconds, based on stress-case timing,
25percent training margin and8hours for evaluation/checkpoints, not an ETA.
Always refresh the grant allowance immediately before the B launch.
Do not apply the discussed residual removal, fusion-depth changes or new
frame-budget pairing to this comparison. Report original-GT F1 as the primary
metric; label experiment size, exposure milestone and panel on every table.

2026-10-15 update: the user stopped the legacy run after10864 exposures
(228 updates); its complete10000 checkpoint and evaluation are preserved.
The latest instruction explicitly defers new frame-budget/batch-pairing
features to a separate later study. This comparison keeps random2..24 inputs
and effective48 assets, not a fixed1152-frame update budget. A starts fresh
from the same official tensors, with the same exposure plan as legacy.
Its completed two-rank9-case capacity confirmation passed with exit0 and
peak allocated memory60.57GiB. Per-rank microbatches are24/12/6 for
T2-6/T7-12/T13-24, accumulated1/2/4 times to keep global effective48.
The reviewed receipt is `reviewed_capacity_a_v1.json` in the normal job
directory. Later B/C/D still need route-specific capacity approval.

The user's2026-10-14 correction supersedes the fixed-small-cohort promotion
sequence below. Fixed8/fixed64 runs are diagnostic history, not a substitute
for normal online training and not an additional prerequisite for this stage.
The legacy64/T4 reference completed all512 updates and its final evaluation
at2026-10-14T12:46:58Z with exit0. Preserve its outputs and allocation129839822.
The user explicitly approved starting the normal sequence at2026-10-14T13:16Z:
legacy -> A -> B -> C -> D, using both physical GPUs5/7 for ONE route at a time.

Use the ALREADY PREPARED immutable5000-row subset, SHA256
`a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`.
Its original rows and root_rt metadata are preserved, with4397 base/603 addition
assets and no validation overlap. Data selection/screening is complete: do not
repeat NPZ censuses, rebuild RT, or introduce another data adapter. The focused
RT consumer review and12 existing CPU tests passed on2026-10-14.

Every training visit selects a fresh input count T uniformly from2..24 and
resamples poses from ALL non-heldout frames. T is shared within each global
optimizer update so both DDP ranks and every route use the same shape schedule.
The existing mesh-only motion-FPS/random pose selector, root-RT removal,
selected-input mesh-union normalization, trackable surface sampling and
all-input-frame main supervision remain unchanged. There is no fixed-pose or
learned-feature cache. Exposure-index seeds make asset order and pose/surface
draws independent of the physical microbatch policy.

Compare legacy query-first, A, B, C and D from the same official initial shared
tensors. Preserve effective batch48, BF16, OneCycle peak1e-4/warmup.1/div5/
final10, weight_decay0, clip1. A/B/C/D retain auxiliary weight.1; legacy has no
auxiliary task. Every route receives EXACTLY20000 asset exposures: four
shuffled visits to each of5000 assets. Evaluate at0/5000/10000/15000/20000.
Each5000 segment is one visit per asset:104 updates of48 plus one update of8,
for420 updates total. The old80016 full-dataset budget does
not apply to this trend comparison. No fit-to-perfect stopping criterion.

Use a deterministic random panel of32 training assets and128 independent
validation assets. Training-pool evaluation uses eight non-heldout poses per
asset; these are not necessarily the exact input sets seen during optimization.
Heldout evaluation uses ALL eight reserved poses for the SAME32 assets. Each
validation asset contributes eight poses. Thus every checkpoint has1536 main
generations (256/256/1024); auxiliary evaluation is once per asset/group only
for A/B/C/D. Failure-as-zero F1 includes every generation; conditional J2J reports
its denominator. This panel monitors trends; it is not the full967 validation
set or a statistical generalization guarantee.

The completed A single-GPU profile measured T24/B6 at55.48GiB and T24/B8
at70.03GiB (above the85-percent limit). It is not a legacy/DDP capacity receipt.
Before training, require two-rank coverage against long-main-token and geometry
extremes, plus longest auxiliary targets for A/B/C/D; include resident gradients
and optimizer state. Use a few physical-batch buckets: profile the maximum T
in each bucket at its actual B. Shorter T is covered only at the SAME B and for
identical stress-asset membership from the existing pinned selection. This
avoids retesting every shorter tensor shape without changing any training input.
Keep at least15 percent measured allocated-memory headroom.
Report physical batch, accumulation, memory and time, not just effective48.
Run each route on physical5/7 with DDP. Profiles never
supply trained weights. Preserve allocation129839822 and check full runtime
plus safety margin before every launch.

New source: `/work/wangyy/virtual_fusion_20261013/normal_code`.
New jobs: `/home/wangyy/jobs/virtual_fusion_normal_20261014_v1`.
New outputs: `/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014`.
The legacy64 source remains frozen at2658d6c in its existing `code` directory.
`train_virtual_fusion_normal.py` reuses the established training/loss loop;
its bounded normal5000 authority is separate from full-data formal release.
Do not launch another full-data80016 run or use historical small-fit gates.
Only fusion-specific C/D parameter additions are excluded from the common
initialization fingerprint; record the actual full fingerprint separately.

## Preserved model and data

Keep the retained-root RT manifests (18002 train / 967 valid), selected-input
mesh-union normalization, 1024 virtual/real anchor tokens, 96 internal
registers and 12 spatial/temporal layers. Keep the complete official UniRig
XYZ autoregressive decoder, original vocabulary and all-frame supervision.
The auxiliary DFS-ID decoder consumes final virtual tokens and contributes
only a loss; its generated structure, length or hidden state never enters
the main prediction path. Fresh trials load official checkpoint SHA
`d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4`.

## Isolation and authorization

- Local code: `C:/Users/namew/.codex/worktrees/virtual-fusion-20261013/evoweave`.
- Local operational artifacts: `D:/evoweave/server_ops/outputs/virtual_fusion_20261013`.
- HGC source/cache/temp: `/work/wangyy/virtual_fusion_20261013/` on verified SSD.
- HGC jobs: `/home/wangyy/jobs/virtual_fusion_20261013_v1`.
- HGC outputs: `/home/wangyy/evorig/outputs/virtual_fusion_20261013/`.
- Initial qlogin129547842 (physicalH100s5/7) completed both fitting runs,
  then expired at the scheduler96h hard limit on2026-10-13 18:37JST.
  Replacement job129839822 was granted on pcg02i at2026-10-14 06:52:02.966JST,
  exactly twoH100s/mem64G, physical5/7 verified by UUID on this NEW node.
  The new /work is verified NVMe RAID0/XFS. Preserve its keepalive; the96h
  limit expires at2026-10-18 06:52:02.966JST. Before every long launch,
  refresh live remaining walltime and require full duration plus safety margin.
  The expired pcg01i grant is never an execution destination.
- User authorized GitHub publication of code/config/result summaries only.
  No datasets, weights or credentials. Every output is exclusive-create.

## Comparison sequence

First audit the existing auxiliary-weight .1 step2048 checkpoint read-only:
greedy auxiliary generation from BOS, natural EOS, exact count and DFS-parent
tree, plus deterministic cross-asset virtual-condition swap. Verify the
checkpoint SHA and unchanged parameter state; perform zero optimizer steps.

Round one is A (`virtual_queries_real`) versus B (`real_queries_virtual`).
A preserves original virtual Q, real K/V and virtual residual; B uses real
Q, virtual K/V and real residual. Other modules and initial tensor bytes match.
Only if neither resolves the problem, round two compares C
(`relative_position`) and D (`token_film`). C adds zero-initialized per-head
anchor-coordinate delta bias, MLP 3->64->8, to A's attention logits. D uses a
same-index virtual MLP 1024->256->2048 to modulate normalized real tokens,
preserving the real residual. Neither consumes GT positions in fusion.

## Training and acceptance

All candidates: seed20261013, auxiliary weight .1, fresh official weights,
same fixed diverse8/T4 inputs, effective batch8 via asset gradient accumulation,
1024 updates = 8192 actual asset / 32768 supervised frame exposures. OneCycle
peak1e-4, warmup.1, div5/final10, weight_decay0, clip1, established BF16 path.
Pair-ready fingerprints gate the first optimizer update. Evaluate updates
0/32/64/128/256/512/1024. Preserve the first passing checkpoint but complete
the same training budget on both cards; do not selectively shorten baseline.

Main acceptance: all32 valid natural EOS, teacher accuracy >=.99, >=29 views
with exact count, normalized J2J <=4/256 and (raw edge F1 >=.95 OR exact
canonical GT tokens). Auxiliary acceptance: all8 valid natural EOS with exact
DFS parent trees. Both must pass at the same checkpoint; CE alone is insufficient.

Passing candidates enter fixed64 train / heldout poses / fixed32 valid,
effective batch48, 512 updates =24576 actual asset exposures, same schedule
and data across comparisons. Evaluate all four input poses per asset at
0/128/256/512, including the eight training/heldout-pose assets and32validation
assets (192main generations per checkpoint). Initial generation is separate
for each fusion mode even when initial parameter tensors are identical.
Success rate and failure-as-zero F1 against continuous originalGT must not
decrease versus A; paired common-success mean J2J may worsen by at most1/256.
J2J uses the quantizedGT, with exactly the same common-success pose set on
both sides; missing common-success evidence does not automatically pass.
This is one-seed bounded validation, not a statistical/generalization claim.

Prefer A when sufficient, otherwise B,C,D in order; stop expanding once a
complete candidate passes. If all fail, diagnose auxiliary structure versus
frame-information loss before any new hypothesis. Full training stays blocked
until primary review, hash-bound real capacity checks across T2..24 and the
existing formal release gates. Formal training uses fresh official weights,
not these fitting checkpoints, and retains the prior80016-exposure budget.

## Completed first round

A and B both completed1024updates/8192actualassetexposures. First complete
main+aux passing checks were A256 andB128; final both reproduce32/32main GT
token sequences and8/8auxiliary parent trees. A already passed at2048actual
exposures, the total exposure count of the older failedB1 constantLR run.
This establishes that the original fusion can learn this fixed task under
the new effectivebatch/OneCycle recipe; it does not isolate the contributions
of batch averaging and schedule, or establish heldout generalization.
Do not run C/D. A remains the preferred architecture pending medium review;
B is the already-tested matched comparator, not a commitment to replacement.

## Completed medium comparison and new reference request

Both A and B completed512 updates, effectivebatch48, or24576 asset exposures
on the same64 training assets and four fixed poses. Each asset therefore had
384 visits. The monitored training-generation subset is8 of64 assets, not
the complete training cohort. Validation uses32 different assets/four poses.
The complete matched receipt is
`../experiments/fusion_ab_medium_v1_review.json`.

At512, both reproduce all32 monitored training-pose GT token streams. Exact
auxiliary training trees are A7/8 andB8/8. Heldout-pose main F1 is
A0.475601/B0.841470; unseen-asset validation F1 is A0.090075/B0.361060,
with failed generation counted zero against continuous originalGT. Valid
generation counts are126/128 and127/128. Exact auxiliary validation trees
remain A1/32 andB0/32. B passes the predeclared relative main-task gates,
but this is not evidence that unseen-asset structure generalization is solved.

On2026-10-14 the user requested an actual old-training comparison before
formal release, to investigate the shared data/training pipeline and small
cohort coverage or overfitting. Next run the proven legacy query-first route
on the SAME raw cache, official initial shared weights,64/32 split,T4,
effective48,512updates and0/128/256/512 full main generation checkpoints.
Each target pose is processed separately by the original query-first encoder.
This matches target exposure across architectures; it is not a historical
single-target sampling-run reproduction. The old route has no virtual/auxiliary
structure branch, so its auxiliary metrics are not applicable rather than zero.
Inspect complete learning curves, not only the final loss. A poor old-reference
result alone does not uniquely identify data, optimization or cohort size.

Formal release and further variants wait for this requested reference.
Only A/B were trained in this fusion round, each at8 and64 assets; C/D have
implementation/tests but no training runs. Earlier8-asset legacy/final-real
controls do not substitute for this matched64-asset reference.
