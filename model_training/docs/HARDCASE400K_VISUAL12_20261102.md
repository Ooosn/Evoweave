# Fixed400k Official12Case Comparison

Status2026-11-02: COMPLETE. Source deployment, audit_v33 and capacity_v33 have
actually completed. The protocol below records that operation, not permission
to repeat it. Read current.json and the immutable receipts before any action.

User2026-11-02 explicitly authorizes replacing the long-lived scratch source
deployment while preserving the old directory and all training results, then
comparing the predeclared400k endpoint on the original12official cases. Do not
restore historical files in the old checkout, restart training, select another
checkpoint, change seeds, run another capacity stress or promote a model.

First use `server_ops/hgc/deploy_stage_checkout.py`: plan with exact committed
source; execute only with idle native keeper/GPU3and4, no process cwd in the old
checkout, verified NVMe and sufficient96h grant. Preserve every existing old
file and its29missing-path state, verify a complete compressed archive and its
durable copy, clone the fixed Git version into a fresh SSD candidate, verify
the Git object store and unchanged keeper implementation, then rename the old
directory to an explicit preserved path and install the clean candidate at the
canonical entry. Do not release/restart the native keeper. No data or outputs
move. Retain the old directory plus its durable verified archive. Keep pinned
new source archives durable. New source must fit the remaining grant and the
5day scratch policy; never disable cleanup, add exclusions or touch around it.
The raw preserved directory remains scratch; its verified durable archive is
the long-term recoverable copy. No raw small-file tree is moved to unknown media.

The endpoint is `outputs/train_v32/weights_100000.pt`, SHA256
`b82aca74b6837437e39f4803025bf996b7cab2425bc681dd49b236f7ff7b98e8`.
Training already passed:400000total visits,70000new,1463new updates. Its full
recovery points, model exports, cache and existing outputs must not change.
Reuse capacity_v28(full300k) and capacity_v31(330k) results, after canonical
input/GT/raw-frame/normalization/official-policy/seed verification for both.

Run fresh audit_v33 CPU-only, then record its actual result/resources and
commit/sync exact admission before capacity_v33. This latter native stage name
is scheduler plumbing: it is zero-update formal inference, not another stress
test. Inherit the accepted native GPU4/FP32/15beam/full2048token proof fromv31,
requiring identical inference math/architecture and policy. Do not regenerate
either baseline. Cases5/15/23/24/30/44/54/61/70/95/105/108; fixed base_seed20261029,
topk5, topp.95, temperature1.5, repetition3, dynamic15beam/no static fallback,
FP32/noTF32/eager; reference0, targets1..7, displays0/2/4/7. No GT prefix for
primary generation, no repair, fitting, automatic retry or alternate seeds.
GT_head and GT_copy remain separately named controls, never end-to-end scores.

Use retained native131170166/keeper2268188, GPU4only and8CPUsubset
11/12/14/15/61/63/64/65 inside the existing16slot dual-H100 grant. Preserve
GPU3and other tasks. Keep64GiB host and85%GPU caps,1800second payload bound;
save-stop2026-11-04T11:40:04Z, actual96h expiry one hour later. Fresh resource
admission must detect occupancy/conflicts and must not stop other tasks,
request more GPUs or release the grant.

Review zero updates, native cleanup, all12references and336target rows,
independent rescoring and matched before/after/GT images. Include every
failure with zero quality; geometry uses explicitly identical common-success
sets and equal asset weighting. Compare400k to both300k and330k, do not use the
results to reselect or tune. Keep105's old exact coordinate-token replay
discrepancy unresolved. Fixed12diagnostics are not locked855/32case population
acceptance, and full300k remains the production reference.

Configuration: `hardcase400k_visual12_20261102.json`.
Work output: `outputs/capacity_v33`.
Durable: `/home/wangyy/evorig/outputs/hardcase3608_20261101/visual12_final400k_v1`.

## Actual Outcome

Deployment source f8bfeb02a6737f127176a2c6368c2cf4f9796727, receipt SHA256
156924e3748161ba999144ca05469ba98a5a234b73f6bef1491bd3342b848ec6.
Old directory `code_preserved_ef4d0ffddbeb_before_f8bfeb02a673` retains all1016
existing files and the29missing paths, with verified work/durable archives.
No historical file was restored in it; all training results remain in place.

Native inference source c5bcd3ec593f63e1d300821ee1a15ca13bb69a13; both stages
exited0 with cleanup and scheduler receipts passed. Result SHA256
68c1e534d3e7a8eb3f4d46a069036fca51f14f5bc089458a2c0571673f0fb2f3;
resources7fb13ced52c265f44decc428bfbf2f1d37be547d75410b5d3aa4e49a6b030ff4.
GPU4/8CPUinside original16slots; no training, baseline generation, new stress,
GPU request, grant release, parameter changes or checkpoint reselection.

All336new target scores and12reference scores were independently recomputed;
all27paired images were checked. Reference parse counts300k/330k/400k are
11/12,12/12,12/12. All12failure-zero predicted-reference target edgeF1:
0.7302501884,0.7560105580,0.7351737799. Same11common-success assets across all
three have normalized targetJ2J2.075863%,1.839580%,2.045971%;105is excluded
only from this conditional geometric comparison, not from quality denominators.
For330kvs400k, the same12assets/84target views haveJ2J1.952829%to2.248990%,
5case improvements and7regressions. LowerJ2Jis better; units are target mesh
bounding-box diagonal. GT-reference controls stay separate.

30:75to87joints versus91GT, but missing lateral span and worse geometry.
70:still52versus122GT, major branches missing.105:139versus136GT, yet worse
connections/positions and edgeF1.108:26versus5GT, excess branches worsen.
The old105exact-coordinate-token replay cause remains unresolved. These12cases
do not support400kover330kand are not locked855/32population acceptance.
full300kremains production reference; no additional compute is admitted.

Local report `D:/evoweave/server_ops/outputs/hardcase400k_visual12_20261102/comparison_report.html`.
Work and durable review, visual inspection, native postrun review and verified
89member evidence archive are retained. Archive SHA256
8e9ce16d39a59ed6c1dccd44453f47b6e2e5f69f716731bb218460dfb221b8c1,
20506996bytes. Native131170166/keeper2268188 and96hgrant remain intact.
