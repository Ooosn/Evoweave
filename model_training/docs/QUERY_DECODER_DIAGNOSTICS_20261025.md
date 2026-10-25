# Frozen Query And Decoder Diagnostics

The human explicitly requested beginning the proposed staged investigation.
This first stage is inference only: no optimizer, backward, checkpoint rewrite,
loss adjustment or fresh training. Completed100k training remains preserved.

## Checkpoint Availability

B65 was previously deleted. A fresh13:04JST read found no weights in its original
normal-B output tree or the known virtual-fusion work/preparation paths; the local
virtual-fusion artifact tree also has no `.pt` copy. Its saved predictions are
not a model. Do not substitute originalJuly80k, ID100k or structure/XYZ-fill for
a B65 architecture ablation. B query-path ablation remains unavailable until
weights are recovered or a separately designed matched model is trained.

ID100k is present at
`/work/wangyy/id_xyz_consistency_20261024/outputs/consistency_100000_cpu4_v1/checkpoint_100000.pt`.
SHA256 `ada70ea7b97b4bdbcbd699b1698696faa0eef0298df09e07637e687eea3a29c8`;
completed model fingerprint `b546677a66e26d321bea1432c4307f7d612cdade2cd5c8d683af08c96835dd23`.
All model/data/runtime imports use the immutable trained `cpu4_code` at `e1ed5bd`.
Only the new diagnostic metrics and entrypoint come from the isolated new checkout.

## Measurements

- Original matched small panel:8 training assets with seen and heldout windows,
  plus16 validation assets;8 real inputs and scored slots0/2/4/6,128 views total.
- Strictly restore every saved tensor and verify the completed model fingerprint.
  Verify raw frame IDs, original continuous GT, parents and target tokens against
  saved ID100k records. Native free generation must reproduce those records exactly.
- Under the correct GT prefix, separate class/new-ID/parent-ID/BRANCH/EOS/X/Y/Z
  NLL and accuracy. Preserve full-vocabulary metrics. Separately report native
  grammar-constrained decisions and sites with more than one legal choice.
- Repeat teacher-forced scoring with another pose's condition from the same asset,
  keeping the target unchanged. This is a pose-dependence diagnostic, not a quality
  score or proof of bone-semantic correspondence.
- Report first full-token divergence and first structure-only divergence. These
  are first positional mismatches, not semantic alignments after insertions.
- Force the entire GT structural sequence, including joint count and EOS, while
  predicting only XYZ from its own generated coordinate prefix. Score resulting
  geometry against original GT, but label it as an oracle diagnostic, never model
  generation quality. Quantized-GT roundtrip is a reference, not an asserted bound.
- Role aggregates are token-weighted and retain denominators; macro view/asset
  quality has four equally weighted views per group asset. Non52 is a joint-count
  stratum and is not automatically synonymous with rare topology.

## Execution Gates

CPU audit covers all32 group-assets without CUDA. Primary accepts the actual
audit before releasing2 validation assets/8 views on GPU. Primary accepts the
smoke before releasing the128-view run. Any input/weight/source/replay mismatch
stops execution and is preserved as a failure receipt; no fallback initialization.

Reuse allocation130309880, GPU5/6, keeper1090013. Actual scheduler grant is1slot;
explicit SSH execution is bounded to4 total CPUs `[58,63,64,65]`,64GiB RAM and
85% GPU memory under the existing supervisor. Do not touch other processes,
release the allocation or modify the trained checkout. Stop before16:45JST.
17 CPU diagnostic/guard/oracle tests passed before real CPU audit.
