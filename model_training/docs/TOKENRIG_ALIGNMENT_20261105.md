# Pretrained TokenRig Interface Test

The user paused the first Qwen run on November 5 and authorized a fresh test of
the pretrained interface. Its last verified full recovery is 250,000 visits,
5,225 updates, with 1,003 finite Adam states and both rank RNG/sampler states.
Both the scratch and durable checkpoint have SHA256
`2d2d84e82fa51b0ce1fb4b194482afdd178652241353bc7e5255c41454d57162`.
The interrupt occurred at 252,064 visits; the final 2,064 were not saved.
The original source, results, native GPU allocation and OPT330k model remain.

The old route loaded the released Qwen and skeleton table strictly, but paired
them with the UniRig geometry encoder and a fresh condition projection. This is
an architectural conditioning mismatch, not an established missed-key defect.
The new isolated route strictly loads the release's eight-layer geometry encoder,
512-to-896 Linear, RMSNorm and Qwen. The first comparison zero-padded native
512-channel features to the unchanged 1024-channel temporal interface; its
published projection occupied the first 512 adapter columns. The accepted
identity-initialized variant below instead duplicates standardized features and
averages two native projection copies. Neither discards native feature channels
or projection rows. Original temporal layers, 1024 anchors, data rules, stage-two
architecture and objectives remain unchanged. This does replace the geometry
weights and depth; it is not a decoder-only ablation anymore.

Static parity of the lifted projection is not dynamic parity. Existing randomly
initialized temporal layers and LayerNorms can still change the pretrained
conditioning distribution. Therefore the first admitted GPU work has zero
optimizer updates and compares four paths on the same twelve fixed validation
assets and the first view of their retained eight-frame inputs: old hybrid
dynamic, native static512, aligned fixed-anchor static1024, aligned dynamic1024.
The input mesh, normalization, GT, Qwen and grammar are paired. The native-static
path intentionally uses the original query sampler; it is a diagnostic control,
not a change to training data or the proposed dynamic method. All generation is
FP32 greedy, not the final15-beam protocol. Inspect results before a separately
recorded capacity test and fresh training. Never resume the old Adam into the
changed architecture or automatically replace OPT330k.

The function-preserving initialization variant keeps every original attention and
FFN layer in the forward path, but initializes their residual output matrices to
zero. The native geometry LayerNorm affine is inverted by a trainable duplicated
512-to-1024 lift and restored at the existing final condition LayerNorm. The
released decoder projection is split equally over the two channel copies.
Internal random attention matrices remain trainable; residual outputs receive
gradients immediately. There is no static fallback, GT-dependent calibration,
extra skip path or frozen motion module. Repeated LayerNorm epsilons and rounding
make this approximately rather than bitwise function-preserving; actual greedy
generation must be measured before using it as a new training initialization.

The first completed matched12-view comparison measured greedy edgeF1 of0 for the
old hybrid and the native-weights/random-motion route,0.77567337 for the native
static512 route and0.43017514 for the fixed-anchor static1024 route. All had zero
optimizer updates, matching input/GT fingerprints and clean native resource
receipts. The next diagnostic tests only the identity-initialized native route,
with both its native512 decoder prefix and a named1024-prefix control. All1024
anchors still enter the encoder and temporal computation; only the decoder's
prefix length is aligned. Data, frames, supervision and stage2 remain unchanged.

The identity/native512 probe completed at2026-11-05T11:49:28Z, without optimizer
updates, on the same12 inputs. Both aligned-static512 and aligned-dynamic512
achieved edgeF1 of0.7798844100010204, with all12 generation sequences ending
normally. Their per-case predictions, topology, coordinates and metrics were
identical. The identity-dynamic1024 control remained0.43017514348256636.
This supports recovering the pretrained prior, not a claim about motion gains,
full-panel validation or stage-two accuracy. Peak whole-device use was9917MiB
on each GPU; supervised payload cleanup and the durable receipt archive passed.
Evidence: `outputs_probe_r5/capacity/result.json`, SHA256
`3ce34ac4dbecc683fce27d0dab1a605b2c874f0b91f218b02fb9537b1697c143`.

The next operation uses fresh Adam and the unchanged300k joint-training recipe.
Its independent capacity stage exercises6/12/24frames at the actual enlarged
physical batches, the largest-joint and largest-geometry cases, FP32 prefix
parity and exact disk restoration of model/Adam/two-rank RNG/sampler state.
The zero-update diagnostic is not capacity evidence. The old hybrid capacity
archives are not admissible for the changed frontend. Reuse the verified SSD
dataset cache without copying or changing data. Formal training is blocked
until these new checks and their durable backups pass; capacity weights will
not initialize formal training.

The independent capacity stage completed normally onNovember5, native source
`459a5a2395eb96ef3ba4d58f5e2f0e844cc87f9d`. Both ranks passed13 records: a
per-rank batch1 joint contract and12 extreme updates across the six declared
frame/stress profiles. FP32 cached-prefix maximum logit errors were3.82e-6 and
5.73e-6, below the unchanged1e-4 absolute/relative tolerances. Twelve separate
representative updates then saved576 visits/12 updates and restored model,
Adam, both-rank RNG and sampler state exactly from disk. Its8,945,962,834-byte
checkpoint has verified scratch/durable copies, SHA256
`82944552222e0b4a503240c405a0a8472662d6f4fb4a6e498855fdab12d4a3af`.
The capacity result SHA256 is
`905930448d9ac65d44e616f43b7aecd7c5114a7dfc007bd67408017e8e244c8d`;
both-rank peaks were73187/72527MiB, at most71.5GiB. Supervisor exit0, complete
payload cleanup and the durable receipt archive are verified. No check is
rerun and no capacity weight is used as formal initialization.

The separate formal recipe is `tokenrig_native_full_20261105.json`, SHA256
`e4fece4995c54dab87318bc13ab2280c47670fe1e920a371c7e63626fe159abb`.
It retains300000 visits, effective batch48, per-rank24 for2-12frames and12 for
13-24frames, original AdamW/OneCycle and all validation/save cadences. Its
`training.resume` is null; the new output is `outputs_native_train/train`.
The old250k optimizer is retained only for the old hybrid model. Launch and
initial durable recovery still require actual native-process evidence.
