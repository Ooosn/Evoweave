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
512-to-896 Linear, RMSNorm and Qwen. Native 512-channel geometry features are
zero-padded to the unchanged 1024-channel temporal interface; the published
projection occupies the first 512 columns of the decoder adapter. No information
or pretrained projection row is discarded. The remaining columns start at zero
and are trainable. Original temporal layers, 1024 anchors, data rules, stage-two
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
