# SkinTokens Published Test-Split Check

This is a twelve-asset paired diagnostic from the public main test split of
Seed3D/Articulation-XL2.0, one of the SkinTokens paper's two skeletal benchmarks.
It is not the full paper table or an assertion that the authors' private
cleaned archive has identical membership. The release download script's
dataset archives were not available. The public test archive has 1,997
entries; source/UUID matching to the fixed 5k fine-tune excluded 41 before
fixed SHA-ordered sampling. Test rigs are never provided to the model.

Both weights run VAST-AI's unmodified SkinTokens demo.py at 273b691, using
its documented transfer flag and generation defaults. Set the same seed
before the official command for each pair. Keep the release's 33,036-token
vocabulary, native generation, FSQ-CVAE, Blender pre/postprocessing, and
normalization. Overlay only the 30k single-pose skeleton weights, including
the first 267 rows of the tied embeddings and output head. Retain all
remaining rows and VAE tensors. This checks the effect of skeleton-only
fine-tuning in the public full rig pipeline, not improved skinning.

Use the README's PyTorch/CUDA configuration, official FlashAttention source
built for the H100, and the same private runtime for both arms. Run the
unmodified Blender helper in a network namespace with loopback only because
its pickled HTTP protocol is not safe to expose on a shared node. Use only
GPU3 and eight CPUs from the retained native grant, with no optimizer steps.

Read exported joint world positions and verify the original mesh coordinate
frame before scoring. J2J/J2B/B2B use symmetric nearest distances, with 100
samples per bone, reported as percentages of the original mesh's longest
bbox side. There is no GT registration. Means are over common scored assets;
preserve every attempted case, input/weight hashes, and original GLB/logs.
