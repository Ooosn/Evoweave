# SkinTokens Single-Pose Baseline

User authorized starting only the first arm on November 9 JST. Fine-tune the
released SkinTokens skeleton path on the existing fixed 5,000 RT-aware train
assets for exactly 30,000 visits. Do not launch the frame-local, multi-target
temporal, or single-query temporal arms until the first result is reviewed.

The native 512-token mesh encoder, 512-to-896 projection with RMSNorm and
released Qwen skeleton embeddings and head are strictly loaded from the pinned
release. The released code does not implement `training_step`. Use the
project's existing skeleton-only next-token CE over the native 267 skeleton
tokens. There is no skin VAE, temporal block, second-stage head, loss or
gradient, and no project checkpoint or optimizer in initialization.

Select one pose per train visit using the existing partition and query seed;
normalize the selected input only. Keep AdamW and the existing OneCycle peak
LR of 1e-4 with the horizon set to 30,000 visits. The two ranks use 24 assets
each except at exact 10k boundaries. Six passes, 627 updates, 30,000 supervised
skeletons. Save full recovery points with both ranks' RNG and optimizer at
10k, 20k and 30k and verify the durable mirror.

At 0, 10k, 20k and 30k generate only slot 0 from each of the retained 128
validation clips. Use the exact archived normalized reference surface and GT
and native point sampling. FP32 greedy decoding with the existing grammar,
no alignment/fitting or skin inference. Record original-GT Edge F1 and
normalized J2J separately; unsuccessful generations remain zero for F1.
This validation has 128 generations, not the historical 1,024-view average.

Use the existing idle HGC grant 131942123, GPU3/4 and 16 CPU slots, with a
stage-scoped checkout and the recent verified NVMe data/dependency cache from
the stopped stage-one continuation. Do not use or refresh the Nov4 scratch
checkout. First run the CPU source/data/plan audit; then admit one bounded
no-update stress check of the exact single-frame architecture and batch;
admit formal training only after both have exited and cleaned up. No retry,
allocation change, or promotion of the resulting weights is automatic.

The first no-update capacity check hit the 76,000 MiB process cap during the
long-skeleton Qwen forward on both ranks. No optimization step ran and both
GPUs were cleaned up. The independent native entry was missing the existing
Qwen training path's non-reentrant activation recomputation. Restore that
same training-only setting, keep the passed eager CPU audit and its unchanged
weights, inputs and config, and run a separate `capacity_recompute` check.
This does not change the model, data, batch, precision, loss or evaluation.
