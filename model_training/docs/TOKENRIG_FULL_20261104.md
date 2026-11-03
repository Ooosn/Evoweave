# TokenRig full joint experiment

The user authorized a new 300,000-asset-visit experiment on November 4.
This is not a continuation of, or a replacement for, the 330k working model.
Use the original joint full-data recipe and the fixed train/validation manifests.
Load the UniRig mesh encoder and its original projection, and load the released
TokenRig Qwen and skeleton vocabulary weights. Learn the new condition projection
and retain the same temporal encoder and stage-two architecture and objectives.
Keep the original OneCycle/AdamW schedule. No capacity weights are used.

Use two H100s, effective batch 48, and per-rank physical batches of 24 for
2–12 frames and 12 for 13–24 frames. The completed real-data probes used both
stress cohorts. The enlarged profiles reached 72,841 and 77,019 MiB respectively.
The run requires an empty native GPU grant, a clean fixed source, and a successful
CPU audit for the exact model and sampler configuration. Save a full recovery
point after the first update, then use the original 10k cadence and durable
retention. Stop for memory or other failures without automatic retries.

The active state, guarded configuration, and run-specific outputs are the
authoritative records. Monitor the live resource receipt and real rank progress;
capacity does not establish model quality or throughput.
