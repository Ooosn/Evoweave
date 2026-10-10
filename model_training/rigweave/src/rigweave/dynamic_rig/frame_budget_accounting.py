"""Actual exposure and token-weighted accumulation for variable asset batches."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


def target_token_weight(batch, eos_token: int, eos_weight: float) -> float:
    if not math.isfinite(eos_weight) or eos_weight < 0:
        raise ValueError("EOS weight must be finite and nonnegative")
    labels = batch["input_ids"][:, 1:]
    valid = batch["attention_mask"][:, 1:].bool()
    count = float(valid.sum())
    count += (eos_weight - 1.0) * float(((labels == eos_token) & valid).sum())
    if not math.isfinite(count) or count <= 0:
        raise ValueError("frame-budget accumulation requires a positive target-token weight")
    return count


def normalize_token_gradients(parameters, world_size: int, global_token_weight: float) -> float:
    if world_size < 1 or not math.isfinite(global_token_weight) or global_token_weight <= 0:
        raise ValueError("invalid global token normalization")
    # DDP has averaged each local token-sum gradient across ranks already.
    factor = world_size / global_token_weight
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)
    return factor


@dataclass
class FrameBudgetProgress:
    samples_seen: int = 0
    input_frames_seen: int = 0
    target_token_weight_seen: float = 0.0
    microbatches_seen: int = 0
    next_epoch: int = 0
    next_batch_in_epoch: int = 0
    last_samples: int = 0
    last_frames: int = 0
    last_token_weight: float = 0.0
    samples_by_frame_count: dict[str, int] = field(default_factory=dict)

    def record_step(self, *, world_size, batches, frames, global_samples, global_frames,
                    global_token_weight, next_epoch, next_batch_in_epoch):
        if len(batches) != len(frames) or not batches or any(value <= 0 for value in batches):
            raise ValueError("invalid microbatch accounting window")
        if world_size * sum(batches) != global_samples:
            raise ValueError("DDP ranks did not follow equal-size frame-batch schedules")
        if world_size * sum(batch * frame for batch, frame in zip(batches, frames)) != global_frames:
            raise ValueError("DDP ranks did not follow equal frame schedules")
        if next_epoch < 0 or next_batch_in_epoch < 0 or not math.isfinite(global_token_weight) or global_token_weight <= 0:
            raise ValueError("invalid frame-budget progress")
        self.samples_seen += int(global_samples)
        self.input_frames_seen += int(global_frames)
        self.target_token_weight_seen += float(global_token_weight)
        self.microbatches_seen += len(batches)
        self.next_epoch, self.next_batch_in_epoch = int(next_epoch), int(next_batch_in_epoch)
        self.last_samples, self.last_frames = int(global_samples), int(global_frames)
        self.last_token_weight = float(global_token_weight)
        for batch, frames_in_batch in zip(batches, frames):
            key = str(frames_in_batch)
            self.samples_by_frame_count[key] = self.samples_by_frame_count.get(key, 0) + world_size * batch

    def fields(self, train_rows):
        return {"sample_seen": self.samples_seen, "optimizer_samples_seen": self.samples_seen,
                "effective_batch": self.last_samples, "samples_in_step": self.last_samples,
                "input_frames_seen": self.input_frames_seen, "input_frames_in_step": self.last_frames,
                "target_token_weight_in_step": self.last_token_weight,
                "target_token_weight_seen": self.target_token_weight_seen,
                "consumed_microbatches": self.microbatches_seen,
                "train_rows": int(train_rows), "epoch_equivalent": self.samples_seen / max(1, train_rows),
                "loss_normalization": "global_target_token_weight"}

    def checkpoint(self):
        return {"schema_version": 1, "mode": "frame_budget", **asdict(self)}

    @classmethod
    def restore(cls, value, *, step, grad_accum_steps):
        if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("mode") != "frame_budget":
            raise ValueError("missing or unsupported frame-budget checkpoint accounting")
        fields = {key: value[key] for key in cls.__dataclass_fields__}
        result = cls(**fields)
        integers = (result.samples_seen, result.input_frames_seen, result.microbatches_seen,
                    result.next_epoch, result.next_batch_in_epoch, result.last_samples, result.last_frames)
        if any(not isinstance(number, int) or number < 0 for number in integers):
            raise ValueError("invalid frame-budget checkpoint counters")
        if result.microbatches_seen != step * grad_accum_steps:
            raise ValueError("checkpoint microbatch cursor disagrees with optimizer step")
        if result.samples_seen != sum(result.samples_by_frame_count.values()):
            raise ValueError("checkpoint sample histogram disagrees with actual exposure")
        if result.input_frames_seen != sum(int(key) * number for key, number in result.samples_by_frame_count.items()):
            raise ValueError("checkpoint frame histogram disagrees with actual exposure")
        if not math.isfinite(result.target_token_weight_seen) or result.target_token_weight_seen <= 0:
            raise ValueError("invalid checkpoint target-token weight")
        return result
