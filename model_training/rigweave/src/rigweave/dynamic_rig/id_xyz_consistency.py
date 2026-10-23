"""Real-frame ID/XYZ autoregression with teacher-forced structural consistency."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .id_xyz_losses import structural_consistency_ce
from .motion_encoder import AnchorWiseAlternatingMotionEncoder
from .motion_evidence import grouped_motion_evidence


@dataclass(frozen=True)
class IdXYZConfig:
    dim: int = 1024
    heads: int = 8
    query_tokens: int = 1024
    register_tokens: int = 96
    motion_depth: int = 12
    max_frames: int = 24
    gradient_checkpointing: bool = True
    motion_evidence_fusion: str = "bias"
    motion_evidence_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("dim", "heads", "query_tokens", "motion_depth", "max_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads or type(self.register_tokens) is not int or self.register_tokens < 0:
            raise ValueError("invalid attention dimensions or register count")
        if self.motion_evidence_fusion not in {"off", "bias"} or self.dropout != 0:
            raise ValueError("only the matched zero-dropout motion recipe is supported")


def extend_decoder_vocabulary(decoder, new_size, *, seed):
    """Append fresh IDs without modifying any existing embedding/head row."""
    old_input = decoder.get_input_embeddings()
    old_output = decoder.get_output_embeddings()
    if (type(new_size) is not int or new_size <= old_input.num_embeddings
            or old_output.weight.shape != old_input.weight.shape):
        raise ValueError("vocabulary extension needs compatible existing input/output tables")
    old_size, dim = old_input.weight.shape
    tied = old_input.weight.data_ptr() == old_output.weight.data_ptr()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        embedding = nn.Embedding(new_size, dim, padding_idx=old_input.padding_idx,
                                 device=old_input.weight.device, dtype=old_input.weight.dtype)
        head = nn.Linear(dim, new_size, bias=old_output.bias is not None,
                         device=old_output.weight.device, dtype=old_output.weight.dtype)
        std = float(getattr(decoder.config, "init_std", .02))
        nn.init.normal_(embedding.weight, std=std)
        nn.init.normal_(head.weight, std=std)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
    with torch.no_grad():
        embedding.weight[:old_size].copy_(old_input.weight)
        head.weight[:old_size].copy_(old_output.weight)
        if head.bias is not None:
            head.bias[:old_size].copy_(old_output.bias)
    if tied:
        head.weight = embedding.weight
    decoder.set_input_embeddings(embedding)
    decoder.set_output_embeddings(head)
    decoder.config.vocab_size = new_size
    if hasattr(decoder, "vocab_size"):
        decoder.vocab_size = new_size
    if not (torch.equal(embedding.weight[:old_size], old_input.weight)
            and torch.equal(head.weight[:old_size], old_output.weight)):
        raise RuntimeError("vocabulary extension changed pretrained rows")
    return {"old_size": old_size, "new_size": new_size, "new_rows": new_size - old_size,
            "old_input_rows_exact": True, "old_output_rows_exact": True,
            "tied": tied, "seed": seed}


class IdXYZRig(nn.Module):
    def __init__(self, surface_tokenizer, decoder, config: IdXYZConfig):
        super().__init__()
        self.config = config
        self.surface_tokenizer = surface_tokenizer
        self.xyz_decoder = decoder
        self.motion_encoder = AnchorWiseAlternatingMotionEncoder(
            dim=config.dim, depth=config.motion_depth, heads=config.heads,
            register_tokens=config.register_tokens, max_frames=config.max_frames,
            dropout=config.dropout, gradient_checkpointing=config.gradient_checkpointing,
            motion_evidence_fusion=config.motion_evidence_fusion,
            motion_evidence_heads=config.motion_evidence_heads,
            register_temporal_mode="aligned", frame_role_policy="all_real")
        if decoder.get_input_embeddings().embedding_dim != config.dim:
            raise ValueError("decoder and motion widths differ")

    def configuration(self):
        return {**asdict(self.config), "virtual_frame": False, "frame_fusion": False,
                "auxiliary_decoder": False, "frame_role_policy": "all_real"}

    def encode(self, surface):
        points = surface["query_points"]
        if points.ndim != 4 or points.shape[2:] != (self.config.query_tokens, 3):
            raise ValueError("surface anchors must retain [B,T,Q,3]")
        tokens = []
        for frame in range(points.shape[1]):
            values = tuple(surface[key][:, frame] for key in
                           ("dense_points", "dense_normals", "query_points", "query_normals"))
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens.append(checkpoint(self.surface_tokenizer, *values, use_reentrant=False))
            else:
                tokens.append(self.surface_tokenizer(*values))
        evidence = (grouped_motion_evidence(surface["dense_points"].detach(), surface["query_indices"])
                    if self.config.motion_evidence_fusion != "off" else None)
        _, encoded = self.motion_encoder(torch.stack(tokens, dim=1), return_all=True,
                                         motion_evidence=evidence)
        return encoded[:, :, 1 + self.config.register_tokens:]

    def logits(self, condition, input_ids, attention_mask):
        if condition.shape[1] + input_ids.shape[1] > self.xyz_decoder.config.max_position_embeddings:
            raise ValueError("complete ID/XYZ target exceeds decoder position capacity")
        embedding = self.xyz_decoder.get_input_embeddings()(input_ids)
        prefix = torch.cat((condition.to(embedding.dtype), embedding), dim=1)
        mask = torch.cat((attention_mask.new_ones(condition.shape[:2]), attention_mask), dim=1)
        output = self.xyz_decoder(inputs_embeds=prefix, attention_mask=mask,
                                  use_cache=False, return_dict=True)
        return output.logits[:, condition.shape[1]:]

    def forward_encoded(self, conditions, targets, *, consistency_weight):
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("consistency weight must be finite and nonnegative")
        ids, mask = targets["input_ids"], targets["attention_mask"]
        frames = targets["frame_mask"].bool()
        if ids.ndim != 3 or ids.shape != mask.shape or ids.shape[:2] != conditions.shape[:2]:
            raise ValueError("all real frames need an aligned full ID/XYZ sequence")
        labels = ids[:, :, 1:].clone().masked_fill(~mask[:, :, 1:].bool(), -100)
        structural = targets["structure_mask"][:, :, 1:].bool()
        values = [self.logits(conditions[:, frame], ids[:, frame, :-1], mask[:, frame, :-1])
                  for frame in range(ids.shape[1])]
        logits = torch.stack(values, dim=1)
        valid = labels != -100
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                             ignore_index=-100, reduction="none").reshape_as(labels)
        per_frame = ce.sum(-1) / valid.sum(-1).clamp_min(1)
        main = (per_frame * frames).sum(-1) / frames.sum(-1).clamp_min(1)
        consistency = structural_consistency_ce(logits, labels, structural, frames)
        per_asset = main + consistency_weight * consistency["loss_per_asset"]
        correct = (logits.detach().argmax(-1) == labels) & valid
        structure_valid = structural & valid & frames[..., None]
        coordinate_valid = targets["coordinate_mask"][:, :, 1:].bool() & valid & frames[..., None]
        return {"loss": per_asset.mean(), "loss_per_asset": per_asset,
                "main_loss": main.mean(), "main_loss_per_frame": per_frame,
                "consistency_ce": consistency["loss"], "consistency_kl": consistency["kl_diagnostic"],
                "structure_ce": (ce * structure_valid).sum() / structure_valid.sum().clamp_min(1),
                "coordinate_ce": (ce * coordinate_valid).sum() / coordinate_valid.sum().clamp_min(1),
                "structure_accuracy": (correct & structure_valid).sum() / structure_valid.sum().clamp_min(1),
                "coordinate_accuracy": (correct & coordinate_valid).sum() / coordinate_valid.sum().clamp_min(1)}

    def forward(self, surface, targets, *, consistency_weight):
        return self.forward_encoded(self.encode(surface), targets, consistency_weight=consistency_weight)

    @torch.no_grad()
    def generate(self, condition, codec, *, max_new_tokens):
        if self.training or condition.shape[0] != 1:
            raise ValueError("generation requires eval mode and one asset/frame")
        ids = torch.tensor([[codec.bos, codec.class_token]], device=condition.device)
        embedding = self.xyz_decoder.get_input_embeddings()(ids)
        mask = torch.ones((1, condition.shape[1] + 2), dtype=torch.long, device=condition.device)
        output = self.xyz_decoder(inputs_embeds=torch.cat((condition.to(embedding.dtype), embedding), dim=1),
                                  attention_mask=mask, use_cache=True, return_dict=True)
        for _ in range(max_new_tokens):
            scores = output.logits[0, -1].float()
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("nonfinite generation logits")
            allowed = codec.allowed_tokens(ids[0].tolist())
            if not allowed:
                raise RuntimeError("ID/XYZ grammar returned no next token")
            candidates = torch.as_tensor(allowed, device=scores.device, dtype=torch.long)
            token = int(candidates[scores[candidates].argmax()])
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
            if token == codec.eos:
                return {"success": True, "status": "eos", "tokens": ids[0].tolist(),
                        "eos_only_legal_choice": len(allowed) == 1}
            if condition.shape[1] + ids.shape[1] >= self.xyz_decoder.config.max_position_embeddings:
                break
            if output.past_key_values is None:
                raise RuntimeError("decoder did not retain its KV cache")
            mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
            output = self.xyz_decoder(input_ids=ids[:, -1:], attention_mask=mask,
                                      past_key_values=output.past_key_values, use_cache=True, return_dict=True)
        return {"success": False, "status": "max_tokens", "tokens": ids[0].tolist()}
