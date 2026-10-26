"""Conditional AR coordinate generation for complete, known reference joints."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import OPTConfig, OPTForCausalLM

from .reference_skeleton_xyz import ReferenceJointConditioner, ReferenceXYZConfig


@dataclass(frozen=True)
class CoordinateGeneratorConfig:
    bins: int = 256
    decoder_layers: int = 4

    def __post_init__(self):
        if (type(self.bins) is not int or self.bins < 2
                or type(self.decoder_layers) is not int or self.decoder_layers < 1):
            raise ValueError('coordinate vocabulary and decoder depth must be positive integers')


def quantize_xyz(xyz, bins):
    if not bool(torch.isfinite(xyz).all()) or bool(((xyz < -1.) | (xyz > 1.)).any()):
        raise ValueError('coordinate labels must be finite within [-1,1]; clipping is forbidden')
    return torch.round((xyz.float() + 1.) * ((bins - 1) / 2.)).long()


def dequantize_xyz(tokens, bins):
    if bool(((tokens < 0) | (tokens >= bins)).any()):
        raise ValueError('generated coordinate token outside the vocabulary')
    return tokens.float() * (2. / (bins - 1)) - 1.


class ReferenceCoordinateGenerator(nn.Module):
    """Generate 3J coordinate tokens; reference joint identity/tree are conditions."""
    def __init__(self, condition_config=ReferenceXYZConfig(), generator_config=CoordinateGeneratorConfig()):
        super().__init__()
        self.config = condition_config
        self.generator_config = generator_config
        self.conditioner = ReferenceJointConditioner(condition_config)
        self.condition_norm = nn.LayerNorm(condition_config.dim)
        self.axis_embedding = nn.Embedding(3, condition_config.dim)
        self.decoder = OPTForCausalLM(OPTConfig(
            vocab_size=generator_config.bins + 1, hidden_size=condition_config.dim,
            word_embed_proj_dim=condition_config.dim, ffn_dim=condition_config.dim * 4,
            num_hidden_layers=generator_config.decoder_layers, num_attention_heads=condition_config.heads,
            max_position_embeddings=condition_config.max_joints * 4,
            dropout=0., attention_dropout=0., layerdrop=0., activation_function='gelu',
            bos_token_id=generator_config.bins, eos_token_id=None, pad_token_id=generator_config.bins,
            use_cache=True, do_layer_norm_before=True))

    def prepare_condition(self, features, points, normals, xyz, parents, mask, reference):
        values = self.conditioner(features, points, normals, xyz, parents, mask, reference)
        batch, frames, joints, dim = values.shape
        values = self.condition_norm(values).reshape(batch * frames, joints, dim)
        prefix_mask = mask[:, None].expand(-1, frames, -1).reshape(batch * frames, joints)
        axis = torch.arange(3, device=values.device)
        query = values[:, :, None] + self.axis_embedding(axis)[None, None]
        return values, prefix_mask, query.reshape(batch * frames, joints * 3, dim)

    def teacher_logits(self, condition, prefix_mask, query, tokens):
        previous = torch.cat((tokens.new_full((tokens.shape[0], 1), self.generator_config.bins), tokens[:, :-1]), 1)
        embedding = self.decoder.get_input_embeddings()(previous) + query
        attention = torch.cat((prefix_mask, prefix_mask.repeat_interleave(3, dim=1)), 1).long()
        output = self.decoder(inputs_embeds=torch.cat((condition.to(embedding.dtype), embedding), 1),
                              attention_mask=attention, use_cache=False, return_dict=True)
        return output.logits[:, condition.shape[1]:, :self.generator_config.bins]

    def forward(self, features, points, normals, xyz, parents, mask, reference, *, targets, return_predictions=False):
        condition, prefix_mask, query = self.prepare_condition(features, points, normals, xyz, parents, mask, reference)
        batch, frames, joints, _ = targets.shape
        if targets.shape != (features.shape[0], features.shape[1], mask.shape[1], 3):
            raise ValueError('teacher labels must match the provided reference joints and observed frames')
        labels = quantize_xyz(targets.masked_fill(~mask[:, None, :, None], 0.), self.generator_config.bins)
        logits = self.teacher_logits(condition, prefix_mask, query, labels.reshape(batch * frames, joints * 3))
        errors = F.cross_entropy(logits.float().reshape(-1, self.generator_config.bins), labels.reshape(-1), reduction='none')
        errors = errors.reshape(batch, frames, joints, 3)
        valid = mask[:, None, :, None] & (torch.arange(frames, device=targets.device)[None, :, None, None] != reference[:, None, None, None])
        valid = valid.expand_as(errors)
        count = valid.sum((1, 2, 3))
        if bool((count == 0).any()):
            raise ValueError('conditional generation needs at least one nonreference target pose')
        loss = errors.masked_fill(~valid, 0.).sum((1, 2, 3)) / count
        predicted_tokens = logits.argmax(-1).reshape_as(labels)
        correct = (predicted_tokens == labels) & valid
        result = {'loss_per_asset': loss, 'teacher_forced_accuracy': correct.sum((1, 2, 3)).float() / count}
        if return_predictions:
            result['teacher_prediction'] = dequantize_xyz(predicted_tokens, self.generator_config.bins)
        return result

    @torch.no_grad()
    def generate(self, features, points, normals, xyz, parents, mask, reference, *, use_cache=True):
        if self.training:
            raise ValueError('free generation requires eval mode')
        condition, prefix_mask, query = self.prepare_condition(features, points, normals, xyz, parents, mask, reference)
        batch, frames, joints = features.shape[0], features.shape[1], mask.shape[1]
        previous = torch.full((batch * frames, 1), self.generator_config.bins, dtype=torch.long, device=features.device)
        first = self.decoder.get_input_embeddings()(previous) + query[:, :1]
        embeddings = torch.cat((condition.to(first.dtype), first), 1)
        attention = torch.cat((prefix_mask, prefix_mask[:, :1]), 1).long()
        output = self.decoder(inputs_embeds=embeddings, attention_mask=attention,
                              use_cache=use_cache, return_dict=True)
        generated = []
        for step in range(joints * 3):
            token = output.logits[:, -1, :self.generator_config.bins].argmax(-1)
            generated.append(token)
            if step == joints * 3 - 1:
                break
            next_embedding = self.decoder.get_input_embeddings()(token[:, None]) + query[:, step + 1:step + 2]
            attention = torch.cat((attention, prefix_mask[:, (step + 1) // 3:(step + 1) // 3 + 1].long()), 1)
            if use_cache:
                if output.past_key_values is None:
                    raise RuntimeError('AR coordinate decoder did not return a KV cache')
                output = self.decoder(inputs_embeds=next_embedding, attention_mask=attention,
                    past_key_values=output.past_key_values, use_cache=True, return_dict=True)
            else:
                embeddings = torch.cat((embeddings, next_embedding), 1)
                output = self.decoder(inputs_embeds=embeddings, attention_mask=attention, use_cache=False, return_dict=True)
        prediction = dequantize_xyz(torch.stack(generated, 1), self.generator_config.bins).reshape(batch, frames, joints, 3)
        is_reference = torch.arange(frames, device=features.device)[None] == reference[:, None]
        prediction = torch.where(is_reference[:, :, None, None], xyz[:, None].float(), prediction)
        return prediction.masked_fill(~mask[:, None, :, None], 0.)

    def configuration(self):
        return {'conditioner': asdict(self.config), 'generator': asdict(self.generator_config)}
