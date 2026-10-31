"""Named diagnostic policies; the established official generator stays unchanged."""
import torch
from transformers import GenerationConfig, LogitsProcessorList

import stage1_official_inference as official
from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor

POLICIES = ('official', 'strict_sampling', 'strict_beam')


def coordinate_state(ids, tokenizer):
    """Track complete coordinate groups, including both endpoints of a branch."""
    remaining, has_joint, ended = 0, False, False
    for token in ids:
        token = int(token)
        if ended:
            if token not in (tokenizer.pad, tokenizer.eos):
                raise ValueError('non-padding token after EOS')
            continue
        if token < tokenizer.num_discrete:
            if remaining == 0:
                remaining = 3
            remaining -= 1
            if remaining == 0:
                has_joint = True
        elif token == tokenizer.eos:
            if remaining or not has_joint:
                raise ValueError('EOS before a complete coordinate structure')
            ended = True
        elif token == tokenizer.token_id_branch:
            if remaining or not has_joint:
                raise ValueError('branch before a complete parent structure')
            remaining = 6
        elif token in tokenizer.part_token_to_name:
            if remaining:
                raise ValueError('part marker inside coordinate group')
        else:
            raise ValueError('unexpected generated token')
    return remaining, has_joint, ended


def strict_processor(tokenizer, start):
    original = original_xyz_vocabulary_processor(tokenizer, start)

    def process(ids, scores):
        output = original(ids, scores)
        for index, suffix in enumerate(ids):
            remaining, has_joint, ended = coordinate_state(suffix.detach().cpu().tolist(), tokenizer)
            if ended:
                continue
            if remaining:
                output[index, tokenizer.num_discrete:] = float('-inf')
            elif not has_joint:
                output[index, tokenizer.eos] = float('-inf')
                output[index, tokenizer.token_id_branch] = float('-inf')
        return output
    return process


def options(policy):
    if policy not in POLICIES:
        raise ValueError('unknown diagnostic decoding policy')
    result = dict(official.OFFICIAL_OPTIONS)
    if policy == 'strict_beam':
        result['do_sample'] = False
        for field in ('top_k', 'top_p', 'temperature'):
            result.pop(field)
    return result


@torch.inference_mode()
def generate(decoder, condition, tokenizer, *, seed, policy):
    if policy == 'official':
        return official.generate_condition(decoder, condition, tokenizer, seed=seed)
    settings = options(policy)
    if decoder.training or condition.shape[0] != 1 or condition.dtype != torch.float32:
        raise ValueError('diagnostic inference requires one FP32 evaluation condition')
    start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')],
                         device=condition.device, dtype=torch.long)
    prompt = torch.cat((condition, decoder.get_input_embeddings()(start[None])), dim=1)
    generation = GenerationConfig(**settings, use_cache=True,
        bos_token_id=tokenizer.bos, eos_token_id=tokenizer.eos, pad_token_id=tokenizer.pad,
        forced_bos_token_id=None, forced_eos_token_id=None)
    devices = [condition.device.index] if condition.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if condition.is_cuda:
            torch.cuda.manual_seed(seed)
        output = decoder.generate(inputs_embeds=prompt, attention_mask=torch.ones(
            prompt.shape[:2], dtype=torch.long, device=prompt.device), generation_config=generation,
            logits_processor=LogitsProcessorList([strict_processor(tokenizer, start)]))
    suffix = output[0].cpu().tolist()
    ended = tokenizer.eos in suffix
    if ended:
        suffix = suffix[:suffix.index(tokenizer.eos)+1]
    return {'success': ended, 'status': 'eos' if ended else 'max_tokens',
        'tokens': start.cpu().tolist()+suffix, 'seed': seed,
        'policy': 'diagnostic_'+policy+'_v1', 'generation_kwargs': settings, 'capacity_only': False}
