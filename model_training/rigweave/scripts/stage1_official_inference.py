"""UniRig's pinned beam-sampling recipe, applied to the dynamic model conditions."""
import hashlib

import torch
from transformers import GenerationConfig, LogitsProcessorList

from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor


OFFICIAL_OPTIONS = dict(max_new_tokens=2048, num_return_sequences=1, num_beams=15,
    do_sample=True, top_k=5, top_p=.95, repetition_penalty=3., temperature=1.5)
POLICY = 'dynamic_unirig_official_beam_sampling_6793c664'


def reference_seed(asset, raw_frame, base_seed=20261029):
    key = f'{POLICY}|{base_seed}|{asset}|{raw_frame}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')


@torch.inference_mode()
def generate_condition(decoder, condition, tokenizer, *, seed, capacity_only=False):
    if decoder.training or condition.shape[0] != 1 or condition.dtype != torch.float32:
        raise ValueError('official inference requires one FP32 evaluation condition')
    start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')],
                         device=condition.device, dtype=torch.long)
    prompt = torch.cat((condition, decoder.get_input_embeddings()(start[None])), dim=1)
    grammar = original_xyz_vocabulary_processor(tokenizer, start)

    def processor(ids, scores):
        scores = grammar(ids, scores)
        if capacity_only:
            scores[:, tokenizer.eos] = float('-inf')
        return scores

    settings = GenerationConfig(**OFFICIAL_OPTIONS, use_cache=True,
        bos_token_id=tokenizer.bos, eos_token_id=tokenizer.eos, pad_token_id=tokenizer.pad,
        forced_bos_token_id=None, forced_eos_token_id=None)
    devices = [condition.device.index] if condition.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if condition.is_cuda:
            torch.cuda.manual_seed(seed)
        generated = decoder.generate(inputs_embeds=prompt, attention_mask=torch.ones(
            prompt.shape[:2], dtype=torch.long, device=prompt.device), generation_config=settings,
            logits_processor=LogitsProcessorList([processor]))
    suffix = generated[0].cpu().tolist()
    ended = tokenizer.eos in suffix
    if ended:
        suffix = suffix[:suffix.index(tokenizer.eos)+1]
    if capacity_only and (ended or len(suffix) != OFFICIAL_OPTIONS['max_new_tokens']):
        raise RuntimeError('beam capacity did not exercise the full token limit')
    return {'success': ended, 'status': 'eos' if ended else 'max_tokens',
        'tokens': start.cpu().tolist() + suffix, 'seed': seed, 'policy': POLICY,
        'generation_kwargs': dict(OFFICIAL_OPTIONS), 'capacity_only': capacity_only}
