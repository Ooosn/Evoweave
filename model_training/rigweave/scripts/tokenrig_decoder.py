"""Released TokenRig Qwen, restricted to its native skeleton vocabulary."""
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType

import torch


class TokenRigSkeletonTokenizer:
    skeleton_class = 'articulation'

    def __init__(self, native, input_type):
        self.native = native
        self.input_type = input_type

    def __getattr__(self, name):
        return getattr(self.native, name)

    def tokenize_skeleton(self, joints, parents, cls):
        if cls != self.skeleton_class:
            raise ValueError(f'unsupported TokenRig skeleton class: {cls}')
        return self.native.tokenize(self.input_type(joints=joints, parents=parents, cls=cls))


def build_tokenrig_decoder(spec, *, load_weights):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from virtual_query_xyz_runtime import sha256

    checkpoint = Path(spec['checkpoint'])
    if sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('TokenRig release checkpoint hash changed')
    root = Path(spec['source']).resolve()
    paths = ('tokenizer/parse.py', 'tokenizer/spec.py', 'tokenizer/tokenizer_part.py',
             'data/order.py', 'data/spec.py')
    hashes = {path: sha256(root/'src'/path) for path in paths}
    if hashes != spec['tokenizer_source_sha256']:
        raise ValueError('TokenRig tokenizer source changed')
    # Both releases use the top-level name "src"; keep UniRig's encoder imports intact.
    package_name = '_evoweave_tokenrig_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(root/'src')]
        sys.modules[package_name] = package
    parse = importlib.import_module(f'{package_name}.tokenizer.parse')
    input_type = importlib.import_module(f'{package_name}.tokenizer.spec').TokenizeInput
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    hp = payload['hyper_parameters']
    codec = dict(hp['tokenizer_config'])
    if (codec['num_discrete'] != 256 or list(codec['continuous_range']) != [-1, 1]
            or 'articulation' not in codec['cls_token_id']):
        raise ValueError('unexpected TokenRig skeleton codec')
    # The generic articulation branch has no named-joint template.
    codec.pop('order_config', None)
    native = parse.get_tokenizer(**codec)
    tokenizer = TokenRigSkeletonTokenizer(native, input_type)
    args = dict(hp['model_config']['llm'])
    args.pop('pretrained_model_name_or_path')
    config_dir = Path(spec['qwen_config_root'])
    if sha256(config_dir/'config.json') != spec['qwen_config_sha256']:
        raise ValueError('TokenRig Qwen base configuration changed')
    config = Qwen3Config.from_pretrained(config_dir, local_files_only=True, **args)
    state = {k.removeprefix('transformer.'): v for k, v in payload['state_dict'].items()
             if k.startswith('transformer.')}
    embedding, head = state['model.embed_tokens.weight'], state['lm_head.weight']
    if (config.hidden_size != 896 or embedding.shape != head.shape
            or tuple(embedding.shape) != (33036, 896) or tokenizer.vocab_size != 267):
        raise ValueError('unexpected released TokenRig embedding/head shape')
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id, config.eos_token_id, config.pad_token_id = tokenizer.bos, tokenizer.eos, tokenizer.pad
    state['model.embed_tokens.weight'] = embedding[:config.vocab_size]
    state['lm_head.weight'] = head[:config.vocab_size]
    # Preserve the release's two effective tables, including untied tables if they differ.
    config.tie_word_embeddings = config.tie_word_embeddings and torch.equal(
        state['model.embed_tokens.weight'], state['lm_head.weight'])
    config._attn_implementation = 'eager' if spec.get('eager') else 'sdpa'
    decoder = Qwen3ForCausalLM(config)
    if load_weights:
        decoder.load_state_dict(state, strict=True)
    return decoder, tokenizer, {'checkpoint_sha256': spec['checkpoint_sha256'],
        'qwen_config_sha256': spec['qwen_config_sha256'], 'tokenizer_source_sha256': hashes,
        'width': config.hidden_size, 'skeleton_vocab_size': config.vocab_size,
        'class_id': tokenizer.cls_name_to_token(tokenizer.skeleton_class),
        'tied_embeddings': config.tie_word_embeddings, 'strict_weights_loaded': load_weights}
