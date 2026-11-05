"""Released TokenRig Qwen, restricted to its native skeleton vocabulary."""
import hashlib
import importlib
from pathlib import Path
import sys
import subprocess
from types import ModuleType

import torch
from torch import nn


class NativeTokenRigProjection(nn.Linear):
    """Lift the released 512-channel projection into the existing motion width."""

    def __init__(self, input_width, native_width, hidden_width):
        if input_width < native_width:
            raise ValueError('the motion width cannot truncate native encoder channels')
        super().__init__(input_width, hidden_width)
        self.native_width = native_width
        self.normalization = nn.RMSNorm(hidden_width)

    def load_native(self, weight, bias, norm_weight):
        if tuple(weight.shape) != (self.out_features, self.native_width):
            raise ValueError('released condition projection shape differs')
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :self.native_width].copy_(weight)
            self.bias.copy_(bias)
            self.normalization.weight.copy_(norm_weight)

    def forward(self, value):
        return self.normalization(super().forward(value))


def tokenrig_package(spec):
    root = Path(spec['source']).resolve()
    package_name = '_evoweave_tokenrig_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(root/'src')]
        sys.modules[package_name] = package
    return package_name


def initialize_native_motion_identity(model):
    """Initialize residual branches at zero while retaining the native feature basis."""
    encoder = model.surface_tokenizer.michelangelo_encoder
    native_norm = encoder.encoder.ln_post
    width = encoder.width
    if model.config.dim != 2 * width or native_norm is None:
        raise ValueError('native identity initialization requires the reviewed doubled-width lift')
    gamma = native_norm.weight.detach().float()
    beta = native_norm.bias.detach().float()
    if not torch.isfinite(gamma).all() or not torch.isfinite(beta).all() or gamma.abs().min() < 1e-4:
        raise ValueError('native LayerNorm cannot be stably mapped into the motion feature basis')
    lift = nn.Linear(width, 2 * width)
    with torch.no_grad():
        block = torch.diag(gamma.reciprocal())
        lift.weight.copy_(torch.cat((block, block), dim=0))
        lift.bias.copy_(torch.cat((-beta/gamma, -beta/gamma)))
        projection = model.decoder_condition_projection
        native_weight = projection.weight[:, :width].clone()
        projection.weight.copy_(torch.cat((native_weight, native_weight), dim=1) * .5)
        model.condition_norm.weight.copy_(torch.cat((gamma, gamma)))
        model.condition_norm.bias.copy_(torch.cat((beta, beta)))
        for block in model.motion_encoder.blocks:
            for layer in (block.pose_inner, block.anchor_temporal):
                layer.self_attn.out_proj.weight.zero_()
                layer.self_attn.out_proj.bias.zero_()
                layer.linear2.weight.zero_()
                layer.linear2.bias.zero_()
        for layer in model.frame_fusion:
            layer.ffn[-1].weight.zero_()
            layer.ffn[-1].bias.zero_()
    model.surface_tokenizer.output_proj = lift
    return {'kind': 'identity_residuals_native_norm_v1',
        'native_LayerNorm_min_absolute_scale': float(gamma.abs().min()),
        'channel_lift': 'duplicate native standardized512 features to1024; restore native affine after temporal path; average the two native decoder projections',
        'new_residual_output_weights_start_at_zero': True,
        'all_modules_remain_in_forward_and_trainable': True,
        'approximation': 'Repeated LayerNorm epsilon and floating-point rounding remain; measure actual initial generation.'}


def build_tokenrig_conditioning(spec, input_width, *, load_weights):
    from virtual_query_xyz_runtime import sha256

    setting = spec['conditioning']
    if setting['kind'] != 'released_mesh_projection_v1' or input_width != 1024:
        raise ValueError('unreviewed TokenRig conditioning route')
    source = Path(spec['source']).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if revision != setting['source_commit'] or dirty:
        raise ValueError('native encoder source must be the pinned clean TokenRig release')
    path = Path(spec['checkpoint'])
    if sha256(path) != spec['checkpoint_sha256']:
        raise ValueError('TokenRig conditioning checkpoint hash changed')
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    args = dict(payload['hyper_parameters']['model_config']['mesh_encoder'])
    if (args.pop('__target__') != 'michelangelo_encoder' or args['width'] != 512
            or args['num_encoder_layers'] != 8 or args['token_num'] != 512
            or args['query_method'] or args['freeze_encoder']):
        raise ValueError('unexpected native geometry encoder architecture')
    if spec.get('eager'):
        args['flash'] = False
    get_model = importlib.import_module(f'{tokenrig_package(spec)}.model.michelangelo.get_model')
    encoder = get_model.get_encoder_simplified(**args)
    projection = NativeTokenRigProjection(input_width, encoder.width, 896)
    state = payload['state_dict']
    mesh = {k.removeprefix('mesh_encoder.'): v for k, v in state.items() if k.startswith('mesh_encoder.')}
    native_keys = {'output_proj.0.weight', 'output_proj.0.bias', 'output_proj.1.weight'}
    if {k for k in state if k.startswith('output_proj.')} != native_keys:
        raise ValueError('the complete native projection and normalization are required')
    if load_weights:
        encoder.load_state_dict(mesh, strict=True)
        projection.load_native(*(state[k] for k in (
            'output_proj.0.weight', 'output_proj.0.bias', 'output_proj.1.weight')))
    return encoder, projection, {
        'kind': setting['kind'], 'source_commit': revision, 'checkpoint_sha256': spec['checkpoint_sha256'],
        'mesh_tensors': len(mesh), 'native_encoder_width': encoder.width, 'motion_width': input_width,
        'native_encoder_layers': 8, 'native_static_prefix_tokens': 512,
        'dynamic_prefix_tokens': setting.get('prefix_tokens', 1024),
        'projection_and_RMSNorm_loaded': load_weights, 'strict_mesh_weights_loaded': load_weights,
        'channel_lift': 'append_512_zero_channels; native projection weights in first512columns; other columns zero',
        'motion_and_stage2_architecture_unchanged': True,
        'native_static_equivalence_does_not_imply_dynamic_equivalence': True,
    }


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
    package_name = tokenrig_package(spec)
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
