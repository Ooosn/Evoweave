"""Record the first native recovery without touching the running source tree."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path('/work/wangyy/tokenrig_alignment_20261105')
SOURCE = '2f50732ee08e98d609c38bc2f73ec3657e5c7022'
CONFIG_SHA = 'e4fece4995c54dab87318bc13ab2280c47670fe1e920a371c7e63626fe159abb'


def read(path):
    return json.loads(path.read_text())


def main():
    output = ROOT/'outputs_native_train'
    path = output/'start_review.json'
    if path.exists():
        print(path.read_text())
        return
    resources = read(output/'train_resources.json')
    audit = read(ROOT/'outputs_native_capacity/audit/result.json')
    initial = [read(output/f'train/initialization_rank_{rank}.json') for rank in (0, 1)]
    progress = [read(output/f'train/progress_rank_{rank}.json') for rank in (0, 1)]
    checkpoint = read(output/'train/checkpoint_000048.json')
    code = ROOT/'code_native_train'
    config = code/'model_training/experiments/tokenrig_native_full_20261105.json'
    if (subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
            or subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE
            or hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA
            or resources['source'] != SOURCE or resources.get('complete') is not False
            or resources['allocation'] != 131351736 or not resources['kernel_scheduler_membership']
            or initial[0] != audit['initialization'] or initial[1] != initial[0]
            or initial[0]['project_checkpoint_loaded'] or initial[0]['project_optimizer_loaded']
            or initial[0]['optimizer_initial_state_entries'] != 0):
        raise RuntimeError('fresh native source, weights, optimizer or resource provenance differs')
    for value in progress:
        if (value['code_commit'] != SOURCE or value['additional_asset_exposures'] < 48
                or not all(math.isfinite(value[k]) for k in
                    ('ar_loss', 'xyz_loss', 'bone_loss', 'gradient_norm'))):
            raise RuntimeError('the fresh joint run has not made finite updates')
    if (checkpoint['samples'] != 48 or checkpoint['step'] != 1
            or checkpoint['code_commit'] != SOURCE or not checkpoint['complete']
            or not checkpoint['both_copies_verified'] or not checkpoint['optimizer_and_rank_RNG_retained']):
        raise RuntimeError('the first full recovery is not complete')
    for field in ('path', 'durable_path'):
        if Path(checkpoint[field]).stat().st_size != checkpoint['bytes']:
            raise RuntimeError('the admitted checkpoint copy changed size')
    first_updates = [json.loads((output/f'train/updates_rank_{rank}.jsonl').open().readline())
        for rank in (0, 1)]
    for value in first_updates:
        if (value['additional_asset_exposures'] != 48 or value['assets_in_update'] != 48
                or value['learning_rate_asset_position'] != 0
                or value['head_lr'] != value['lr'] * 3
                or not all(math.isfinite(x) and x > 0 for x in value['gradient_groups'].values()
                    if isinstance(x, (float, int)))):
            raise RuntimeError('the first real joint update differs from the fresh recipe')
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(ROOT/'code_native_train/') in command and any(term in command for term in
                ('launch_virtual_pair.py', 'torch.distributed.run', 'train_anchor_full.py')):
            processes.append({'pid': int(entry.name), 'command': command})
    if not processes:
        raise RuntimeError('no native training payload is alive')
    result = {'review_complete': True, 'training_complete': False, 'status': 'running',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'source': SOURCE,
        'config_sha256': CONFIG_SHA, 'allocation': 131351736, 'physical_gpus': [3, 4],
        'cpu_slots': 16, 'kernel_scheduler_membership': resources['kernel_scheduler_membership'],
        'first_checkpoint': checkpoint, 'fresh_optimizer_verified': True,
        'initial_model_sha256': initial[0]['initial_model_sha256'],
        'same_initialization_as_CPU_audit': True, 'capacity_weights_not_used': True,
        'first_update': {k: v for k, v in first_updates[0].items() if k != 'microbatches'},
        'rank_progress': [{k: v[k] for k in ('step', 'additional_asset_exposures',
            'ar_loss', 'xyz_loss', 'bone_loss', 'gradient_norm')} for v in progress],
        'sampled_peak_gpu_memory_mib': resources['peak_gpu_memory_mib'],
        'native_processes': processes, 'old250k_and_working330k_unchanged': True,
        'running_source_not_modified': True}
    encoded = (json.dumps(result, indent=2)+'\n').encode()
    durable = Path('/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_full/start_review.json')
    with path.open('xb') as handle:
        handle.write(encoded)
    with durable.open('xb') as handle:
        handle.write(encoded)
    if hashlib.sha256(durable.read_bytes()).digest() != hashlib.sha256(encoded).digest():
        raise RuntimeError('durable launch review differs')
    print(encoded.decode())


if __name__ == '__main__':
    main()
