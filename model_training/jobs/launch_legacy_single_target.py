#!/usr/bin/env python3
"""Launch and supervise a payload only inside the approved HGC allocation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from supervise_structure_xyz_resources import descendants


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def cpu_ticks(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    return int(fields[11]) + int(fields[12])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'capacity', 'train'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = config['runtime']
    root = Path(runtime['repo'])
    sys.path.insert(0, str(root / 'model_training/rigweave/scripts'))
    from train_legacy_single_target import require_job_membership, verify_launch
    # Read actual kernel provenance before setting CUDA visibility or importing torch.
    require_job_membership(Path('/proc/self/cgroup').read_text(), runtime['allocation'])
    if (os.environ.get('JOB_ID') != str(runtime['allocation'])
            or int(os.environ.get('NSLOTS', '0')) != runtime['cpu_slots']):
        raise RuntimeError('the launcher must inherit the real job ID and CPU grant')
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < runtime['cpu_slots']:
        raise RuntimeError('insufficient allowed CPUs')
    binding = set(allowed[:runtime['cpu_slots']])
    os.sched_setaffinity(0, binding)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, runtime['physical_gpus']))
    os.environ.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'RIGWEAVE_DISABLE_OPEN3D',
        'RIGWEAVE_DISABLE_LIGHTNING_IMPORT')})
    os.environ.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_MODULE_LOADING='LAZY',
        EVOWEAVE_UNIRIG_ROOT=runtime['unirig_root'], EVOWEAVE_OPT_CONFIG_ROOT=runtime['opt_config_root'])
    os.environ['PYTHONPATH'] = os.pathsep.join((str(root / 'model_training/rigweave/src'),
        str(root / 'model_training/rigweave/scripts'), runtime['unirig_root'], runtime['extra_pythonpath']))
    source = verify_launch(args, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_name(args.output.name + '_resources.json')
    if receipt_path.exists() or args.output.exists():
        raise RuntimeError('refuse to overwrite a prior stage or resource receipt')
    command = [runtime['python'], '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc_per_node=2', str(root / 'model_training/rigweave/scripts/train_legacy_single_target.py'),
        '--config', str(args.config), '--output', str(args.output), '--mode', args.mode]
    receipt = {'complete': False, 'passed': False, 'source': source, 'mode': args.mode,
        'allocation': runtime['allocation'], 'kernel_scheduler_membership': True,
        'physical_gpus': runtime['physical_gpus'], 'cpu_affinity': sorted(binding),
        'rss_limit_bytes': runtime['rss_limit_bytes'], 'peak_tree_rss_bytes': 0,
        'aggregate_cpu_seconds': 0., 'supervisor_pid': os.getpid()}
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    receipt['child_pid'] = child.pid
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    last_ticks, total_ticks, next_grant = {}, 0, 0.
    try:
        while child.poll() is None:
            if interrupted:
                raise RuntimeError(f'supervisor interrupted by signal {interrupted[0]}')
            now = time.monotonic()
            if now >= next_grant:
                grant = subprocess.check_output(['qstat', '-j', str(runtime['allocation'])], text=True, timeout=30)
                if f'pcg01i: /dev/nvidia5 /dev/nvidia6' not in grant:
                    raise RuntimeError('the authorized GPU grant changed')
                next_grant = now + 60
            processes = descendants(child.pid)
            rss = sum(value[1] for value in processes.values())
            if rss > runtime['rss_limit_bytes']:
                raise RuntimeError('the training process tree exceeded64GiB RAM')
            for pid in processes:
                try:
                    require_job_membership((Path('/proc') / str(pid) / 'cgroup').read_text(), runtime['allocation'])
                    for task in (Path('/proc') / str(pid) / 'task').iterdir():
                        if not os.sched_getaffinity(int(task.name)) <= binding:
                            raise RuntimeError('a child thread escaped its CPU affinity')
                    ticks = cpu_ticks(pid)
                    total_ticks += max(0, ticks - last_ticks.get(pid, 0))
                    last_ticks[pid] = ticks
                except (FileNotFoundError, ProcessLookupError):
                    pass
            text = subprocess.check_output(['nvidia-smi', '-i', '5,6',
                '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                text=True, timeout=30)
            memory = [[int(v.strip()) for v in line.split(',')] for line in text.strip().splitlines()]
            if sorted(row[0] for row in memory) != [5, 6] or any(used > total*.85 for _, used, total in memory):
                raise RuntimeError('the authorized GPUs exceeded85percent memory or changed identity')
            receipt.update(last_observed_utc=datetime.now(timezone.utc).isoformat(), tree_rss_bytes=rss,
                peak_tree_rss_bytes=max(rss, receipt['peak_tree_rss_bytes']), process_count=len(processes),
                aggregate_cpu_seconds=total_ticks/os.sysconf('SC_CLK_TCK'), gpu_memory_mib=memory)
            write_json(receipt_path, receipt)
            for _ in range(5):
                if child.poll() is not None or interrupted:
                    break
                time.sleep(1)
        receipt.update(complete=True, passed=child.returncode == 0, exit_code=child.returncode)
    except BaseException as exc:
        receipt['error'] = repr(exc)
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        except ProcessLookupError:
            pass
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    return child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
