"""Run unmodified SkinTokens demo.py with a private Blender loopback service."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import time

from skintokens_original_test_20261109 import (
    CASES, OVERLAY, PANEL, RELEASE, RELEASE_SHA, ROOT, SCOPE, UPSTREAM,
    UPSTREAM_COMMIT, VAE, VAE_SHA, digest, require_digest, score_case, write_new,
)


CODE = Path(__file__).resolve().parents[2]
ENTRY = 'server_ops/hgc/run_skintokens_original_test_20261109.py'
CONFIG = CODE/'model_training/experiments'/f'{SCOPE}.json'
DURABLE = Path('/home/wangyy/evorig/outputs/tokenrig_alignment_20261105') / SCOPE
OUTPUT = ROOT/'outputs'
PYTHON = ROOT/'venv/bin/python'
DEVICE = 'GPU-1fb4221b-91b0-0df6-8615-b36fbdd55bf7'
JOB = 131942123
CPUS = {6,7,9,10,11,12,14,15}
STOP_UTC = '2026-11-11T04:50:00+00:00'
MAX_MIB = 79872
MAX_RSS = 64*1024**3
TIMEOUT_SECONDS = 75*60


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def command(args, *, cwd=CODE):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=35).strip()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    write_new(tmp, value)
    tmp.replace(path)


def verify_config(config):
    expected = {
        'scope': SCOPE, 'upstream_commit': UPSTREAM_COMMIT, 'cases': CASES,
        'weights_review_sha256': digest(ROOT/'weights_review.json'),
        'panel_manifest_sha256': digest(PANEL/'manifest.json'),
        'release_sha256': RELEASE_SHA, 'overlay_sha256': read(ROOT/'weights_review.json')['overlay_sha256'],
        'VAE_sha256': VAE_SHA, 'python': str(PYTHON), 'job': JOB,
        'physical_gpu': 3, 'gpu_uuid': DEVICE, 'affinity': sorted(CPUS),
        'stop_utc': STOP_UTC, 'max_gpu_memory_mib': MAX_MIB, 'max_rss_bytes': MAX_RSS,
        'timeout_seconds': TIMEOUT_SECONDS, 'input_root': str(PANEL), 'output_root': str(OUTPUT),
        'durable_root': str(DURABLE),
        'cli': {'top_k': 5, 'top_p': .95, 'temperature': 1., 'repetition_penalty': 2., 'num_beams': 10,
                'do_sample': True, 'max_length': 2048, 'use_transfer': True, 'use_skeleton': False,
                'use_postprocess': False, 'hf_path': None},
    }
    if config != expected:
        raise ValueError('official comparison differs from its admitted inputs, defaults or allocation')


def verify_launch(config):
    sys.path.insert(0, str(CODE/'model_training/rigweave/scripts'))
    from virtual_pair_guard import require_job_membership, verify_allocation

    verify_config(config)
    state = read(CODE/'model_training/state/current.json')
    op = state['active_operation']
    head = command(['git','rev-parse','HEAD'])
    if (CODE != ROOT/'code' or socket.gethostname().split('.')[0] != 'pcg01'
            or os.environ.get('JOB_ID') != str(JOB) or os.environ.get('NSLOTS') != '16'
            or not CPUS <= os.sched_getaffinity(0)
            or resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY
            or op['experiment'] != SCOPE or op['explicit_human_authorized'] is not True
            or op['complete'] or op['formal_training_allowed'] is not False
            or op['required_entrypoint'] != ENTRY or op['config_sha256'] != digest(CONFIG)
            or op['output_root'] != str(OUTPUT) or op['optimizer_updates_allowed'] != 0
            or command(['git','status','--porcelain']) or command(['git','status','--porcelain'],cwd=UPSTREAM)
            or command(['git','rev-parse','HEAD'],cwd=UPSTREAM) != UPSTREAM_COMMIT):
        raise RuntimeError('source, path, state or native environment differs')
    require_job_membership(Path('/proc/self/cgroup').read_text(), JOB)
    recipe = read(CODE/'model_training/experiments/skintokens_static_5k_20261109.json')
    verify_allocation(recipe, command(['qstat','-j',str(JOB)]))
    subprocess.run(['git','merge-base','--is-ancestor',op['code_commit'],head], cwd=CODE, check=True)
    subprocess.run(['git','diff','--quiet',op['code_commit'],head,'--','server_ops/hgc'], cwd=CODE, check=True)
    branch = 'refs/heads/codex/skintokens-original-test-20261109'
    if command(['git','ls-remote','origin',branch]).split() != [head, branch]:
        raise RuntimeError('exact inference admission was not published privately')
    if command(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader']) != DEVICE:
        raise RuntimeError('H100 identity changed')
    if command(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader']):
        raise RuntimeError('approved inference GPU is occupied; no process was stopped')
    for path, sha in ((RELEASE, RELEASE_SHA), (VAE, VAE_SHA), (OVERLAY, config['overlay_sha256'])):
        require_digest(path, sha)
    return head


def loopback():
    links = json.loads(command(['ip','-j','link','show']))
    if (os.readlink('/proc/self/ns/net') == os.environ['EVOWEAVE_HOST_NETNS']
            or [x['ifname'] for x in links] != ['lo'] or 'UP' not in links[0]['flags']):
        raise RuntimeError('refuse to start the pickled Blender service on a shared network')


def measure_pair(outputs):
    metrics = ('J2J_percent','J2B_percent','B2B_percent')
    paired = [row for row in outputs if all('metrics' in row[k] for k in ('release','fine_tuned'))]
    means = {}
    if paired:
        for name in metrics:
            left, right = ([row[k]['metrics'][name] for row in paired] for k in ('release','fine_tuned'))
            means[name] = {'release': statistics.fmean(left), 'fine_tuned': statistics.fmean(right),
                'fine_tuned_minus_release': statistics.fmean(b-a for a,b in zip(left,right,strict=True)),
                'fine_tuned_better_assets': sum(b<a for a,b in zip(left,right,strict=True))}
    return {'paired_assets': len(paired), 'pairs_attempted': len(outputs), 'mean_on_paired': means,
        'valid_by_arm': {k:sum('metrics' in row[k] for row in outputs) for k in ('release','fine_tuned')}}


def worker():
    loopback()
    config = read(CONFIG)
    panel = read(PANEL/'manifest.json')
    if len(panel['cases']) != CASES:
        raise ValueError('panel size changed')
    with (OUTPUT/'upstream_help.log').open('xb') as stream:
        subprocess.run([str(PYTHON), '-B', 'demo.py', '--help'], cwd=UPSTREAM,
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
    outputs = []
    for case in panel['cases']:
        mesh = Path(case['mesh'])
        require_digest(mesh, case['mesh_sha256'])
        row = {'asset_id': case['id'], 'seed': case['seed'], 'mesh_sha256': case['mesh_sha256']}
        for arm, checkpoint in (('release',RELEASE), ('fine_tuned',OVERLAY)):
            path = OUTPUT/f'{case["ordinal"]:02d}_{arm}.glb'
            log = path.with_suffix('.log')
            atomic(OUTPUT/'progress.json', {'case':case['ordinal'], 'total':CASES, 'arm':arm, 'start_utc':now()})
            args = [str(PYTHON), '-B', str(CODE/'server_ops/hgc/skintokens_original_test_20261109.py'),
                    'demo', str(case['seed']), '--input', str(mesh), '--output', str(path),
                    '--model_ckpt', str(checkpoint), '--use_transfer']
            started = time.monotonic()
            with log.open('xb') as stream:
                result = subprocess.run(args, cwd=UPSTREAM, stdout=stream, stderr=subprocess.STDOUT, timeout=480)
            elapsed = time.monotonic()-started
            record = {'returncode': result.returncode, 'seconds': elapsed, 'output': path.name,
                      'log': log.name, 'argv': args}
            if result.returncode or not path.is_file():
                content = log.read_text(errors='replace')
                if any(flag in content for flag in ('ImportError:', 'ModuleNotFoundError:',
                        'bpy_server failed to start', 'CUDA out of memory', 'Torch not compiled with CUDA')):
                    raise RuntimeError('official runtime failed before valid inference: '+log.name)
                record['error'] = 'official pipeline did not export a rig'
            else:
                try:
                    record.update(metrics=score_case(case,path), sha256=digest(path))
                except ValueError as exc:
                    record['error'] = str(exc)
            row[arm] = record
            # The CLI owns its service. Wait only for its atexit cleanup.
            for _ in range(50):
                with socket.socket() as client:
                    client.settimeout(.2)
                    if client.connect_ex(('127.0.0.1', 59876)) != 0:
                        break
                time.sleep(.1)
            else:
                raise RuntimeError('a previous Blender service survived official CLI exit')
        outputs.append(row)
        atomic(OUTPUT/'paired_rows.json', outputs)
    write_new(OUTPUT/'result.json', {'complete': True, 'finished_utc': now(),
        'configuration_sha256': digest(CONFIG), 'upstream_commit': UPSTREAM_COMMIT, 'decode': config['cli'],
        'normalization': 'Distances as percent of longest side of original mesh bounding box; no fitting to GT',
        **measure_pair(outputs)})


def supervise():
    sys.path.insert(0, str(CODE/'model_training/jobs'))
    from launch_legacy_single_target import cleanup_payload, descendants, resource_cgroups, verify_child_cgroups

    config = read(CONFIG)
    head = verify_launch(config)
    if OUTPUT.exists() or DURABLE.exists():
        raise FileExistsError('official test outputs already exist')
    OUTPUT.mkdir()
    DURABLE.mkdir(parents=True)
    os.sched_setaffinity(0, CPUS)
    actual_cgroup = Path('/proc/self/cgroup').read_text()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('cannot retain ownership of the Blender process tree')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': DEVICE, 'PYTHONDONTWRITEBYTECODE': '1',
        'EVOWEAVE_HOST_NETNS': os.readlink('/proc/self/ns/net'), 'TMPDIR': str(ROOT/'tmp'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false'}
    for name in tuple(env):
        if name.lower().endswith('_proxy') or name in ('PYTHONPATH', 'USE_FLASH3'):
            env.pop(name)
    cmd = ['unshare','--user','--map-root-user','--net','/bin/bash','-c',
           'set -e; ip link set lo up; exec "$@"','loopback',str(PYTHON),'-B',str(CODE/ENTRY),'worker']
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    started = time.monotonic()
    receipt = {'source':head,'job':JOB,'kernel_scheduler_membership':True,'actual_cgroup':actual_cgroup,
               'physical_gpu':3,'gpu_uuid':DEVICE,'cpu_affinity':sorted(CPUS),'complete':False,'passed':False,
               'started_utc':now(),'peak_gpu_memory_mib':0,'peak_rss_bytes':0}
    log = OUTPUT/'runner.log'
    with log.open('xb') as stream:
        child = subprocess.Popen(cmd,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['child_pid'] = child.pid
        try:
            while child.poll() is None:
                if (interrupted or time.monotonic()-started >= TIMEOUT_SECONDS
                        or datetime.now(timezone.utc) >= datetime.fromisoformat(STOP_UTC)):
                    raise RuntimeError('resource deadline or interruption reached')
                rss = 0
                for pid, (_, amount) in descendants(child.pid).items():
                    try:
                        if verify_child_cgroups(pid,resource_cgroups(actual_cgroup)):
                            if not os.sched_getaffinity(pid) <= CPUS:
                                raise RuntimeError('a process escaped the CPU budget')
                            rss += amount
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                memory = int(command(['nvidia-smi','-i',DEVICE,'--query-gpu=memory.used','--format=csv,noheader,nounits']))
                receipt.update(peak_gpu_memory_mib=max(receipt['peak_gpu_memory_mib'],memory),
                               peak_rss_bytes=max(receipt['peak_rss_bytes'],rss),observed_utc=now())
                atomic(OUTPUT/'resources.json', receipt)
                if memory >= MAX_MIB or rss > MAX_RSS:
                    raise RuntimeError('inference exceeded resource ceiling')
                time.sleep(5)
            receipt['passed'] = child.returncode == 0 and (OUTPUT/'result.json').is_file()
        except BaseException as exc:
            receipt['error'] = repr(exc)
            raise
        finally:
            try:
                cleaned = cleanup_payload(child, None)
                receipt.update(cleanup_complete=True, cleaned_payload_pids=cleaned)
            except BaseException as exc:
                receipt.update(cleanup_complete=False, cleanup_error=repr(exc), passed=False)
                raise
            finally:
                receipt.update(complete=True, exit_code=child.poll(), finished_utc=now())
                atomic(OUTPUT/'resources.json', receipt)
                for path in (CONFIG,PANEL/'manifest.json',ROOT/'weights_review.json',*OUTPUT.glob('*.json')):
                    shutil.copyfile(path, DURABLE/path.name)
                pack = DURABLE/'outputs.tar.gz'
                with tarfile.open(pack, 'x:gz') as tar:
                    tar.add(OUTPUT, arcname='outputs')
                write_new(DURABLE/'archive.json', {'path':str(pack),'sha256':digest(pack)})
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('supervise','worker'))
    mode = parser.parse_args().mode
    if mode == 'worker':
        worker()
    else:
        raise SystemExit(supervise())
