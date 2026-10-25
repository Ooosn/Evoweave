#!/usr/bin/env python3
"""Keep the granted shell alive and launch approved stages as actual job children."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_request(request, directory):
    if set(request) != {'stage', 'mode', 'config', 'output', 'source'}:
        raise ValueError('unexpected launch request fields')
    stage, mode = request['stage'], request['mode']
    if mode not in ('audit', 'capacity', 'train') or not re.fullmatch(mode + r'_v[1-9][0-9]*', stage):
        raise ValueError('unexpected stage')
    config, output = Path(request['config']), Path(request['output'])
    if config.resolve().parent != ROOT / 'model_training/experiments':
        raise ValueError('config escaped the isolated checkout')
    if output.resolve() != directory.parent / 'outputs' / stage:
        raise ValueError('output escaped the approved run root')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if source != request['source']:
        raise ValueError('launch commit differs from the request')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('launch checkout is dirty')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    matches = [r for r in state['active_operation']['runs'] if r.get('approved')
               and r['mode'] == mode and r['config'] == str(config) and r['output'] == str(output)]
    if len(matches) != 1 or matches[0]['config_sha256'] != hashlib.sha256(config.read_bytes()).hexdigest():
        raise ValueError('the exact stage is not approved')
    runtime = json.loads(config.read_text())['runtime']
    if (str(runtime['allocation']) != os.environ.get('JOB_ID')
            or runtime['cpu_slots'] != int(os.environ.get('NSLOTS', '0'))
            or runtime['execution_mode'] != 'scheduler_job'):
        raise ValueError('stage is not bound to this real scheduler grant')
    return [runtime['python'], '-B', str(ROOT / 'model_training/jobs/launch_virtual_pair.py'),
            '--config', str(config), '--output', str(output), '--mode', mode]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory.parent != ROOT.parent or not directory.name.startswith('allocation_'):
        raise ValueError('allocation directory escaped the isolated run root')
    if not os.environ.get('JOB_ID', '').isdigit() or os.environ.get('NSLOTS') != '16':
        raise RuntimeError('keeper requires an actual16-slot scheduler job')
    directory.mkdir(parents=True, exist_ok=False)
    requests = directory / 'requests'
    requests.mkdir()
    provenance = {'job_id': int(os.environ['JOB_ID']), 'cpu_slots': int(os.environ['NSLOTS']),
        'hostname': socket.gethostname(), 'pid': os.getpid(), 'ppid': os.getppid(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cgroups': Path('/proc/self/cgroup').read_text(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'scheduler_environment': {k: v for k, v in os.environ.items()
            if k in ('JOB_ID', 'NSLOTS', 'QUEUE', 'HOSTNAME', 'CUDA_VISIBLE_DEVICES')
            or k.startswith('SGE_HGR_')}}
    write_json(directory / 'grant.json', provenance)
    print(json.dumps(provenance), flush=True)
    while True:
        for path in sorted(requests.glob('*.json')):
            receipt = directory / (path.stem + '_launch.json')
            if receipt.exists():
                continue
            record = {'request': str(path), 'started_utc': datetime.now(timezone.utc).isoformat(),
                      'complete': False, 'keeper_pid': os.getpid()}
            write_json(receipt, record)
            try:
                request = json.loads(path.read_text())
                command = validate_request(request, directory)
                with (directory / (path.stem + '_guard_begin.json')).open('x') as handle:
                    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                                    'begin'], check=True, stdout=handle)
                record['command'] = command
                with (directory.parent / (request['stage'] + '.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    record['supervisor_pid'] = child.pid
                    write_json(receipt, record)
                    record['exit_code'] = child.wait()
            except Exception as exc:
                record['error'] = repr(exc)
            record.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(receipt, record)
        time.sleep(5)


if __name__ == '__main__':
    main()
