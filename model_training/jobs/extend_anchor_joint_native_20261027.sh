#!/usr/bin/env bash
set -euo pipefail
source ~/1.sh
test "$(sha256sum ~/HGC_GPU_QLOGIN_QSUB_README.sh | cut -d' ' -f1)" = bfe3d5b9b54a39fa903590d5694211630a4f5310d944b212fd134d8cbfb28730
ssh -o BatchMode=yes pcg01i /bin/bash -s <<'NODE'
set -euo pipefail
source ~/1.sh
/home/wangyy/miniconda3/envs/mygs/bin/python -B - <<'PY'
from pathlib import Path
import json
import os
import subprocess
root=Path('/work/wangyy/virtual_query_matched_5k_20261025')
repo=root/'code'
proc=Path('/proc/1108368')
groups=proc.joinpath('cgroup').read_text()
assert ':devices:/AGE/130675914.1/master' in groups and ':cpuset:/AGE/130675914.1/master' in groups
assert not proc.joinpath('task/1108368/children').read_text().strip()
resource=json.loads((root/'outputs/capacity_v18_resources.json').read_text())
assert resource['complete'] and resource['passed'] and resource['exit_code']==0 and resource['payload_cleanup_complete']
uuids=['GPU-1fb4221b-91b0-0df6-8615-b36fbdd55bf7','GPU-0544830a-4dac-9fe3-a099-95ddc2240b72']
gpu=subprocess.check_output(['nvidia-smi','-i',','.join(uuids),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
assert len(gpu.splitlines())==2 and all(int(line.split(',')[1])==0 for line in gpu.splitlines())
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
op=json.loads((repo/'model_training/state/current.json').read_text())['active_operation']
extension=op['allocation_lifetime_extension']
assert extension['approved'] and extension['allocation']==130675914 and extension['requested_h_rt_seconds']==345600
assert not op['gpu_work_allowed'] and not op['formal_training_allowed']
for arguments in (['begin'],['check','--operation','preflight','--entrypoint','model_training/jobs/extend_anchor_joint_native_20261027.sh']):
    check=subprocess.run(['/home/wangyy/miniconda3/envs/mygs/bin/python','-B',str(repo/'model_training/tools/agent_work_guard.py'),*arguments],cwd=repo,capture_output=True,text=True)
    if check.returncode:
        raise RuntimeError(check.stdout+check.stderr)
assert os.sched_getaffinity(1108368)=={6,7,9,10,11,12,14,15,61,63,64,65,74,76,77,78}
print('Verified retained native keeper, idle GPUs, clean capacity exit and committed lifetime-change approval.')
PY
NODE
qalter -when now -l h100=2,h_rt=96:00:00,mem_req=4G 130675914
python3 - <<'PY'
from datetime import datetime, timezone
import re
import subprocess
grant=subprocess.check_output(['qstat','-j','130675914'],text=True)
line=re.search(r'^hard_resource_list:\s*(.+)$',grant,re.M).group(1)
values=dict(field.strip().split('=',1) for field in line.split(','))
assert values=={'h100':'2','h_rt':'345600','mem_req':'4G'}, values
assert re.search(r'job_state\s+1:\s+r\b',grant)
assert re.search(r'exec_host_list\s+1:\s+pcg01i:16',grant)
assert len(re.findall(r'^granted_devices.*pcg01i: /dev/nvidia3 /dev/nvidia4\s*$',grant,re.M))==16
print('Observed UTC:',datetime.now(timezone.utc).isoformat())
print('\n'.join(line for line in grant.splitlines() if re.match(r'^(job_number|hard_resource_list|start_time|job_state|exec_host_list)\b',line)))
PY
ssh -o BatchMode=yes pcg01i 'test -d /proc/1108368 && cat /proc/1108368/cgroup'
