"""Check the authors' Windows executable on its bundled example, without UI."""
import ctypes
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SOURCE=Path('C:/Users/namew/.codex/tmp/evoweave-paper-20261102/SkeRig/SkeRig_v1.0')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('D:/evoweave/server_ops/outputs/paper_v1_20261102/skerig_demo'))
    parser.add_argument('--timeout',type=int,default=90);args=parser.parse_args()
    if not 10<=args.timeout<=900:raise ValueError('bounded demo timeout required')
    OUTPUT=args.output
    OUTPUT.mkdir(parents=True,exist_ok=False)
    ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    args=[str(SOURCE/'SkeRig.exe'),'-DBG','1',str(SOURCE/'cat-poses.txt'),str(SOURCE/'cat-poses.obj'),str(OUTPUT/'cat_generated.rig.txt')]
    t=time.monotonic()
    with (OUTPUT/'program.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(args,cwd=OUTPUT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code=p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            p.kill();p.wait();code=124
    result={'example_only':True,'exit_code':code,'seconds':time.monotonic()-t,
        'executable_sha256':hashlib.sha256((SOURCE/'SkeRig.exe').read_bytes()).hexdigest(),
        'output_exists':(OUTPUT/'cat_generated.rig.txt').exists(),'GPU_used':False,'threads':1}
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))
    raise SystemExit(0 if code==0 and result['output_exists'] else 1)


if __name__=='__main__':main()
