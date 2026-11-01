"""Recover our already-running demo after its parent wrapper errored; never restart it."""
import ctypes
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime
import subprocess

ROOT=Path('D:/evoweave/server_ops/outputs/paper_v1_20261102/skerig_demo600')
EXE=Path('C:/Users/namew/.codex/tmp/evoweave-paper-20261102/SkeRig/SkeRig_v1.0/SkeRig.exe')
PID=176656


def main():
    command=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {PID}'; if(!$p){{exit 3}}; [pscustomobject]@{{exe=$p.ExecutablePath;command=$p.CommandLine;started=$p.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
    process=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True))
    if Path(process['exe']).resolve()!=EXE.resolve() or str(ROOT/'cat_generated.rig.txt').replace('\\','/').lower() not in process['command'].replace('\\','/').lower():
        raise ValueError('not the exact owned SkeRig demo')
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000|0x1000,False,PID)
    if not handle:raise ctypes.WinError()
    try:
        left=max(0.,datetime.fromisoformat(process['started']).timestamp()+600-time.time())
        status=kernel.WaitForSingleObject(ctypes.c_void_p(handle),int(left*1000))
        if status==258:
            # This handle belongs to the verified process, even if its numeric PID is later reused.
            terminate=kernel.OpenProcess(0x0001|0x00100000,False,PID)
            if terminate:
                kernel.TerminateProcess(ctypes.c_void_p(terminate),124)
                kernel.WaitForSingleObject(ctypes.c_void_p(terminate),10000)
                kernel.CloseHandle(ctypes.c_void_p(terminate))
            code=124
        elif status==0:
            value=ctypes.c_ulong();kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(value));code=int(value.value)
        else:raise ctypes.WinError()
    finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    result={'example_only':True,'exit_code':code,'recovered_existing_pid':PID,'not_restarted':True,
        'executable_sha256':hashlib.sha256(EXE.read_bytes()).hexdigest(),
        'output_exists':(ROOT/'cat_generated.rig.txt').exists(),'GPU_used':False,'threads':1}
    (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))
    raise SystemExit(0 if code==0 and result['output_exists'] else 1)


if __name__=='__main__':main()
