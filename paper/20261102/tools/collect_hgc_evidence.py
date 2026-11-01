"""Download only bounded paper evidence using the existing HGC credential helper."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0,str(Path.home()/'.codex/skills/hgc-server/scripts'))
from hgc_exec import connect_client,read_dpapi_secret,DEFAULT_SECRET

RUN='/work/wangyy/virtual_query_matched_5k_20261025'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('audit_v36','audit_v37','capacity_v36'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    secret=read_dpapi_secret(DEFAULT_SECRET)
    key=str(Path.home()/'.ssh/id_rsa')
    with ExitStack() as stack:
        jump=connect_client('slogin.hgc.jp','wangyy',key,secret,20);stack.callback(jump.close)
        channel=jump.get_transport().open_channel('direct-tcpip',('pcl03',22),('127.0.0.1',0))
        login=connect_client('pcl03','wangyy',key,secret,20,sock=channel);stack.callback(login.close)
        channel=login.get_transport().open_channel('direct-tcpip',('pcg01i',22),('127.0.0.1',0))
        compute=connect_client('pcg01i','wangyy',key,secret,20,sock=channel);stack.callback(compute.close)
        ftp=compute.open_sftp();stack.callback(ftp.close)
        remote=f'{RUN}/outputs/{args.stage}'
        candidates=[(f'{remote}/{x.filename}',Path(args.stage)/x.filename) for x in ftp.listdir_attr(remote)
            if x.filename.endswith(('.json','.log','.npz'))]
        candidates.extend([(f'{RUN}/outputs/{args.stage}_resources.json',Path(args.stage+'_resources.json')),
            (f'{RUN}/{args.stage}.log',Path(args.stage+'.log')),
            (f'{RUN}/allocation_binding_native_20261031/{args.stage}_launch.json',Path(args.stage+'_launch.json'))])
        receipts=[]
        for source,relative in candidates:
            size=ftp.stat(source).st_size
            if size>100*1024**2:
                raise ValueError('evidence exceeded100MiB per-file bound')
            _,stdout,stderr=compute.exec_command('sha256sum -- '+shlex.quote(source),timeout=60)
            expected=stdout.read().decode().split()[0]
            if stdout.channel.recv_exit_status()!=0:raise RuntimeError('remote digest failed')
            target=args.output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                    raise ValueError('immutable collected evidence changed: '+str(target))
            else:
                partial=target.with_suffix(target.suffix+'.part')
                ftp.get(source,str(partial))
                if hashlib.sha256(partial.read_bytes()).hexdigest()!=expected:
                    raise ValueError('download hash differs')
                partial.replace(target)
            receipts.append({'remote':source,'relative_path':relative.as_posix(),'bytes':size,'sha256':expected})
    path=args.output/(args.stage+'_collection.json')
    path.write_text(json.dumps({'verified':True,'files':receipts},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'verified':True,'files':len(receipts),'output':str(args.output)}))


if __name__=='__main__':main()
