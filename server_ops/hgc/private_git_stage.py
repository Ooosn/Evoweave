"""Transfer a reviewed Git bundle to private HGC storage and optionally stage it."""
import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path, PurePosixPath
import shlex
import sys

sys.path.insert(0, str(Path.home()/'.codex/skills/hgc-server/scripts'))
from hgc_exec import connect_client, read_dpapi_secret, DEFAULT_SECRET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--commit')
    parser.add_argument('--bare')
    parser.add_argument('--checkout')
    args = parser.parse_args()
    destination = PurePosixPath(args.destination)
    if '..' in destination.parts or not str(destination).startswith((
            '/home/wangyy/evorig/outputs/tokenrig_', '/work/wangyy/tokenrig_alignment_20261105/',
            '/home/wangyy/claude_paper_exp/runner/queue/')):
        raise ValueError('transfer destination is outside this experiment')
    if bool(args.commit) != bool(args.bare):
        raise ValueError('private Git import needs both a commit and a bare repository')
    secret = read_dpapi_secret(DEFAULT_SECRET)
    key = str(Path.home()/'.ssh/id_rsa')
    expected = hashlib.sha256(args.file.read_bytes()).hexdigest()
    with ExitStack() as stack:
        jump = connect_client('slogin.hgc.jp', 'wangyy', key, secret, 20)
        stack.callback(jump.close)
        channel = jump.get_transport().open_channel('direct-tcpip', ('pcl03', 22), ('127.0.0.1', 0))
        login = connect_client('pcl03', 'wangyy', key, secret, 20, sock=channel)
        stack.callback(login.close)
        channel = login.get_transport().open_channel('direct-tcpip', ('pcg01i', 22), ('127.0.0.1', 0))
        compute = connect_client('pcg01i', 'wangyy', key, secret, 20, sock=channel)
        stack.callback(compute.close)
        ftp = compute.open_sftp()
        stack.callback(ftp.close)

        def command(argv):
            _, out, err = compute.exec_command(shlex.join(argv), timeout=120)
            value, error = out.read().decode(), err.read().decode()
            if out.channel.recv_exit_status():
                raise RuntimeError(error[-3000:])
            return value.strip()

        for parent in reversed(destination.parents):
            if str(parent) == '/':
                continue
            try:
                ftp.stat(str(parent))
            except FileNotFoundError:
                ftp.mkdir(str(parent))
        try:
            ftp.stat(str(destination))
        except FileNotFoundError:
            temporary = str(destination)+'.partial'
            ftp.put(str(args.file), temporary)
            if command(['sha256sum', temporary]).split()[0] != expected:
                raise RuntimeError('uploaded artifact checksum differs')
            ftp.rename(temporary, str(destination))
        if command(['sha256sum', str(destination)]).split()[0] != expected:
            raise RuntimeError('refuse to overwrite a different remote artifact')
        if args.commit:
            command(['git', '--git-dir='+args.bare, 'bundle', 'verify', str(destination)])
            command(['git', '--git-dir='+args.bare, 'fetch', str(destination), 'HEAD'])
            if command(['git', '--git-dir='+args.bare, 'rev-parse', 'FETCH_HEAD']) != args.commit:
                raise RuntimeError('private source import differs')
            command(['git', '--git-dir='+args.bare, 'update-ref',
                     'refs/heads/codex/virtual-fusion-20261013', args.commit])
            if args.checkout:
                target = PurePosixPath(args.checkout)
                if not str(target).startswith('/work/wangyy/tokenrig_alignment_20261105/') or '..' in target.parts:
                    raise ValueError('checkout must be a new experiment-scoped SSD directory')
                command(['git', 'clone', '--shared', '--no-checkout', args.bare, str(target)])
                command(['git', '-C', str(target), 'checkout', '--detach', args.commit])
                if command(['git', '-C', str(target), 'status', '--porcelain']):
                    raise RuntimeError('staged source is not clean')
        print('private transfer verified', destination, expected, args.commit or '')


if __name__ == '__main__':
    main()
