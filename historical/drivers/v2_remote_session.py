"""Interactive SSH/SFTP session; password stays in memory and is never logged."""
import argparse
import getpass
import json
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    known = Path.home()/'.ssh/known_hosts'
    known.parent.mkdir(exist_ok=True)
    known.touch(exist_ok=True)
    client = paramiko.SSHClient()
    client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = getpass.getpass('SSH password: ')
    client.connect(args.host, port=args.port, username='root', password=password,
                   timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
    del password
    client.get_transport().set_keepalive(20)
    print('CONNECTED', flush=True)
    try:
        while True:
            request = json.loads(input())
            action = request['action']
            if action == 'close':
                break
            if action == 'exec':
                _, stdout, stderr = client.exec_command(request['command'], timeout=request.get('timeout', 120))
                out, err = stdout.read().decode('utf-8', errors='replace'), stderr.read().decode('utf-8', errors='replace')
                print(json.dumps(dict(exit_code=stdout.channel.recv_exit_status(), stdout=out, stderr=err), ensure_ascii=True), flush=True)
            elif action in ('put', 'get'):
                with client.open_sftp() as sftp:
                    if action == 'put':
                        sftp.put(request['local'], request['remote'])
                    else:
                        Path(request['local']).parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(request['remote'], request['local'])
                print(json.dumps(dict(completed=True, action=action)), flush=True)
            else:
                raise ValueError('Unknown action')
    finally:
        client.close()


if __name__ == '__main__':
    main()
