"""One SSH command with a secret delivered on stdin, never in argv or output."""
import argparse
import getpass
import json
from pathlib import Path
import paramiko

parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--port',type=int,required=True)
parser.add_argument('--command',required=True);args=parser.parse_args()
client=paramiko.SSHClient();client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
password=getpass.getpass('SSH password: ')
client.connect(args.host,port=args.port,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=30)
client.get_transport().set_keepalive(20)
del password
secret=getpass.getpass('SwanLab API key: ')
stdin,stdout,stderr=client.exec_command(args.command,timeout=120)
stdin.write(secret+'\n');stdin.flush();stdin.channel.shutdown_write();del secret
print(json.dumps(dict(exit_code=stdout.channel.recv_exit_status(),stdout=stdout.read().decode(errors='replace'),stderr=stderr.read().decode(errors='replace'))))
client.close()
