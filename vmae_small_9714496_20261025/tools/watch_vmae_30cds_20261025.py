"""Incremental local backup for the 30-CDS comparison; password only in memory."""
import argparse
import csv
import getpass
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time

import paramiko
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.watch_latent_fullfold_backups import download_checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True);parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--local-root',type=Path,default=Path('runs/vmae_30cds_20261025'))
    args=parser.parse_args()
    local=args.local_root.resolve();local.mkdir(parents=True,exist_ok=True)
    remote=PurePosixPath('/root/gpufree-data/vmae_30cds_20261025')
    password=getpass.getpass('SSH password: ')
    cache={}
    while True:
        client=paramiko.SSHClient();client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(args.host,port=args.port,username='root',password=password,
                look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20)
            client.get_transport().set_keepalive(20)
            with client.open_sftp() as sftp:
                def copy(relative):
                    path=remote/relative
                    try:stamp=sftp.stat(str(path))
                    except FileNotFoundError:return None
                    signature=(stamp.st_mtime,stamp.st_size)
                    destination=local/relative
                    if cache.get(str(path))==signature and destination.exists():
                        return json.loads(destination.read_text()) if destination.suffix=='.json' else None
                    with sftp.open(str(path),'rb') as stream:content=stream.read()
                    parsed=json.loads(content) if str(path).endswith('.json') else None
                    if str(path).endswith('.csv'):list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    temporary=destination.with_suffix(destination.suffix+'.part');temporary.write_bytes(content);temporary.replace(destination)
                    cache[str(path)]=signature
                    return parsed
                def names(relative):
                    try:return sftp.listdir(str(remote/relative))
                    except FileNotFoundError:return []
                while True:
                    state=copy('status.json')
                    for name in names('.'):
                        if name.endswith(('.json','.csv','.fasta')):copy(name)
                    for name in names('native_receipts'):
                        if name.endswith('.json'):copy('native_receipts/'+name)
                    for identity in names('metrics'):
                        if identity.startswith('cds_'):
                            for name in names('metrics/'+identity):
                                if name.endswith('.json'):copy('metrics/'+identity+'/'+name)
                    for identity in names('search'):
                        if identity.startswith('cds_'):
                            for name in names('search/'+identity):
                                if name.endswith('.json'):copy('search/'+identity+'/'+name)
                    for name in names('preflight'):
                        if name.endswith('.json'):copy('preflight/'+name)
                        elif name in ('parity_tail','parity_all','mixed_tail','mixed_all','long_preflight'):
                            copy('preflight/'+name+'/completion_receipt.json')
                            copy('preflight/'+name+'/runtime_protocol.json')
                    for name in names('native_archives'):
                        if name.endswith('.receipt.json'):
                            receipt=copy('native_archives/'+name)
                            archive=name[:-len('.receipt.json')]+'.gz'
                            download_checked(sftp,str(remote/'native_archives'/archive),local/'native_archives'/archive,receipt['sha256'])
                    record=dict(state='monitoring',experiment=state,archive_backups=len(list((local/'native_archives').glob('*.tar.gz'))) if (local/'native_archives').exists() else 0)
                    (local/'backup_monitor_status.json').write_text(json.dumps(record,indent=2)+'\n')
                    print(json.dumps(record),flush=True)
                    if state and state['state'] in ('completed','failed'):return
                    time.sleep(60)
        except (OSError,paramiko.SSHException,ValueError) as error:
            print(json.dumps(dict(event='backup_retry',error=type(error).__name__)),flush=True)
            time.sleep(30)
        finally:client.close()


if __name__=='__main__':main()
