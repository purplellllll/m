"""Read-only remote progress watcher; verified atomic local backups, password in memory only."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

import paramiko


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def download_checked(sftp, remote, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(destination) == expected:
        return
    partial = destination.with_suffix('.part')
    sftp.get(remote, str(partial))
    if digest(partial) != expected:
        raise ValueError('Checkpoint checksum mismatch; previous destination preserved')
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--baseline-cv', action='store_true', help='Back up completed uniform scratch baselines for folds1-4')
    parser.add_argument('--include-fold0', action='store_true', help='Include fold0 for a new five-fold baseline run')
    args = parser.parse_args()
    assert not args.include_fold0 or args.baseline_cv
    root = PurePosixPath(args.remote_root)
    assert root.is_absolute() and '..' not in root.parts
    local = args.local_root.resolve()
    local.mkdir(parents=True, exist_ok=True)
    password = getpass.getpass('SSH password: ')
    manifest_path = local/'fullfold_backup_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cache = {}
    def save():
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(manifest_path)
    while True:
        client = paramiko.SSHClient()
        known = Path.home()/'.ssh/known_hosts'
        client.load_host_keys(str(known))
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(args.host, port=args.port, username='root', password=password,
                timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
            client.get_transport().set_keepalive(20)
            with client.open_sftp() as sftp:
                def read(relative):
                    try:
                        with sftp.open(str(root/relative), 'rb') as stream:
                            return json.loads(stream.read())
                    except FileNotFoundError:
                        return None
                def copy_jsons(relative):
                    try:
                        files = sftp.listdir_attr(str(root/relative))
                    except FileNotFoundError:
                        return
                    for item in files:
                        if not item.filename.endswith('.json'):
                            continue
                        remote = root/relative/item.filename
                        stamp = item.st_mtime, item.st_size
                        if cache.get(str(remote)) == stamp:
                            continue
                        with sftp.open(str(remote), 'rb') as stream:
                            content = stream.read()
                        json.loads(content)
                        destination = local/relative/item.filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        partial = destination.with_suffix('.json.part')
                        partial.write_bytes(content)
                        partial.replace(destination)
                        cache[str(remote)] = stamp
                while True:
                    state = read('status.json' if args.baseline_cv else 'fullfold_recovery/status.json')
                    assert state is not None
                    for fold in (range(1, 5) if args.baseline_cv and not args.include_fold0 else range(5)):
                        for arm in (('uniform',) if args.baseline_cv else ('control','candidate','compact')):
                            relative = PurePosixPath(f'results/fold_{fold}/{arm}' if args.baseline_cv else f'fullfold_recovery/fold_{fold}/{arm}')
                            complete = read(relative/'complete.json')
                            if complete is None:
                                continue
                            remote = root/relative/'best.pt' if args.baseline_cv else PurePosixPath(complete['selected_path'])
                            assert remote.parent == root/relative and '..' not in remote.parts
                            assert remote.name in ('best.pt', 'best_joint.pt')
                            expected = complete['best_sha256'] if args.baseline_cv else complete['selected_sha256']
                            key = f'fold_{fold}/{arm}'
                            destination = local/('checkpoints' if args.baseline_cv else 'fullfold_recovery/checkpoints')/f'fold{fold}_{arm}.pt'
                            if manifest.get(key, {}).get('sha256') != expected or not destination.exists():
                                download_checked(sftp, str(remote), destination, expected)
                                manifest[key] = dict(sha256=expected, bytes=destination.stat().st_size,
                                    local=str(destination), remote=str(remote), metrics=complete if args.baseline_cv else complete['metrics'])
                                save()
                                print(json.dumps(dict(event='checkpoint_backed_up',fold=fold,arm=arm,sha256=expected)),flush=True)
                            copy_jsons(relative)
                    copy_jsons(PurePosixPath('.') if args.baseline_cv else PurePosixPath('fullfold_recovery'))
                    if not args.baseline_cv:
                        copy_jsons(PurePosixPath('fullfold_recovery/final_automation'))
                    final = None if args.baseline_cv else read('fullfold_recovery/final_validation/status.json')
                    if final and final['state'] in ('completed','failed'):
                        copy_jsons(PurePosixPath('fullfold_recovery/final_validation'))
                    receipt = dict(event='backup_monitor',training=state,backed_up=len(manifest),final=final)
                    (local/'backup_monitor_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
                    print(json.dumps(receipt),flush=True)
                    if state['state'] == 'failed' or (args.baseline_cv and state['state'] == 'completed') or (final and final['state'] in ('completed','failed')):
                        return
                    time.sleep(60)
        except (OSError, paramiko.SSHException, ValueError) as error:
            print(json.dumps(dict(event='backup_connection_retry',error=type(error).__name__)),flush=True)
            time.sleep(30)
        finally:
            client.close()


if __name__ == '__main__':
    main()
