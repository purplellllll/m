"""Snapshot the verified experiment without copying the multi-GB environment."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    assert json.loads((ROOT/'verification.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'status.json').read_text())['stage']=='verified'
    paths=[ROOT/name for name in ['runs','references','data','source','physics_scorer']]
    paths += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.csv','.npz','.md','.png','.log','.txt']
              and p.name not in ['get-pip.py','archive_receipt.json','archive_file_manifest.json']]
    files=[]
    for p in paths:
        files.extend(sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p])
    assert not any(p.is_symlink() for p in files)
    manifest=[dict(relative_path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in files]
    (ROOT/'archive_file_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=ROOT.with_name(ROOT.name+'_result.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for p in paths:tar.add(p,arcname=ROOT.name+'/'+str(p.relative_to(ROOT)))
        tar.add(ROOT/'archive_file_manifest.json',arcname=ROOT.name+'/archive_file_manifest.json')
    receipt=dict(path=str(archive),sha256=sha(archive),size_bytes=archive.stat().st_size,n_files=len(files))
    (ROOT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
