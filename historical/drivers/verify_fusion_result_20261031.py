"""Verify downloaded archive, all copied files and every checkpoint hash."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]/'runs/fusion_encoders_20261031/gpu_result'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    receipt=json.loads((ROOT/'archive_receipt.json').read_text());archive=ROOT/'fusion_encoders_20261031_result.tar.gz'
    assert archive.stat().st_size==receipt['size_bytes'] and sha(archive)==receipt['sha256']
    dest=(ROOT/'completed_snapshot').resolve();assert dest.is_relative_to(ROOT.resolve());dest.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar.getmembers():assert (dest/m.name).resolve().is_relative_to(dest) and not m.issym() and not m.islnk()
        tar.extractall(dest,filter='data')
    snap=dest/'fusion_encoders_20261031';manifest=json.loads((snap/'archive_file_manifest.json').read_text())
    for r in manifest:
        p=(snap/r['relative_path']).resolve();assert p.is_relative_to(snap.resolve()) and sha(p)==r['sha256']
    models=json.loads((snap/'all_checkpoints_locked.json').read_text())['models'];prefix='/root/fusion_encoders_20261031/'
    for r in models:
        assert r['path'].startswith(prefix);assert sha(snap/r['path'][len(prefix):])==r['sha256']
    for name in ['predictions.npz','metrics.csv','per_transcript.csv','verification.json']:
        assert sha(snap/name)==sha(ROOT/name)
    result=dict(status='passed',n_weights=len(models),n_files=len(manifest),archive_sha256=receipt['sha256'],snapshot_path=str(snap))
    assert result['n_weights']==30
    (ROOT/'local_archive_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
