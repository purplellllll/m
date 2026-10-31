"""Transfer only the fixed historical panel and cached reference predictions."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'runs/scorer_benchmark_breakdown_20261031'
SOURCE=WORK/'runs/profile_model_comparison_20261030/completed_snapshot'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True);transfer=OUT/'transfer';transfer.mkdir(exist_ok=False);(transfer/'features').mkdir()
    original=json.loads((SOURCE/'panel.json').read_text())
    rows=[r for r in original['records'] if r['split']=='fresh_test'];assert len(rows)==120
    fit={s:[[r['gene'],r['dna']] for r in original['records'] if r['split']==s] for s in ['train','validation']}
    panel=dict(records=rows,fit_records=fit,physics_parameters=json.loads((SOURCE/'physics_parameters_train_only.json').read_text()),
        historical_source_panel_sha256=sha(SOURCE/'panel.json'),historical_reference_predictions_sha256=sha(SOURCE/'predictions.npz'))
    (transfer/'panel.json').write_text(json.dumps(panel,indent=2))
    for r in rows:shutil.copy2(SOURCE/f'features/{r["gene"]}.npz',transfer/f'features/{r["gene"]}.npz')
    shutil.copy2(SOURCE/'predictions.npz',transfer/'reference_predictions.npz')
    manifest=[dict(path=str(p.relative_to(transfer)).replace('\\','/'),sha256=sha(p)) for p in transfer.rglob('*') if p.is_file()]
    (transfer/'transfer_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=OUT/'transfer.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:tar.add(transfer,arcname='scorer_breakdown_20261031')
    print(json.dumps(dict(archive=str(archive),size_bytes=archive.stat().st_size,sha256=sha(archive))))


if __name__=='__main__':main()
