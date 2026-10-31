"""Snapshot receipts and prediction tables without touching models or secrets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source_parity={name:sha(ROOT/'physics_scorer'/name)==sha(Path('/root/graph_polish_20261031/physics_scorer')/name)
        for name in ['model.py','profile_model.py','fusion_encoder_model.py','graph_polish_model.py']}
    assert all(source_parity.values()),'Transferred profile code changed from checkpoint source'
    state=json.loads(subprocess.check_output(['python3',str(ROOT/'monitor_scalar_tasks_20261031.py')],text=True))
    state['frozen_source_code_matches_prior_model']=source_parity
    (ROOT/'progress_snapshot.json').write_text(json.dumps(state,indent=2,allow_nan=False))
    selected=['progress_snapshot.json','protocol_locked.json','evaluation_reporting_amendment.json','runtime_status.json',
        'scalar_gpu_preflight.json','mamba_backend_parity.json','scalar_reference_preflight.json',
        'protein_evaluation_locked.json','protein_ours_locked_test_audit.json','protein_ours_locked_test_predictions.csv',
        'protein_baseline_status.json','protein_training_complete.json','protein_evaluation_complete.json',
        'te_features_status.json','rpf_features_status.json','status.json','data/manifest.json','local_data_audit.json','source_audit.json']
    files=[ROOT/p for p in selected if (ROOT/p).is_file()]
    for folder in (ROOT/'runs').glob('*'):
        files += [p for p in folder.iterdir() if p.name in ['locked.json','inputs_locked.json','history.jsonl','status.json']]
    receipt=dict(unix=time.time(),files={str(p.relative_to(ROOT)):sha(p) for p in files},
        no_checkpoint_weights_in_this_snapshot=True,background_controller_preserved=True)
    (ROOT/'progress_files_manifest.json').write_text(json.dumps(receipt,indent=2))
    target=ROOT/'progress_snapshot.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'progress_files_manifest.json']:z.write(p,str(p.relative_to(ROOT)))
    (ROOT/'progress_archive_receipt.json').write_text(json.dumps(dict(sha256=sha(target),bytes=target.stat().st_size,n_files=len(files)+1),indent=2))
    print(json.dumps(dict(archive=str(target),bytes=target.stat().st_size,n_files=len(files)+1,sha256=sha(target))))


if __name__=='__main__':main()
