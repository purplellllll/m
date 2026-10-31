"""Register a pre-RPF-evaluation reporting correction, preserving old hash history."""
import hashlib
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent


def main():
    assert not (ROOT/'rpf_evaluation_locked.json').exists()
    p=ROOT/'evaluate_scalar_rpf_20261031.py';locked=json.loads((ROOT/'protocol_locked.json').read_text())
    result=dict(unix=time.time(),source=p.name,prior_source_sha256=locked['source_sha256'][p.name],
        amended_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        reason='Separate genes held out for our validation from truly seen training genes in unseen environments; add MSE and paired environment bootstrap',
        rpf_test_evaluated=False,training_unchanged=True,protein_selection_unchanged=True,
        additional_readonly_checks=['preflight_scalar_reference_20261031.py','audit_locked_protein_predictions_20261031.py','monitor_scalar_tasks_20261031.py'])
    (ROOT/'evaluation_reporting_amendment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
