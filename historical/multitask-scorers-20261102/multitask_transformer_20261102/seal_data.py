"""Derive trainer-only development rows and separately sealed test targets."""
from pathlib import Path
import json,hashlib
from collections import Counter
HERE=Path(__file__).resolve().parent
LABEL_FIELDS={'targets','target_mask','standardized_targets'}

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_rows(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf8')

def main():
    rows=[json.loads(line) for line in (HERE/'records.jsonl').read_text(encoding='utf8').splitlines()]
    # Explicit assay metadata belongs in inputs, independently of label availability.
    rows=[dict(r,assay_cell_free=(r['task']=='solubility')) for r in rows]
    dev=[r for r in rows if r['split'] in {'train','val'}]
    tests=[r for r in rows if r['split']=='test']
    inputs=[{k:v for k,v in r.items() if k not in LABEL_FIELDS} for r in tests]
    targets={r['record_id']:r['targets'] for r in tests}
    assert len(targets)==len(tests) and all(not LABEL_FIELDS&set(r) for r in inputs)
    assert all(r['split']!='test' for r in dev)
    write_rows(HERE/'train_validation.jsonl',dev)
    write_rows(HERE/'test_inputs.jsonl',inputs)
    save(HERE/'sealed_test_targets.json',targets)
    anomalies=[]
    for r in rows:
        value=r['targets'].get('solubility')
        if value is not None and not 0<=value<=100:
            anomalies.append(dict(record_id=r['record_id'],split=r['split'],measured_solubility_percent=value))
    derived=[HERE/name for name in ['train_validation.jsonl','test_inputs.jsonl','sealed_test_targets.json']]
    save(HERE/'sealed_data_audit.json',dict(status='passed',development_n=len(dev),test_n=len(inputs),
        test_target_fields_removed=sorted(LABEL_FIELDS),test_labels_in_training_input=False,
        out_of_physical_range_solubility=dict(n=len(anomalies),by_split=dict(Counter(r['split'] for r in anomalies)),
            records=anomalies,decision='Retain exact original measured values without clipping; values outside0..100 are assay/calibration artifacts and must not be interpreted as physically valid fractions'),
        solubility_min=min(r['targets']['solubility'] for r in rows if 'solubility' in r['targets']),
        solubility_max=max(r['targets']['solubility'] for r in rows if 'solubility' in r['targets']),
        assay_cell_free=True,host_note='eSOL host ecoli identifies source translation apparatus; assay is PURE cell-free, not living bacterial cells'))
    save(HERE/'derived_data_protocol.json',dict(status='locked_before_training',
        parent_protocol_sha256=sha(HERE/'data_protocol_locked.json'),parent_records_sha256=sha(HERE/'records.jsonl'),
        script_sha256=sha(__file__),derived_sha256={p.name:sha(p) for p in derived},
        training_access='Trainer reads train_validation.jsonl only; test inference uses test_inputs.jsonl; labels opened only by evaluator after checkpoint selection',
        added_metadata='assay_cell_free boolean; false for expression/half_life, true for PURE solubility',
        clipping=False,old_locked_protocol_modified=False))
    print(json.dumps(dict(development=len(dev),test=len(inputs),solubility_out_of_range=len(anomalies),solubility_max=max(r['targets']['solubility'] for r in rows if 'solubility' in r['targets'])),indent=2))

if __name__=='__main__':main()
