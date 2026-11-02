"""Preserve the finite-budget native MFE surrogate's actual fitting diagnostics."""
from pathlib import Path
import re,json,hashlib,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'te_stability_20261102/native_compare'
def main():
    records=[]
    for i,name in [(1,'case1.log'),(2,'case2.log'),(3,'case3_retry_ext4.log')]:
        path=ROOT/name;text=path.read_bytes().decode('utf-8',errors='replace')
        values=[float(x.strip()) for x in re.findall(r'^R2 of train data for MFE:([^\r\n]+)',text,re.M)]
        assert len(values)==3 and all(math.isfinite(x) for x in values),(i,values)
        records.append(dict(case=i,training_r2_by_round=values,log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(status='completed',cases=records,negative_training_r2_observed=any(v<0 for r in records for v in r['training_r2_by_round']),
        interpretation='Native printed R2 pools predictions made throughout parameter updates; it is not final frozen-model training R2 or independent validation. See native_surrogate_holdout_audit.json for frozen-model diagnostics. An MFE advantage over this finite-budget run is not proof of superiority to a fully trained native optimizer.')
    (ROOT/'native_surrogate_fit_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
