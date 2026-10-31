"""Independent centered-dot Pearson for every saved head and interior metric."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'runs/fusion_encoders_20261031/gpu_result'


def pearson(x,y):
    a=x.astype(float)-x.astype(float).mean();b=y.astype(float)-y.astype(float).mean()
    denominator=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/denominator) if denominator>0 else np.nan


def main():
    verification=json.loads((ROOT/'verification.json').read_text());assert verification['status']=='passed'
    digest=hashlib.sha256((ROOT/'predictions.npz').read_bytes()).hexdigest();assert digest==verification['predictions_sha256']
    p=pd.read_csv(ROOT/'per_transcript.csv');m=pd.read_csv(ROOT/'metrics.csv').set_index(['split','model'])
    rows=json.loads((ROOT/'data/panel.json').read_text())['records'];errors=[]
    with np.load(ROOT/'predictions.npz',allow_pickle=False) as pp:
        for (split,model),frame in p.groupby(['split','model'],sort=False):
            expected=[r['gene'] for r in rows if r['split']==split];assert frame.gene.tolist()==expected
            values=[]
            for j,r in enumerate(frame.itertuples()):
                y=pp[f'{split}__label__{j}'];mask=pp[f'{split}__observed__{j}'];inside=mask.copy();inside[:5]=False;inside[-5:]=False
                for head,column,which in [('main','pcc',mask),('main','interior_pcc',inside),('physics','physics_pcc',mask),('gnn','gnn_pcc',mask),('context','context_pcc',mask)]:
                    q=pp[f'{split}__{model}__{head}__{j}'];assert q.shape==y.shape and np.isfinite(q).all()
                    value=pearson(y[which],q[which]);saved=getattr(r,column)
                    assert np.isfinite(value)==np.isfinite(saved)
                    if np.isfinite(value):errors.append(abs(value-saved));assert abs(value-saved)<1e-7
                    if column=='pcc':values.append(value)
            assert abs(np.nanmean(values)-m.loc[(split,model),'mean_pcc'])<1e-7
    result=dict(status='passed',implementation='centered dot product, independent from np.corrcoef evaluator',
        checked_metric_values=len(errors),max_error=max(errors),predictions_sha256=digest)
    (ROOT/'metric_recheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
