"""Independent saved-array and sequence-exclusion audit on the local machine."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from prepare_fusion_data_20261031 import protein

BASE=Path(__file__).resolve().parent
ROOT=BASE if (BASE/'metrics.csv').exists() else BASE.parent/'runs/graph_polish_20261031/gpu_result'


def centered(a,b):
    a=a.astype(float)-a.astype(float).mean();b=b.astype(float)-b.astype(float).mean()
    denominator=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/denominator) if denominator>0 else np.nan


def main():
    verification=json.loads((ROOT/'verification.json').read_text());assert verification['status']=='passed'
    digest=hashlib.sha256((ROOT/'predictions.npz').read_bytes()).hexdigest();assert digest==verification['predictions_sha256']
    panel=json.loads((ROOT/'data/panel.json').read_text())['records']
    known=json.loads((ROOT/'data/exclusions.json').read_text())['records'];assert len(known)==1500
    assert len(panel)==720 and len({r['gene'] for r in panel})==720
    forbidden=[{r['gene'] for r in known},{r['dna'] for r in known},{protein(r['dna']) for r in known}]
    for r in panel:
        if r['split']=='fresh_test':
            assert all(v not in s for v,s in zip([r['gene'],r['dna'],protein(r['dna'])],forbidden))
    p=pd.read_csv(ROOT/'per_transcript.csv');summary=pd.read_csv(ROOT/'metrics.csv').set_index(['split','model'])
    # Zip lookup avoids repeatedly scanning the large NPZ member list.
    with zipfile.ZipFile(ROOT/'predictions.npz') as z:
        arrays={name[:-4]:np.load(io.BytesIO(z.read(name)),allow_pickle=False) for name in z.namelist()}
    errors=[];mass_errors=[]
    for (split,model),frame in p.groupby(['split','model'],sort=False):
        expected=[r['gene'] for r in panel if r['split']==split];assert frame.gene.tolist()==expected
        values=[]
        for j,r in enumerate(frame.itertuples()):
            y=arrays[f'{split}__label__{j}'];observed=arrays[f'{split}__observed__{j}']
            for head,column in [('main','pcc'),('main','interior_pcc'),('physics','physics_pcc'),('gnn','gnn_pcc'),('context','context_pcc')]:
                mask=observed.copy()
                if column=='interior_pcc':mask[:5]=False;mask[-5:]=False
                q=arrays[f'{split}__{model}__{head}__{j}'];assert q.shape==y.shape and np.isfinite(q).all()
                mass_errors.append(abs(q.sum()-1));assert mass_errors[-1]<1e-10
                value=centered(y[mask],q[mask]);saved=getattr(r,column)
                assert np.isfinite(value)==np.isfinite(saved)
                if np.isfinite(value):errors.append(abs(value-saved));assert errors[-1]<1e-7
                if column=='pcc':values.append(value)
            if not model.endswith(('_30101','_30102','_30103')):
                for head in ['main','physics','gnn','context']:
                    mean=np.mean([arrays[f'{split}__{model}_{s}__{head}__{j}'] for s in [30101,30102,30103]],0)
                    np.testing.assert_array_equal(mean,arrays[f'{split}__{model}__{head}__{j}'])
        assert abs(np.nanmean(values)-summary.loc[(split,model),'mean_pcc'])<1e-7
    result=dict(status='passed',implementation='centered dot product independent of np.corrcoef',
        checked_metric_values=len(errors),max_error=max(errors),max_mass_error=max(mass_errors),
        n_records=720,n_prior_excluded=1500,fresh_genes=120,predictions_sha256=digest,ensembles_recomputed=True)
    (ROOT/'metric_recheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
