"""Freeze another unused 120 genes; keep all fitting inputs train-only."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from prepare_fusion_data_20261031 import CODONS, protein
from run_aligned_profile_ablation_20261030 import structure_job, simulation_job

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'fusion_encoders_20261031'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    data=ROOT/'data';data.mkdir(exist_ok=False)
    (data/'frozen_inputs/features').mkdir(parents=True);(data/'features').mkdir()
    source=PRIOR/'related_liver/LIVER.csv'
    assert sha(source)=='70a6621d0779a0cef17ce49427560c0c3c1b3b237e005a1d1038f047aff8aa7f'
    previous=json.loads((PRIOR/'data/panel.json').read_text())['records']
    known=json.loads((PRIOR/'data/exclusions.json').read_text())['records']
    known += [r for r in previous if r['split']=='fresh_test']
    assert len(known)==1500
    save(data/'exclusions.json',dict(records=known))
    genes={r['gene'] for r in known};sequences={r['dna'] for r in known};proteins={protein(x) for x in sequences}
    candidates=[]
    for r in pd.read_csv(source).itertuples(index=False):
        ids=np.fromstring(r.codon_sequence.strip('[]'),sep=',',dtype=int)
        y=np.fromstring(r.annotations.strip('[]'),sep=',')
        if not 80<=len(ids)<=250 or len(ids)!=len(y) or (ids<0).any() or (ids>=64).any():continue
        observed=np.isfinite(y)
        if observed.mean()<.8 or (y[observed]<0).any() or np.mean(y[observed]>0)<.3 or np.ptp(y[observed])==0:continue
        longest=current=0
        for v in y:current=current+1 if v==0 else 0;longest=max(longest,current)
        if longest>20:continue
        dna=''.join(CODONS[i] for i in ids);gene=str(r.gene).split('.')[0];aa=protein(dna)
        if not dna.startswith('ATG') or dna[-3:] not in {'TAA','TAG','TGA'} or '*' in aa[:-1]:continue
        if gene in genes or dna in sequences or aa in proteins:continue
        rank=hashlib.sha256(('graph_polish_fresh_20261031:'+gene+':'+str(r.transcript)).encode()).hexdigest()
        candidates.append(dict(gene=gene,transcript=str(r.transcript),dna=dna,source_ids=ids.tolist(),
            y=[float(v) if np.isfinite(v) else None for v in y],rank=rank,split='fresh_test'))
    fresh=[]
    for r in sorted(candidates,key=lambda x:x['rank']):
        aa=protein(r['dna'])
        if r['gene'] in genes or r['dna'] in sequences or aa in proteins:continue
        fresh.append(r);genes.add(r['gene']);sequences.add(r['dna']);proteins.add(aa)
        if len(fresh)==120:break
    assert len(fresh)==120
    rows=[r for r in previous if r['split'] in ['train','validation']]
    supplementary=[dict(r,split='supplementary_test') for r in previous if r['split']=='fresh_test']
    rows += fresh+supplementary
    save(data/'panel.json',dict(records=rows,n_prior_excluded=1500,near_homology_clustered=False,primary_test='fresh_test'))
    params_path=PRIOR/'data/physics_parameters_train_only.json';shutil.copy2(params_path,data/params_path.name)
    params=json.loads(params_path.read_text())
    for r in [r for r in rows if r['split']!='fresh_test']:
        original=PRIOR/'data/frozen_inputs/features'/f'{r["gene"]}.npz'
        if not original.exists():original=PRIOR/'data/features'/f'{r["gene"]}.npz'
        dest=data/('frozen_inputs/features' if r['split'] in ['train','validation'] else 'features')/original.name
        shutil.copy2(original,dest);assert sha(original)==sha(dest)
    for i,r in enumerate(fresh):
        f=structure_job((str(ROOT/'source'),r['dna']))
        seed=int(hashlib.sha256(('profile_sim:'+r['gene']).encode()).hexdigest()[:8],16)
        f.update(simulation_job((str(ROOT/'source'),r['dna'],f,params,seed)))
        y=np.array([np.nan if v is None else v for v in r['y']],dtype=np.float32)
        f['observed']=np.isfinite(y);f['label']=np.log1p(np.nan_to_num(y,nan=0)).astype(np.float32)
        np.savez_compressed(data/'features'/f'{r["gene"]}.npz',**f)
        if (i+1)%10==0:save(ROOT/'prepare_status.json',dict(stage='features',completed=i+1,total=120,unix=time.time()))
    manifest=[]
    for r in rows:
        p=data/'frozen_inputs/features'/f'{r["gene"]}.npz'
        if not p.exists():p=data/'features'/f'{r["gene"]}.npz'
        manifest.append(dict(gene=r['gene'],sha256=sha(p)))
    assert len(rows)==720 and len({r['gene'] for r in rows})==720
    save(data/'features_locked.json',dict(panel_sha256=sha(data/'panel.json'),features=manifest))
    save(ROOT/'prepare_status.json',dict(stage='completed',unix=time.time(),n_records=720))


if __name__=='__main__':main()
