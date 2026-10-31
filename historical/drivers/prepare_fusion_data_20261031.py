"""Freeze a fresh panel and compute aligned simulator inputs before training."""
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from run_aligned_profile_ablation_20261030 import structure_job,simulation_job

ROOT=Path(__file__).resolve().parent
CODONS=[''.join(c) for c in itertools.product('ATCG',repeat=3)]
CODE=dict(zip(map(''.join,itertools.product('TCAG',repeat=3)),'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def protein(dna):return ''.join(CODE[dna[i:i+3]] for i in range(0,len(dna),3))
def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def main():
    data=ROOT/'data';assert not (data/'panel.json').exists()
    source=ROOT/'related_liver/LIVER.csv'
    assert sha(source)=='70a6621d0779a0cef17ce49427560c0c3c1b3b237e005a1d1038f047aff8aa7f'
    known=json.loads((data/'exclusions.json').read_text())['records'];assert len(known)==1380
    genes={r['gene'] for r in known};dna_set={r['dna'] for r in known};aa_set={protein(x) for x in dna_set};pool=[]
    for r in pd.read_csv(source).itertuples(index=False):
        ids=np.fromstring(r.codon_sequence.strip('[]'),sep=',',dtype=int);y=np.fromstring(r.annotations.strip('[]'),sep=',')
        if not 80<=len(ids)<=250 or len(ids)!=len(y) or (ids<0).any() or (ids>=64).any():continue
        m=np.isfinite(y)
        if m.mean()<.8 or (y[m]<0).any() or np.mean(y[m]>0)<.3 or np.ptp(y[m])==0:continue
        longest=current=0
        for v in y:current=current+1 if v==0 else 0;longest=max(longest,current)
        if longest>20:continue
        dna=''.join(CODONS[i] for i in ids);gene=str(r.gene).split('.')[0];aa=protein(dna)
        if not dna.startswith('ATG') or dna[-3:] not in {'TAA','TAG','TGA'} or '*' in aa[:-1]:continue
        if gene in genes or dna in dna_set or aa in aa_set:continue
        rank=hashlib.sha256(('common_profile_fresh_test_20261030:'+gene+':'+str(r.transcript)).encode()).hexdigest()
        pool.append(dict(gene=gene,transcript=str(r.transcript),dna=dna,source_ids=ids.tolist(),
            y=[float(v) if np.isfinite(v) else None for v in y],rank=rank,split='fresh_test'))
    new=[]
    for r in sorted(pool,key=lambda v:v['rank']):
        aa=protein(r['dna'])
        if r['gene'] in genes or r['dna'] in dna_set or aa in aa_set:continue
        new.append(r);genes.add(r['gene']);dna_set.add(r['dna']);aa_set.add(aa)
        if len(new)==120:break
    assert len(new)==120,f'Insufficient unused genes: {len(new)}'
    rows=json.loads((data/'panel_train_validation.json').read_text())['records']
    supplementary=[dict(r,split='supplementary_test') for r in json.loads((data/'supplementary_panel.json').read_text())['records']]
    rows += new+supplementary;assert len(rows)==720
    save(data/'panel.json',dict(records=rows,n_prior_excluded=1380,n_fresh_test=120,source_sha256=sha(source),
        primary_test='fresh_test',near_homology_clustered=False))
    (data/'features').mkdir();params=json.loads((data/'physics_parameters_train_only.json').read_text())
    for i,r in enumerate(new+supplementary):
        f=structure_job((str(ROOT/'source'),r['dna']))
        seed=int(hashlib.sha256(('profile_sim:'+r['gene']).encode()).hexdigest()[:8],16)
        f.update(simulation_job((str(ROOT/'source'),r['dna'],f,params,seed)))
        y=np.array([np.nan if v is None else v for v in r['y']],dtype=np.float32)
        f['observed']=np.isfinite(y);f['label']=np.log1p(np.nan_to_num(y,nan=0)).astype(np.float32)
        np.savez_compressed(data/'features'/f'{r["gene"]}.npz',**f)
        if (i+1)%10==0:save(ROOT/'prepare_status.json',dict(stage='features',completed=i+1,total=240,unix=time.time()))
    manifest=[]
    for r in rows:
        path=data/'frozen_inputs/features'/f'{r["gene"]}.npz'
        if not path.exists():path=data/'features'/f'{r["gene"]}.npz'
        manifest.append(dict(gene=r['gene'],sha256=sha(path)))
    save(data/'features_locked.json',dict(panel_sha256=sha(data/'panel.json'),features=manifest))
    save(ROOT/'prepare_status.json',dict(stage='completed',unix=time.time(),n_records=720))


if __name__=='__main__':main()
