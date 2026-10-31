"""Prepare the explicitly chosen related-study source, without claiming the original RiboGL split."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from prepare_fusion_data_20261031 import CODONS, protein
from run_aligned_profile_ablation_20261030 import structure_job, simulation_job, fit_physics

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/fusion_encoders_20261031/related_liver/LIVER.csv')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def fold_one(row):
    f=None
    from run_aligned_profile_ablation_20261030 import native
    utils,_,_=native(ROOT/'source')
    for folder in ['/root/graph_polish_20261031/data/features','/root/graph_polish_20261031/data/frozen_inputs/features',
                   '/root/fusion_encoders_20261031/data/features']:
        old=Path(folder)/f'{row["gene"]}.npz'
        if not old.exists():continue
        with np.load(old,allow_pickle=False) as z:
            dna=''.join(utils.CODONS[int(i)] for i in z['cod']).replace('U','T')
            if dna==row['dna']:
                keys=['cod','angles','paired','abin','bucket','edges','structure','mfe','dotbracket']
                f={k:z[k].copy() for k in keys};break
    if f is None:f=structure_job((str(ROOT/'source'),row['dna']))
    path=ROOT/'data/structure'/f'{row["gene"]}.npz'
    np.savez_compressed(path,**f)
    return row['gene'],sha(path)


def simulate_one(row,params):
    with np.load(ROOT/'data/structure'/f'{row["gene"]}.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
    seed=int(hashlib.sha256(('riboseq_graph_v2:'+row['gene']).encode()).hexdigest()[:8],16)
    f.update(simulation_job((str(ROOT/'source'),row['dna'],f,params,seed)))
    y=np.array([np.nan if v is None else v for v in row['y']],dtype=np.float32)
    f['observed']=np.isfinite(y);f['label']=np.log1p(np.nan_to_num(y,nan=0)).astype(np.float32)
    path=ROOT/'data/features'/f'{row["gene"]}.npz';np.savez_compressed(path,**f)
    return dict(gene=row['gene'],sha256=sha(path))


def main():
    approval=json.loads((ROOT/'data_choice.json').read_text())
    assert approval['choice']=='related_liver_repartition','No implicit fallback to a different source'
    assert sha(SOURCE)=='70a6621d0779a0cef17ce49427560c0c3c1b3b237e005a1d1038f047aff8aa7f'
    data=ROOT/'data';data.mkdir(exist_ok=False)
    (data/'structure').mkdir();(data/'features').mkdir()
    prior=json.loads(Path('/root/graph_polish_20261031/data/exclusions.json').read_text())['records']
    previous=json.loads(Path('/root/graph_polish_20261031/data/panel.json').read_text())['records']
    prior += [r for r in previous if r['split']=='fresh_test']
    old_genes={r['gene'] for r in prior};old_dna={r['dna'] for r in prior};old_aa={protein(r['dna']) for r in prior}
    candidates=[];rejections={}
    def reject(reason):rejections[reason]=rejections.get(reason,0)+1
    for r in pd.read_csv(SOURCE).itertuples(index=False):
        ids=np.fromstring(r.codon_sequence.strip('[]'),sep=',',dtype=int);y=np.fromstring(r.annotations.strip('[]'),sep=',')
        if len(ids)<40 or len(ids)!=len(y) or (ids<0).any() or (ids>=64).any():reject('invalid_length_or_ids');continue
        m=np.isfinite(y)
        if m.mean()<.95 or (y[m]<0).any() or np.mean(y[m]>0)<.3 or np.ptp(y[m])==0:reject('coverage_or_constant');continue
        longest=current=0
        for v in y:current=current+1 if v==0 else 0;longest=max(longest,current)
        if longest>20:reject('long_zero_run');continue
        dna=''.join(CODONS[i] for i in ids);aa=protein(dna);gene=str(r.gene).split('.')[0]
        if not dna.startswith('ATG') or dna[-3:] not in {'TAA','TGA','TAG'} or '*' in aa[:-1]:reject('incomplete_or_invalid_cds');continue
        rank=hashlib.sha256(('expanded_riboseq_v2:'+gene+':'+str(r.transcript)).encode()).hexdigest()
        known=gene in old_genes or dna in old_dna or aa in old_aa
        candidates.append(dict(gene=gene,transcript=str(r.transcript),dna=dna,protein_sha256=hashlib.sha256(aa.encode()).hexdigest(),
            y=[float(v) if np.isfinite(v) else None for v in y],rank=rank,previously_used=known))
    rows=[];genes=set();dnas=set();aas=set()
    for r in sorted(candidates,key=lambda x:x['rank']):
        if r['gene'] in genes or r['dna'] in dnas or r['protein_sha256'] in aas:reject('exact_duplicate_gene_cds_protein');continue
        genes.add(r['gene']);dnas.add(r['dna']);aas.add(r['protein_sha256']);rows.append(r)
    unused=[r for r in rows if not r['previously_used']]
    ntest=max(120,round(len(rows)*.1));assert len(unused)>ntest
    test={r['gene'] for r in unused[:ntest]}
    rest=[r for r in rows if r['gene'] not in test]
    nval=max(120,round(len(rows)*.1));validation={r['gene'] for r in rest[:nval]}
    for r in rows:r['split']='test' if r['gene'] in test else 'validation' if r['gene'] in validation else 'train'
    counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}
    assert all(not r['previously_used'] for r in rows if r['split']=='test')
    save(data/'panel.json',dict(source='related-study processed GSE73553 LIVER, not original RiboGL train.csv',
        source_doi='10.5281/zenodo.17100113',source_sha256=sha(SOURCE),counts=counts,rejections=rejections,
        no_length_truncation=True,near_homology_clustered=False,original_paper_split_reproduced=False,records=rows))
    workers=min(6,os.cpu_count() or 1);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fold_one,r) for r in rows]
        for i,future in enumerate(as_completed(futures)):
            future.result()
            if (i+1)%10==0:save(ROOT/'prepare_status.json',dict(stage='folding',completed=i+1,total=len(rows),seconds=time.monotonic()-started,unix=time.time()))
    trainrows=[r for r in rows if r['split']=='train'];features=[]
    for r in trainrows:
        with np.load(data/'structure'/f'{r["gene"]}.npz',allow_pickle=False) as z:features.append({k:z[k].copy() for k in z.files})
    params=fit_physics(trainrows,features);save(data/'physics_parameters_train_only.json',params)
    manifest=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(simulate_one,r,params) for r in rows]
        for i,future in enumerate(as_completed(futures)):
            manifest.append(future.result())
            if (i+1)%10==0:save(ROOT/'prepare_status.json',dict(stage='simulating',completed=i+1,total=len(rows),seconds=time.monotonic()-started,unix=time.time()))
    lookup={r['gene']:r for r in manifest}
    save(data/'features_locked.json',dict(panel_sha256=sha(data/'panel.json'),features=[lookup[r['gene']] for r in rows]))
    save(ROOT/'prepare_status.json',dict(stage='completed',counts=counts,n_records=len(rows),seconds=time.monotonic()-started,unix=time.time()))


if __name__=='__main__':main()
