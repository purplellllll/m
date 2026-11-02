"""External24 luciferase + chemical-HL panel; never cellular-HL acceptance."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
import train as tr
import features as ft
from model import MultiTaskTransformer,TASKS
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1];OUT=HERE/'paired_external';OUT.mkdir(exist_ok=True)
def main():
    from Bio.Seq import Seq
    source=WORK/'runs/multitask_transformer_20261102/sources/paired_assays/persist_24_expression_chemical_HL_paired.csv';d=pd.read_csv(source);rows=[]
    ft.save(OUT/'protocol.json',dict(source_sha256=ft.sha(source),script_sha256=ft.sha(__file__),
        panel='24 identical constructs with measured human-cell Nluc/Fluc6/24h AND accelerated in-solution chemical HL; unmodified RNA/HBB context',
        predictions='Frozen multi-task expression head trained Ecoli mRFP, HL head trained cellularHEK293; both are out-of-assay transfer here',
        target='log2(Nluc/Fluc24h)+log2(chemicalHL), fixed unit weights; chemicalHL NOT cellularHL',
        joint_prediction='Half-sum source-training-standardized expression and HL head outputs; no fitting on external24',
        acceptance='Secondary external stress diagnostic only; cannot satisfy cellularHL or solubility acceptance'))
    params=ft.read(HERE/'human_parameters.json');lock=ft.sha(HERE/'features_v2_protocol_locked.json');kept=[];excluded=[]
    for _,r in d.iterrows():
        dna=str(r['CDS_sequence']).upper().replace('U','T');full=str(r['sequence_with_UTRs']).upper().replace('U','T')
        if full.count(dna)!=1:
            excluded.append(dict(id=r['Sequence ID'],reason='CDS_sequence disagrees with sequence_with_UTRs; do not silently repair source'));continue
        pos=full.index(dna);uid=hashlib.sha256(dna.encode()).hexdigest();rec=dict(record_id=str(r['Sequence ID']),uid=uid,dna=dna,aa=str(Seq(dna[:-3]).translate()),utr5=full[:pos],utr3=full[pos+len(dna):],host='human',task='expression',split='external')
        path=ft.cache_path(rec)
        if not path.exists():
            v=ft.dynamics.extract(dna,'human',params=params,geometry=ft.geometry(dna),seed=20261102,n_runs=2,burn_steps=8192,observe_steps=4096)
            with path.open('wb') as f:np.savez_compressed(f,node=v['node'],global_features=v['global'],protocol_sha256=lock,dna_sha256=uid,audit_json=json.dumps(v['audit']))
        rows.append(rec)
        kept.append(r)
    d=pd.DataFrame(kept);ft.save(OUT/'identity_audit.json',dict(raw_n=24,exact_CDS_context_n=len(rows),excluded=excluded))
    bank=ft.load(rows);tr.setup();p=[]
    for seed in tr.SEEDS:
        ck=torch.load(HERE/'models'/f'dynamics_{seed}'/'best.pt',map_location='cpu',weights_only=True);model=MultiTaskTransformer(**ck['config']).cuda().eval();model.load_state_dict(ck['model'],strict=True);p.append(tr.predict(model,rows,bank));del model;torch.cuda.empty_cache()
    p=np.mean(p,axis=0);actual={'expression24h':np.log2(d.Nluc_Fluc_24h_mean.to_numpy()),'expression6h':np.log2(d.Nluc_Fluc_6h_mean.to_numpy()),'chemical_HL':np.log2(d.half_life.to_numpy())};actual['joint24h_chemicalHL']=actual['expression24h']+actual['chemical_HL']
    predicts={'expression24h':p[:,0],'expression6h':p[:,0],'chemical_HL':p[:,1],'joint24h_chemicalHL':p[:,:2].mean(1)};rng=np.random.default_rng(20261102);ix=rng.integers(len(rows),size=(2000,len(rows)));metrics=[]
    for t,y in actual.items():
        a=predicts[t];boot=np.array([tr.corr(y[i],a[i]) for i in ix],dtype=float);metrics.append(dict(task=t,n=len(y),pearson=tr.corr(y,a),ci95=np.nanquantile(boot,[.025,.975]).tolist()));d['pred_'+t]=a
    d.to_csv(OUT/'predictions.csv',index=False);ft.save(OUT/'summary.json',dict(status='completed',metrics=metrics,acceptance='Not cellularHL; no solubility labels; not primary acceptance'))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
