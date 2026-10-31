"""Evaluate predeclared frozen candidates on an identical, historical mouse panel."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
FUSION=Path('/root/fusion_encoders_20261031')
POLISH=Path('/root/graph_polish_20261031')
sys.path.insert(0,str(POLISH))
import numpy as np
import pandas as pd
import torch
from physics_scorer.fusion_encoder_model import FusionEncoderModel
from physics_scorer.graph_polish_model import GraphPolishModel
from train_fusion_encoders_20261031 import collate

SEEDS=[30101,30102,30103]
CANDIDATES={'bilstm_context_x2':(FUSION,'context_x2',FusionEncoderModel),
    'gnn_transformer_polish':(POLISH,'physics_gnn_polish',GraphPolishModel),
    'parallel_transformer':(POLISH,'parallel',GraphPolishModel)}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')


def corr(a,b):
    a=a.astype(float)-a.astype(float).mean();b=b.astype(float)-b.astype(float).mean()
    den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else None


@torch.no_grad()
def predict(model,features):
    model.eval();result=[]
    for start in range(0,len(features),8):
        batch=features[start:start+8];x,_,_=collate(batch);q,_=model(*(v.cuda() for v in x))
        for j,f in enumerate(batch):
            v=q[j,:len(f['cod'])].cpu().numpy().astype(float)
            assert np.isfinite(v).all() and v.sum()>0;result.append(v/v.sum())
    return result


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((ROOT/'transfer_manifest.json').read_text())
    for r in manifest:assert sha(ROOT/r['path'])==r['sha256']
    panel=json.loads((ROOT/'panel.json').read_text());rows=panel['records'];assert len(rows)==120
    features=[]
    for r in rows:
        with np.load(ROOT/f'features/{r["gene"]}.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        y=np.array([np.nan if v is None else v for v in r['y']],dtype=np.float32)
        assert np.array_equal(np.isfinite(y),f['observed'])
        np.testing.assert_allclose(np.log1p(np.nan_to_num(y,nan=0)),f['label'],atol=1e-7)
        features.append(f)
    models=[]
    for name,(project,arm,_) in CANDIDATES.items():
        old=json.loads((project/'data/panel.json').read_text())['records']
        for split in ['train','validation']:
            old_set={(r['gene'],r['dna']) for r in old if r['split']==split}
            assert old_set=={tuple(v) for v in panel['fit_records'][split]}
        assert json.loads((project/'data/physics_parameters_train_only.json').read_text())==panel['physics_parameters']
        for seed in SEEDS:
            record=json.loads((project/f'runs/{arm}_{seed}/locked.json').read_text())
            assert sha(record['path'])==record['sha256']
            models.append(dict(record,name=name))
    save(ROOT/'evaluation_locked.json',dict(unix=time.time(),models=models,candidates=list(CANDIDATES),
        source_hashes={str(project/'physics_scorer'/name):sha(project/'physics_scorer'/name)
            for project in [FUSION,POLISH] for name in ['fusion_encoder_model.py','profile_model.py','model.py']},
        panel='historical previously inspected120, NOT a fresh confirmatory test',
        no_new_training=True,ensemble='unit-mass per seed before arithmetic mean',n_train=400,n_validation=80))
    output={};reload_errors={}
    for r in models:
        cls=CANDIDATES[r['name']][2];model=cls(**r['setting']).cuda()
        model.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['state_dict'])
        name=f'{r["name"]}_{r["seed"]}';output[name]=predict(model,features)
        if r['seed']==SEEDS[0]:
            again=predict(model,features[:8]);error=max(float(np.max(np.abs(a-b))) for a,b in zip(again,output[name][:8]))
            assert error<1e-7;reload_errors[r['name']]=error
        del model;torch.cuda.empty_cache()
    with np.load(ROOT/'reference_predictions.npz',allow_pickle=False) as z:
        for method in ['seq2ribo','ribogl']:
            for seed in SEEDS:
                values=[]
                for j,f in enumerate(features):
                    q=z[f'fresh_test__{method}_{seed}__{j}'].astype(float)
                    assert q.shape==f['label'].shape and np.isfinite(q).all() and q.sum()>0
                    values.append(q/q.sum())
                output[f'{method}_{seed}']=values
    arms=list(CANDIDATES)+['seq2ribo','ribogl']
    for name in arms:output[name]=[np.mean([output[f'{name}_{s}'][j] for s in SEEDS],0) for j in range(120)]
    per=[];archive={}
    for name,values in output.items():
        for j,(r,f,q) in enumerate(zip(rows,features,values)):
            mask=f['observed'];inside=mask.copy();inside[:5]=False;inside[-5:]=False
            item=dict(model=name,gene=r['gene'],transcript=r['transcript'],n_codons=len(f['cod']),
                observed_fraction=float(mask.mean()),gc_fraction=(r['dna'].count('G')+r['dna'].count('C'))/len(r['dna']),
                pcc=corr(f['label'][mask],q[mask]),interior_pcc=corr(f['label'][inside],q[inside]))
            per.append(item);archive[f'{name}__{j}']=q
    for j,f in enumerate(features):archive[f'label__{j}']=f['label'];archive[f'observed__{j}']=f['observed']
    pd.DataFrame(per).to_csv(ROOT/'common_panel_per_transcript.csv',index=False)
    np.savez_compressed(ROOT/'common_panel_predictions.npz',**archive)
    save(ROOT/'verification.json',dict(status='passed',n_genes=120,n_current_weights=len(models),n_reference_weights=6,
        n_train=400,n_validation=80,fit_sets_exactly_equal=True,physics_parameters_exactly_equal=True,
        reproducibility_errors=reload_errors,predictions_sha256=sha(ROOT/'common_panel_predictions.npz'),torch_version=torch.__version__))
    print(json.dumps(dict(status='completed',current_models=list(CANDIDATES))))


if __name__=='__main__':main()
