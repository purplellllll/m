"""Lock all weights and validation selection, then evaluate held-out profiles."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from physics_scorer.graph_polish_model import GraphPolishModel, MODES
from physics_scorer.fusion_encoder_model import FusionEncoderModel
from train_graph_polish_20261031 import SEEDS, collate, sha, save

ROOT=Path(__file__).resolve().parent
HEADS=['main','physics','gnn','context']


def corr(y,q):
    return float(np.corrcoef(y.astype(float),q.astype(float))[0,1]) if len(y)>1 and np.ptp(y)>0 and np.ptp(q)>0 else np.nan


def load_data():
    rows=json.loads((ROOT/'data/panel.json').read_text())['records'];features=[]
    manifest=json.loads((ROOT/'data/features_locked.json').read_text())
    assert manifest['panel_sha256']==sha(ROOT/'data/panel.json')
    for r,locked in zip(rows,manifest['features']):
        path=ROOT/'data/frozen_inputs/features'/f'{r["gene"]}.npz'
        if not path.exists():path=ROOT/'data/features'/f'{r["gene"]}.npz'
        assert locked['gene']==r['gene'] and locked['sha256']==sha(path)
        with np.load(path,allow_pickle=False) as z:features.append({k:z[k].copy() for k in z.files})
        y=np.array([np.nan if v is None else v for v in r['y']],dtype=np.float32)
        assert np.array_equal(np.isfinite(y),features[-1]['observed'])
        np.testing.assert_allclose(np.log1p(np.nan_to_num(y,nan=0)),features[-1]['label'],atol=1e-7)
    assert len(rows)==len(features)==len(manifest['features'])==720
    return rows,features


def make_model(record):
    model=(GraphPolishModel(**record['setting']) if record['kind']=='polish' else FusionEncoderModel(**record['setting'])).cuda()
    model.load_state_dict(torch.load(record['path'],map_location='cpu',weights_only=True)['state_dict'])
    return model.eval()


@torch.no_grad()
def predict(model,features,intervention=None):
    output={h:[] for h in HEADS}
    for start in range(0,len(features),8):
        examples=features[start:start+8];x,_,_=collate(examples)
        kwargs=dict(physics_override=intervention) if intervention is not None else {}
        q,aux=model(*(v.cuda() for v in x),**kwargs)
        for j,f in enumerate(examples):
            for h,value in [('main',q),*aux.items()]:
                a=value[j,:len(f['cod'])].cpu().numpy().astype(float)
                assert np.isfinite(a).all() and a.sum()>0
                output[h].append(a/a.sum())
    return output


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not (ROOT/'metrics.csv').exists()
    protocol=json.loads((ROOT/'protocol_locked.json').read_text())
    for name,digest in protocol['source_hashes'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'data/features_locked.json')==protocol['data_features_manifest_sha256']
    records=[json.loads((ROOT/f'runs/{a}_{s}/locked.json').read_text()) for a in MODES for s in SEEDS]
    records+=json.loads((ROOT/'references_locked.json').read_text())['models']
    for r in records:assert r['sha256']==sha(r['path'])
    save(ROOT/'all_checkpoints_locked.json',dict(unix=time.time(),models=records))
    eligible=['parallel','physics_polish','gnn_polish','physics_gnn_polish']
    values={a:float(np.mean([r['validation_pcc'] for r in records if r['arm']==a])) for a in eligible}
    save(ROOT/'validation_choice.json',dict(selected_arm=max(values,key=values.get),mean_seed_validation_pcc=values,
        unix=time.time(),test_metrics_existed=False))
    rows,features=load_data();indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['validation','fresh_test','supplementary_test']}
    pred={s:{} for s in indices};per=[];summary=[];archive={}
    arms=list(MODES)+sorted({r['arm'] for r in records if r['kind']=='fusion'})
    for k,r in enumerate(records):
        model=make_model(r);name=f'{r["arm"]}_{r["seed"]}'
        for split,ids in indices.items():
            examples=[features[i] for i in ids];pred[split][name]=predict(model,examples)
            if r['arm']=='physics_gnn_polish':
                for intervention in ['zero','shuffle']:
                    pred[split][f'early_{intervention}_{r["seed"]}']=predict(model,examples,intervention)
        del model;torch.cuda.empty_cache()
        save(ROOT/'status.json',dict(stage='evaluating',completed_models=k+1,total_models=len(records),unix=time.time()))
    for split,ids in indices.items():
        for arm in arms+['early_zero','early_shuffle']:
            pred[split][arm]={h:[np.mean([pred[split][f'{arm}_{s}'][h][j] for s in SEEDS],0) for j in range(len(ids))] for h in HEADS}
        for name,outputs in pred[split].items():
            values=[];internal=[]
            for j,i in enumerate(ids):
                f=features[i];mask=f['observed'];inside=mask.copy();inside[:5]=False;inside[-5:]=False
                value=corr(f['label'][mask],outputs['main'][j][mask]);iv=corr(f['label'][inside],outputs['main'][j][inside])
                values.append(value);internal.append(iv)
                item=dict(split=split,model=name,gene=rows[i]['gene'],pcc=value,interior_pcc=iv)
                for h in HEADS[1:]:item[h+'_pcc']=corr(f['label'][mask],outputs[h][j][mask])
                per.append(item)
                for h in HEADS:archive[f'{split}__{name}__{h}__{j}']=outputs[h][j]
            boot=np.random.default_rng(94031).integers(0,len(ids),(2000,len(ids)))
            lo,hi=np.nanpercentile(np.nanmean(np.array(values)[boot],1),[2.5,97.5])
            summary.append(dict(split=split,model=name,n_defined=int(np.isfinite(values).sum()),mean_pcc=float(np.nanmean(values)),
                ci_low=lo,ci_high=hi,mean_interior_pcc=float(np.nanmean(internal))))
        for j,i in enumerate(ids):archive[f'{split}__label__{j}']=features[i]['label'];archive[f'{split}__observed__{j}']=features[i]['observed']
    pd.DataFrame(per).to_csv(ROOT/'per_transcript.csv',index=False);pd.DataFrame(summary).to_csv(ROOT/'metrics.csv',index=False)
    np.savez_compressed(ROOT/'predictions.npz',**archive)
    reload_errors={}
    for arm in ['parallel','gnn_polish','physics_gnn_polish','diagonal_attention']:
        r=next(r for r in records if r['arm']==arm and r['seed']==SEEDS[0]);model=make_model(r)
        output=predict(model,[features[i] for i in indices['fresh_test']])
        error=max(float(np.max(np.abs(q-pred['fresh_test'][f'{arm}_{SEEDS[0]}']['main'][j]))) for j,q in enumerate(output['main']))
        assert error<1e-7;reload_errors[arm]=error;del model;torch.cuda.empty_cache()
    # Read each NPZ array once; centered-dot PCC differs from evaluator np.corrcoef.
    errors=[]
    for row in per:
        split=row['split'];j=next(j for j,i in enumerate(indices[split]) if rows[i]['gene']==row['gene'])
        y=archive[f'{split}__label__{j}'];mask=archive[f'{split}__observed__{j}']
        for head,column in [('main','pcc'),('main','interior_pcc'),('physics','physics_pcc'),('gnn','gnn_pcc'),('context','context_pcc')]:
            m=mask.copy()
            if column=='interior_pcc':m[:5]=False;m[-5:]=False
            a=y[m].astype(float);b=archive[f'{split}__{row["model"]}__{head}__{j}'][m].astype(float)
            a-=a.mean();b-=b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b));v=np.dot(a,b)/den if den>0 else np.nan
            assert np.isfinite(v)==np.isfinite(row[column])
            if np.isfinite(v):errors.append(abs(v-row[column]));assert errors[-1]<1e-7
    save(ROOT/'verification.json',dict(status='passed',n_new_weights=24,n_total_weights=len(records),n_records=len(rows),
        checked_metric_values=len(errors),max_pearson_error=max(errors),reload_max_errors=reload_errors,
        predictions_sha256=sha(ROOT/'predictions.npz')))
    save(ROOT/'status.json',dict(stage='verified',n_new_weights=24,n_total_weights=len(records),unix=time.time()))


if __name__=='__main__':main()
