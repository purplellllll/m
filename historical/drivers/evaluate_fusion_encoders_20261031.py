"""Equal-mass, aligned Pearson evaluation after all weights are locked."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from physics_scorer.fusion_encoder_model import FusionEncoderModel
from train_fusion_encoders_20261031 import ARMS,collate,sha,save
SEEDS=[30101,30102,30103]


def corr(y,q):
    return float(np.corrcoef(y.astype(float),q.astype(float))[0,1]) if len(y)>1 and np.ptp(y)>0 and np.ptp(q)>0 else np.nan


def load_data():
    rows=json.loads((ROOT/'data/panel.json').read_text())['records'];fs=[]
    for r in rows:
        p=ROOT/'data/frozen_inputs/features'/f'{r["gene"]}.npz'
        if not p.exists():p=ROOT/'data/features'/f'{r["gene"]}.npz'
        with np.load(p,allow_pickle=False) as z:fs.append({k:z[k].copy() for k in z.files})
    return rows,fs


@torch.no_grad()
def predict(model,examples):
    model.eval();outputs={k:[] for k in ['main','physics','gnn','context']}
    for start in range(0,len(examples),8):
        batch=examples[start:start+8];x,_,_=collate(batch);q,a=model(*(v.cuda() for v in x))
        for j,f in enumerate(batch):
            n=len(f['cod'])
            for key,value in [('main',q),*a.items()]:
                p=value[j,:n].cpu().numpy().astype(float);assert np.isfinite(p).all() and p.sum()>0
                outputs[key].append(p/p.sum())
    return outputs


def make_model(record):
    model=FusionEncoderModel(**record['setting']).cuda()
    state=torch.load(record['path'],map_location='cpu',weights_only=True)['state_dict']
    if record['arm']=='frozen_bilstm':
        missing=model.load_state_dict(state,strict=False)
        assert set(missing.missing_keys)=={'fixed_gains','gain_logits'} and not missing.unexpected_keys
    else:model.load_state_dict(state)
    return model.eval()


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows,fs=load_data();records=[json.loads((ROOT/f'runs/{a}_{s}/locked.json').read_text()) for a in ARMS for s in SEEDS]
    protocol=json.loads((ROOT/'protocol_locked.json').read_text())
    for name,digest in protocol['source_hashes'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'data/features_locked.json')==protocol['data_features_manifest_sha256']
    assert len(rows)==len({r['gene'] for r in rows})==720
    for row,f in zip(rows,fs):
        y=np.array([np.nan if v is None else v for v in row['y']],dtype=np.float32)
        assert np.array_equal(np.isfinite(y),f['observed'])
        np.testing.assert_allclose(np.log1p(np.nan_to_num(y,nan=0)),f['label'],atol=1e-7)
    for seed in SEEDS:
        path=ROOT/f'references/bilstm_{seed}.pt'
        records.append(dict(arm='frozen_bilstm',seed=seed,path=str(path),sha256=sha(path),setting=dict(encoder='bilstm')))
    for r in records:assert sha(r['path'])==r['sha256']
    assert not (ROOT/'metrics.csv').exists()
    save(ROOT/'all_checkpoints_locked.json',dict(unix=time.time(),models=records))
    val={a:float(np.mean([r['validation_pcc'] for r in records if r['arm']==a])) for a in ARMS}
    save(ROOT/'validation_choice.json',dict(unix=time.time(),selected_arm=max(val,key=val.get),mean_seed_validation_pcc=val,test_metrics_existed=False))
    indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['validation','fresh_test','supplementary_test']}
    predicted={s:{} for s in indices};per=[];summary=[];archive={}
    for k,record in enumerate(records):
        model=make_model(record);name=f'{record["arm"]}_{record["seed"]}'
        for split,ids in indices.items():predicted[split][name]=predict(model,[fs[i] for i in ids])
        del model;torch.cuda.empty_cache();save(ROOT/'status.json',dict(stage='evaluating',completed_models=k+1,total_models=len(records),unix=time.time()))
    for split,ids in indices.items():
        for arm in list(ARMS)+['frozen_bilstm']:
            predicted[split][arm]={head:[np.mean([predicted[split][f'{arm}_{s}'][head][j] for s in SEEDS],0) for j in range(len(ids))] for head in ['main','physics','gnn','context']}
        for name,outputs in predicted[split].items():
            values=[];internal=[]
            for j,i in enumerate(ids):
                f=fs[i];mask=f['observed'];inside=mask.copy();inside[:5]=False;inside[-5:]=False
                q=outputs['main'][j];r=corr(f['label'][mask],q[mask]);ir=corr(f['label'][inside],q[inside]);values.append(r);internal.append(ir)
                item=dict(split=split,model=name,gene=rows[i]['gene'],pcc=r,interior_pcc=ir)
                for head in ['physics','gnn','context']:
                    item[head+'_pcc']=corr(f['label'][mask],outputs[head][j][mask])
                per.append(item)
                for head in outputs:archive[f'{split}__{name}__{head}__{j}']=outputs[head][j]
            boot=np.random.default_rng(301031).integers(0,len(ids),(2000,len(ids)))
            lo,hi=np.nanpercentile(np.nanmean(np.array(values)[boot],1),[2.5,97.5])
            summary.append(dict(split=split,model=name,n_defined=int(np.isfinite(values).sum()),mean_pcc=float(np.nanmean(values)),
                ci_low=lo,ci_high=hi,mean_interior_pcc=float(np.nanmean(internal))))
        for j,i in enumerate(ids):archive[f'{split}__label__{j}']=fs[i]['label'];archive[f'{split}__observed__{j}']=fs[i]['observed']
    pd.DataFrame(per).to_csv(ROOT/'per_transcript.csv',index=False);pd.DataFrame(summary).to_csv(ROOT/'metrics.csv',index=False)
    np.savez_compressed(ROOT/'predictions.npz',**archive)
    # Independent float64 metric/ensemble checks and representative weight reload.
    table=pd.DataFrame(per).set_index(['split','model','gene']);errors=[];reloaded={}
    with np.load(ROOT/'predictions.npz',allow_pickle=False) as pp:
        for split,ids in indices.items():
            for name in predicted[split]:
                for j,i in enumerate(ids):
                    f=fs[i];mask=f['observed'];q=pp[f'{split}__{name}__main__{j}'];v=corr(f['label'][mask],q[mask])
                    saved=float(table.loc[(split,name,rows[i]['gene']),'pcc']);assert abs(v-saved)<1e-7;errors.append(abs(v-saved))
                    assert abs(q.sum()-1)<1e-10
                    if name in list(ARMS)+['frozen_bilstm']:
                        expected=np.mean([pp[f'{split}__{name}_{s}__main__{j}'] for s in SEEDS],0);np.testing.assert_allclose(expected,q,atol=0,rtol=0)
        for arm in ['bilstm_baseline','learned_fusion','transformer','bert','t5']:
            record=next(r for r in records if r['arm']==arm and r['seed']==30101);model=make_model(record)
            outputs=predict(model,[fs[i] for i in indices['fresh_test']])['main']
            error=max(float(np.max(np.abs(q-pp[f'fresh_test__{arm}_30101__main__{j}']))) for j,q in enumerate(outputs))
            assert error<1e-7;reloaded[arm]=error;del model;torch.cuda.empty_cache()
    save(ROOT/'verification.json',dict(status='passed',n_new_weights=27,n_total_weights=30,n_records=len(rows),
        max_pearson_error=max(errors),reload_max_errors=reloaded,predictions_sha256=sha(ROOT/'predictions.npz')))
    save(ROOT/'status.json',dict(stage='verified',n_new_weights=27,n_total_weights=30,unix=time.time()))


if __name__=='__main__':main()
