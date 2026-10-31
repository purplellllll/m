"""Final held-out evaluation only after every planned model has been locked."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from train_expanded_riboseq_20261031 import ARMS,model_for,load_features,evaluate,save,sha,pearson

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((ROOT/'protocol_locked.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert sha(ROOT/path)==digest
    assert sha(ROOT/'data/features_locked.json')==protocol['features_sha256']
    records=[json.loads((ROOT/f'runs/{arm}_{seed}/locked.json').read_text()) for arm in ARMS for seed in [30101,30102,30103]]
    for r in records:assert sha(r['path'])==r['sha256']
    save(ROOT/'all_checkpoints_locked.json',dict(models=records,unix=time.time()))
    validation={arm:float(np.mean([r['best_validation_pcc'] for r in records if r['arm']==arm])) for arm in ARMS}
    assert not (ROOT/'test_metrics.json').exists()
    save(ROOT/'validation_choice.json',dict(selected_arm=max(validation,key=validation.get),validation=validation,
        unix=time.time(),test_metrics_existed=False))
    rows,features=load_features(ROOT/'data',{'test'});indices=list(range(len(rows)))
    assert all(r['split']=='test' and not r['previously_used'] for r in rows)
    result={};outputs={};(ROOT/'test_predictions').mkdir()
    for r in records:
        model=model_for(r['setting']).cuda()
        # Locally generated checkpoints contain RNG/optimizer state, including NumPy arrays.
        state=torch.load(r['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict']);name=f'{r["arm"]}_{r["seed"]}'
        result[name]=evaluate(model,rows,features,indices,ROOT/'test_predictions'/name)
        with np.load(ROOT/'test_predictions'/f'{name}.npz',allow_pickle=False) as z:outputs[name]={k:z[k].copy() for k in z.files}
        del model;torch.cuda.empty_cache()
    for arm in ARMS:
        values=[];per=[]
        for r,f in zip(rows,features):
            gene=r['gene'];q=np.mean([outputs[f'{arm}_{s}'][gene+'__main'] for s in [30101,30102,30103]],0)
            assert abs(q.sum()-1)<1e-10
            value=pearson(f['label'][f['observed']],q[f['observed']])
            per.append(dict(gene=gene,pcc=value))
            if value is not None:values.append(value)
        rng=np.random.default_rng(1031);array=np.asarray(values);boot=rng.integers(0,len(array),(2000,len(array)))
        lo,hi=np.percentile(array[boot].mean(1),[2.5,97.5])
        result[arm]=dict(pcc=float(array.mean()),ci_low=float(lo),ci_high=float(hi),n_transcripts=len(per),n_defined=len(values))
        save(ROOT/'test_predictions'/f'{arm}_ensemble.json',dict(summary=result[arm],per_transcript=per))
    save(ROOT/'test_metrics.json',result)
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'],save=False)
    run=swanlab.init(project='RiboSeq-GNN-Transformer-20261031',experiment_name='locked-held-out-evaluation',
        public=False,mode='online',logdir=str(ROOT/'swanlog'),config=dict(n_test=len(rows),
            selected_arm=max(validation,key=validation.get),selection_before_test=True,n_weights=len(records)))
    swanlab.log({f'test/{name}/pcc':metrics['pcc'] for name,metrics in result.items()},step=0)
    save(ROOT/'test_swanlab_run.json',dict(id=run.id,url=getattr(run,'url',None)))
    swanlab.finish()
    save(ROOT/'status.json',dict(stage='completed',n_weights=len(records),n_test=len(rows),unix=time.time()))


if __name__=='__main__':main()
