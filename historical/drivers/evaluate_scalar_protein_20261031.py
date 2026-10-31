"""Locked validation selection, then paired measured protein regression evaluation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_scalar_tasks_20261031 import load,make_model,evaluate,corr,sha,save


def bootstrap(y,a,b):
    rng=np.random.default_rng(20261031);delta=[]
    for _ in range(5000):
        ix=rng.integers(len(y),size=len(y));ca=corr(y[ix],a[ix]);cb=corr(y[ix],b[ix])
        if ca is not None and cb is not None:delta.append(ca-cb)
    return np.quantile(delta,[.025,.975]).tolist()


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seeds=[30101,30102,30103];arms=['frozen_probe','finetune'];locks={}
    for arm in arms:
        locks[arm]=[json.loads((ROOT/f'runs/protein_{arm}_{s}/locked.json').read_text()) for s in seeds]
        for lock in locks[arm]:assert sha(lock['path'])==lock['sha256']
    selected=max(arms,key=lambda a:np.mean([r['best_validation_pcc'] for r in locks[a]]))
    save(ROOT/'protein_evaluation_locked.json',dict(selected=selected,selection='mean validation scalar Pearson over3 seeds',
        locks=locks,baseline_primary='seq2ribo ipsc protein checkpoint, chosen from paper, not test scores',
        baseline_other_cells=['hek293','lcl','rpe'],n_stasep_runs=100,seed=20261031,unix=time.time()))
    protocol,rows,fs=load('protein',True);test=[r for r in rows if r['split']=='test'];assert len(test)==219
    pred=[]
    for seed in seeds:
        lock=next(r for r in locks[selected] if r['seed']==seed);ckpt=torch.load(lock['path'],map_location='cpu',weights_only=True)
        model,_=make_model('protein',seed,1);model.load_state_dict(ckpt['state_dict']);model=model.cuda()
        mean=torch.tensor(ckpt['target_mean'],device='cuda');scale=torch.tensor(ckpt['target_scale'],device='cuda')
        _,p=evaluate(model,test,fs,mean,scale,'cuda');pred.append(p[:,0]);del model;torch.cuda.empty_cache()
    table=pd.DataFrame(dict(uid=[r['uid'] for r in test],observed=[r['target'][0] for r in test],ours=np.mean(pred,0)))
    for seed,p in zip(seeds,pred):table[f'ours_{seed}']=p
    table['clean_test']=~table.uid.isin(protocol['normalized_cds_train_test_overlap_ids'])
    # Require the compiled execution backend to agree with the author's
    # unmodified Mamba2.2.4 class and Torch reference scan before comparison.
    parity=json.loads((ROOT/'mamba_backend_parity.json').read_text())
    assert parity['status']=='passed','Baseline needs a numerically checked backend'
    sys.path.insert(0,str(ROOT/'seq2ribo_source'))
    from seq2ribo import Seq2Ribo
    import RNA
    baseline_meta={}
    for cell in ['ipsc','hek293','lcl','rpe']:
        path=ROOT/'seq2ribo_source/weights'/f'{cell}_mamba_expr_full_final.pt'
        baseline_meta[cell]=dict(checkpoint_sha256=sha(path),backend='Mamba2.2.6.post3 compiled CUDA, checked against2.2.4 Torch reference',
            backend_parity=parity,viennarna=RNA.__version__,torch=torch.__version__)
        predictor=Seq2Ribo(cell_line=cell,device='cuda',weights_dir=str(path.parent),cache_dir=str(ROOT/'baseline_geometry'))
        results=[];begun=time.monotonic()
        for start in range(0,len(test),4):
            chunk=test[start:start+4]
            with contextlib.redirect_stdout(io.StringIO()):
                p=predictor.predict([r['dna'] for r in chunk],task='protein',n_stasep_runs=100,init_p=.01,seed=20261031,batch_size=32)
            assert np.isfinite(p).all();results.extend(p)
            save(ROOT/'protein_baseline_status.json',dict(stage='inference',cell=cell,completed=len(results),total=len(test),seconds=time.monotonic()-begun,unix=time.time()))
        table['seq2ribo_'+cell]=results
        table.to_csv(ROOT/'protein_test_predictions.csv',index=False)
        del predictor;torch.cuda.empty_cache()
    summary=[];differences=[]
    for subset,g in [('native219',table),('clean217',table[table.clean_test])]:
        for name in ['ours','seq2ribo_ipsc','seq2ribo_hek293','seq2ribo_lcl','seq2ribo_rpe']:
            summary.append(dict(subset=subset,model=name,n=len(g),pearson=corr(g.observed.to_numpy(),g[name].to_numpy()),
                mse=float(np.mean((g.observed-g[name])**2))))
        for cell in ['ipsc','hek293','lcl','rpe']:
            a=g.ours.to_numpy();b=g['seq2ribo_'+cell].to_numpy();y=g.observed.to_numpy()
            differences.append(dict(subset=subset,baseline=cell,delta_pearson=corr(y,a)-corr(y,b),
                paired_bootstrap_95ci=bootstrap(y,a,b),unit='sequence variant; no claim of novel-protein generalization'))
    pd.DataFrame(summary).to_csv(ROOT/'protein_metrics.csv',index=False)
    save(ROOT/'protein_paired_comparison.json',dict(differences=differences,baseline_runtime=baseline_meta,
        split_identity=protocol['split_note'],strict_native_sota_certified=False,
        remaining_requirements='Full seq2ribo split identity and compiled-backend parity, independent broader SOTA baselines'))
    save(ROOT/'protein_evaluation_complete.json',dict(unix=time.time(),n=219,clean_n=217,selected=selected,
        predictions_sha256=sha(ROOT/'protein_test_predictions.csv')))


if __name__=='__main__':main()
