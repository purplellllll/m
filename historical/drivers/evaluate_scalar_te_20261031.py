"""Paired, target-matched held-out TE regression on the common locked split."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_scalar_tasks_20261031 import load,make_model,evaluate,sha,save,corr
from evaluate_scalar_protein_20261031 import bootstrap


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seeds=[30101,30102,30103];locks={}
    for utr in [False,True]:
        for target in range(4):
            for kind in ['ours','seq2ribo']:
                name='te'+('_utr' if utr else '')+f'_target{target}'+('_seq2ribo' if kind=='seq2ribo' else '')
                key=f'{kind}_utr{utr}_target{target}'
                locks[key]=[json.loads((ROOT/f'runs/{name}_finetune_{seed}/locked.json').read_text()) for seed in seeds]
                for lock in locks[key]:assert sha(lock['path'])==lock['sha256']
    save(ROOT/'te_evaluation_locked.json',dict(unix=time.time(),locks=locks,
        selection='Best validation scalar Pearson checkpoint per run; no test-based target or input-setting selection',
        scope='Single-target training for both methods; same cached4-run simulation, CDS, UTRs and gene folds',
        limitation='Common-input retraining; NOT recovered native seq2ribo stratified split or native simulation recipe'))
    summary=[];differences=[];all_predictions=[]
    for utr in [False,True]:
        for target in range(4):
            protocol,rows,fs=load('te',True,target)
            test=[r for r in rows if r['split']=='test'];actual=np.array([r['target'][0] for r in test]);predictions={}
            for kind in ['ours','seq2ribo']:
                key=f'{kind}_utr{utr}_target{target}'
                for lock in locks[key]:
                    seed=lock['seed'];checkpoint=torch.load(lock['path'],map_location='cpu',weights_only=True)
                    model,_=make_model('te',seed,1,utr,kind,target,checkpoint['target_normalization'])
                    model.load_state_dict(checkpoint['state_dict']);model=model.cuda()
                    mean=torch.tensor(checkpoint['target_mean'],device='cuda');scale=torch.tensor(checkpoint['target_scale'],device='cuda')
                    _,values=evaluate(model,test,fs,mean,scale,'cuda',utr);predictions[f'{kind}_{seed}']=values[:,0]
                    del model;torch.cuda.empty_cache()
                predictions[kind]=np.mean([predictions[f'{kind}_{s}'] for s in seeds],0)
            for kind,value in predictions.items():
                summary.append(dict(input='CDS+UTR' if utr else 'CDS-only',target=protocol['labels'][target],model=kind,
                    n=len(test),pearson=corr(actual,value),mse=float(np.mean((actual-value)**2))))
            differences.append(dict(input='CDS+UTR' if utr else 'CDS-only',target=protocol['labels'][target],n=len(test),
                delta_pearson=corr(actual,predictions['ours'])-corr(actual,predictions['seq2ribo']),
                paired_gene_bootstrap_95ci=bootstrap(actual,predictions['ours'],predictions['seq2ribo'])))
            for i,r in enumerate(test):all_predictions.append(dict(input='CDS+UTR' if utr else 'CDS-only',target=protocol['labels'][target],
                gene=r['gene'],uid=r['uid'],observed=float(actual[i]),**{k:float(v[i]) for k,v in predictions.items()}))
    pd.DataFrame(summary).to_csv(ROOT/'te_metrics.csv',index=False)
    pd.DataFrame(all_predictions).to_csv(ROOT/'te_test_predictions.csv',index=False)
    save(ROOT/'te_paired_comparison.json',dict(differences=differences,strict_native_sota_certified=False,
        scope='Matched single-target common-input refit; four-trajectory mean, frozen mouse-proxy physics for both',
        required_for_native_reproduction='Native seq2ribo train/validation/test identifiers and simulation/training protocol'))
    save(ROOT/'te_evaluation_complete.json',dict(unix=time.time(),conditions=8,independent_test_genes=True,
        predictions_sha256=sha(ROOT/'te_test_predictions.csv')))


if __name__=='__main__':main()
