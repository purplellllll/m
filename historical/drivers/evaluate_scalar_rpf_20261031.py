"""Pair current scalar probes and original RiboDecode on identical RPF rows."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_scalar_rpf_20261031 import prepare_inputs,prediction
from train_scalar_tasks_20261031 import sha,save,corr
from physics_scorer.scalar_downstream import ScalarHeads


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seeds=[30101,30102,30103];arms=['full_probe','rna_context_only'];locks={}
    for arm in arms:
        locks[arm]=[json.loads((ROOT/f'runs/rpf_{arm}_{s}/locked.json').read_text()) for s in seeds]
        for lock in locks[arm]:assert sha(lock['path'])==lock['sha256']
    save(ROOT/'rpf_evaluation_locked.json',dict(locks=locks,primary_model='full_probe three-seed ensemble',
        primary_metric='within-environment Pearson across author-test genes, macro average; seen and unseen environments separately',unix=time.time()))
    preds={};index=None
    for seed in seeds:
        protocol,d,inputs=prepare_inputs(seed)
        gs=d['gene_split'][d['gene_index']];es=d['environment_split'][d['environment_index']]
        current=np.flatnonzero((gs==2)|(es==2))
        if index is None:index=current
        else:np.testing.assert_array_equal(index,current)
        for arm in arms:
            lock=next(r for r in locks[arm] if r['seed']==seed)
            head=ScalarHeads(424,'rpf').cuda();head.load_state_dict(torch.load(lock['path'],map_location='cpu',weights_only=True)['state_dict']);head.eval()
            values=[]
            with torch.no_grad():
                emb=head.environment(inputs['environment'])
                for start in range(0,len(index),2048):
                    ix=torch.as_tensor(index[start:start+2048],device='cuda')
                    values.append(prediction(head,inputs,ix,emb,arm=='rna_context_only').cpu().numpy())
            preds[f'{arm}_{seed}']=np.concatenate(values);del head
        del inputs;torch.cuda.empty_cache()
    for arm in arms:preds[arm]=np.mean([preds[f'{arm}_{s}'] for s in seeds],0)
    # Original author's architecture and exact frozen checkpoint.
    spec=importlib.util.spec_from_file_location('native_ribodecode',ROOT/'ribodecode_source/inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((ROOT/'ribodecode_source/model_config.json').read_text())
    net=module.CNN_Encoder_Model(config,module.InferenceModel.input_shape).cuda()
    checkpoint=ROOT/'ribodecode_source/ribodecode_inference_only.pt'
    assert sha(checkpoint)=='c68b9c8a0fd9a4cc66fd008e669bff6f728374da9315eae9651c13bb6364a731'
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']);net.eval()
    alphabet=np.full(256,4,dtype=np.int64)
    for j,c in enumerate('ACGT'):alphabet[ord(c)]=j
    baseline=[];started=time.monotonic()
    contexts=torch.from_numpy(d['environment']).cuda()
    for start in range(0,len(index),32):
        ix=index[start:start+32];ids=np.full((len(ix),4500),4,dtype=np.int64)
        for j,si in enumerate(d['sequence_index'][ix]):
            dna=protocol['records'][int(si)]['dna'];ids[j,:len(dna)]=alphabet[np.frombuffer(dna.encode(),dtype=np.uint8)]
        x=torch.nn.functional.one_hot(torch.from_numpy(ids).cuda(),5)[...,:4].float().transpose(1,2)
        env=torch.as_tensor(d['environment_index'][ix],device='cuda',dtype=torch.long)
        rna=torch.from_numpy(d['log_rna'][ix,None]).cuda()
        with torch.no_grad():p=net(x,contexts[env],rna).flatten().cpu().numpy()
        assert np.isfinite(p).all();baseline.append(p)
        if start%3200==0:save(ROOT/'rpf_baseline_status.json',dict(completed=min(start+32,len(index)),total=len(index),seconds=time.monotonic()-started,unix=time.time()))
    preds['RiboDecode']=np.concatenate(baseline);preds['RNA_only']=d['log_rna'][index]
    selected_gs=gs[index];selected_es=es[index]
    cohorts=np.where(selected_gs==2,np.where(selected_es==0,'unseen_gene_seen_environment','unseen_gene_unseen_environment'),
        np.where(selected_gs==1,'validation_gene_unseen_environment','seen_gene_unseen_environment'))
    table=pd.DataFrame(dict(row_index=index,gene_index=d['gene_index'][index],environment_index=d['environment_index'][index],cohort=cohorts,target=d['log_rpf'][index]))
    per=[]
    for name,p in preds.items():
        for (cohort,env),group in table.groupby(['cohort','environment_index']):
            ix=group.index.to_numpy();per.append(dict(model=name,cohort=cohort,environment_index=int(env),n=len(ix),
                pearson=corr(group.target.to_numpy(),p[ix]),mse=float(np.mean((group.target.to_numpy()-p[ix])**2))))
    per=pd.DataFrame(per);per.to_csv(ROOT/'rpf_per_environment.csv',index=False)
    per.groupby(['model','cohort']).agg(pearson=('pearson','mean'),macro_mse=('mse','mean'),environments=('environment_index','count'),n=('n','sum')).reset_index().to_csv(ROOT/'rpf_metrics.csv',index=False)
    rng=np.random.default_rng(20261031);paired=[]
    for cohort,g in per[per.model.isin(['full_probe','RiboDecode','rna_context_only'])].groupby('cohort'):
        pivot=g.pivot(index='environment_index',columns='model',values='pearson').dropna()
        for reference in ['RiboDecode','rna_context_only']:
            delta=(pivot.full_probe-pivot[reference]).to_numpy()
            boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
            paired.append(dict(cohort=cohort,reference=reference,n_environments=len(delta),delta_pearson=float(delta.mean()),
                environment_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                limitation='Environment-level bootstrap; repeated genes and shared studies are not independent. This is exploratory inference.'))
    save(ROOT/'rpf_paired_comparison.json',dict(comparisons=paired,unit='within-environment Pearson; paired environments',strict_native_sota_certified=False))
    table.to_csv(ROOT/'rpf_test_rows.csv',index=False)
    np.savez_compressed(ROOT/'rpf_test_predictions.npz',**preds,target=d['log_rpf'][index],row_index=index)
    save(ROOT/'rpf_evaluation_complete.json',dict(unix=time.time(),n=len(index),
        baseline_checkpoint_sha256=sha(checkpoint),strict_native_sota_certified=False,
        limitation='Common canonical90..4500nt full-CDS subset, not certified full paper population; frozen-representation first-stage probe'))


if __name__=='__main__':main()
