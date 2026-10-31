"""Full200-environment RPF scalar probe of frozen current-model representations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_scalar_tasks_20261031 import make_model,batch,sha,save,corr
from physics_scorer.scalar_downstream import ScalarHeads


@torch.no_grad()
def cache_representations(seed,protocol):
    path=ROOT/f'rpf_representations_{seed}.npz';lockpath=ROOT/f'rpf_representations_{seed}_locked.json'
    if path.exists():
        lock=json.loads(lockpath.read_text());assert sha(path)==lock['sha256']
        with np.load(path) as z:return z['pooled'],z['global_features']
    features_lock=json.loads((ROOT/'rpf_features_locked.json').read_text());model,_=make_model('rpf',seed,1);model=model.cuda().eval()
    rows=protocol['records'];pooled=np.empty((len(rows),3,848),dtype=np.float32);global_values=np.empty((len(rows),6),dtype=np.float32)
    # Length sorting bounds attention padding without truncating a single CDS.
    order=sorted(range(len(rows)),key=lambda i:len(rows[i]['dna']))
    for start in range(0,len(order),4):
        ix=order[start:start+4];rr=[dict(rows[i],target=[None]) for i in ix];fs={}
        for r in rr:
            p=ROOT/'features'/f'{r["uid"]}.npz';assert sha(p)==features_lock['files'][r['uid']]
            with np.load(p) as z:fs[r['uid']]={k:z[k].copy() for k in z.files}
        x,g,_,_=batch(rr,fs,'cuda');v,_=model.representations(x)
        pooled[ix]=v.cpu().numpy();global_values[ix]=g.cpu().numpy()
        if start%100==0:save(ROOT/'rpf_cache_status.json',dict(seed=seed,completed=min(start+4,len(order)),total=len(order),unix=time.time()))
    np.savez_compressed(path,pooled=pooled,global_features=global_values)
    save(lockpath,dict(sha256=sha(path),feature_lock_sha256=sha(ROOT/'rpf_features_locked.json'),seed=seed,
        backbone_frozen=True,label_access_for_representation=False))
    del model;torch.cuda.empty_cache();return pooled,global_values


def prepare_inputs(seed,device='cuda'):
    protocol=json.loads((ROOT/'data/rpf.json').read_text())
    assert sha(ROOT/'data/rpf_rows.npz')==protocol['row_archive_sha256']
    with np.load(ROOT/'data/rpf_rows.npz') as z:d={k:z[k].copy() for k in z.files}
    pooled,global_values=cache_representations(seed,protocol)
    result={k:torch.from_numpy(d[k]).to(device) for k in ['sequence_index','environment_index','gene_index','log_rna','environment']}
    result.update(pooled=torch.from_numpy(pooled).to(device),global_features=torch.from_numpy(global_values).to(device))
    return protocol,d,result


def prediction(head,inputs,ix,environment_embeddings=None,ablate=False):
    seq=inputs['sequence_index'][ix].long();env=inputs['environment_index'][ix].long()
    p=inputs['pooled'][seq];g=inputs['global_features'][seq]
    if ablate:p=torch.zeros_like(p);g=torch.zeros_like(g)
    env= head.environment(inputs['environment']) [env] if environment_embeddings is None else environment_embeddings[env]
    return head(p,g,log_rna=inputs['log_rna'][ix,None],environment_embedding=env).squeeze(-1)


@torch.no_grad()
def validate(head,inputs,d,index,ablate=False):
    head.eval();emb=head.environment(inputs['environment']);out=[]
    for start in range(0,len(index),2048):
        ix=torch.as_tensor(index[start:start+2048],device='cuda');out.append(prediction(head,inputs,ix,emb,ablate).cpu().numpy())
    pred=np.concatenate(out);per=[];samples=d['environment_index'][index]
    for env in np.unique(samples):
        m=samples==env;value=corr(d['log_rpf'][index[m]],pred[m])
        per.append(dict(environment=int(env),n=int(m.sum()),pearson=value))
    return float(np.mean([v['pearson'] for v in per if v['pearson'] is not None])),per


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    protocol,d,inputs=prepare_inputs(args.seed)
    gs=d['gene_split'][d['gene_index']];es=d['environment_split'][d['environment_index']]
    train=np.flatnonzero((gs==0)&(es==0));val=np.flatnonzero((gs==1)&(es==0))
    assert set(d['environment_index'][train])==set(np.flatnonzero(d['environment_split']==0))
    head=ScalarHeads(424,'rpf').cuda();ablate=args.arm=='rna_context_only'
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01)
    out=ROOT/'runs'/f'rpf_{args.arm}_{args.seed}';out.mkdir(parents=True,exist_ok=False)
    targets=torch.from_numpy(d['log_rpf']).cuda();variance=max(float(np.var(d['log_rpf'][train])),1e-6)
    locked=dict(task='rpf',arm=args.arm,seed=args.seed,training_rows=len(train),validation_rows=len(val),training_environments=200,
        backbone_frozen=True,label='log1p(5*RPF_RPKM)',max_steps=3000,validation_every=100,n_test_evaluated=0,
        selection='mean within-environment validation-gene Pearson',protocol_sha256=sha(ROOT/'data/rpf.json'),
        representation_sha256=sha(ROOT/f'rpf_representations_{args.seed}.npz'),source_sha256=sha(__file__))
    save(out/'inputs_locked.json',locked);swan=None
    if os.environ.get('SWANLAB_API_KEY'):
        import swanlab
        swan=swanlab.init(project='RiboSeq-GNN-Transformer-20261031',experiment_name=f'scalar_rpf_{args.arm}_{args.seed}',config=locked,logdir=str(out/'swanlog'))
    best=-float('inf');stale=0;losses=[];started=time.monotonic()
    for step in range(1,3001):
        head.train();ix=torch.as_tensor(rng.choice(train,512,replace=True),device='cuda')
        p=prediction(head,inputs,ix,ablate=ablate);loss=(p-targets[ix]).square().mean()/variance
        optimizer.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(),1.));optimizer.step();losses.append(float(loss.detach()))
        if step%100==0:
            pcc,metrics=validate(head,inputs,d,val,ablate);event=dict(step=step,train_mse=float(np.mean(losses)),validation_pcc=pcc,seconds=time.monotonic()-started)
            losses=[]
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            save(out/'status.json',dict(stage='training',unix=time.time(),**event));print(json.dumps(event),flush=True)
            if swan:swan.log({'validation/scalar_pearson':pcc,'train/scaled_mse':event['train_mse']},step=step)
            if pcc>best:
                best=pcc;stale=0;path=out/'best.pt';torch.save(dict(state_dict=head.state_dict(),**locked,best_step=step),path)
                save(out/'locked.json',dict(path=str(path),sha256=sha(path),best_validation_pcc=pcc,best_step=step,**locked))
            else:stale+=1
            if stale>=10:break
    save(out/'status.json',dict(stage='completed',steps=step,best_validation_pcc=best,unix=time.time(),n_test_evaluated=0))
    if swan:swan.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['full_probe','rna_context_only'],required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    try:main(args)
    except BaseException as error:
        save(ROOT/f'failure_rpf_{args.arm}_{args.seed}.json',dict(error=repr(error),unix=time.time()));raise
