"""Train/validate current-model scalar heads, without reading test labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.append('/root/graph_polish_20261031')
from physics_scorer.graph_polish_model import GraphPolishModel
from physics_scorer.scalar_downstream import TaskScalarModel
from run_aligned_profile_ablation_20261030 import collate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def corr(a,b):
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a.astype(float),b.astype(float))[0,1])


def global_features(row,f):
    dna=row['dna'];mean=f['simulation_mean'];sd=f['simulation_sd']
    return np.array([np.log1p(len(f['cod']))/8,(dna.count('G')+dna.count('C'))/len(dna),
        float(f['mfe'])/len(dna),np.log1p(mean.mean())/8,np.log1p(sd.mean())/8,
        sd.mean()/max(float(mean.mean()),1e-8)],dtype=np.float32)


def load(task,include_test=False,target_index=None):
    p=ROOT/'data'/f'{task}.json';protocol=json.loads(p.read_text())
    lock=json.loads((ROOT/f'{task}_features_locked.json').read_text());assert lock['task_sha256']==sha(p)
    rows=[r for r in protocol['records'] if (include_test or r['split']!='test') and not r.get('exclude_from_clean_fit',False)];features={}
    if target_index is not None:
        rows=[dict(r,target=[r['target'][target_index]]) for r in rows if r['target'][target_index] is not None]
    for r in rows:
        if r['uid'] in features:continue
        path=ROOT/'features'/f'{r["uid"]}.npz';assert sha(path)==lock['files'][r['uid']]
        with np.load(path,allow_pickle=False) as z:features[r['uid']]={k:z[k].copy() for k in z.files}
    return protocol,rows,features


def make_model(task,seed,outputs,use_utr=False,model_kind='ours',target_index=None,normalization=None):
    if model_kind=='seq2ribo':
        assert task=='te' and outputs==1 and target_index in range(4)
        sys.path.insert(0,str(ROOT/'seq2ribo_source'))
        from seq2ribo import Seq2Ribo
        from physics_scorer.seq2ribo_scalar_reference import Seq2RiboScalarReference
        cell=['ipsc','hek293','lcl','rpe'][target_index]
        predictor=Seq2Ribo(cell_line=cell,device='cpu',weights_dir=str(ROOT/'seq2ribo_source/weights'),cache_dir=str(ROOT/'baseline_geometry'))
        base=predictor._load_model('riboseq')
        path=ROOT/'seq2ribo_source/weights'/f'{cell}_mamba_final_unscaled.pt'
        metadata=dict(path=str(path),sha256=sha(path),cell=cell,
            training='New scalar head; ONLY profile pretrained weights, no pretrained TE checkpoint',
            split='Locked common RiboNN folds; NOT native seq2ribo split',
            simulations='Same cached mouse-proxy four-trajectory mean for both models; common-input controlled retraining')
        return Seq2RiboScalarReference(base,use_utr,cell!='hek293',normalization),metadata
    lockpath=Path(f'/root/graph_polish_20261031/runs/physics_gnn_polish_{seed}/locked.json')
    lock=json.loads(lockpath.read_text());assert sha(lock['path'])==lock['sha256']
    base=GraphPolishModel(**lock['setting']);base.load_state_dict(torch.load(lock['path'],map_location='cpu',weights_only=True)['state_dict'])
    return TaskScalarModel(base,task,outputs,use_utr=use_utr),lock


def batch(rows,fs,device,use_utr=False):
    examples=[fs[r['uid']] for r in rows];inputs,_,_=collate(examples)
    x=tuple(v.to(device) for v in inputs)
    g=torch.from_numpy(np.stack([global_features(r,f) for r,f in zip(rows,examples)])).to(device)
    labels=np.array([[np.nan if v is None else v for v in r['target']] for r in rows],dtype=np.float32)
    extras={}
    if use_utr:
        utrs=[];alphabet={c:i for i,c in enumerate('ACGT')}
        for key in ['utr5','utr3']:
            n=max(1,max(len(r[key]) for r in rows));ids=torch.full((len(rows),n),4,dtype=torch.long)
            for i,r in enumerate(rows):
                if r[key]:ids[i,:len(r[key])]=torch.tensor([alphabet[c] for c in r[key]])
            utrs.append((ids.to(device),(ids!=4).to(device)))
        extras['utrs']=utrs
    return x,g,torch.from_numpy(labels).to(device),extras


def microbatches(rows,use_utr=False):
    current=[];longest=0;largest_utr=0
    for row in rows:
        n=len(row['dna'])//3;u=max(len(row.get('utr5','')),len(row.get('utr3',''))) if use_utr else 0
        size=len(current)+1;length=max(longest,n);utr=max(largest_utr,u)
        if current and (size>8 or size*length>2048 or size*length*length>1500000 or size*utr>8192):
            yield current;current=[];longest=0;largest_utr=0
        current.append(row);longest=max(longest,n);largest_utr=max(largest_utr,u)
    if current:yield current


@torch.no_grad()
def evaluate(model,rows,fs,mean,scale,device,use_utr=False):
    model.eval();pred=[];actual=[]
    for group in microbatches(rows,use_utr):
        x,g,y,e=batch(group,fs,device,use_utr)
        pred.append((model(x,g,**e)*scale+mean).cpu().numpy());actual.append(y.cpu().numpy())
    a=np.concatenate(actual);b=np.concatenate(pred);metrics=[]
    for j in range(a.shape[1]):
        m=np.isfinite(a[:,j]);metrics.append(dict(output=j,n=int(m.sum()),pearson=corr(a[m,j],b[m,j]),mse=float(np.mean((a[m,j]-b[m,j])**2))))
    return metrics,b


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=args.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    protocol,rows,fs=load(args.task,target_index=args.target_index);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    y=np.array([[np.nan if v is None else v for v in r['target']] for r in train],dtype=float)
    mean=torch.tensor(np.nanmean(y,0),dtype=torch.float32,device='cuda')[None]
    scale=torch.tensor(np.nanstd(y,0),dtype=torch.float32,device='cuda').clamp_min(1e-6)[None]
    normalization=dict(target_low=float(np.nanmin(y)),target_high=float(np.nanmax(y)),target_mean=float(mean[0,0]),target_scale=float(scale[0,0]))
    model,pretrained=make_model(args.task,seed,y.shape[1],args.use_utr,args.model,args.target_index,normalization);model=model.cuda()
    task_name=args.task+('_utr' if args.use_utr else '')+('' if args.target_index is None else f'_target{args.target_index}')
    if args.model!='ours':task_name+='_'+args.model
    out=ROOT/'runs'/f'{task_name}_{args.arm}_{seed}';out.mkdir(parents=True,exist_ok=False)
    if args.arm=='frozen_probe':model.backbone.requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':model.heads.parameters(),'lr':3e-4}]+
        ([{'params':model.backbone.parameters(),'lr':3e-5}] if args.arm=='finetune' else []),weight_decay=.01)
    locked=dict(task=args.task,arm=args.arm,seed=seed,n_outputs=y.shape[1],use_utr=args.use_utr,model_kind=args.model,target_index=args.target_index,
        target_normalization=normalization,
        n_train=len(train),n_validation=len(val),n_test_evaluated=0,task_sha256=sha(ROOT/'data'/f'{args.task}.json'),
        pretrained_checkpoint=pretrained,pretrained_species='mouse liver' if args.model=='ours' else 'human cell-line profile pretraining',
        target_mean=mean.cpu().tolist(),target_scale=scale.cpu().tolist(),
        max_steps=args.max_steps,validation_every=100,early_stop_checks=10,selection='macro validation scalar Pearson',
        source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'physics_scorer/scalar_downstream.py',ROOT/'physics_scorer/seq2ribo_scalar_reference.py']})
    save(out/'inputs_locked.json',locked)
    swan=None
    if os.environ.get('SWANLAB_API_KEY'):
        import swanlab
        swan=swanlab.init(project='RiboSeq-GNN-Transformer-20261031',experiment_name=f'scalar_{task_name}_{args.arm}_{seed}',
                          config=locked,logdir=str(out/'swanlog'))
    order=[];best=-float('inf');stale=0;begun=time.monotonic();losses=[]
    for step in range(1,args.max_steps+1):
        if len(order)<8:order+=np.random.permutation(len(train)).tolist()
        ix=order[:8];order=order[8:];batch_rows=[train[i] for i in ix]
        model.train()
        if args.arm=='frozen_probe':model.backbone.eval()
        counts=torch.tensor(np.sum(np.isfinite([[np.nan if v is None else v for v in r['target']] for r in batch_rows]),axis=0),device='cuda').clamp_min(1)
        optimizer.zero_grad(set_to_none=True);total_loss=0.
        for group in microbatches(batch_rows,args.use_utr):
            x,g,target,extras=batch(group,fs,'cuda',args.use_utr)
            pred=model(x,g,**extras);mask=torch.isfinite(target);scaled=(target-mean)/scale
            residual=(pred-scaled.masked_fill(~mask,0)).square().masked_fill(~mask,0)
            loss=(residual.sum(0)/counts).mean()
            assert torch.isfinite(loss);loss.backward();total_loss+=float(loss.detach())
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));optimizer.step();losses.append(total_loss)
        if step%100==0:
            metrics,_=evaluate(model,val,fs,mean,scale,'cuda',args.use_utr)
            pcc=float(np.mean([m['pearson'] for m in metrics if m['pearson'] is not None]))
            event=dict(step=step,train_mse=float(np.mean(losses)),validation_pcc=pcc,per_output=metrics,
                       branch_gains=(3*model.heads.branch_logits.softmax(0)).detach().cpu().tolist() if hasattr(model.heads,'branch_logits') else None,seconds=time.monotonic()-begun)
            losses=[]
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
            save(out/'status.json',dict(stage='training',unix=time.time(),**event));print(json.dumps(event),flush=True)
            if swan:
                logs={'train/mse':event['train_mse'],'validation/scalar_pearson':pcc}
                for m in metrics:logs[f'validation/output_{m["output"]}_pearson']=m['pearson']
                swan.log(logs,step=step)
            if pcc>best:
                best=pcc;stale=0
                path=out/'best.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    **locked,best_step=step,best_validation_pcc=pcc),path)
                save(out/'locked.json',dict(path=str(path),sha256=sha(path),best_validation_pcc=pcc,best_step=step,**locked))
            else:stale+=1
            if stale>=10:break
    save(out/'status.json',dict(stage='completed',steps=step,best_validation_pcc=best,unix=time.time(),n_test_evaluated=0))
    if swan:swan.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['protein','te'],required=True)
    p.add_argument('--arm',choices=['frozen_probe','finetune'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-steps',type=int,default=2000);p.add_argument('--use-utr',action='store_true')
    p.add_argument('--model',choices=['ours','seq2ribo'],default='ours');p.add_argument('--target-index',type=int,choices=range(4))
    args=p.parse_args()
    if args.task!='te' and (args.target_index is not None or args.model!='ours'):raise ValueError('Reference model and target selection belong to TE')
    try:main(args)
    except BaseException as error:
        save(ROOT/f'failure_{args.task}_{args.arm}_{args.seed}.json',dict(error=repr(error),unix=time.time()));raise
