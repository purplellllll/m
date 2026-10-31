"""Full-panel training with optimizer-step validation and private SwanLab metrics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault('USE_TF','0')
BASE=Path(__file__).resolve().parent
if not (BASE/'physics_scorer').exists():BASE=BASE.parent
sys.path.insert(0,str(BASE))
import numpy as np
import torch
from physics_scorer.graph_polish_model import GraphPolishModel
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.diagnostic_model import native_profile_loss
from run_aligned_profile_ablation_20261030 import collate

PROJECT='RiboSeq-GNN-Transformer-20261031'
ARMS={
    'baseline':dict(kind='old_polish',mode='physics_gnn_polish',width=192),
    'v2_control':dict(kind='reliable',width=192,sequence_v2=False,structure_gate=False,anchored_head=False),
    'sequence_graph':dict(kind='reliable',width=192,sequence_v2=True,structure_gate=False,anchored_head=False),
    'gated_graph':dict(kind='reliable',width=192,sequence_v2=True,structure_gate=True,anchored_head=False),
    'context_anchor':dict(kind='reliable',width=192,sequence_v2=True,structure_gate=True,anchored_head=True),
}
RECIPE=dict(max_steps=3000,validate_every=100,lr=.0003,warmup_steps=200,weight_decay=.01,
            max_batch_size=8,max_padded_codons=4096,max_attention_cells=4500000,early_stop_checks=12)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def model_for(setting):
    setting=dict(setting);kind=setting.pop('kind')
    return GraphPolishModel(**setting) if kind=='old_polish' else ReliableGraphContext(**setting)


def load_features(root,allowed_splits):
    rows=json.loads((root/'panel.json').read_text())['records'];selected=[];features=[]
    locked=json.loads((root/'features_locked.json').read_text());assert locked['panel_sha256']==sha(root/'panel.json')
    hashes={r['gene']:r['sha256'] for r in locked['features']}
    for row in rows:
        if row['split'] not in allowed_splits:continue
        path=root/'features'/f'{row["gene"]}.npz';assert sha(path)==hashes[row['gene']]
        with np.load(path,allow_pickle=False) as z:feature={k:z[k].copy() for k in z.files}
        y=np.array([np.nan if v is None else v for v in row['y']],dtype=np.float32)
        assert np.array_equal(np.isfinite(y),feature['observed'])
        np.testing.assert_allclose(np.log1p(np.nan_to_num(y,nan=0)),feature['label'],atol=1e-7)
        assert len(feature['cod'])*3==len(row['dna'])
        selected.append(row);features.append(feature)
    return selected,features


def batches(indices,features,seed=None):
    order=list(indices)
    if seed is not None:
        rng=np.random.default_rng(seed);rng.shuffle(order)
        order=[i for start in range(0,len(order),128) for i in sorted(order[start:start+128],key=lambda i:len(features[i]['cod']))]
    else:order.sort(key=lambda i:len(features[i]['cod']))
    result=[];batch=[];length=0
    for i in order:
        new_length=max(length,len(features[i]['cod']));size=len(batch)+1
        if batch and (size>8 or size*new_length>4096 or size*new_length**2>4500000):
            result.append(batch);batch=[];length=0
        batch.append(i);length=max(length,len(features[i]['cod']))
    if batch:result.append(batch)
    if seed is not None:rng.shuffle(result)
    assert sorted(i for group in result for i in group)==sorted(indices)
    return result


def pearson(a,b):
    a=a.astype(float)-a.astype(float).mean();b=b.astype(float)-b.astype(float).mean()
    den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else None


@torch.no_grad()
def evaluate(model,rows,features,indices,save_path=None):
    was_training=model.training;model.eval();metrics=[];losses=[];cache={};gate_sum=0.;gate_count=0
    for batch in batches(indices,features):
        x,y,mask=collate([features[i] for i in batch]);x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
        q,aux=model(*x)
        for layer in model.graph:
            count=getattr(layer,'last_gate_count',0)
            if count:
                gate_sum+=float(layer.last_gate_mean)*count;gate_count+=count
        for j,i in enumerate(batch):
            f=features[i];n=len(f['cod']);observed=f['observed'];target=f['label'];inside=observed.copy();inside[:5]=False;inside[-5:]=False
            prediction=q[j,:n].cpu().numpy().astype(float)
            if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
            item=dict(gene=rows[i]['gene'],pcc=pearson(target[observed],prediction[observed]),
                interior_pcc=pearson(target[inside],prediction[inside]))
            for h,v in aux.items():item[h+'_pcc']=pearson(target[observed],v[j,:n].cpu().numpy()[observed])
            metrics.append(item);losses.append(float(native_profile_loss(q[j:j+1],y[j:j+1],mask[j:j+1])))
            if save_path is not None:
                cache[f'{rows[i]["gene"]}__main']=prediction/prediction.sum()
                cache[f'{rows[i]["gene"]}__label']=target;cache[f'{rows[i]["gene"]}__observed']=observed
    values={key:float(np.mean([r[key] for r in metrics if r[key] is not None])) for key in ['pcc','interior_pcc','physics_pcc','gnn_pcc','context_pcc']}
    values.update(native_loss=float(np.mean(losses)),n_transcripts=len(metrics),n_defined=sum(r['pcc'] is not None for r in metrics))
    if gate_count:values['structure_gate_mean']=gate_sum/gate_count
    if save_path is not None:
        save(save_path.with_suffix('.json'),dict(summary=values,per_transcript=metrics));np.savez_compressed(save_path.with_suffix('.npz'),**cache)
    model.train(was_training)
    return values


def atomic_checkpoint(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def main(args):
    import swanlab
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    root=Path(args.root);out=root/'runs'/f'{args.arm}_{args.seed}';out.mkdir(parents=True,exist_ok=False)
    settings=ARMS[args.arm];model=model_for(settings).cuda()
    rows,features=load_features(root/'data',{'train','validation'})
    tr=[i for i,r in enumerate(rows) if r['split']=='train'];va=[i for i,r in enumerate(rows) if r['split']=='validation']
    assert tr and va and not any(r['split']=='test' for r in rows)
    config=dict(arm=args.arm,seed=args.seed,setting=settings,recipe=dict(RECIPE,max_steps=args.max_steps),n_train=len(tr),n_validation=len(va),
        panel_sha256=sha(root/'data/panel.json'),source='related GSE73553 repartition; original RiboGL split unavailable',
        parameters=sum(v.numel() for v in model.parameters()),driver_sha256=sha(__file__),n_test_evaluated=0)
    save(out/'inputs_locked.json',config)
    if args.swanlab_mode=='online':swanlab.login(api_key=os.environ['SWANLAB_API_KEY'],save=False)
    run=swanlab.init(project=PROJECT,experiment_name=f'{args.arm}-seed{args.seed}',mode=args.swanlab_mode,
        public=False,logdir=str(root/'swanlog'),config=config)
    save(out/'swanlab_run.json',dict(id=run.id,url=getattr(run,'url',None),mode=args.swanlab_mode))
    optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'],weight_decay=.01)
    step=0;epoch=0;seen=0;stale=0;best=-float('inf');begun=time.monotonic();last_eval=0
    try:
        while step<args.max_steps and stale<RECIPE['early_stop_checks']:
            groups=batches(tr,features,args.seed+epoch)
            for group in groups:
                step+=1;model.train();x,y,mask=collate([features[i] for i in group]);x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
                lr=.0003*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-200)/max(1,args.max_steps-200))))
                for g in optimizer.param_groups:g['lr']=lr
                optimizer.zero_grad(set_to_none=True);q,aux=model(*x)
                loss=native_profile_loss(q,y,mask)+.1*torch.stack([native_profile_loss(v,y,mask) for v in aux.values()]).mean()
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(norm);optimizer.step();seen+=len(group)
                event=dict(step=step,epoch=epoch+1,seen_transcripts=seen,loss=float(loss.detach()),lr=lr,batch_size=len(group),seconds=time.monotonic()-begun)
                logged={'train/loss':event['loss'],'train/lr':lr,'train/seen_transcripts':seen,'train/gradient_norm':float(norm)}
                if step%100==0 or step==args.max_steps:
                    validation=evaluate(model,rows,features,va);last_eval=step
                    event['validation']=validation
                    logged.update({'validation/'+k:v for k,v in validation.items()})
                    if 'structure_gate_mean' in validation:
                        logged['graph/mean_structure_gate']=validation['structure_gate_mean']
                    payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),arm=args.arm,seed=args.seed,
                        setting=settings,step=step,epoch=epoch,validation=validation,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
                    if validation['pcc']>best:
                        best=validation['pcc'];best_step=step;stale=0;atomic_checkpoint(out/'best.pt',payload)
                    else:stale+=1
                    atomic_checkpoint(out/'latest.pt',payload)
                    with (out/'history.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    save(out/'status.json',dict(stage='training',unix=time.time(),best_pcc=best,best_step=best_step,**event))
                    print(json.dumps(event),flush=True)
                if step%10==0 or step==args.max_steps:swanlab.log(logged,step=step)
                if step>=args.max_steps or stale>=RECIPE['early_stop_checks']:break
            epoch+=1
        checkpoint=out/'best.pt'
        save(out/'locked.json',dict(path=str(checkpoint),sha256=sha(checkpoint),arm=args.arm,seed=args.seed,
            setting=settings,best_validation_pcc=best,best_step=best_step,last_step=step,n_test_evaluated=0,parameters=config['parameters']))
        save(out/'status.json',dict(stage='completed',step=step,best_pcc=best,best_step=best_step,unix=time.time()))
    finally:swanlab.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--seed',type=int,default=30101);parser.add_argument('--max-steps',type=int,default=3000)
    parser.add_argument('--swanlab-mode',choices=['online','offline'],default='online');main(parser.parse_args())
