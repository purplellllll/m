"""Bounded GPU training/validation; this module never reads test targets."""
from pathlib import Path
import copy
import math
import sys
import time
import numpy as np
import torch
from common import ROOT,WORK,REPO,PILOT,SEEDS,ARMS,MAX_STEPS,EVERY,read,save,sha,module,verify_sources
sys.path.insert(0,str(PILOT))
from run_pilot import Head,ReliableGraphContext,TaskScalarModel

def batches(rows,seed=None):
    order=list(range(len(rows)))
    if seed is not None:np.random.default_rng(seed).shuffle(order)
    order=[j for start in range(0,len(order),128) for j in sorted(order[start:start+128],key=lambda j:rows[j]['n_codons'])]
    groups=[];current=[];longest=0
    for j in order:
        n=max(longest,rows[j]['n_codons']);b=len(current)+1
        if current and (b>8 or b*n>4096 or b*n*n>4500000):groups.append(current);current=[];longest=0
        current.append(j);longest=max(longest,rows[j]['n_codons'])
    if current:groups.append(current)
    if seed is not None:np.random.default_rng(seed+1).shuffle(groups)
    assert sorted(j for group in groups for j in group)==list(range(len(rows)))
    return groups

def model_for(seed):
    payload=torch.load(WORK/f'runs/scorer_compare_search_20261031/search_outputs/scorer_backbones/gated_graph_{seed}.pt',map_location='cpu',weights_only=True)
    base=ReliableGraphContext(**payload['metadata']['spec']['config']);base.load_state_dict(payload['state_dict'],strict=True)
    return TaskScalarModel(base,'te').cuda(),Head().cuda(),payload['metadata']['spec']['config']
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def setup_gpu():
    torch.set_num_threads(2);assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.65);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)

def preflight():
    setup_gpu();torch.manual_seed(91);model,head,_=model_for(SEEDS[0]);model.train();head.train()
    cod=torch.randint(0,64,(2,1500),device='cuda');valid=torch.ones_like(cod,dtype=torch.bool)
    x=(cod,valid,torch.rand(2,1500,2,device='cuda'),torch.rand(2,1500,5,device='cuda'),torch.tensor([[0,1500],[1499,2999]],device='cuda'))
    p,_=model.representations(x);loss=head(p[:,2],torch.rand(2,3,device='cuda')).square().mean();loss.backward()
    assert torch.isfinite(loss) and model.backbone.context.layers[-1].self_attn.in_proj_weight.grad.abs().sum()>0
    groups=batches([dict(n_codons=n) for n in [80,99,499,900,1500,1490,300,400,801]],17)
    for group in groups:
        n=max([80,99,499,900,1500,1490,300,400,801][i] for i in group)
        assert len(group)<=8 and len(group)*n<=4096 and len(group)*n*n<=4500000
    save(ROOT/'gpu_preflight.json',dict(status='passed',gpu=torch.cuda.get_device_name(0),batch=2,n_codons=1500,
        last_layer_gradient=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),dynamic_batch_limits_checked=True))

def train():
    protocol=verify_sources();setup_gpu();rows=read(ROOT/'train_validation.json');assert sha(ROOT/'train_validation.json')==protocol['train_validation_sha256']
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='val'];assert len(train_rows)==7373 and len(val_rows)==922
    features={};manifest={}
    for r in rows:
        path=ROOT/'features'/f'{r["uid"]}.npz'
        with np.load(path,allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        assert str(f['dna_sha256'])==r['uid'] and len(f['cod'])*3==len(r['dna']);features[r['uid']]=f;manifest[r['uid']]=sha(path)
    save(ROOT/'development_features_locked.json',dict(files=manifest,protocol_sha256=sha(ROOT/'protocol_locked.json')))
    probe=module('expanded_batch',WORK/'runs/mrfp_branch_probe_20261031/run_probe.py');cai=read(WORK/'runs/vmae_tiny_30cds_20261025/cai_weights.json')
    def stats(r):
        dna=r['dna'];return [(dna.count('G')+dna.count('C'))/len(dna),float(features[r['uid']]['mfe'])/len(dna),
            math.exp(np.mean([math.log(cai[dna[i:i+3]]) for i in range(0,len(dna)-3,3)]))]
    raw=np.array([stats(r) for r in train_rows]);gm=raw.mean(0);gs=raw.std(0).clip(1e-6)
    target=np.array([r['target'] for r in train_rows]);ym=float(target.mean());ys=max(float(target.std()),1e-6)
    normalized={r['uid']:((np.array(stats(r))-gm)/gs).astype(np.float32) for r in rows}
    def batch(group,arm):
        x,_=probe.batch(group,features,'cuda');g=np.stack([normalized[r['uid']] for r in group])
        if arm=='no_stats':g*=0
        return x,torch.tensor(g,device='cuda')
    @torch.no_grad()
    def evaluate(model,head,arm):
        model.eval();head.eval();pred=np.empty(len(val_rows))
        for indices in batches(val_rows):
            group=[val_rows[j] for j in indices];x,g=batch(group,arm);p,_=model.representations(x)
            pred[indices]=(head(p[:,2],g)*ys+ym).cpu().numpy()
        actual=np.array([r['target'] for r in val_rows]);r=float(np.corrcoef(actual,pred)[0,1]);assert math.isfinite(r)
        return r,float(np.mean((actual-pred)**2)),pred
    completed=[]
    for seed in SEEDS:
        for arm in ARMS:
            out=ROOT/'models'/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():completed.append(read(out/'complete.json'));continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model,head,config=model_for(seed)
            optimizer=torch.optim.AdamW([dict(params=model.parameters(),lr=1e-5),dict(params=head.parameters(),lr=3e-4)],weight_decay=.01)
            start=1;best=-float('inf');stale=0;epoch=0;pointer=0;history=[];seen=0
            if (out/'latest.pt').exists():
                checkpoint=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
                assert checkpoint['protocol_sha256']==sha(ROOT/'protocol_locked.json')
                model.backbone.load_state_dict(checkpoint['backbone']);head.load_state_dict(checkpoint['head']);optimizer.load_state_dict(checkpoint['optimizer'])
                start=checkpoint['step']+1;best=checkpoint['best'];stale=checkpoint['stale'];epoch=checkpoint['epoch'];pointer=checkpoint['pointer'];seen=checkpoint['seen'];history=checkpoint['history']
                torch.set_rng_state(checkpoint['rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
            groups=batches(train_rows,seed+epoch)
            for step in range(start,MAX_STEPS+1):
                if pointer==len(groups):epoch+=1;pointer=0;groups=batches(train_rows,seed+epoch)
                indices=groups[pointer];pointer+=1;group=[train_rows[j] for j in indices];seen+=len(group)
                model.train();head.train();x,g=batch(group,arm);optimizer.zero_grad(set_to_none=True)
                factor=min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*min(1.,max(0,step-200)/(MAX_STEPS-200)))))
                optimizer.param_groups[0]['lr']=1e-5*factor;optimizer.param_groups[1]['lr']=3e-4*factor
                p,_=model.representations(x);prediction=head(p[:,2],g);truth=torch.tensor([(r['target']-ym)/ys for r in group],device='cuda',dtype=torch.float32)
                loss=(prediction-truth).square().mean();assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.);assert torch.isfinite(norm);optimizer.step()
                if step%EVERY==0 or step==MAX_STEPS:
                    r,mse,pred=evaluate(model,head,arm);history.append(dict(step=step,seen_transcripts=seen,validation_pearson=r,validation_mse=mse))
                    improved=r>best
                    if improved:best=r;stale=0
                    else:stale+=1
                    checkpoint=dict(backbone=cpu_state(model.backbone),head=cpu_state(head),config=config,arm=arm,seed=seed,
                        target_mean=ym,target_scale=ys,stats_mean=gm.tolist(),stats_scale=gs.tolist(),protocol_sha256=sha(ROOT/'protocol_locked.json'),
                        step=step,validation_pearson=r,validation_predictions=pred.tolist())
                    if improved:torch.save(checkpoint,out/'best.pt')
                    torch.save(dict(**checkpoint,optimizer=optimizer.state_dict(),best=best,stale=stale,epoch=epoch,pointer=pointer,seen=seen,
                        history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'latest.pt')
                    save(out/'history.json',history);save(ROOT/'training_status.json',dict(stage='training',arm=arm,seed=seed,step=step,max_steps=MAX_STEPS,validation_pearson=r,best=best,stale=stale,completed_models=len(completed),total_models=4))
                    print(arm,seed,step,r,flush=True)
                    if stale>=8:break
            best_checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
            result=dict(status='completed',arm=arm,seed=seed,best_step=best_checkpoint['step'],validation_pearson=best_checkpoint['validation_pearson'],
                best_path=str(out/'best.pt'),checkpoint_sha256=sha(out/'best.pt'),last_step=step,seen_transcripts=seen)
            save(out/'complete.json',result);completed.append(result);del model,head,optimizer;torch.cuda.empty_cache()
    scores={arm:np.mean([r['validation_pearson'] for r in completed if r['arm']==arm]) for arm in ARMS}
    chosen=sorted(ARMS,key=lambda arm:(-scores[arm],arm))[0]
    save(ROOT/'selection_locked.json',dict(arm=chosen,validation_mean_per_seed={k:float(v) for k,v in scores.items()},models=completed,
        selected_models=[r for r in completed if r['arm']==chosen],test_predicted=False,protocol_sha256=sha(ROOT/'protocol_locked.json')))
    save(ROOT/'training_complete.json',dict(status='completed',models=4,test_predicted=False))

if __name__=='__main__':
    try:preflight() if '--preflight' in sys.argv else train()
    except BaseException as error:save(ROOT/'training_failure.json',dict(error=repr(error),unix=time.time()));raise
