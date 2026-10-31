"""Exercise longest-input CUDA, saved metric formula and dynamic-batch boundaries."""
import json
from pathlib import Path
import numpy as np
import torch
from train_expanded_riboseq_20261031 import ARMS,model_for,batches,pearson,RECIPE,save
from physics_scorer.diagnostic_model import native_profile_loss

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.65)
    assert [s for s in range(1,301) if s%RECIPE['validate_every']==0]==[100,200,300]
    rng=np.random.default_rng(91);a=rng.normal(size=100);b=rng.normal(size=100)
    assert abs(pearson(a,b)-np.corrcoef(a,b)[0,1])<1e-12
    fs=[dict(cod=np.zeros(n)) for n in [40,80,250,512,1024,1500,1998]]
    groups=batches(list(range(len(fs))),fs,1)
    for group in groups:
        n=max(len(fs[i]['cod']) for i in group)
        assert len(group)<=8 and len(group)*n<=4096 and len(group)*n*n<=4500000
    results={};n=1998
    cod=torch.randint(0,64,(1,n),device='cuda');valid=torch.ones(1,n,dtype=torch.bool,device='cuda')
    p=torch.rand(1,n,2,device='cuda');s=torch.rand(1,n,5,device='cuda');edges=torch.tensor([[0,100],[n-1,n-100]],device='cuda')
    y=torch.rand(1,n,device='cuda')
    for arm,setting in ARMS.items():
        torch.manual_seed(30101);model=model_for(setting).cuda();model.train()
        q,aux=model(cod,valid,p,s,edges)
        loss=native_profile_loss(q,y,valid)+.1*torch.stack([native_profile_loss(v,y,valid) for v in aux.values()]).mean()
        assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        results[arm]=dict(parameters=sum(v.numel() for v in model.parameters()),finite_loss=float(loss.detach()))
        del model,q,aux;torch.cuda.empty_cache()
    save(ROOT/'pipeline_preflight.json',dict(status='passed',source='synthetic numeric tensors; no biological metric',
        longest_codons=n,arms=results,pearson_formula_verified=True,validation_schedule=[100,200,300],
        peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2))
    print(json.dumps(dict(status='passed',arms=len(results),longest_codons=n)))


if __name__=='__main__':main()
