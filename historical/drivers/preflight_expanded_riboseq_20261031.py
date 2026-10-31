"""Check dynamic batches and real CUDA gradients for every architecture before formal runs."""
import json
from pathlib import Path
import torch
from train_expanded_riboseq_20261031 import ARMS,model_for,load_features,batches,collate,native_profile_loss,save

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.65)
    rows,fs=load_features(ROOT/'data',{'train','validation'})
    indices=[i for i,r in enumerate(rows) if r['split']=='train']
    groups=batches(indices,fs,30101);longest=max(groups,key=lambda g:max(len(fs[i]['cod']) for i in g))
    assert len(longest)<=8
    result={}
    for arm,setting in ARMS.items():
        torch.manual_seed(30101);model=model_for(setting).cuda()
        x,y,mask=collate([fs[i] for i in longest]);x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
        model.train();q,a=model(*x)
        loss=native_profile_loss(q,y,mask)+.1*torch.stack([native_profile_loss(v,y,mask) for v in a.values()]).mean()
        assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        opt=torch.optim.AdamW(model.parameters(),lr=.0003);opt.step()
        result[arm]=dict(parameters=sum(v.numel() for v in model.parameters()),loss=float(loss.detach()))
        del opt,model,q,a;torch.cuda.empty_cache()
    save(ROOT/'gpu_preflight.json',dict(status='passed',arms=result,max_length=max(len(fs[i]['cod']) for i in longest),
        long_batch_size=len(longest),peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2))


if __name__=='__main__':main()
