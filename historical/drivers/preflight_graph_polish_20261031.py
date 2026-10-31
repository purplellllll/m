"""Real batch8 CUDA checks before any formal model training."""
import json
from pathlib import Path
import torch
from physics_scorer.graph_polish_model import GraphPolishModel, MODES
from train_graph_polish_20261031 import data, collate, native_profile_loss, save

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows,fs=data(ROOT/'data');x,y,mask=collate(fs[:8]);x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
    results={};shared=None
    for arm in MODES:
        torch.manual_seed(30101);model=GraphPolishModel(arm).cuda()
        current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if shared is None:shared=current
        else:assert all(torch.equal(shared[k],v) for k,v in current.items())
        model.train();q,aux=model(*x)
        loss=native_profile_loss(q,y,mask)+.1*torch.stack([native_profile_loss(v,y,mask) for v in aux.values()]).mean()
        assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        assert model.polish_input[0].weight.grad.abs().sum()>0
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);optimizer.step();model.eval()
        with torch.no_grad():
            output,_=model(*x);assert torch.isfinite(output).all() and output[~x[1]].abs().sum()==0
        results[arm]=dict(parameters=sum(v.numel() for v in model.parameters()),finite_loss=float(loss.detach()))
        del optimizer,model,q,aux,output;torch.cuda.empty_cache()
    assert len({r['parameters'] for r in results.values()})==1
    save(ROOT/'gpu_preflight.json',dict(status='passed',batch_size=8,arms=results,
        identical_initialization=True,torch_version=torch.__version__,peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2))


if __name__=='__main__':main()
