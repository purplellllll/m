"""Exercise every intended arm on real aligned CUDA batch8 inputs."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from train_fusion_encoders_20261031 import ARMS,data,collate,native_profile_loss
from physics_scorer.fusion_encoder_model import FusionEncoderModel


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows,fs=data(ROOT/'data');x,y,mask=collate(fs[:8]);x=tuple(v.cuda() for v in x);y=y.cuda();mask=mask.cuda()
    results={};shared=None
    for arm,setting in ARMS.items():
        torch.manual_seed(30101);model=FusionEncoderModel(**setting).cuda()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('codons.','physics.','structure.','graph.','fusion.'))}
        if shared is None:shared=state
        else:assert all(torch.equal(shared[k],v) for k,v in state.items()),'Shared initialization differs'
        model.train();q,a=model(*x)
        loss=native_profile_loss(q,y,mask)+.1*torch.stack([native_profile_loss(v,y,mask) for v in a.values()]).mean()
        assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if setting.get('learned_gains'):assert model.gain_logits.grad is not None and model.gain_logits.grad.abs().sum()>0
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.context.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.step();model.eval()
        with torch.no_grad():
            p,_=model(*x);assert torch.isfinite(p).all() and p[~x[1]].abs().sum()==0
        results[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),ff=model.context_ff,finite_training_loss=float(loss.detach()))
        del optimizer,model,q,a,p;torch.cuda.empty_cache()
    import transformers,RNA
    result=dict(status='passed',batch_size=8,arms=results,shared_branch_initialization_identical=True,
        peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2,torch_version=torch.__version__,transformers_version=transformers.__version__,viennarna_version=RNA.__version__)
    (ROOT/'gpu_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
