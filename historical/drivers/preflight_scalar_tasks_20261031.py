"""Exercise the real424-width scalar transfer model on measured training records."""
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_scalar_tasks_20261031 import load,make_model,batch


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.20);torch.manual_seed(31)
    _,rows,features=load('protein');train=[r for r in rows if r['split']=='train'][:8]
    model,_=make_model('protein',30101,1);model=model.cuda().train()
    x,g,y,_=batch(train,features,'cuda');p=model(x,g)
    assert p.shape==y.shape==(8,1) and torch.isfinite(p).all()
    loss=(p-y).square().mean();loss.backward()
    gradients={}
    for name in ['physics','graph','context']:
        gradients[name]=sum(float(q.grad.abs().sum()) for q in getattr(model.backbone,name).parameters() if q.grad is not None)
        assert gradients[name]>0
    model.eval()
    with torch.no_grad():expected=model(x,g)
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    model.load_state_dict(torch.load(stream,map_location='cpu',weights_only=True))
    with torch.no_grad():actual=model(x,g)
    torch.testing.assert_close(expected,actual,rtol=1e-6,atol=1e-6)
    result=dict(status='passed',actual_train_batch=8,full_cds_codons=226,profile_backbone_width=424,
        scalar_shape=[8,1],gradient_l1_by_branch=gradients,checkpoint_roundtrip_max_error=float((expected-actual).abs().max()),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),test_records_evaluated=0)
    (ROOT/'scalar_gpu_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
