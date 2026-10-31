"""Check task heads and sim-feature alignment on training samples only."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'seq2ribo_source'))
from train_scalar_tasks_20261031 import load,batch,make_model,global_features
from physics_scorer.seq2ribo_scalar_reference import Seq2RiboScalarReference


def main():
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12);torch.manual_seed(31)
    from seq2ribo import Seq2Ribo
    _,rows,features=load('protein');rows=[r for r in rows if r['split']=='train'][:2]
    x,g,_,_=batch(rows,features,'cuda');results=[]
    for cell in ['ipsc','hek293','lcl','rpe']:
        p=Seq2Ribo(cell_line=cell,device='cuda',weights_dir=str(ROOT/'seq2ribo_source/weights'),cache_dir=str(ROOT/'baseline_geometry'))
        base=p._load_model('riboseq')
        for utr in [False,True]:
            model=Seq2RiboScalarReference(base,utr,cell!='hek293',dict(target_low=-3.,target_high=3.,target_mean=0.,target_scale=1.)).cuda().train()
            u5=torch.tensor([[0,1,2,3],[0,1,4,4]],device='cuda');m5=u5!=4
            u3=torch.tensor([[0,3,4],[1,2,3]],device='cuda');m3=u3!=4
            pred=model(x,g,utrs=[(u5,m5),(u3,m3)] if utr else None)
            assert pred.shape==(2,1) and torch.isfinite(pred).all()
            pred.square().mean().backward()
            gradients=sum(float(q.grad.abs().sum()) for q in model.backbone.parameters() if q.grad is not None)
            assert gradients>0;results.append(dict(cell=cell,use_utr=utr,shape=[2,1],backbone_gradient_l1=gradients))
        del p,base,model;torch.cuda.empty_cache()
    # Reconstructing raw simulation mean must preserve all valid codon values.
    reconstructed=(torch.expm1(x[2][...,0])*torch.expm1(8*g[:,3:4])).cpu().numpy()
    for j,r in enumerate(rows):np.testing.assert_allclose(reconstructed[j,:len(features[r['uid']]['cod'])],features[r['uid']]['simulation_mean'],rtol=1e-6,atol=1e-6)
    result=dict(status='passed',conditions=results,raw_sim_mean_reconstruction=True,tests_evaluated=0,
        note='Task heads are newly initialized; pretrained profile weights alone loaded')
    (ROOT/'scalar_reference_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
