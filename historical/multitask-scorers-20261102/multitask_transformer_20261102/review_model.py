"""Independent CPU boundary checks; does not edit or train the model."""
from pathlib import Path
import json
import torch
import model

def main():
    model.selfcheck();torch.set_num_threads(1);torch.manual_seed(9)
    b,n=2,20
    x=dict(codons=torch.randint(0,64,(b,n)),amino=torch.randint(0,20,(b,n)),valid=torch.ones(b,n,dtype=torch.bool),
        physics=torch.randn(b,n,12),physics_present=torch.ones(b),physical_global=torch.randn(b,36),physics_calibrated=torch.zeros(b),
        utr=torch.full((b,2,48),4,dtype=torch.long),utr_present=torch.zeros(b,2),stats=torch.randn(b,7),host=torch.tensor([0,1]))
    net=model.MultiTaskTransformer(7,width=32,max_tokens=16).eval()
    with torch.no_grad():
        a=net(x);assert torch.isfinite(a).all()
        xx={k:v.clone() for k,v in x.items()}
        for key,value in [('codons',64),('amino',21),('valid',False)]:
            xx[key]=torch.nn.functional.pad(xx[key],(0,10),value=value)
        xx['physics']=torch.nn.functional.pad(xx['physics'],(0,0,0,10),value=123)
        aa=net(xx);torch.testing.assert_close(a,aa,rtol=0,atol=1e-7)
    t=torch.randn(2,4,requires_grad=True);y=torch.randn(2,4)
    mask=torch.tensor([[1,0,0,0],[0,0,1,0]],dtype=torch.bool)
    loss=model.masked_loss(t,y,mask);loss.backward();assert (t.grad[~mask]==0).all()
    result=dict(status='passed',checks=['original_masked_label_gradient_check','missing_UTR_forward_finite',
        'long_CDS_compression_forward','right_padding_CDS_invariance','unobserved_label_gradient_exact_zero'],
        padding_max_error=float((a-aa).abs().max()),model_modified=False)
    (Path(__file__).resolve().parent/'model_independent_review.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result)
if __name__=='__main__':main()
