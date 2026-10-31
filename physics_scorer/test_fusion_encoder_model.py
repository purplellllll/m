"""Check fusion interventions, context families and masked padded sequences."""
import json
import torch
from .fusion_encoder_model import FusionEncoderModel
from .profile_model import AlignedProfileModel
from .diagnostic_model import native_profile_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(317)
    cod=torch.tensor([[1,2,3,4],[3,2,1,64]]);valid=cod!=64
    p=torch.rand(2,4,2);s=torch.rand(2,4,5);edges=torch.tensor([[0,4],[3,6]]);x=(cod,valid,p,s,edges)
    y=torch.rand(2,4);observed=valid.clone();observed[0,1]=False;y[0,1]=float('nan')
    torch.manual_seed(317);base=AlignedProfileModel(encoder='bilstm',width=32).eval()
    torch.manual_seed(317);control=FusionEncoderModel(width=32).eval()
    with torch.no_grad():torch.testing.assert_close(base(*x)[0],control(*x)[0],atol=0,rtol=0)
    for family in ['bilstm','bilstm_ln','transformer','bert','t5']:
        model=FusionEncoderModel(encoder=family,width=32,learned_gains=True).eval()
        q,aux=model(*x);changed,other=model(*x,gain_override=[2,1,1])
        assert (q-changed).abs().max()>1e-7
        for key in aux:torch.testing.assert_close(aux[key],other[key],atol=0,rtol=0)
        native_profile_loss(q,y,observed).backward()
        assert model.gain_logits.grad is not None and model.gain_logits.grad.abs().sum()>0
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in model.context.parameters())
        assert q[~valid].abs().sum()==0 and not model._active and model._codon_cache is None
        with torch.no_grad():
            for i,n in enumerate([4,3]):
                e=edges[:,edges[0]//4==i]-i*4
                single=model(cod[i:i+1,:n],valid[i:i+1,:n],p[i:i+1,:n],s[i:i+1,:n],e)[0]
                torch.testing.assert_close(single[0],q[i,:n],atol=5e-6,rtol=5e-6)
    print(json.dumps(dict(status='passed',families=5,checks=['unit gains equal baseline','gain intervention preserves branch heads',
        'learned gains and encoder gradients','padding and batch isolation','cache cleanup'])))


if __name__=='__main__':main()
