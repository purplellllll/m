"""Check graph-only pretraining gradients and masked-target isolation."""
import json
import torch
from .pretrain_profile_model import PretrainProfileModel,masked_reconstruction_loss
from .diagnostic_model import native_profile_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(310)
    cod=torch.tensor([[1,2,3,4],[3,2,1,64]]);valid=cod!=64
    p=torch.rand(2,4,2);s=torch.rand(2,4,5);edges=torch.tensor([[0,4],[3,6]])
    mask=torch.tensor([[0,1,0,0],[0,1,0,0]],dtype=torch.bool)
    corrupted=p.clone();corrupted[mask]=.5
    model=PretrainProfileModel(width=32).eval()
    q,a,reconstruction=model.forward_with_reconstruction(cod,valid,p,s,edges,graph_physics=corrupted)
    y=torch.rand(2,4);observed=valid.clone();observed[0,0]=False;y[0,0]=float('nan')
    rec=masked_reconstruction_loss(reconstruction,p,mask)
    changed=p.clone();changed[~mask]=float('nan')
    torch.testing.assert_close(rec,masked_reconstruction_loss(reconstruction,changed,mask),rtol=0,atol=0)
    (native_profile_loss(a['gnn'],y,observed)+.1*rec).backward()
    assert all(x.grad is None for x in model.context.parameters())
    assert all(x.grad is None for x in model.fusion.parameters())
    assert model.reconstruction_head.weight.grad.abs().sum()>0
    assert model.graph[0].self_proj.weight.grad.abs().sum()>0
    assert model.physics[0].weight.grad.abs().sum()>0
    assert reconstruction[~valid].abs().sum()==0
    assert model._reconstruction_hidden is None and not model._capture_reconstruction
    with torch.no_grad():
        x=(cod,valid,p,s,edges)
        normal=model(*x)[0]
        torch.testing.assert_close(model.forward_with_reconstruction(*x)[0],normal,rtol=0,atol=0)
        torch.testing.assert_close(model(*x,graph_physics=corrupted)[0],q,rtol=0,atol=0)
        # The late physical slot is disabled, so true vs corrupted physical inputs
        # cannot bypass the graph-only reconstruction input.
        q2=model(cod,valid,p*2,s,edges,graph_physics=corrupted)[0]
        torch.testing.assert_close(q2,q,rtol=0,atol=0)
    print(json.dumps(dict(status='passed',checks=['graph/physics/reconstruction gradients','no LSTM/fusion pretraining gradients',
        'masked targets only','padding','no true-physics late bypass','cache cleanup'])))


if __name__=='__main__':main()
