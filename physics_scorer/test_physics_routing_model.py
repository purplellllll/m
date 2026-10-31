"""Check route isolation, initial equivalence and gradients on small padded graphs."""
import json
import torch
from .profile_model import AlignedProfileModel
from .diagnostic_model import native_profile_loss
from .physics_routing_model import PhysicsRoutingProfileModel


def main():
    torch.set_num_threads(1);torch.manual_seed(305)
    cod=torch.tensor([[1,2,3,4],[3,2,1,64]]);valid=cod!=64
    physics=torch.rand(2,4,2);structure=torch.rand(2,4,5);edges=torch.tensor([[0,4],[3,6]])
    base=AlignedProfileModel(encoder='bilstm').eval()
    state=base.state_dict();model=PhysicsRoutingProfileModel(width=32,graph_physics='gated').eval()
    missing=model.load_state_dict(state,strict=False)
    assert missing.missing_keys==['physical_gate'] and not missing.unexpected_keys
    x=(cod,valid,physics,structure,edges)
    with torch.no_grad():torch.testing.assert_close(model(*x)[0],base(*x)[0],rtol=0,atol=0)
    model.physical_gate.data.fill_(.5)
    q,aux=model(*x);qoff,aoff=model(*x,disable_graph_physics=True)
    assert (q-qoff).abs().max()>1e-6 and (aux['gnn']-aoff['gnn']).abs().max()>1e-6
    torch.testing.assert_close(aux['physics'],aoff['physics'],atol=0,rtol=0)
    torch.testing.assert_close(aux['context'],aoff['context'],atol=0,rtol=0)
    independent=physics.clone().requires_grad_()
    prediction,_=model(*x,graph_physics=independent)
    gradient=torch.autograd.grad(prediction.sum(),independent)[0]
    assert gradient[valid].abs().sum()>0 and gradient[~valid].abs().sum()==0
    y=torch.rand(2,4);observed=valid.clone();observed[0,1]=False;y[0,1]=float('nan')
    native_profile_loss(q,y,observed).backward()
    assert model.physical_gate.grad is not None and model.physical_gate.grad.abs()>0
    with torch.no_grad():
        for i,n in enumerate([4,3]):
            e=edges[:,edges[0]//4==i]-i*4
            single=model(cod[i:i+1,:n],valid[i:i+1,:n],physics[i:i+1,:n],structure[i:i+1,:n],e)[0]
            torch.testing.assert_close(single[0],q[i,:n],rtol=3e-6,atol=3e-6)
    model.late_physics=False;model.graph_physics='none'
    with torch.no_grad():torch.testing.assert_close(model(*x)[0],model(cod,valid,physics*3,structure,edges)[0],atol=0,rtol=0)
    assert not model._active and model._physical is None and model._override is None
    print(json.dumps(dict(status='passed',checks=['zero gate equals baseline','graph-only intervention leaves other branches fixed',
        'graph input gradient','padding and sample isolation','late mask independent from physical auxiliary','no retained cache'])))


if __name__=='__main__':main()
