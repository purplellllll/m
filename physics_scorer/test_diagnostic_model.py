"""Check unchanged routing controls, changed information paths and gradients."""
import itertools
import json
import torch
from .profile_model import AlignedProfileModel,masked_transcript_mse
from .diagnostic_model import RoutingProfileModel,native_profile_loss


def main():
    torch.set_num_threads(1)
    cod=torch.tensor([[14,2,4,48],[14,3,48,64]]);valid=cod!=64
    phy=torch.rand(2,4,2);struct=torch.rand(2,4,5);edges=torch.tensor([[0,4],[3,6]])
    y=torch.rand(2,4);mask=valid.clone();mask[0,1]=False;y[0,1]=float('nan')
    torch.manual_seed(31);base=AlignedProfileModel(encoder='bilstm').eval()
    torch.manual_seed(31);control=RoutingProfileModel(encoder='bilstm').eval()
    assert set(base.state_dict())==set(control.state_dict())
    with torch.no_grad():
        reference=base(cod,valid,phy,struct,edges)[0]
        torch.testing.assert_close(reference,control(cod,valid,phy,struct,edges)[0],atol=0,rtol=0)
    for pg,gc in itertools.product([False,True],repeat=2):
        model=RoutingProfileModel(encoder='bilstm',physics_into_graph=pg,graph_into_context=gc).eval()
        model.load_state_dict(base.state_dict());pred,aux=model(cod,valid,phy,struct,edges)
        if pg or gc:assert (pred-reference).abs().max()>1e-6
        loss=native_profile_loss(pred,y,mask)+.1*torch.stack([native_profile_loss(a,y,mask) for a in aux.values()]).mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model._captured_graph is None and model._captured_physics is None
        with torch.no_grad():
            for i,n in enumerate([4,3]):
                e=edges[:,edges[0]//4==i]-i*4
                single=model(cod[i:i+1,:n],valid[i:i+1,:n],phy[i:i+1,:n],struct[i:i+1,:n],e)[0]
                torch.testing.assert_close(single[0],pred[i,:n],atol=3e-6,rtol=3e-6)
    assert torch.isfinite(masked_transcript_mse(reference,y,mask))
    print(json.dumps(dict(status='passed',route_combinations=4,unchanged_control_exact=True,
                         checks=['branch gradients','native loss with missing labels','batch isolation','no retained graph cache'])))


if __name__=='__main__':main()
