"""Check padding, path interventions, graph isolation and attention reach."""
import json
import torch
from .graph_polish_model import GraphPolishModel, MODES
from .diagnostic_model import native_profile_loss


def main():
    torch.set_num_threads(1)
    cod = torch.tensor([[1,2,3,4,5,6],[3,2,1,4,64,64]])
    valid = cod != 64
    torch.manual_seed(817)
    p = torch.rand(2,6,2); s = torch.rand(2,6,5)
    edges = torch.tensor([[0,6],[5,9]])
    x = (cod, valid, p, s, edges)
    y = torch.rand(2,6)
    states = None; counts = set()
    for mode in MODES:
        torch.manual_seed(817)
        model = GraphPolishModel(mode, width=32).eval()
        current = model.state_dict()
        if states is None:
            states = {k:v.clone() for k,v in current.items()}
        else:
            assert all(torch.equal(states[k],v) for k,v in current.items())
        counts.add(sum(v.numel() for v in model.parameters()))
        q, aux = model(*x)
        assert torch.isfinite(q).all() and q[~valid].abs().sum() == 0
        native_profile_loss(q,y,valid).backward()
        assert model.polish_input[0].weight.grad.abs().sum() > 0
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in model.context.parameters())
        assert not model._polish_cache and model._polish_valid is None
        with torch.no_grad():
            for j,n in enumerate([6,4]):
                pairs = edges[:,edges[0]//6 == j]-6*j
                single = model(cod[j:j+1,:n],valid[j:j+1,:n],p[j:j+1,:n],s[j:j+1,:n],pairs)[0]
                torch.testing.assert_close(single[0],q[j,:n],atol=4e-6,rtol=4e-6)
            zero, changed = model(*x, physics_override='zero')
            for head in ['physics','gnn']:
                torch.testing.assert_close(changed[head],aux[head],atol=0,rtol=0)
            if model.early_physics:
                assert (zero-q).abs().max() > 1e-7
            else:
                torch.testing.assert_close(zero,q,atol=1e-6,rtol=1e-6)
        roundtrip = GraphPolishModel(mode,width=32).eval()
        roundtrip.load_state_dict(model.state_dict())
        torch.testing.assert_close(roundtrip(*x)[0],q,atol=0,rtol=0)
    assert len(counts) == 1
    attention = model.context.eval()
    z = torch.randn(1,6,32); zz=z.clone(); zz[:,0] += torch.randn(32)
    full_change=(attention.forward(z)[:,5]-attention.forward(zz)[:,5]).abs().max()
    mask=~torch.eye(6,dtype=torch.bool)
    local_change=(attention.forward(z,mask=mask)[:,5]-attention.forward(zz,mask=mask)[:,5]).abs().max()
    assert full_change>1e-6 and local_change<1e-7
    try:
        model(cod,valid,p,s,torch.tensor([[0],[6]]))
        raise AssertionError('Cross-sample edge accepted')
    except ValueError:
        assert not model._polish_cache
    print(json.dumps(dict(status='passed',modes=len(MODES),equal_initialization=True,
        checks=['padding/batch isolation','early-physics path intervention','nonzero gradients',
                'weight reload','global versus diagonal attention','cross-sample edge rejection','cache cleanup'])))


if __name__=='__main__':
    main()
