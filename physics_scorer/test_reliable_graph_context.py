"""Check gates retain scale, sequence direction, padding and direct context supervision."""
import torch
from .reliable_graph_context import ReliableGraphContext, ReliableGraphLayer
from .diagnostic_model import native_profile_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(91)
    cod=torch.tensor([[1,2,3,4,5,6,7,8],[2,3,1,4,5,64,64,64]])
    valid=cod!=64;p=torch.rand(2,8,2);s=torch.rand(2,8,5);edges=torch.tensor([[0,8],[6,12]])
    model=ReliableGraphContext(width=32).eval();q,aux=model(cod,valid,p,s,edges)
    torch.testing.assert_close(q,aux['context'],atol=0,rtol=0)
    native_profile_loss(q,torch.rand_like(q),valid).backward()
    assert model.context_head.weight.grad.abs().sum()>0
    assert model.graph[0].edge_gate[-1].weight.grad.abs().sum()>0
    for j,n in enumerate([8,5]):
        e=edges[:,edges[0]//8==j]-j*8
        with torch.no_grad():single=model(cod[j:j+1,:n],valid[j:j+1,:n],p[j:j+1,:n],s[j:j+1,:n],e)[0]
        torch.testing.assert_close(single[0],q[j,:n],atol=5e-6,rtol=5e-6)
    assert q[~valid].abs().sum()==0
    shared=None
    for sequence,gate,anchor in [(False,False,False),(True,False,False),(True,True,False),(True,True,True)]:
        torch.manual_seed(30101)
        control=ReliableGraphContext(width=32,sequence_v2=sequence,structure_gate=gate,anchored_head=anchor)
        current=control.state_dict()
        if shared is None:shared={k:v.clone() for k,v in current.items()}
        else:assert all(torch.equal(shared[k],v) for k,v in current.items())
    layer=ReliableGraphLayer(32).eval();x=torch.randn(1,8,32);vv=torch.ones(1,8,dtype=torch.bool);ss=torch.rand(1,8,5)
    ee=torch.tensor([[0,7],[7,0]])
    with torch.no_grad():
        layer.edge_gate[-1].weight.zero_();layer.edge_gate[-1].bias.fill_(-30)
        gated=layer(x,vv,ss,ee);none=layer(x,vv,ss,ee,True)
        torch.testing.assert_close(gated,none,atol=1e-6,rtol=1e-6)
        layer.edge_gate[-1].bias.fill_(30)
        assert (layer(x,vv,ss,ee)-none).abs().max()>1e-4
    try:model(cod,valid,p,s,torch.tensor([[0],[8]]));raise AssertionError('Cross-sample edges accepted')
    except ValueError:pass
    print('passed: direct-context head, gate gradients, gate scale, padding/batch isolation, invalid-edge rejection')


if __name__=='__main__':main()
