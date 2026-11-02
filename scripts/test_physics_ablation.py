"""Synthetic regression checks for every scalar physics entrance; no data or optimizer."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physics_scorer.reliable_graph_context import ReliableGraphContext
from physics_scorer.scalar_downstream import TaskScalarModel, ScalarHeads
from physics_scorer.diagnostic_model import native_profile_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(91)
    cod=torch.tensor([[1,2,3,4,5,6,7,8],[2,3,1,4,5,64,64,64]])
    valid=cod!=64
    x=(cod,valid,torch.rand(2,8,2),torch.rand(2,8,5),torch.tensor([[0,8],[6,12]]))
    changed=(*x[:2],x[2]+7,*x[3:])
    g=torch.randn(2,6);g2=g.clone();g2[:,3:]+=9
    for task in ['te','protein','rpf']:
        model=TaskScalarModel(ReliableGraphContext(width=32,physics=False),task,environment_dim=7).eval()
        # LayerNorm biases must not resurrect a disabled branch after pooling.
        with torch.no_grad():model.heads.branch_norm[0].bias.fill_(2)
        extras=dict(log_rna=torch.randn(2,1),environment=torch.randn(2,7)) if task=='rpf' else {}
        p,q=model.representations(x);p2,q2=model.representations(changed)
        assert p[:,0].count_nonzero()==0
        torch.testing.assert_close(p,p2,atol=0,rtol=0)
        torch.testing.assert_close(q,q2,atol=0,rtol=0)
        torch.testing.assert_close(model(x,g,**extras),model(changed,g2,**extras),atol=0,rtol=0)
        probe=x[2].clone().requires_grad_();globals_probe=g.clone().requires_grad_()
        model((*x[:2],probe,*x[3:]),globals_probe,**extras).square().mean().backward()
        assert probe.grad is None or probe.grad.count_nonzero()==0
        assert globals_probe.grad[:,3:].count_nonzero()==0
        assert all(t.grad is None or t.grad.count_nonzero()==0 for t in model.backbone.physics.parameters())
        assert model.heads.branch_logits.grad[0]==0
        assert model.heads.branch_norm[0].bias.grad.count_nonzero()==0
        model.zero_grad(set_to_none=True)
        profile,aux=model.backbone(*x)
        assert set(aux)=={'gnn','context'}
        loss=native_profile_loss(profile,torch.rand_like(profile),valid)
        loss+=.1*torch.stack([native_profile_loss(v,torch.rand_like(v),valid) for v in aux.values()]).mean()
        loss.backward()
        assert all(t.grad is None for t in model.backbone.physics_head.parameters())
    enabled=TaskScalarModel(ReliableGraphContext(width=32,physics=True),'te').eval()
    assert (enabled(x,g)-enabled(changed,g)).abs().max()>1e-6
    assert (enabled(x,g)-enabled(x,g2)).abs().max()>1e-6
    # Same shape/state schema; enabled-by-default reproduces the pre-fix scalar formula.
    head=ScalarHeads(32,'te').eval();pooled=torch.rand(2,3,64)
    expected=head.readout(torch.cat([norm(pooled[:,j])*(3*head.branch_logits.softmax(0))[j]
                                   for j,norm in enumerate(head.branch_norm)]+[g],-1))
    torch.testing.assert_close(head(pooled,g),expected,atol=0,rtol=0)
    clone=TaskScalarModel(ReliableGraphContext(width=32),'te')
    clone.load_state_dict(enabled.state_dict(),strict=True)
    cached=head(pooled,g,physics_enabled=False)
    changed_pool=pooled.clone();changed_pool[:,0]+=13
    torch.testing.assert_close(cached,head(changed_pool,g2,physics_enabled=False),atol=0,rtol=0)
    restored=head.state_dict()
    bare_clone=ScalarHeads(32,'te');bare_clone.load_state_dict(restored,strict=True)
    print('PASS: TE/protein/RPF disabled-input invariance, zero gradients, no physics auxiliary loss, enabled sensitivity and state schema')


if __name__=='__main__':main()
