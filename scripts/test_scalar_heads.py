"""Check masked scalar pooling, true scalar gradients and required covariates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from physics_scorer.graph_polish_model import GraphPolishModel
from physics_scorer.scalar_downstream import TaskScalarModel,ScalarHeads,masked_pool


def main():
    torch.set_num_threads(1);torch.manual_seed(31)
    valid=torch.tensor([[True,True,True,False],[True,True,False,False]])
    cod=torch.tensor([[1,2,3,64],[2,3,64,64]])
    x=(cod,valid,torch.rand(2,4,2),torch.rand(2,4,5),torch.tensor([[0,1],[2,0]]))
    model=TaskScalarModel(GraphPolishModel(width=32),'te',4).eval()
    global_features=torch.rand(2,6)
    result=model(x,global_features);assert result.shape==(2,4)
    changed=list(x);changed[2]=x[2].masked_fill(~valid[...,None],float('nan'))
    changed[3]=x[3].masked_fill(~valid[...,None],float('nan'))
    torch.testing.assert_close(result,model(tuple(changed),global_features),atol=1e-6,rtol=1e-6)
    # Scalar supervision must reach all three representation paths.
    result.square().mean().backward()
    for branch in [model.backbone.physics,model.backbone.graph,model.backbone.context]:
        assert sum(p.grad.abs().sum().item() for p in branch.parameters() if p.grad is not None)>0
    p=torch.rand(2,3,64);g=torch.rand(2,6)
    rpf=ScalarHeads(32,'rpf',environment_dim=7)
    try:rpf(p,g)
    except ValueError:pass
    else:raise AssertionError('Missing RNA/context silently accepted')
    assert rpf(p,g,log_rna=torch.ones(2,1),environment=torch.ones(2,7)).shape==(2,1)
    utr=ScalarHeads(32,'te',4,use_utr=True).eval()
    ids=torch.tensor([[0,1,4],[4,4,4]]);mask=ids!=4
    assert torch.isfinite(utr(p,g,utrs=[(ids,mask),(ids,mask)])).all()
    a=torch.randn(2,4,5);base=masked_pool(a,valid);a[~valid]=10000
    torch.testing.assert_close(base,masked_pool(a,valid))
    print('PASS: scalar shapes, padding invariance, branch gradients, RPF covariates and empty UTRs')


if __name__=='__main__':main()
