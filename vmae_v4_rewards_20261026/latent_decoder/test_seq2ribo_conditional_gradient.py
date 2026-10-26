import unittest
import torch
from torch import nn
from torch.nn import functional as F

from .seq2ribo_conditional_gradient import conditional_value


class TinyTE(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Module();self.base.cod_emb=nn.Embedding(65,3)
        with torch.no_grad():self.base.cod_emb.weight.normal_(0,.1)
    def forward(self,codons,simulation,mask):
        x=self.base.cod_emb(codons).sum(-1)*mask
        scaled=torch.sigmoid(x.mean(1)+simulation.mean(1))
        return x,scaled


class ConditionalChecks(unittest.TestCase):
    def test_embedding_permutation_forward_gradient_and_cleanup(self):
        torch.manual_seed(31);model=TinyTE().eval().requires_grad_(False)
        permutation=torch.arange(64).flip(0);source=torch.tensor([[2,7,11]])
        args=(permutation[source],torch.zeros(1,3),torch.ones(1,3,dtype=torch.bool))
        p=F.one_hot(source,64).float().requires_grad_();table=model.base.cod_emb.weight[permutation]
        transform=dict(lo=-2.,hi=3.,eps=1e-6)
        value=conditional_value(model,args,{},p,table,transform)
        expected=-2+5*model(*args)[1]
        self.assertTrue(torch.equal(value,expected))
        gradient,=torch.autograd.grad(value.sum(),p)
        direction=torch.zeros_like(p);direction[0,1,7]=-1;direction[0,1,9]=1
        shifted=conditional_value(model,args,{},p.detach()+.001*direction,table,transform)
        finite_difference=float((shifted-value.detach())/.001)
        self.assertAlmostEqual(finite_difference,float((gradient*direction).sum()),delta=.001)
        self.assertEqual(len(model.base.cod_emb._forward_hooks),0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_hook_removed_when_model_raises(self):
        model=TinyTE()
        def fail(*args,**kwargs):raise RuntimeError('synthetic failure')
        model.forward=fail
        args=(torch.zeros(1,2,dtype=torch.long),torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool))
        with self.assertRaises(RuntimeError):
            conditional_value(model,args,{},torch.zeros(1,2,64),model.base.cod_emb.weight[:64],dict(lo=0,hi=1,eps=1e-6))
        self.assertEqual(len(model.base.cod_emb._forward_hooks),0)


if __name__=='__main__':unittest.main()
