import unittest
import torch
from torch.nn import functional as F
from .mlm_objective_ablation import objective
from .model import synonymous_logits,vae_loss
from .tokenizer import TOKEN,PAD


class VisibleAblationChecks(unittest.TestCase):
    def test_exact_original_and_visible_gradient_removal(self):
        torch.manual_seed(31)
        tokens=torch.tensor([[TOKEN['ATG'],TOKEN['GCT'],TOKEN['GCC'],TOKEN['TAA'],PAD]])
        selected=torch.tensor([[False,True,False,False,False]])
        logits=torch.randn(1,5,64,requires_grad=True);mu=torch.randn(1,5,3,requires_grad=True);lv=torch.zeros_like(mu)
        original,_=vae_loss(logits,tokens,mu,lv,selected,.001)
        control,_=objective(logits,tokens,mu,lv,selected,.001,1.)
        self.assertTrue(torch.equal(original,control))
        ablated,_=objective(logits,tokens,mu,lv,selected,.001,0.)
        ce=F.cross_entropy(synonymous_logits(logits,tokens).transpose(1,2),tokens.clamp_max(63),reduction='none')
        kl=.5*mu[:,:4].square().sum(-1).mean()
        self.assertTrue(torch.allclose(ablated,1.25*ce[0,1]+.001*kl))
        g_original,=torch.autograd.grad(original,logits,retain_graph=True)
        g_ablated,=torch.autograd.grad(ablated,logits)
        self.assertTrue(torch.allclose(g_original[0,1],g_ablated[0,1]))
        self.assertGreater(float(g_original[0,2].abs().sum()),0)
        self.assertEqual(float(g_ablated[0,2].abs().sum()),0)

    def test_no_mask_and_padding(self):
        tokens=torch.tensor([[TOKEN['ATG'],TOKEN['TAA'],PAD]])
        selected=torch.zeros_like(tokens,dtype=torch.bool)
        logits=torch.zeros(1,3,64);mu=torch.zeros(1,3,2);lv=torch.zeros_like(mu)
        value,_=objective(logits,tokens,mu,lv,selected,.001,0.)
        self.assertEqual(float(value),0.)
        with self.assertRaises(ValueError):objective(logits,tokens,mu,lv,selected,.001,.5)


if __name__=='__main__':unittest.main()
