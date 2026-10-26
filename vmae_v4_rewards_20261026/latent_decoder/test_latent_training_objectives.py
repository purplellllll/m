import unittest

import torch

from .latent_recovery_gpu import BASE, RecoveryVMAE, batch_mask, reconstruction_objective as frozen
from .latent_training_objectives import reconstruction_objective
from .model import PAD


class TrainingObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        self.model=RecoveryVMAE(BASE,d_model=16,n_heads=2,encoder_layers=2,decoder_layers=1,
            ff_dim=32,latent_dim=8,dropout=0.,max_codons=64,max_positions=64).eval()
        rows=[dict(sequence_id='objective',n_codons=34,cds_dna='ATG'+'GCTGCC'*16+'TAA')]
        self.tokens,self.inputs,self.selected=batch_mask(rows,175,'cpu')

    def test_default_objective_matches_frozen_formula_and_gradient(self):
        for weight in (0.,.5):
            recipe=dict(BASE,mean_weight=weight,training_beta=.001)
            new=reconstruction_objective(self.model,self.tokens,self.inputs,self.selected,recipe,7,False)[0]
            old=frozen(self.model,self.tokens,self.inputs,self.selected,recipe,7,False)[0]
            torch.testing.assert_close(new,old)
            a=torch.autograd.grad(new,self.model.logvar.weight,retain_graph=True)[0]
            b=torch.autograd.grad(old,self.model.logvar.weight)[0]
            torch.testing.assert_close(a,b)

    def test_beta_changes_differentiable_kl_without_changing_samples(self):
        low=reconstruction_objective(self.model,self.tokens,self.inputs,self.selected,dict(BASE,training_beta=0.),9,False)[0]
        high=reconstruction_objective(self.model,self.tokens,self.inputs,self.selected,dict(BASE,training_beta=.003),9,False)[0]
        mu,lv,_=self.model.encode_features(self.inputs,self.tokens)
        expected=.003*.5*(mu.square()+lv.exp()-1-lv)[self.tokens<PAD].mean(0).sum()
        torch.testing.assert_close(high-low,expected,atol=1e-6,rtol=1e-5)
        actual_grad=torch.autograd.grad(high-low,self.model.mu.weight,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected,self.model.mu.weight)[0]
        torch.testing.assert_close(actual_grad,expected_grad)
        self.assertGreater(float(actual_grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
