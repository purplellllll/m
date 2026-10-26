import unittest

import torch

from .latent_capacity_gpu import compress_state
from .model import CodonVMAE,collate_records


class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.config=dict(d_model=16,n_heads=2,encoder_layers=2,decoder_layers=1,ff_dim=32,
            latent_dim=8,dropout=0.,max_codons=64,max_positions=64)
        self.teacher=CodonVMAE(**self.config).eval()
        self.state=dict(config=dict(model=self.config),model=self.teacher.state_dict())
        self.tokens=collate_records([dict(cds_dna='ATG'+'GCTGCC'*16+'TAA')])
        self.scores=torch.tensor([1.,7.,3.,5.,2.,8.,4.,6.])

    def test_full_capacity_preserves_initial_function(self):
        state,indices=compress_state(self.state,8,self.scores,torch.randn(8))
        model=CodonVMAE(**state['config']['model']).eval();model.load_state_dict(state['model'])
        torch.testing.assert_close(model(self.tokens,self.tokens,sample=False)[0],
                                   self.teacher(self.tokens,self.tokens,sample=False)[0])
        self.assertEqual(indices.tolist(),list(range(8)))

    def test_smaller_model_keeps_exact_selected_posteriors(self):
        mean=torch.randn(8)
        original=self.teacher.from_latent.bias.detach().clone()
        state,indices=compress_state(self.state,4,self.scores,mean)
        model=CodonVMAE(**state['config']['model']).eval();model.load_state_dict(state['model'],strict=True)
        mu,lv=self.teacher.encode(self.tokens,self.tokens)
        actual_mu,actual_lv=model.encode(self.tokens,self.tokens)
        torch.testing.assert_close(actual_mu,mu[...,indices])
        torch.testing.assert_close(actual_lv,lv[...,indices])
        removed=torch.ones(8,dtype=torch.bool);removed[indices]=False
        residual=self.teacher.from_latent.weight[:,removed]@(mu[...,removed]-mean[removed])[0].T
        torch.testing.assert_close((self.teacher.from_latent(mu)-model.from_latent(actual_mu))[0].T,residual)
        torch.testing.assert_close(self.teacher.from_latent.bias,original)
        self.assertEqual(self.state['config']['model']['latent_dim'],8)

    def test_invalid_dimensions_rejected(self):
        for dimensions in (0,9):
            with self.assertRaises(AssertionError):compress_state(self.state,dimensions,self.scores,torch.zeros(8))


if __name__=='__main__':
    unittest.main()
