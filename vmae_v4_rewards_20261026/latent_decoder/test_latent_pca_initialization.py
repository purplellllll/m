import unittest

import torch

from .latent_pca_initialization import compress_pca
from .latent_recovery_gpu import BASE
from .model import CodonVMAE


class PCAInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        config=dict(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,ff_dim=32,
            latent_dim=8,dropout=0.,max_codons=64,max_positions=64)
        self.model=CodonVMAE(**config)
        with torch.no_grad():
            self.model.logvar.weight.zero_();self.model.logvar.bias.fill_(2*torch.log(torch.tensor(.7)))
        self.state=dict(model=self.model.state_dict(),config=dict(model=config),recipe=BASE)
        self.h=torch.randn(80,16,dtype=torch.float64)
        self.mu=torch.nn.functional.linear(self.h,self.model.mu.weight.double(),self.model.mu.bias.double())
        self.lv=torch.full_like(self.mu,float(self.model.logvar.bias[0].detach()))
        self.weights=torch.ones(80,dtype=torch.float64)

    def test_full_dimension_is_exact(self):
        result,report=compress_pca(self.state,self.h,self.mu,self.lv,self.weights,8)
        self.assertTrue(report['full_capacity_exact'])
        for name,value in self.state['model'].items():torch.testing.assert_close(result['model'][name],value,rtol=0,atol=0)

    def test_head_projection_and_decoder_compensation_match_the_same_subspace(self):
        result,report=compress_pca(self.state,self.h,self.mu,self.lv,self.weights,3)
        basis=torch.tensor(report['basis'],dtype=torch.float64);center=torch.tensor(report['mean'],dtype=torch.float64)
        new=result['model']
        actual=torch.nn.functional.linear(self.h,new['mu.weight'].double(),new['mu.bias'].double())
        torch.testing.assert_close(actual,(self.mu-center)@basis,atol=1e-6,rtol=1e-5)
        decoded=torch.nn.functional.linear(actual,new['from_latent.weight'].double(),new['from_latent.bias'].double())
        expected=torch.nn.functional.linear(center+actual@basis.T,self.model.from_latent.weight.double(),self.model.from_latent.bias.double())
        torch.testing.assert_close(decoded,expected,atol=1e-6,rtol=1e-5)
        model=CodonVMAE(**result['config']['model']);model.load_state_dict(new,strict=True)

    def test_isotropic_teacher_marginals_remain_exact_with_unused_hidden_features(self):
        result,report=compress_pca(self.state,self.h,self.mu,self.lv,self.weights,4)
        predicted=torch.nn.functional.linear(self.h,result['model']['logvar.weight'].double(),result['model']['logvar.bias'].double())
        torch.testing.assert_close(predicted,torch.full_like(predicted,float(self.model.logvar.bias[0].detach())),atol=1e-6,rtol=1e-6)
        self.assertLess(report['training_logvar_rmse'],1e-8)


if __name__=='__main__':unittest.main()
