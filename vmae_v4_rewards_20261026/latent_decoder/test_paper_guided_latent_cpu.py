import math
import unittest

import torch

from .model import CodonVMAE, collate_records
from .paper_guided_latent_cpu import beta_for, cache_prefix, centered_aa, decode, encode_tail, posterior, vicreg
from .tokenizer import TOKEN


class PaperPilotTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_cache_and_tail_backward(self):
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=2,decoder_layers=1,
                        ff_dim=32,latent_dim=4,dropout=0.,max_codons=64,max_positions=64).cuda().eval()
        row=dict(cds_dna='ATG'+'GCTGCC'*15+'TAA',sequence_id='cuda-test',panel='train')
        item=cache_prefix(model,[row])[0]
        self.assertEqual(item['tokens'].device.type,'cuda')
        mu,lv=encode_tail(model,item['views'][0]['h'],item['tokens'],'control',None)
        z=mu+torch.randn(mu.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(7))*(.5*lv).exp()
        decode(model,z,item['tokens'],'control',None).square().mean().backward()
        self.assertTrue(torch.isfinite(model.mu.weight.grad).all())

    def test_prefix_tail_equals_original_encoder(self):
        torch.manual_seed(7)
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=2,decoder_layers=1,
                        ff_dim=32,latent_dim=4,dropout=0.,max_codons=16,max_positions=16).eval()
        tokens=collate_records([dict(cds_dna='ATGGCTGCCGCGTAA')])
        h=model.embedding(tokens)+model.aa_embedding(model.aa_ids[tokens])
        h=model.encoder.layers[0](h,tokens==64)
        actual=encode_tail(model,h,tokens,'control',None)
        expected=model.encode(tokens,tokens)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b)
        torch.testing.assert_close(decode(model,actual[0],tokens,'control',None),model.decode(actual[0],tokens))

    def test_fixed_variance_cannot_prove_information(self):
        mu=torch.zeros(2,3,256)
        _,lv=posterior(mu,mu,'fixed_sigma_half',None)
        kl=.5*(mu.square()+lv.exp()-1-lv)
        self.assertGreater(float(kl.min()),.3)
        self.assertEqual(int((mu.flatten(0,1).var(0)>.01).sum()),0)
        self.assertAlmostEqual(float(lv[0,0,0]),math.log(.25),places=6)

    def test_annealing_same_final_beta_and_different_speed(self):
        for arm in ('anneal_fast','anneal_slow'):
            self.assertAlmostEqual(beta_for(arm,0,24),0.)
            self.assertAlmostEqual(beta_for(arm,24,24),.001)
            self.assertLess(beta_for(arm,1,24),.001)
        self.assertGreater(beta_for('anneal_fast',6,24),beta_for('anneal_slow',6,24))

    def test_aa_residualization_and_vicreg_gradients(self):
        tokens=torch.tensor([TOKEN['GCT'],TOKEN['GCC'],TOKEN['GAA'],TOKEN['GAG']]*8)
        x=torch.randn(32,4,requires_grad=True)
        aa_shift=torch.tensor([10.,10.,-20.,-20.]*8)[:,None]
        torch.testing.assert_close(centered_aa(x+aa_shift,tokens),centered_aa(x,tokens),atol=2e-6,rtol=2e-5)
        loss=vicreg(.05*x,.05*x+.01*torch.randn_like(x),tokens)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_inactive_latents_cannot_affect_decoder(self):
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
                        ff_dim=32,latent_dim=4,dropout=0.,max_codons=8,max_positions=8).eval()
        tokens=collate_records([dict(cds_dna='ATGGCTTAA')])
        active=torch.tensor([1.,0.,1.,0.])
        z=torch.randn(1,3,4)
        altered=z+100*(1-active)
        torch.testing.assert_close(decode(model,z,tokens,'active64',active),
                                   decode(model,altered,tokens,'active64',active))


if __name__=='__main__':
    unittest.main()
