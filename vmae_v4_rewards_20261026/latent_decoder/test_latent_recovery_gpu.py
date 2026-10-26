import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .latent_recovery_gpu import (BASE,RECIPES,RecoveryVMAE,batch_mask,calibrate_skip,initialize,
    eligible_for_selection,geometry,reconstruction_objective,training_batches)
from .model import CodonVMAE,collate_records,mutable_positions,vae_loss
from .paper_guided_latent_cpu import vicreg

CONFIG=dict(d_model=16,n_heads=2,encoder_layers=2,decoder_layers=1,ff_dim=32,
            latent_dim=8,dropout=0.,max_codons=64,max_positions=64)


class RecoveryTests(unittest.TestCase):
    def test_gain_preserves_initial_function(self):
        torch.manual_seed(9)
        baseline=CodonVMAE(**CONFIG).eval()
        tokens=collate_records([dict(cds_dna='ATG'+'GCTGCC'*8+'TAA')])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'base.pt'
            torch.save(dict(model=baseline.state_dict(),config=dict(model=CONFIG)),checkpoint)
            for gain in (.5,2.):
                model,_=initialize(checkpoint,dict(BASE,gain=gain),device='cpu')
                model.eval()
                torch.testing.assert_close(model(tokens,tokens,sample=False)[0],
                                           baseline(tokens,tokens,sample=False)[0])

    def test_zero_initialized_skip_preserves_function_and_learns(self):
        torch.manual_seed(17)
        baseline=CodonVMAE(**CONFIG).eval()
        rows=[dict(sequence_id=f'skip-{i}',n_codons=34,cds_dna='ATG'+'GCTGCC'*16+'TAA') for i in range(2)]
        tokens=collate_records(rows)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'base.pt'
            torch.save(dict(model=baseline.state_dict(),config=dict(model=CONFIG)),checkpoint)
            model,_=initialize(checkpoint,dict(BASE,skip_layer=1),device='cpu')
            model.eval()
            with patch('latent_decoder.latent_recovery_gpu.pick',return_value=rows):
                calibrate_skip(model,rows)
            self.assertGreater(float(model.skip_scale),0.)
            torch.testing.assert_close(model(tokens,tokens,sample=False)[0],baseline(tokens,tokens,sample=False)[0])
            logits,_,_=model(tokens,tokens,sample=False)
            logits.square().mean().backward()
            self.assertTrue(torch.isfinite(model.latent_skip.weight.grad).all())
            self.assertGreater(float(model.latent_skip.weight.grad.abs().sum()),0.)

    def test_fixed_sigma_has_zero_variance_kl(self):
        model=RecoveryVMAE(RECIPES['sigma1'],**CONFIG).eval()
        tokens=collate_records([dict(cds_dna='ATGGCTGCCTAA')])
        mu,lv,mid=model.encode_features(tokens,tokens)
        self.assertTrue(torch.equal(lv,torch.zeros_like(lv)))
        torch.testing.assert_close(.5*(mu.square()+lv.exp()-1-lv),.5*mu.square())
        self.assertEqual(mid.shape[-1],16)

    def test_length_batches_pair_orders_and_masks(self):
        rows=[dict(sequence_id=f'{g}-{i}',n_codons=n,cds_dna='ATG'+'GCT'*(n-2)+'TAA')
              for g,n in [('short',90),('medium',420),('long',710)] for i in range(6)]
        a=training_batches(rows,173,27)
        b=training_batches(list(reversed(rows)),173,27)
        self.assertEqual([(g,[r['sequence_id'] for r in rws]) for g,rws in a],
                         [(g,[r['sequence_id'] for r in rws]) for g,rws in b])
        self.assertEqual([g for g,_ in a[:3]],['short','medium','long'])
        x=batch_mask([rows[0],rows[6]],175,'cpu')
        y=batch_mask(rows[:1],175,'cpu')
        torch.testing.assert_close(x[1][0,:90],y[1][0])
        self.assertTrue((x[1][0,90:]==64).all())
        self.assertTrue((x[2]<=mutable_positions(x[0])).all())

    def test_clean_geometry_and_posterior_noise(self):
        config=dict(CONFIG,max_codons=1024,max_positions=1024)
        model=RecoveryVMAE(RECIPES['sigma1'],**config).eval()
        rows=[dict(sequence_id=f'geometry-{n}',v2_group=n,n_codons=n,cds_dna='ATG'+'GCT'*(n-2)+'TAA') for n in (90,420,710)]
        result,_=geometry(model,rows,limit=1)
        for group in ('short','medium','long'):
            self.assertIn('within_aa_geometry',result[group]['clean'])
            self.assertLess(result[group]['masked_only']['tokens'],result[group]['tokens'])
            torch.testing.assert_close(torch.tensor(result[group]['posterior_noise']['mean_variance_by_dim']),torch.ones(8))

    def test_encoder_retains_input_guards(self):
        model=RecoveryVMAE(BASE,**CONFIG).eval()
        tokens=collate_records([dict(cds_dna='ATGGCTTAA')])
        with self.assertRaises(ValueError):model.encode(tokens[:,:-1],tokens)
        with self.assertRaises(ValueError):model.encode(tokens,tokens.fill_(65))

    def test_deterministic_ablation_matches_fixed_eval_objective(self):
        recipe=RECIPES['deterministic_sigma1']
        model=RecoveryVMAE(recipe,**CONFIG).eval()
        rows=[dict(sequence_id='deterministic',n_codons=34,cds_dna='ATG'+'GCTGCC'*16+'TAA')]
        tokens,inputs,selected=batch_mask(rows,173,'cpu')
        loss,_,_,_=reconstruction_objective(model,tokens,inputs,selected,recipe,7,False)
        other,_,_,_=reconstruction_objective(model,tokens,inputs,selected,recipe,777,False)
        logits,mu,lv=model(inputs,tokens,sample=False)
        expected,_=vae_loss(logits,tokens,mu,lv,selected,beta=.001)
        torch.testing.assert_close(loss,expected)
        torch.testing.assert_close(loss,other)

    def test_noise_warmup_returns_to_original_sampling(self):
        model=RecoveryVMAE(BASE,**CONFIG).eval()
        rows=[dict(sequence_id='noise-schedule',n_codons=34,cds_dna='ATG'+'GCTGCC'*16+'TAA')]
        tokens,inputs,selected=batch_mask(rows,173,'cpu')
        baseline=reconstruction_objective(model,tokens,inputs,selected,BASE,7,False)[0]
        endpoint=reconstruction_objective(model,tokens,inputs,selected,RECIPES['learned_noise'],7,False,1.)[0]
        start=reconstruction_objective(model,tokens,inputs,selected,RECIPES['learned_noise'],7,False,0.)[0]
        torch.testing.assert_close(baseline,endpoint)
        self.assertGreater(float((start-endpoint).detach().abs()),1e-6)

    def test_noise_warmup_cannot_select_low_noise_shortcut(self):
        recipe=RECIPES['learned_noise']
        self.assertTrue(eligible_for_selection(recipe,0,1152))
        self.assertFalse(eligible_for_selection(recipe,576,1152))
        self.assertFalse(eligible_for_selection(recipe,720,1152))
        self.assertTrue(eligible_for_selection(recipe,864,1152))
        self.assertTrue(eligible_for_selection(BASE,144,1152))

    def check_all_gradients(self,device,amp):
        torch.manual_seed(5)
        recipe=dict(RECIPES['sigma1_mid_mu_mean'],skip_layer=1)
        model=RecoveryVMAE(recipe,**CONFIG).to(device).eval().requires_grad_(True)
        rows=[dict(sequence_id=f'gradient-{i}',n_codons=34,cds_dna='ATG'+'GCTGCC'*16+'TAA') for i in range(2)]
        tokens,inputs,selected=batch_mask(rows,173,device)
        loss,_,mu,mid=reconstruction_objective(model,tokens,inputs,selected,recipe,7,amp)
        _,other_inputs,other_selected=batch_mask(rows,174,device)
        other_mu,_,other_mid=model.encode_features(other_inputs,tokens)
        common=mutable_positions(tokens)&~selected&~other_selected
        loss=loss+.001*vicreg(mu[common],other_mu[common],tokens[common])
        loss=loss+.0001*vicreg(mid.float()[common],other_mid[common],tokens[common])
        loss.backward()
        for parameter in (model.encoder.layers[0].attention.qkv.weight,model.mu.weight,
                          model.from_latent.weight,model.decoder.layers[0].attention.qkv.weight,model.output.weight,
                          model.latent_skip.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_full_network_gradient_cpu(self):
        self.check_all_gradients('cpu',False)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_bfloat16_full_network_gradient_cuda(self):
        self.check_all_gradients('cuda',True)


if __name__=='__main__':
    unittest.main()
