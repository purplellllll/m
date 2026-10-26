"""Small controls for the CPU diagnosis, independent of large checkpoints."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .latent_collapse_cpu import Pool, gradient_probe, interventions, replay_heads, trace_cache
from .model import CodonVMAE, collate_records
from .latent_rank_cpu import measure_rank


class CollapseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=32,latent_dim=8,dropout=0.,max_codons=900,max_positions=900).eval().requires_grad_(False)
        self.records=[dict(sequence_id=f'{panel}_{g}',panel=panel,n_codons=n+2,cds_dna='ATG'+'GCT'*n+'TAA')
                      for panel in ('train','val') for g,n in [('short',31),('medium',310),('long',700)]]

    def test_covariance_correlation_and_pad(self):
        tokens=collate_records([dict(cds_dna='ATGGCTGCCTAA'),dict(cds_dna='ATGGCTTAA')])
        x=torch.randn(*tokens.shape,3)
        x[tokens==64]=1e6
        x[...,2]=1.
        first,second=Pool(3,True),Pool(3,True)
        first.add(x,tokens);second.add(x*torch.tensor([10.,.1,2.]),tokens)
        a,b=first.summary(),second.summary()
        self.assertEqual(a['tokens'],7)
        self.assertAlmostEqual(a['correlation_participation'],b['correlation_participation'],places=5)
        self.assertEqual(a['pooled']['variance_by_dim'][2],0.)
        self.assertGreaterEqual(a['covariance_dimensions_95pct'],1)

    def test_hooks_gradients_interventions_preserve_frozen_model(self):
        state={k:v.clone() for k,v in self.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            cache,pools=trace_cache(self.model,self.records[:1],Path(directory))
            self.assertIn('train/short/clean/encoder_norm',pools)
            item=cache[1]
            probe=gradient_probe(self.model,item)
            self.assertEqual(len(probe['contraction_by_dim']['kl']),8)
            result=interventions(self.model,item,item['mu'][0].var(0,unbiased=False)>.01)
            self.assertLess(result['rescale2_projection_max_error'],2e-6)
            self.assertTrue(all(torch.equal(v,state[k]) for k,v in self.model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_decay_control_has_expected_variance_and_replay_restores_weights(self):
        initial={k:v.clone() for k,v in self.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cache,_=trace_cache(self.model,self.records,root)
            settings=dict(beta=0.,decay=.01,reconstruction=False,sample=False)
            with patch('latent_decoder.latent_collapse_cpu.ARMS',dict(decay_only=settings)):
                result=replay_heads(self.model,cache,root,3)['decay_only']
            start=result['trajectory'][0]['latent']['val/short']['pooled']['variance_by_dim']
            end=result['trajectory'][-1]['latent']['val/short']['pooled']['variance_by_dim']
            for a,b in zip(start,end):
                self.assertAlmostEqual(b,a*(1-1e-6)**6,delta=abs(a)*2e-6+1e-10)
            self.assertTrue(all(torch.equal(v,initial[k]) for k,v in self.model.state_dict().items()))

    def test_layer_covariance_supplement_counts_channels_and_preserves_model(self):
        initial={k:v.clone() for k,v in self.model.state_dict().items()}
        result=measure_rank(self.model,self.records[:1])
        before=result['train/short/clean/encoder_norm']
        after=result['train/short/clean/mu']
        self.assertEqual((before['channels'],after['channels']),(16,8))
        self.assertTrue(0 < before['within_aa_geometry']['covariance_participation'] <= 16+1e-5)
        self.assertTrue(all(torch.equal(v,initial[k]) for k,v in self.model.state_dict().items()))


if __name__=='__main__':
    unittest.main()
