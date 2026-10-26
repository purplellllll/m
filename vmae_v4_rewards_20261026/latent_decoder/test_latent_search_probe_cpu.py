import tempfile
import unittest
from pathlib import Path

import torch

from .dual_space_ablation import CountedScorer
from .latent_optimization import state_hash
from .latent_optimization import design_probabilities
from .latent_search_probe_cpu import expanded_radius, latent_delta, project, search, validate_native
from .model import MASK, CodonVMAE, collate_records, mutable_positions


class ToyScore(torch.nn.Module):
    def forward(self, probabilities, tokens):
        weights=torch.linspace(0.,1.,64,device=probabilities.device)
        return (probabilities*weights).sum((1,2))/tokens.shape[1]


class ProbeChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_actual_cuda_search_with_cpu_mask_input(self):
        torch.manual_seed(17)
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=24,latent_dim=8,dropout=0.,max_codons=32,max_positions=32).cuda().eval().requires_grad_(False)
        scorer=CountedScorer(ToyScore()).cuda()
        row=dict(sequence_id='cuda',n_codons=6,cds_dna='ATGGCTGCCGCGGCATAA')
        tokens=collate_records([row]); inputs=tokens.masked_fill(mutable_positions(tokens),MASK)
        frozen=state_hash(model)
        with tempfile.TemporaryDirectory() as tmp:
            result=search(model,scorer,row,2,torch.tensor([True]*4+[False]*4),Path(tmp),
                          arms=('latent_radius8',),encoder_tokens=inputs)
        self.assertEqual(result[0]['search_scorer_queries'],6)
        self.assertEqual(result[0]['encoder_masked_positions'],4)
        self.assertEqual(frozen,state_hash(model))

    def test_temperature_preserves_hard_codons_and_reduces_saturation(self):
        tokens=collate_records([dict(cds_dna='ATGGCTGCCTAA')])
        logits=torch.zeros(1,4,64,requires_grad=True)
        with torch.no_grad():logits.scatter_(-1,tokens[...,None],14.)
        cold=design_probabilities(logits,tokens)
        warm=design_probabilities(logits/5,tokens)
        self.assertTrue(torch.equal(cold.argmax(-1),warm.argmax(-1)))
        self.assertTrue(torch.equal(warm[:,0],cold[:,0]))
        self.assertTrue(torch.equal(warm[:,-1],cold[:,-1]))
        cold_gradient,=torch.autograd.grad(cold[0,1,tokens[0,1]],logits,retain_graph=True)
        warm_gradient,=torch.autograd.grad(warm[0,1,tokens[0,1]],logits)
        self.assertGreater(float(warm_gradient.norm()),float(cold_gradient.norm()))

    def test_native_stop_change_is_reported_but_protein_change_is_rejected(self):
        valid=validate_native('ATGGCTTAA','ATGGCTTGA')
        self.assertTrue(valid['protein_preserved'])
        self.assertFalse(valid['stop_preserved'])
        with self.assertRaises(AssertionError):validate_native('ATGGCTTAA','ATGTTTTGA')

    def test_expansion_needs_stalled_hard_and_improving_soft(self):
        self.assertEqual(expanded_radius(.25,5,.1),.5)
        self.assertEqual(expanded_radius(.25,4,.1),.25)
        self.assertEqual(expanded_radius(.25,5,0.),.25)
        self.assertEqual(expanded_radius(8.,5,.1),8.)

    def test_radius_is_independent_of_sequence_duplication(self):
        original=torch.arange(12,dtype=torch.float32).reshape(1,3,4)
        first,second=original.clone(),original.repeat(1,2,1)
        project(first,torch.tensor([[True,True,False]]),.25)
        project(second,torch.tensor([[True,True,False]]).repeat(1,2),.25)
        self.assertTrue(torch.equal(first,second[:,:3]))
        self.assertEqual(float(first[:,2].abs().sum()),0.)
        self.assertAlmostEqual(float(first[:,:2].square().mean().sqrt()),.25,places=6)

    def test_actual_search_preserves_weights_and_counts_calls(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=24,latent_dim=8,dropout=0.,max_codons=32,max_positions=32).eval().requires_grad_(False)
        scorer=CountedScorer(ToyScore())
        frozen=state_hash(model)
        row=dict(sequence_id='synthetic',n_codons=6,cds_dna='ATGGCTGCCGCGGCATAA')
        with tempfile.TemporaryDirectory() as tmp:
            results=search(model,scorer,row,2,torch.tensor([True]*4+[False]*4),Path(tmp))
        self.assertEqual(len(results),5)
        self.assertEqual([r['search_scorer_queries'] for r in results],[6]*5)
        self.assertTrue(all(r['best']['hard_score']>=r['original_score'] for r in results))
        self.assertTrue(all(r['best']['hard_score']==max(c['hard_score'] for c in r['hard_candidates']) for r in results))
        self.assertEqual(frozen,state_hash(model))

    def test_masked_encoder_initialization_and_archive(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=24,latent_dim=8,dropout=0.,max_codons=32,max_positions=32).eval().requires_grad_(False)
        frozen=state_hash(model)
        row=dict(sequence_id='masked',n_codons=6,cds_dna='ATGGCTGCCGCGGCATAA')
        tokens=collate_records([row]);active=mutable_positions(tokens)
        inputs=tokens.masked_fill(active,MASK)
        with tempfile.TemporaryDirectory() as tmp:
            results=search(model,CountedScorer(ToyScore()),row,2,torch.tensor([True]*4+[False]*4),
                Path(tmp),arms=('latent_radius8',),encoder_tokens=inputs)
            invalid=inputs.clone();invalid[:,0]=MASK
            with self.assertRaises(AssertionError):
                search(model,CountedScorer(ToyScore()),row,2,torch.tensor([True]*8),Path(tmp),encoder_tokens=invalid)
        r=results[0]
        self.assertEqual(r['encoder_masked_positions'],int(active.sum()))
        self.assertEqual(r['best']['hard_score'],max(c['hard_score'] for c in r['hard_candidates']))
        self.assertEqual(len({c['cds_dna'] for c in r['hard_candidates']}),len(r['hard_candidates']))
        self.assertEqual(frozen,state_hash(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_basis_search_keeps_the_radius_in_original_latent_coordinates(self):
        torch.set_num_threads(1);torch.manual_seed(18)
        basis=torch.randn(8,2)*4
        variable=torch.ones(1,3,2)
        active=torch.tensor([[True,True,False]])
        project(variable,active,.25,basis)
        self.assertAlmostEqual(float(latent_delta(variable,basis)[active].square().mean().sqrt()),.25,places=6)
        self.assertEqual(float(variable[:,2].abs().sum()),0.)
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=24,latent_dim=8,dropout=0.,max_codons=32,max_positions=32).eval().requires_grad_(False)
        frozen=state_hash(model)
        row=dict(sequence_id='basis',n_codons=6,cds_dna='ATGGCTGCCGCGGCATAA')
        with tempfile.TemporaryDirectory() as tmp:
            results=search(model,CountedScorer(ToyScore()),row,2,torch.tensor([True]*4+[False]*4),
                Path(tmp),arms=('latent_radius8',),latent_basis=basis)
        self.assertEqual(results[0]['search_scorer_queries'],6)
        self.assertEqual(results[0]['search_coordinate_dimensions'],2)
        self.assertEqual(len(results[0]['best_displacement']['energy_by_dim']),8)
        self.assertEqual(frozen,state_hash(model))


if __name__=='__main__':unittest.main()
