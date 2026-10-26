"""Software checks for the frozen-model optimization diagnostic."""
import unittest
import torch
from torch.nn import functional as F

from .latent_optimization import design_probabilities, predictor_input, standardized_rms, project_radius
from .model import AA_IDS, PAD, TOKEN, CodonVMAE, collate_records, mutable_positions


class OptimizationChecks(unittest.TestCase):
    def test_translation_adapter_and_immutable_codons(self):
        tokens=collate_records([dict(cds_dna='ATGGCTTAG'),dict(cds_dna='ATGGCCGCTTAA')])
        logits=torch.randn(*tokens.shape,64,requires_grad=True)
        probs=design_probabilities(logits,tokens)
        hard=F.one_hot(tokens.clamp_max(63),64).float()*(tokens<PAD)[...,None]
        nt=predictor_input(hard,tokens)
        self.assertEqual(nt.shape,(2,4,4500))
        for row,sequence in enumerate(['ATGGCT','ATGGCCGCT']):
            expected=F.one_hot(torch.tensor(['ACGT'.index(c) for c in sequence]),4).float().T
            self.assertTrue(torch.equal(nt[row,:,:len(sequence)],expected))
            self.assertEqual(float(nt[row,:,len(sequence):].abs().sum()),0.)
        self.assertTrue(torch.equal(probs[~mutable_positions(tokens)],hard[~mutable_positions(tokens)]))
        nt=predictor_input(probs,tokens)
        (nt*torch.randn_like(nt)).sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_radius_and_frozen_decoder_gradients(self):
        torch.set_num_threads(2)
        tokens=collate_records([dict(cds_dna='ATGGCTGCCTAA')])
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
                        ff_dim=32,latent_dim=4,dropout=0,max_codons=16).eval().requires_grad_(False)
        with torch.no_grad():
            mu,logvar=model.encode(tokens,tokens)
        variable=torch.randn_like(mu,requires_grad=True)
        active=mutable_positions(tokens)
        project_radius(variable,active,.5)
        self.assertLessEqual(float(standardized_rms(variable.detach(),active).max()),.500001)
        self.assertEqual(float(variable.detach()[~active].abs().sum()),0.)
        p=design_probabilities(model.decode(mu+variable*torch.exp(.5*logvar),tokens),tokens)
        (p*torch.randn_like(p)).sum().backward()
        self.assertGreater(float(variable.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
