import unittest
import torch
from .dual_space_ablation import (center_direct, feasible, masks, mean_square, phase,
                                  project_joint, regularizers)
from .latent_optimization import design_probabilities
from .model import CodonVMAE, collate_records


class DualSpaceChecks(unittest.TestCase):
    def test_joint_radius_gauge_and_phase_budget(self):
        tokens = collate_records([dict(cds_dna='ATGGCTGCCTAA'), dict(cds_dna='ATGGCGTGA')])
        active, support = masks(tokens)
        latent = torch.randn(*tokens.shape, 4)
        direct = torch.randn(*tokens.shape, 64)
        project_joint(latent, direct, active, support)
        energy = mean_square(latent, active[..., None]) + mean_square(direct, support)
        self.assertLessEqual(float(energy.max()), 1.000001)
        self.assertEqual(float(direct[~support].abs().sum()), 0.)
        self.assertTrue(torch.allclose(direct.sum(-1), torch.zeros_like(tokens, dtype=torch.float), atol=1e-6))
        self.assertEqual([phase('B4', s, 80) for s in (1, 40, 41, 80)], ['latent', 'latent', 'direct', 'direct'])
        project_joint(latent, direct, active, support, component_radius=2**-.5)
        frozen = latent.clone()
        direct.mul_(100)
        project_joint(latent, direct, active, support, component_radius=2**-.5, frozen_latent=True)
        self.assertTrue(torch.equal(frozen, latent))
        self.assertLessEqual(float(mean_square(direct, support).max()), .500001)

    def test_regularization_and_both_gradient_paths(self):
        torch.set_num_threads(2)
        tokens = collate_records([dict(cds_dna='ATGGCTGCCTAA')])
        active, support = masks(tokens)
        model = CodonVMAE(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                          ff_dim=32, latent_dim=4, dropout=0, max_codons=16).eval().requires_grad_(False)
        with torch.no_grad(): mu, _ = model.encode(tokens, tokens)
        z = torch.zeros_like(mu, requires_grad=True)
        d = torch.zeros(*tokens.shape, 64, requires_grad=True)
        logits = model.decode(mu + z * active[..., None], tokens)
        decoder = design_probabilities(logits, tokens)
        p = design_probabilities(logits + d * support, tokens)
        terms = regularizers(p, decoder, z, d, active, support)
        self.assertTrue(all(abs(float(v.detach())) < 1e-6 for v in terms.values()))
        (p * torch.randn_like(p)).sum().backward()
        self.assertGreater(float(z.grad.abs().sum()), 0.)
        self.assertGreater(float(d.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(float(d.grad[~support].abs().sum()), 0.)
        config = dict(max_mutable_hamming=.2, max_gc_change=.05, max_js=.05)
        row = dict(mutable_hamming=.21, gc_fraction=.5, js_from_initial=0.)
        self.assertFalse(feasible(row, dict(gc_fraction=.5), config))


if __name__ == '__main__':
    unittest.main()
