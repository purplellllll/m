import unittest

import torch

from .latent_posterior_audit import audit, posterior_noise
from .latent_recovery_gpu import BASE, RecoveryVMAE


class PosteriorAuditTests(unittest.TestCase):
    def test_common_noise_is_deterministic_without_changing_global_rng(self):
        torch.manual_seed(5)
        before = torch.get_rng_state().clone()
        first = posterior_noise('cds', 7, (1, 20, 8))
        torch.testing.assert_close(first, posterior_noise('cds', 7, first.shape))
        torch.testing.assert_close(before, torch.get_rng_state())
        self.assertFalse(torch.equal(first, posterior_noise('cds', 8, first.shape)))

    def test_sampling_changes_reconstruction_but_not_kl(self):
        model = RecoveryVMAE(BASE, d_model=16, n_heads=2, encoder_layers=2,
            decoder_layers=1, ff_dim=32, latent_dim=8, dropout=0., max_codons=64,
            max_positions=64).eval()
        record = dict(sequence_id='posterior-test', cds_dna='ATG'+'GCTGCC'*16+'TAA', n_codons=34)
        result = audit(model, [record], draws=(7, 8))
        self.assertEqual(len(result['rows']), 3)
        self.assertEqual(result['posterior_minus_mean']['kl'], 0.)
        self.assertNotEqual(result['posterior_minus_mean']['ce'], 0.)
        for row in result['rows']:
            self.assertAlmostEqual(row['fixed_loss'], row['ce']+row['full_ce']+.001*row['kl'], places=6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
