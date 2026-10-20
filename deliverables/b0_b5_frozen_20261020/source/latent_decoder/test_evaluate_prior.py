import unittest

import torch

from .evaluate_prior import evaluate_prior, shuffle_latents
from .model import AA_IDS, CodonVMAE, collate_records, mutable_positions
from .train import codon_frequency


class PriorChecks(unittest.TestCase):
    def test_shuffle_preserves_aa_visibility_and_padding(self):
        records = [{'cds_dna': 'ATG'+'GCTGCCTTCTTT'*8+'TAG'}, {'cds_dna': 'ATGGCTTAG'}]
        tokens = collate_records(records)
        selected = mutable_positions(tokens) & (torch.arange(tokens.shape[1]) % 3 == 0)
        mu = torch.arange(tokens.numel()*2).reshape(*tokens.shape, 2).float()
        result = shuffle_latents(mu, tokens, selected, torch.Generator().manual_seed(8))
        self.assertFalse(torch.equal(mu, result))
        self.assertTrue(torch.equal(mu[~mutable_positions(tokens)], result[~mutable_positions(tokens)]))
        for amino_acid in range(20):
            for masked in (False, True):
                at = mutable_positions(tokens) & (AA_IDS[tokens] == amino_acid) & (selected == masked)
                self.assertTrue(torch.equal(mu[at].sort(dim=0).values, result[at].sort(dim=0).values))

    def test_model_without_latent_dependency_has_zero_shuffle_gap(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        records = [{'cds_dna':'ATG'+'GCTGCCTTCTTT'*20+'TAG','n_codons':82}]
        model = CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
                          ff_dim=32,latent_dim=4,dropout=0,max_codons=100)
        with torch.no_grad():
            model.from_latent.weight.zero_()
        result = evaluate_prior(model, records, codon_frequency(records), torch.device('cpu'))
        self.assertGreater(result['masked_sites'], 0)
        self.assertAlmostEqual(result['shuffle_masked_ce_increase'], 0., places=7)
        self.assertEqual(len(result['shuffled_masked_ce_repeats']), 3)


if __name__ == '__main__':
    unittest.main()
