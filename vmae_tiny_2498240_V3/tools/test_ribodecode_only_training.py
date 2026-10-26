"""Small checks for official-only provenance, batching and scratch-model compatibility."""
import unittest

import torch

from latent_decoder.latent_recovery_gpu import BASE, RecoveryVMAE
from latent_decoder.model import CodonVMAE, collate_records, corrupt, vae_loss
from latent_decoder.v2_train import windows
from tools.prepare_ribodecode_only_training import official_matches


class RiboDecodeOnlyChecks(unittest.TestCase):
    def test_provenance_rejects_nonofficial_cds(self):
        row = dict(sequence_id='example', transcript_ids=['TX'], cds_dna='ATGGCTTAA', protein_sequence='MA')
        self.assertEqual(official_matches(row, {'TX': 'ATGGCT'}), ['TX'])
        with self.assertRaises(ValueError):
            official_matches(row, {'TX': 'ATGGCC'})

    def test_windows_cover_all_rows_once(self):
        records = [dict(n_codons=n) for n in (30, 301, 302, 700, 701, 1500)*15]
        batches = windows(records, 20261023)
        self.assertEqual(sorted(i for _, rows in batches for i in rows), list(range(len(records))))

    def test_train_and_standard_decoder_loading(self):
        torch.set_num_threads(1)
        config = dict(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                      ff_dim=32, latent_dim=8, dropout=0., max_codons=64, max_positions=64)
        model = RecoveryVMAE(dict(BASE), **config)
        ordinary = CodonVMAE(**config)
        ordinary.load_state_dict(model.state_dict(), strict=True)
        tokens = collate_records([dict(cds_dna='ATG'+'GCT'*20+'TAA')])
        inputs, selected = corrupt(tokens, .8, torch.Generator().manual_seed(7))
        logits, mu, lv = model(inputs, tokens)
        loss, terms = vae_loss(logits, tokens, mu, lv, selected, beta=.001)
        loss.backward()
        self.assertTrue(torch.isfinite(model.mu.weight.grad).all())
        self.assertAlmostEqual(float(loss.detach()), float(terms['reconstruction_ce']+terms['masked_ce']+.001*terms['kl_per_token']), places=5)
        model.eval(); ordinary.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens, tokens, sample=False)[0], ordinary(tokens, tokens, sample=False)[0]))


if __name__ == '__main__':
    unittest.main()
