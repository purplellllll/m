import unittest

import torch

from .latent_correlated_projection import load_projected, project_state
from .latent_functional_validation import functional_validation
from .latent_recovery_gpu import BASE
from .model import CodonVMAE


class FunctionalValidationTest(unittest.TestCase):
    def test_actual_diagonal_and_correlated_models_use_same_fixed_targets(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        config = dict(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1, ff_dim=32,
            latent_dim=8, dropout=0., max_codons=96, max_positions=96)
        original = CodonVMAE(**config).eval()
        original.recipe = BASE
        basis = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64))[0]
        state = project_state(dict(config=dict(model=config), model=original.state_dict(), recipe=BASE),
            dict(basis=basis.tolist(), mean=[0.]*8))
        compact = load_projected(state, 'cpu')
        dna = 'ATG'+'CTTCTACTGCTA'*12+'TAA'
        record = dict(sequence_id='fixed_target_toy', cds_dna=dna, n_codons=len(dna)//3)
        for model in (original, compact):
            result = functional_validation(model, [record], torch.ones(64), repeats=1)
            recodings = [r for r in result['rows'] if 'repeat' in r]
            self.assertEqual(len(recodings), 2)
            self.assertTrue(all(r['same_complete_protein'] and r['hidden_targets_unchanged'] for r in recodings))
            self.assertEqual(len({r['target_count'] for r in result['rows']}), 1)
            eigen = result['geometry']['short']['hidden']['signal_noise']['generalized_eigenvalues']
            self.assertEqual(len(eigen), model.latent_dim)
            self.assertTrue(torch.isfinite(torch.tensor(eigen)).all())


if __name__ == '__main__':
    unittest.main()
