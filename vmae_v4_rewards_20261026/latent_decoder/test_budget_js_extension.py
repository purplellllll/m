import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from .budget_js_extension import latent_probe, run_trial, summarize
from .dual_space_ablation import CountedScorer
from .latent_optimization import design_probabilities
from .model import collate_records


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.randn(256, 64) * .03)

    def decode(self, z, tokens):
        return z @ self.weight + F.one_hot(tokens, 64).float()


class TinyScorer(torch.nn.Module):
    def forward(self, p, tokens):
        return (p * torch.arange(64)).sum(-1).mean(-1) / 64


class BudgetJSTests(unittest.TestCase):
    def test_trial_budget_js_probe_and_reproducibility(self):
        torch.manual_seed(7)
        model = TinyModel()
        tokens = collate_records([dict(cds_dna='ATG'+'GCT'*12+'TAA')])
        mu = torch.zeros(1, tokens.shape[1], 256)
        sigma = torch.ones_like(mu)
        reference = design_probabilities(model.decode(mu, tokens), tokens)
        config = dict(lr=.1, arm_lr_multipliers={'B5': .25}, direct_lr_factor=1.6014172141870504,
            latent_scale=.5, direct_scale=1.2517843842506409,
            regularization=dict(latent_l2=.1, direct_l2=.1, decoder_kl=.1))
        task = dict(sequence_id='test', seed=7, budget_percent=20, js_limit=.05)
        high = torch.arange(256) < 21
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                scorer = CountedScorer(TinyScorer())
                baseline = float(scorer(F.one_hot(tokens, 64).float(), tokens))
                run = Path(directory)/str(index)
                run.mkdir()
                result = run_trial(model, scorer, tokens, mu, sigma, reference, config, task,
                                   baseline, run, high, probe=False)
                self.assertEqual(scorer.forward_sequences, 161)
                self.assertLessEqual(result['best']['changed_codons'], 2)
                self.assertLessEqual(result['best']['soft_js'], .05+1e-7)
                self.assertGreaterEqual(result['filter_gap'], 0)
                outputs.append(result)
            self.assertEqual(outputs[0]['best'], outputs[1]['best'])
            self.assertEqual(outputs[0]['initial_latent_sha256'], outputs[1]['initial_latent_sha256'])
            state = torch.load(Path(directory)/'0/best_state.pt', weights_only=True)
            probe = latent_probe(model, scorer, tokens, mu, sigma, reference, state['latent'], state['direct'],
                                 state['initial_latent'], config, 2, .05, high)
            self.assertEqual(scorer.forward_sequences, 167)
            self.assertEqual(len(probe['interventions']), 5)
            self.assertTrue(all(r['changed_from_original'] <= 2 for r in probe['interventions']))
            self.assertLessEqual(probe['displacement_spectral_participation'], 256)

    def test_summary_requires_complete_cds_and_separates_archives(self):
        baseline = [dict(sequence_id='a', seed=s, budget_percent=b, arm=arm, best={'hard_score': score})
                    for s in (1, 2, 3) for b in (5, 10, 20) for arm, score in [('B5', 2.), ('RiboDecode', 1.)]]
        rows = [dict(sequence_id='a', seed=s, budget_percent=b, js_limit=.05,
                     best={'hard_score': 3.}, best_budget_only={'hard_score': 4.})
                for s in (1, 2, 3) for b in (5, 10, 20)]
        summary = summarize(rows, baseline)
        self.assertEqual(len(summary['comparisons']), 6)
        self.assertEqual(summary['comparisons'][0]['js_minus_b5']['mean'], 1.)
        self.assertEqual(summary['comparisons'][1]['js_minus_b5']['mean'], 2.)
        self.assertIsNone(summary['comparisons'][0]['js_minus_b5']['exploratory_ci95'])
        self.assertEqual(summarize(rows[:-1], baseline)['comparisons'][-1]['budget_percent'], 10)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
