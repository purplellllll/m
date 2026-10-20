import json
import tempfile
import time
import unittest
from pathlib import Path

import torch

from .b6_reproducibility import frozen_config
from .dual_space_ablation import CountedScorer, optimize_batch, phase
from .model import CodonVMAE
from .test_b6_reproducibility import LinearScorer


class ReverseScheduleChecks(unittest.TestCase):
    def test_reverse_freeze_and_zero_regularization_equivalence(self):
        torch.set_num_threads(2)
        torch.manual_seed(58)
        model = CodonVMAE(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                          ff_dim=32, latent_dim=4, dropout=0, max_codons=16).eval().requires_grad_(False)
        records = [dict(sequence_id='toy', n_codons=4, cds_dna='ATGGCTGCCTAA')]
        _, config = frozen_config(Path('deliverables/b0_b5_frozen_20261020'), ['B7', 'B8'])
        config.update(steps=4, tuning=True, random_start_seed=71, random_start_rms=.05)
        config['handoff'] = dict(method='kl', steps=3, lr=.01, eta=.1, temperature=1.)
        config['regularization'] = dict(latent_l2=0., direct_l2=0., decoder_kl=0.)
        self.assertEqual([phase('B8', s, 80) for s in (1, 40, 41, 80)], ['direct', 'direct', 'latent', 'latent'])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            scorer = CountedScorer(LinearScorer())
            result = optimize_batch(model, scorer, records, torch.ones(64), config, out, time.monotonic())
            self.assertEqual(scorer.forward_sequences, 6 + 2 * (4 * 3 + 2))
            for key in ('initial', 'best', 'final', 'best_hard_gain'):
                self.assertEqual(result[1][key], result[2][key])
            steps = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
            for arm in ('B7', 'B8'):
                rows = [r for r in steps if r['arm'] == arm]
                self.assertAlmostEqual(rows[0]['R_z'], .05**2, places=7)
                self.assertEqual(rows[0]['R_z'], rows[1]['R_z'])
                self.assertEqual(rows[2]['R_delta'], 0.)
                self.assertEqual(rows[2]['R_delta'], rows[3]['R_delta'])
                self.assertNotEqual(rows[1]['R_z'], rows[3]['R_z'])
            handoffs = [json.loads(line) for line in (out/'handoff.jsonl').read_text().splitlines()]
            self.assertEqual(len(handoffs), 2)
            for row in handoffs:
                self.assertLessEqual(row['trace'][-1]['objective'], row['trace'][0]['objective'])
                self.assertAlmostEqual(row['delta_projection'], row['after']['hard_score']-row['before']['hard_score'])
            config['handoff']['method'] = 'reencode'
            config['run_arms'] = ['B8']
            other = out/'reencode'; other.mkdir()
            optimize_batch(model, CountedScorer(LinearScorer()), records, torch.ones(64), config, other, time.monotonic())
            self.assertEqual(json.loads((other/'handoff.jsonl').read_text())['method'], 'reencode')


if __name__ == '__main__':
    unittest.main()
