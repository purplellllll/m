"""Small invariant checks for grouped folds, length updates, selection and search."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from .dual_space_ablation import CountedScorer
from .length_diagnostics import length_group
from .model import CodonVMAE, collate_records
from .v2_data import assign_folds
from .v2_experiment import COEFFICIENTS, SEEDS, select, summarize
from .v2_search import native, search
from .v2_train import GROUPS, PROFILES, windows


class ToyScorer(torch.nn.Module):
    def forward(self, probabilities, tokens):
        return (probabilities*torch.arange(64, device=tokens.device)/64).sum(-1).mean(-1)


class V2Tests(unittest.TestCase):
    def test_folds_and_windows(self):
        records = [dict(sequence_id=str(i), cluster_id=str(i//2), n_codons=(101, 401, 801)[i//2%3]) for i in range(360)]
        with tempfile.TemporaryDirectory() as directory:
            hits = Path(directory)/'hits.tsv'
            hits.write_text('0\t20\t0.5\t0.9\t0.9\n')
            result = assign_folds(records, hits)
        mapping = result['assignments']
        self.assertEqual(mapping['0'], mapping['20'])
        self.assertEqual(mapping['0'], mapping['1'])
        tested = []
        for split in result['splits'].values():
            sets = [set(mapping[i]['group'] for i in split[name]) for name in ('train', 'val', 'test')]
            self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
            tested.extend(split['test'])
        self.assertEqual(sorted(tested, key=int), [str(i) for i in range(360)])
        batches = windows(records, 5)
        self.assertEqual(batches, windows(records, 5))
        self.assertEqual(sorted(i for _, ids in batches for i in ids), list(range(360)))
        for group, ids in batches:
            self.assertEqual({length_group(records[i]['n_codons']-1) for i in ids}, {group})

    def test_selection_uses_complete_inner_grid(self):
        rows = [dict(profile=p, group=g, coefficients=list(c), gain=float(p=='long_fast' and c==(.5, 2.)))
                for p in PROFILES for g in GROUPS for c in COEFFICIENTS for _ in range(9)]
        chosen = select(rows)
        self.assertEqual(chosen['profile'], 'long_fast')
        self.assertEqual(chosen['by_length']['long']['coefficients'], [.5, 2.])
        with self.assertRaises(AssertionError):
            select(rows[:-1])

    def test_partial_summary_does_not_claim_larger_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'fold_0'/'outer'
            output.mkdir(parents=True)
            for arm in ('selected_js05', 'selected_relaxed', 'uniform_default', 'RiboDecode'):
                for seed in SEEDS:
                    row = dict(fold=0, sequence_id='toy', seed=seed, arm=arm, budget_percent=5,
                               group='short', baseline=1., best=dict(hard_score=2.))
                    (output/f'{arm}_{seed}.json').write_text(json.dumps(row))
            summarize(root)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual([r['budget_percent'] for r in summary['comparisons']], [5])

    def test_search_budget_queries_frozen_weights_and_determinism(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = CodonVMAE(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                          ff_dim=32, latent_dim=8, dropout=0., max_codons=50, max_positions=50).eval().requires_grad_(False)
        tokens = collate_records([dict(cds_dna='ATG'+'GCT'*20+'TAA')])
        scorer = CountedScorer(ToyScorer())
        state = {k:v.clone() for k,v in model.state_dict().items()}
        first = search(model, scorer, tokens, 'toy', 17, 10, steps=3)
        second = search(model, scorer, tokens, 'toy', 17, 10, steps=3)
        self.assertEqual(first['best'], second['best'])
        self.assertEqual(first['scorer_queries'], 6)
        self.assertLessEqual(first['best']['changed_codons'], 2)
        self.assertLessEqual(first['best']['soft_js'], .05+1e-7)
        self.assertEqual(len(first['encoder']['variance_by_dim']), 8)
        self.assertTrue(all(torch.equal(v, state[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        control = native(scorer, tokens, 'toy', 17, 10, steps=3)
        self.assertEqual(control['scorer_queries'], 6)
        self.assertLessEqual(control['best']['changed_codons'], 2)


if __name__ == '__main__':
    unittest.main()
