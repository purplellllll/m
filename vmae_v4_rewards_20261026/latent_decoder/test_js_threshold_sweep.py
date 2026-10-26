import unittest

from .js_threshold_sweep import BUDGETS, LEVELS, contrasts, select_thresholds


class ThresholdTests(unittest.TestCase):
    def test_nonmonotonic_crossing_best_and_no_crossing(self):
        rows = []
        for budget in BUDGETS:
            differences = [-.2, -.1, .01, .05, -.02, .03, .04, .04] if budget == 10 else [-.3]*8
            for threshold, difference in zip(LEVELS, differences):
                rows.append(dict(budget_percent=budget, js_limit=threshold, n_cds=8,
                    means={'B5_JS': 3+difference}, vs_native={'mean': difference}))
        selected = select_thresholds(rows, 8)
        self.assertEqual(selected['10']['validate'], [.1, .15])
        self.assertEqual(selected['20']['validate'], [.05])
        self.assertIsNone(selected['20']['first_screen_crossing'])
        rows[0]['n_cds'] = 7
        with self.assertRaises(AssertionError):
            select_thresholds(rows, 8)

    def test_matched_seeds_no_cross_threshold_pooling(self):
        baseline = [dict(sequence_id=i, seed=s, budget_percent=10, arm=a, best={'hard_score': 2.})
                    for i in ('a', 'b') for s in (1, 2, 3) for a in ('B5', 'RiboDecode')]
        rows = [dict(sequence_id=i, seed=s, budget_percent=10, js_limit=t, best={'hard_score': score})
                for i in ('a', 'b') for s in (1, 2, 3) for t, score in ((.1, 2.1), (.2, 9.))]
        r = contrasts(rows, baseline, ['a', 'b'], [2, 3], 10, .1)
        self.assertEqual(r['n_cds'], 2)
        self.assertAlmostEqual(r['vs_native']['mean'], .1)
        rows = [r for r in rows if not (r['sequence_id'] == 'b' and r['seed'] == 3)]
        self.assertEqual(contrasts(rows, baseline, ['a', 'b'], [2, 3], 10, .1)['n_cds'], 1)


if __name__ == '__main__':
    unittest.main()
