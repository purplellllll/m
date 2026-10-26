import unittest
from .analyze_random_validation import sequence_metrics,synonym_changes,paired_interval
from .tokenizer import CODONS


class PosthocChecks(unittest.TestCase):
    def test_synonymous_metrics_and_direction(self):
        weights={c:1. for c in CODONS}
        self.assertEqual(sequence_metrics('ATGGCTGCCTAA',weights)['train_reference_cai'],1.)
        delta,tv=synonym_changes('ATGGCTGCCTAA','ATGGCCGCCTAA')
        self.assertEqual(delta['A']['GCT'],-.5)
        self.assertEqual(delta['A']['GCC'],.5)
        self.assertAlmostEqual(tv,1/3)
        with self.assertRaises(AssertionError):synonym_changes('ATGGCTGCCTAA','ATGGTTGCCTAA')
        result=paired_interval([.1,.1],['a','b'])
        self.assertEqual(result['wins'],2)
        self.assertAlmostEqual(result['ci95'][0],.1)
        self.assertAlmostEqual(result['ci95'][1],.1)


if __name__=='__main__':unittest.main()
