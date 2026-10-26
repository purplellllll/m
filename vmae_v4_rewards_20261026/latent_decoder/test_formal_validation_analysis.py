import unittest
from .formal_validation_analysis import matches,endpoint_values


class FormalAnalysisChecks(unittest.TestCase):
    def test_matching_does_not_use_scores_and_empty_is_explicit(self):
        limits=dict(max_pairwise_gc_fraction=.01,max_pairwise_mfe_per_nt=.01,max_pairwise_mutable_hamming=.02)
        rows=[dict(gc_fraction=.5+i*.003,rnafold_mfe_per_nt=-.2,mutable_hamming=.05,ribodecode_score=i*1e9) for i in range(3)]
        self.assertTrue(matches(rows,limits));rows[2]['mutable_hamming']=.071
        self.assertFalse(matches(rows,limits))
        data=[dict(n_codons=n,matched_seeds=0,internal_B5_minus_B4=g,internal_B5_minus_B6=g) for n,g in [(100,.1),(800,.02)]]
        result=endpoint_values(data)
        self.assertAlmostEqual(result['length_B5_minus_B4'],-.08)
        self.assertIsNone(result['external_B5_minus_B4'])


if __name__=='__main__':unittest.main()
