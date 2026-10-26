import copy
import unittest
from .analyze_b6 import compare


class ReplayComparisonChecks(unittest.TestCase):
    def examples(self):
        row=dict(arm='B6',sequence_id='toy',best_hard_gain=.1,final_hard_gain=.1,
            initial=dict(hard_score=1.),best=dict(hard_score=1.1,cds_dna='ATGGCTTAA',step=4),
            final=dict(hard_score=1.1,cds_dna='ATGGCTTAA'),elapsed_seconds=1.)
        step=dict(arm='B6',sequence_id='toy',step=1,feasible=True,hard_score=1.1)
        return row,step

    def test_ignores_only_timing_and_distinguishes_float_difference(self):
        source,step=self.examples()
        current=copy.deepcopy(source);current['elapsed_seconds']=100.
        self.assertTrue(compare([source],[current],[step],[step])['strict_match'])
        current['best']['hard_score']+=1e-7
        result=compare([source],[current],[step],[step])
        self.assertFalse(result['strict_match'])
        self.assertTrue(result['score_tolerance_diagnostic'])

    def test_changed_dna_or_large_score_error_cannot_pass(self):
        source,step=self.examples()
        current=copy.deepcopy(source);current['best']['cds_dna']='ATGGCCTAA'
        self.assertFalse(compare([source],[current],[step],[step])['score_tolerance_diagnostic'])
        current=copy.deepcopy(source);current['best_hard_gain']+=2e-5
        self.assertFalse(compare([source],[current],[step],[step])['score_tolerance_diagnostic'])


if __name__=='__main__':unittest.main()
