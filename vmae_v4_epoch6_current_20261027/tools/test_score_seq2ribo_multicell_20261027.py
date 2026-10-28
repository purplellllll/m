import unittest
from tools.score_seq2ribo_multicell_20261027 import summarize,PROPERTIES,CELLS


class PairingTest(unittest.TestCase):
    def test_pairing_excludes_unmatched_and_keeps_cell_scales_separate(self):
        def row(identity,method,value):
            return dict(sequence_id=identity,method=method,**{k:value for k in (*PROPERTIES,*CELLS)})
        result=summarize([row('a','original',1),row('b','original',100),row('a','V4_epoch6_current',3)])
        self.assertEqual(result['methods']['original']['n'],2)
        paired=result['paired']['V4_epoch6_current minus original']
        self.assertEqual(paired['sequence_ids'],['a'])
        self.assertEqual(paired['mean_delta'],dict.fromkeys(CELLS,2))
        self.assertEqual(paired['higher_count'],dict.fromkeys(CELLS,1))


if __name__=='__main__':unittest.main()
