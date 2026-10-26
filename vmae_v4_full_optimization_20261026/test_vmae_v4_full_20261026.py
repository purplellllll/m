"""The conditional continuation cannot use partial panels or independent seq2ribo."""
import unittest

from tools.run_vmae_v4_full_20261026 import choose_checkpoint


class DecisionTests(unittest.TestCase):
    def test_latest_checkpoint_wins_complete_validation(self):
        scores={e:{'a':float(e),'b':float(e)+1} for e in range(4)}
        result=choose_checkpoint(scores,['a','b'])
        self.assertEqual(result['best_epoch'],3)
        self.assertEqual(result['additional_epochs'],5)
        self.assertFalse(result['seq2ribo_used_for_selection'])

    def test_other_epoch_does_not_extend(self):
        scores={e:{'a':float(e)} for e in range(4)}
        scores[2]['a']=10.
        self.assertEqual(choose_checkpoint(scores,['a'])['additional_epochs'],0)

    def test_missing_case_or_identity_mismatch_rejected(self):
        scores={e:{'a':1.,'b':2.} for e in range(4)}
        scores[3]={'a':3.}
        with self.assertRaises(ValueError):choose_checkpoint(scores,['a','b'])
        scores[3]={'a':1.,'c':2.}
        with self.assertRaises(ValueError):choose_checkpoint(scores,['a','b'])

    def test_tie_and_nonfinite(self):
        scores={e:{'a':1.} for e in range(4)}
        self.assertEqual(choose_checkpoint(scores,['a'])['best_epoch'],3)
        scores[3]['a']=float('nan')
        with self.assertRaises(ValueError):choose_checkpoint(scores,['a'])


if __name__=='__main__':
    unittest.main()
