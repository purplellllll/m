import unittest
from .formal_validation_plan import select_formal


class FormalPlanChecks(unittest.TestCase):
    def test_order_invariant_component_disjoint_selection(self):
        records=[dict(n_codons=n,sequence_id=str(n)+'_'+str(i),cluster_id=str(n)+'_'+str(i))
                 for n in [100,300,700,1200] for i in range(70)]
        excluded={'100_0','700_0'}
        a=select_formal(records,excluded);b=select_formal(list(reversed(records)),excluded)
        self.assertEqual(a,b)
        self.assertEqual(len(a),256)
        self.assertFalse({r['cluster_id'] for r in a}&excluded)


if __name__=='__main__':unittest.main()
