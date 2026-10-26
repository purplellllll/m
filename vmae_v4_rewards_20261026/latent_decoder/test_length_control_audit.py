import unittest
import numpy as np
from .length_control_audit import estimates


class LengthChecks(unittest.TestCase):
    def test_recovers_length_and_adjusted_effect(self):
        rng=np.random.default_rng(3);rows=[]
        for i,n in enumerate(np.geomspace(40,1400,40)):
            cov=rng.normal(size=3)
            y=.08-.02*np.log2(n/512)+cov@[.01,.02,.03]
            r=dict(n_codons=n,B5_minus_B4=y)
            for k,v in zip(['gc_fraction','rnafold_mfe_per_nt','mutable_hamming'],cov):r['B5_minus_B4_'+k]=v*.01
            rows.append(r)
        result=estimates(rows,'B4')
        self.assertAlmostEqual(result['adjusted_slope'],-.02)
        expected=.08-.02*np.log2(np.array([r['n_codons'] for r in rows])/512).mean()
        self.assertAlmostEqual(result['adjusted_equal_covariates_mean'],expected)


if __name__=='__main__':unittest.main()
