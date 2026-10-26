from collections import Counter
import unittest

import torch
from torch.nn import functional as F

from .joint_property_proxy import BASES, fit_delta_ridge, sequence_features, synonymous_sample
from .model import AA_IDS, PAD, collate_records
from .tokenizer import TOKEN


class ProxyChecks(unittest.TestCase):
    def test_delta_ridge_recovers_a_known_change_with_constant_feature(self):
        x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[2.,1.,0.],[-1.,2.,0.],[1.,3.,0.],[-2.,-1.,0.]])
        truth=torch.tensor([[2.,-1.,0.],[-.5,3.,0.]])
        y=F.linear(x,truth)
        weight,info=fit_delta_ridge(x,y,['a']*3+['b']*3,1e-8)
        audit=torch.tensor([[3.,2.,0.],[-1.,-2.,0.]],dtype=torch.float64)
        self.assertTrue(torch.allclose(F.linear(audit,weight),F.linear(audit,truth.double()),atol=1e-5))
        self.assertTrue(torch.equal(F.linear(torch.zeros(1,3,dtype=torch.float64),weight),torch.zeros(1,2,dtype=torch.float64)))
        self.assertEqual(info['groups'],2)

    def test_hard_features_match_empirical_counts_and_ignore_padding(self):
        dna='ATGGCTGCCTAA'
        tokens=collate_records([dict(cds_dna=dna),dict(cds_dna='ATGGCCTAA')])
        p=F.one_hot(tokens.clamp_max(63),64).float()
        features=sequence_features(p,tokens)
        counts=Counter(dna[i:i+2] for i in range(len(dna)-1))
        expected=torch.tensor([counts[a+b]/(len(dna)-1) for a in BASES for b in BASES])
        self.assertEqual(features.shape,(2,208))
        self.assertTrue(torch.allclose(features[0,-16:],expected))
        for start,size in ((0,64),(64,64),(128,64),(192,16)):
            self.assertTrue(torch.allclose(features[:,start:start+size].sum(1),torch.ones(2)))

    def test_soft_features_equal_independent_codon_expectation_and_have_gradients(self):
        dnas=['ATG'+a+b+'TAA' for a in ('GCT','GCC') for b in ('GCT','GCC')]
        tokens=collate_records([dict(cds_dna=dnas[0])])
        hard=torch.cat([F.one_hot(collate_records([dict(cds_dna=d)]),64).float() for d in dnas])
        soft=hard.mean(0,keepdim=True).requires_grad_()
        expected=sequence_features(hard,tokens.expand(4,-1)).mean(0,keepdim=True)
        self.assertTrue(torch.allclose(sequence_features(soft,tokens),expected,atol=1e-7))
        gradient,=torch.autograd.grad(sequence_features(soft,tokens).square().sum(),soft)
        self.assertTrue(torch.isfinite(gradient).all() and gradient.abs().sum()>0)

    def test_sampling_is_synonymous_deterministic_and_keeps_boundaries(self):
        original='ATGGCTGCCCTGAGATAA';counts=torch.arange(1,65).float()
        dna=synonymous_sample(original,counts,gc_bias=-1.,seed=17)
        self.assertEqual(dna,synonymous_sample(original,counts,gc_bias=-1.,seed=17))
        before=collate_records([dict(cds_dna=original)]);after=collate_records([dict(cds_dna=dna)])
        self.assertTrue(torch.equal(AA_IDS[before],AA_IDS[after]))
        self.assertEqual((dna[:3],dna[-3:]),(original[:3],original[-3:]))
        self.assertEqual(synonymous_sample(original,counts,fraction=0.,seed=17),original)


if __name__=='__main__':unittest.main()
