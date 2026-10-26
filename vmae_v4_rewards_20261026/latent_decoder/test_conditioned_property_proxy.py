import unittest
import torch
from torch.nn import functional as F

from .conditioned_property_proxy import conditioned_features
from .model import collate_records


class ConditionedChecks(unittest.TestCase):
    def test_zero_change_anchor_and_expected_edit_context(self):
        reference=collate_records([dict(cds_dna='ATGGCTGCCTAA')])
        changed=collate_records([dict(cds_dna='ATGGCCGCCTAA')])
        original=F.one_hot(reference,64).float();mutant=F.one_hot(changed,64).float()
        self.assertTrue(torch.equal(conditioned_features(original,reference,reference,torch.tensor([2.])),torch.zeros(1,210)))
        features=conditioned_features(mutant,reference,reference,torch.tensor([2.]))
        self.assertTrue(torch.allclose(features[0,-2:],torch.tensor([.5,1.])))
        soft=(.75*original+.25*mutant).requires_grad_()
        mixture=conditioned_features(soft,reference,reference,torch.tensor([2.]))
        self.assertTrue(torch.allclose(mixture[0,-2:],torch.tensor([.125,.25])))
        g,=torch.autograd.grad(mixture.sum(),soft)
        self.assertTrue(torch.isfinite(g).all() and g.abs().sum()>0)


if __name__=='__main__':unittest.main()
