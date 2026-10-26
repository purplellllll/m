import math
import unittest
import torch
from torch.nn import functional as F
from .mlm_temperature_calibration import inverse_temperature


class TemperatureChecks(unittest.TestCase):
    def test_known_binary_probability_optimum_with_forbidden_class(self):
        logits=torch.tensor([[0.,math.log(9.),-torch.inf]]*10,dtype=torch.float64)
        target=torch.tensor([1]*8+[0]*2);weights=torch.ones(10)
        fitted=inverse_temperature(logits,target,weights)
        self.assertAlmostEqual(fitted,math.log(4)/math.log(9),places=7)
        self.assertLess(float(F.cross_entropy(logits*fitted,target)),float(F.cross_entropy(logits,target)))
        self.assertTrue(torch.equal(logits.argmax(-1),(logits*fitted).argmax(-1)))

    def test_boundaries_and_invalid_target(self):
        logits=torch.tensor([[0.,2.]])
        self.assertEqual(inverse_temperature(logits,torch.tensor([1]),torch.ones(1)),4.)
        self.assertEqual(inverse_temperature(logits,torch.tensor([0]),torch.ones(1)),.25)
        with self.assertRaises(ValueError):inverse_temperature(torch.tensor([[0.,-torch.inf]]),torch.tensor([1]),torch.ones(1))
        with self.assertRaises(ValueError):inverse_temperature(logits,torch.tensor([1]),torch.zeros(1))


if __name__=='__main__':unittest.main()
