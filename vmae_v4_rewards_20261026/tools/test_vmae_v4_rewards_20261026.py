"""Reward direction checks; mocked folding tests do not claim biological measurements."""
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from latent_decoder.tokenizer import CODONS, PAD, TOKEN, encode_cds
from tools import vmae_v4_rewards_20261026 as core


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.weights = {c:1. for c in CODONS}
        self.weights.update(TTT=.2,TTC=.8)
        self.tokens = torch.tensor([encode_cds('ATGTTTTAA')])

    def test_gc_cai_rewards_are_monotonic_and_zero_at_parent(self):
        original = F.one_hot(self.tokens,64).double()
        parent = core.property_rewards(original,self.tokens,self.weights)
        self.assertEqual(float(parent['gc_reward']),0.)
        self.assertEqual(float(parent['cai_reward']),0.)
        candidate = F.one_hot(torch.tensor([encode_cds('ATGTTCTAA')]),64).double()
        better = core.property_rewards(candidate,self.tokens,self.weights)
        self.assertAlmostEqual(float(better['gc_reward']),1/9/.05)
        self.assertAlmostEqual(float(better['cai_reward']),(math.sqrt(.8)-math.sqrt(.2))/.05)
        # Moving past a five-percentage-point GC change still earns a positive reward.
        self.assertGreater(float(better['gc_reward']),1.)

    def test_direct_reward_gradient_and_padding(self):
        target = torch.cat([self.tokens,torch.tensor([[PAD]])],-1)
        logits = torch.zeros((1,4,64),dtype=torch.float64,requires_grad=True)
        soft = logits.softmax(-1)
        hard = F.one_hot(torch.tensor([[TOKEN['ATG'],TOKEN['TTT'],TOKEN['TAA'],0]]),64).double()
        props = core.property_rewards(hard+soft-soft.detach(),target,self.weights)
        (-props['gc_reward']-props['cai_reward']).sum().backward()
        # Compare the two synonymous directions; unconstrained test logits include other codons.
        self.assertLess(float(logits.grad[0,1,TOKEN['TTC']]),float(logits.grad[0,1,TOKEN['TTT']]))
        self.assertEqual(float(logits.grad[:,3].abs().sum()),0.)

    def test_mfe_reward_keeps_increasing_for_more_negative_energy(self):
        self.assertAlmostEqual(core.mfe_reward(-.3,-.2),2.)
        self.assertAlmostEqual(core.mfe_reward(-.4,-.2),4.)
        self.assertAlmostEqual(core.mfe_reward(-.1,-.2),-2.)
        with self.assertRaises(ValueError):
            core.mfe_reward(float('nan'),-.2)

    def test_scst_encourages_better_mfe_and_discourages_worse(self):
        for better in (True,False):
            logits = torch.zeros((1,3,64),dtype=torch.float64,requires_grad=True)
            with torch.no_grad():
                logits[0,1,TOKEN['TTT']] = math.log(3.)
                logits[0,-1,TOKEN['TAG']] = 8.
            def fold(dna, cache):
                energy = (-.4 if better else -.1) if dna[3:6]=='TTC' else -.2
                return dict(mfe=energy*len(dna),mfe_per_nt=energy)
            sampled = torch.tensor([TOKEN['ATG'],TOKEN['TTC'],TOKEN['TAA']])
            with patch.object(core,'exact_fold',fold),patch('torch.distributions.Categorical.sample',return_value=sampled):
                loss,diagnostics = core.scst_loss(logits.softmax(-1),self.tokens,-.2,{})
            loss.backward()
            self.assertLess(float(logits.grad[0,1,TOKEN['TTC']])*(1 if better else -1),0.)
            self.assertEqual(float(logits.grad[0,0].abs().sum()),0.)
            self.assertEqual(float(logits.grad[0,-1].abs().sum()),0.)
            self.assertEqual(diagnostics['sample_dna'],'ATGTTCTAA')
            self.assertEqual(diagnostics['greedy_dna'],'ATGTTTTAA')

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            core.property_rewards(F.one_hot(self.tokens,64).float()*.5,self.tokens,self.weights)


if __name__ == '__main__':
    unittest.main()
