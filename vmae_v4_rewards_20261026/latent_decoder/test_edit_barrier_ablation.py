import math
import unittest

import torch

from .edit_barrier_ablation import ARMS, crossing_metrics, settings
from .js_constraint import JSConstraint


class AblationTests(unittest.TestCase):
    def test_single_term_settings_do_not_modify_shared_configuration(self):
        config=dict(regularization=dict(latent_l2=.1,direct_l2=.1,decoder_kl=.1),lr=.1)
        for arm in ARMS:
            changed,limit=settings(config,arm)
            self.assertEqual(changed['lr'],.1)
            self.assertEqual(changed['regularization']['latent_l2'],.1)
            self.assertEqual(changed['regularization']['decoder_kl'],0. if arm=='no_decoder_kl' else .1)
            self.assertEqual(changed['regularization']['direct_l2'],0. if arm=='no_direct_l2' else .1)
            self.assertEqual(limit>math.log(2),arm=='no_js_penalty')
        self.assertEqual(config['regularization'],dict(latent_l2=.1,direct_l2=.1,decoder_kl=.1))

    def test_disabled_controller_has_zero_penalty_and_gradient_even_for_disjoint_distributions(self):
        reference=torch.tensor([[[1.,0.],[1.,0.]]])
        p=torch.tensor([[[0.,1.],[.5,.5]]],requires_grad=True)
        _,limit=settings(dict(regularization={}), 'no_js_penalty')
        controller=JSConstraint(reference,torch.ones(1,2,dtype=torch.bool),limit,1.)
        for _ in range(5):
            penalty=controller.penalty(p)
            self.assertEqual(float(penalty.detach()),0.)
            self.assertEqual(float(torch.autograd.grad(penalty.sum(),p)[0].abs().sum()),0.)
            controller.update(p)
            self.assertEqual(float(controller.dual),0.)

    def test_crossings_count_reverted_positions_and_censor_missing_thresholds(self):
        original='ATGGCTGCTTAA'
        trace=[dict(step=i,cds_dna=original,changed_codons=0,soft_js=0.) for i in range(1,81)]
        trace[2]=dict(step=3,cds_dna='ATGGCCGCTTAA',changed_codons=1,soft_js=.1)
        metric=crossing_metrics(trace,original)
        self.assertEqual(metric['first_step_exceeding_fraction']['0'],3)
        self.assertEqual(metric['ever_edited_fraction'],.25)
        self.assertEqual(metric['endpoint_edited_fraction'],0.)
        self.assertEqual(metric['js05_feasible_step_fraction'],79/80)


if __name__=='__main__':
    unittest.main()
