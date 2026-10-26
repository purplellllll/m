import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

import torch

from .b6_reproducibility import frozen_config
from .dual_space_ablation import CountedScorer, optimize_batch
from .js_constraint import JSConstraint, mutable_js
from .latent_optimization import design_probabilities, snapshot
from .model import CodonVMAE, collate_records, mutable_positions
from .test_b6_reproducibility import LinearScorer
from .js_constraint_validation import report


class JSConstraintTests(unittest.TestCase):
    def test_matches_existing_js_and_has_restoring_gradient(self):
        tokens = collate_records([dict(cds_dna='ATGGCTGCCTAA')])
        active = mutable_positions(tokens)
        torch.manual_seed(73)
        reference = design_probabilities(torch.randn(1,4,64),tokens)
        logits = (torch.randn(1,4,64)*3).requires_grad_()
        p = design_probabilities(logits,tokens)
        original = snapshot(p,tokens,LinearScorer(),torch.ones(64),reference)[0]['js_from_initial']
        self.assertAlmostEqual(float(mutable_js(p,reference,active).detach()),original,places=7)
        controller = JSConstraint(reference,active,.001,1.)
        penalty = controller.penalty(p).sum()
        gradient = torch.autograd.grad(penalty,logits)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        norm = gradient.norm().clamp_min(1e-12)
        q = design_probabilities(logits.detach()-.001*gradient/norm,tokens)
        self.assertLess(float(mutable_js(q,reference,active).detach()),original)
        controller.update(p)
        self.assertGreater(float(controller.dual[0]),0)
        controller.update(reference)
        self.assertGreaterEqual(float(controller.dual[0]),0)
        controller.dual.zero_()
        self.assertEqual(float(controller.penalty(reference)),0)
        for limit,rho in ((0,1),(.05,0),(.05,float('nan'))):
            with self.assertRaises(ValueError): JSConstraint(reference,active,limit,rho)

    def test_inactive_constraint_keeps_trajectory_and_call_count(self):
        torch.set_num_threads(2);torch.manual_seed(58)
        model = CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=32,latent_dim=4,dropout=0,max_codons=16).eval().requires_grad_(False)
        records = [dict(sequence_id='toy',n_codons=4,cds_dna='ATGGCTGCCTAA')]
        _,base = frozen_config(Path('deliverables/b0_b5_frozen_20261020'),['B5'])
        base.update(steps=6,tuning=True,random_start_rms=.05,random_start_seed=71,max_js=.7)
        results=[]
        with tempfile.TemporaryDirectory() as directory:
            for index,settings in enumerate((None,dict(rho=1.))):
                out=Path(directory)/str(index);out.mkdir()
                config=copy.deepcopy(base)
                if settings: config['js_constraint']=settings
                scorer=CountedScorer(LinearScorer())
                results.append(optimize_batch(model,scorer,records,torch.ones(64),config,out,time.monotonic()))
                self.assertEqual(scorer.forward_sequences,6+6*3)
            for a,b in zip(*results):
                for key in ('arm','initial','best_hard_gain','final_hard_gain','unique_candidates'):
                    self.assertEqual(a[key],b[key])
                for key in ('cds_dna','hard_score','js_from_initial'):
                    self.assertEqual(a['best'][key],b['best'][key])
            rows=[json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['js_dual_after']==0 for r in rows))

    def test_smaller_gap_alone_does_not_pass_success_rule(self):
        before=dict(legal_gain=.3,raw_gain=1.4,filter_gap=1.1,legal_js=.04,legal_edit=.08,
                    feasible_step_fraction=.5,js_rejection_fraction=.5)
        after=dict(before,raw_gain=.4,filter_gap=.1)
        rows=[dict(stage='long_test',sequence_id='toy',seed=seed,baseline=before,constrained=after) for seed in (1,2,3)]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            report(out,rows)
            result=json.loads((out/'summary.json').read_text())
            self.assertFalse(result['stages']['long_test']['descriptive_success'])
            after['legal_gain']=.4
            report(out,rows)
            result=json.loads((out/'summary.json').read_text())
            self.assertTrue(result['stages']['long_test']['descriptive_success'])


if __name__ == '__main__':
    unittest.main()
