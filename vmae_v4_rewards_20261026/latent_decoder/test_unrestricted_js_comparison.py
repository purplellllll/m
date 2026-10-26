import unittest

import torch
from torch.nn import functional as F

from .budget_comparison import project_budget
from .model import AA_IDS, CODONS, collate_records, synonymous_logits
from .unrestricted_js_comparison import ARMS, aggregate, validate_winner


class UnrestrictedComparisonTests(unittest.TestCase):
    def test_full_budget_equals_all_synonymous_argmax_changes(self):
        tokens=collate_records([dict(cds_dna='ATG'+'GCT'*20+'TAA')])
        logits=torch.zeros((*tokens.shape,64))
        # Give a synonymous alternative a higher score at every alanine.
        alternative=CODONS.index('GCC')
        logits[:,:,alternative]=5
        p=synonymous_logits(logits,tokens).softmax(-1)
        selected=project_budget(p,tokens,tokens.shape[1]).argmax(-1)
        self.assertEqual(int((selected!=tokens).sum()),20)
        self.assertTrue(torch.equal(AA_IDS[selected],AA_IDS[tokens]))
        self.assertTrue(torch.equal(selected[:,[0,-1]],tokens[:,[0,-1]]))

    def test_complete_seed_pairs_and_no_final_js_filter(self):
        tokens=collate_records([dict(cds_dna='ATGGCTTAA')])
        rows=[]
        for arm in ARMS:
            for seed in (1,2,3):
                r=dict(sequence_id='toy',seed=seed,arm=arm,n_aa=2,baseline=1.,gain=1.,changed_fraction=0.,scorer_queries=160,
                       best=dict(cds_dna='ATGGCTTAA',hard_score=2.,changed_codons=0,soft_js=.2),
                       best_if_js05=dict(hard_score=1.))
                validate_winner(r,tokens)
                rows.append(r)
        self.assertEqual(aggregate(rows[:-1],(1,2,3))['complete_cds'],0)
        summary=aggregate(rows,(1,2,3))
        self.assertEqual(summary['complete_cds'],1)
        self.assertEqual(summary['per_cds'][0]['B5_JS_soft_only']['js05_violation_runs'],3)


if __name__=='__main__':
    unittest.main()
