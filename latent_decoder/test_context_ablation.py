import unittest
import torch
from .context_ablation import context_inputs
from .model import AA_IDS, MASK, collate_records, corrupt, mutable_positions
from .train import codon_frequency


class ContextChecks(unittest.TestCase):
    def test_targets_aa_and_visible_counts_are_preserved(self):
        records=[dict(cds_dna='ATG'+'GCTGCCTTCTTT'*12+'TAG')]
        tokens=collate_records(records)
        inputs,selected=corrupt(tokens,.15,torch.Generator().manual_seed(173))
        variants=context_inputs(tokens,inputs,selected,codon_frequency(records),torch.Generator().manual_seed(5))
        for variant in variants.values():
            self.assertTrue((variant[selected]==MASK).all())
            visible=variant<64
            self.assertTrue(torch.equal(AA_IDS[variant[visible]],AA_IDS[tokens[visible]]))
            self.assertTrue(torch.equal(variant[~mutable_positions(tokens)],tokens[~mutable_positions(tokens)]))
        self.assertTrue(torch.equal(inputs[~selected].sort().values,variants['shuffled_visible'][~selected].sort().values))
        self.assertFalse(torch.equal(inputs,variants['shuffled_visible']))


if __name__=='__main__':
    unittest.main()
