import unittest

import torch

from .latent_gene_style_probe import visible_prior
from .model import AA_IDS, collate_records


class GeneStyleProbeTests(unittest.TestCase):
    def test_hidden_synonymous_labels_do_not_enter_counts(self):
        tokens=collate_records([dict(cds_dna='ATGGCTGCCGCAGCGTAA')])
        hidden=torch.zeros_like(tokens,dtype=torch.bool);hidden[:,1:3]=True
        changed=tokens.clone();changed[:,1]=tokens[:,4];changed[:,2]=tokens[:,3]
        torch.testing.assert_close(AA_IDS[changed],AA_IDS[tokens])
        torch.testing.assert_close(visible_prior(tokens,hidden,torch.ones(64),5),
                                   visible_prior(changed,hidden,torch.ones(64),5))

    def test_probabilities_normalize_per_amino_acid_and_ignore_padding(self):
        tokens=collate_records([dict(cds_dna='ATGGCTGCCGCATAG')]);hidden=torch.zeros_like(tokens,dtype=torch.bool)
        p=visible_prior(tokens,hidden,torch.arange(1,65).float(),20)
        for aa in AA_IDS[:64].unique():self.assertAlmostEqual(float(p[AA_IDS[:64]==aa].sum()),1.)
        padded=torch.nn.functional.pad(tokens,(0,4),value=64)
        torch.testing.assert_close(p,visible_prior(padded,torch.zeros_like(padded,dtype=torch.bool),torch.arange(1,65).float(),20))


if __name__=='__main__':unittest.main()
