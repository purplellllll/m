import unittest

import torch

from .diagnose_mask_plateau import audit,build_neighbor_counts,cached,measure
from .model import CodonVMAE
from .tokenizer import encode_cds
from .train import codon_frequency


class PlateauChecks(unittest.TestCase):
    def test_baselines_and_loss_on_same_masked_positions(self):
        torch.set_num_threads(2)
        records=[]
        for index,sequence in enumerate(['ATG'+'GCTGCCGCGTTCTTA'*8+'TAG','ATG'+'GCCGCATTTCCTCTG'*8+'TAA']):
            records.append(dict(sequence_id=str(index),cds_dna=sequence,n_codons=len(sequence)//3,codon_ids=encode_cds(sequence)))
        counts=build_neighbor_counts(records)
        self.assertEqual(int(counts.sum()),sum(r['n_codons'] for r in records))
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,ff_dim=32,latent_dim=4,dropout=.1,max_codons=64)
        data=cached(records,torch.device('cpu'),173)
        result=audit(model,data,codon_frequency(records),counts)
        direct=measure(model,data)
        self.assertAlmostEqual(result['model_ce'],direct['masked_ce'],places=6)
        self.assertAlmostEqual(result['clean_ce'],direct['clean_on_same_sites_ce'],places=6)
        self.assertEqual(sum(x['sites'] for x in result['family'].values()),result['sites'])
        for name in ('global','sequence','neighbor'):
            self.assertTrue(0<=result[name+'_correct']<=1)
            self.assertGreaterEqual(result[name+'_prior_ce'],0)


if __name__=='__main__':
    unittest.main()
