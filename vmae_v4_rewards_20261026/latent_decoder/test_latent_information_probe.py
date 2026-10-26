import unittest

import torch

from .latent_information_probe import probe,projection_geometry,recode_visible
from .latent_collapse_cpu import Pool
from .latent_recovery_gpu import BASE,RecoveryVMAE
from .model import AA_IDS,collate_records,mutable_positions


class InformationProbeTests(unittest.TestCase):
    def setUp(self):
        self.tokens=collate_records([dict(cds_dna='ATG'+'GCTGCCGCAGCG'*8+'TAA')])
        self.selected=torch.zeros_like(self.tokens,dtype=torch.bool)
        self.selected[:,3::7]=True
        self.selected&=mutable_positions(self.tokens)

    def test_permutation_preserves_full_protein_and_hidden_targets(self):
        changed=recode_visible(self.tokens,self.selected,'permutation',7,torch.ones(64))
        torch.testing.assert_close(AA_IDS[changed],AA_IDS[self.tokens])
        torch.testing.assert_close(changed[self.selected],self.tokens[self.selected])
        torch.testing.assert_close(torch.bincount(changed.flatten(),minlength=64),
                                   torch.bincount(self.tokens.flatten(),minlength=64))
        self.assertTrue((changed!=self.tokens).any())

    def test_frequency_draw_preserves_full_protein_and_padding(self):
        padded=torch.nn.functional.pad(self.tokens,(0,5),value=64)
        mask=torch.nn.functional.pad(self.selected,(0,5),value=False)
        changed=recode_visible(padded,mask,'train_frequency',7,torch.ones(64))
        torch.testing.assert_close(AA_IDS[changed],AA_IDS[padded])
        torch.testing.assert_close(changed[mask],padded[mask])
        self.assertTrue((changed[:,-5:]==64).all())

    def test_all_hidden_has_no_visible_codon_effect(self):
        hidden=mutable_positions(self.tokens)
        for kind in ('permutation','train_frequency'):
            torch.testing.assert_close(recode_visible(self.tokens,hidden,kind,7,torch.ones(64)),self.tokens)

    def test_projection_geometry_matches_explicit_response(self):
        torch.manual_seed(3)
        x=torch.randn(1,50,8,dtype=torch.float64)
        tokens=collate_records([dict(cds_dna='ATG'+'GCT'*48+'TAA')])
        weight=torch.randn(16,8,dtype=torch.float64)
        pool=Pool(8,covariance=True);pool.add(x,tokens)
        result=projection_geometry(pool,weight,gain=.5)
        actual=(x[0]@weight.T)*.5
        centered=actual-actual.mean(0)
        covariance=centered.T@centered/len(centered)
        expected=float(covariance.trace().square()/covariance.square().sum())
        self.assertAlmostEqual(result['covariance_participation'],expected,places=10)
        self.assertAlmostEqual(result['rms'],float(actual.square().mean().sqrt()),places=10)

    def check_probe(self,device):
        config=dict(d_model=16,n_heads=2,encoder_layers=2,decoder_layers=1,ff_dim=32,
                    latent_dim=8,dropout=0.,max_codons=64,max_positions=64)
        model=RecoveryVMAE(BASE,**config).to(device).eval()
        record=dict(sequence_id='probe',n_codons=34,cds_dna='ATG'+'GCTGCC'*16+'TAA')
        result=probe(model,[record],torch.ones(64),torch.tensor([1.,1.,1.,1.,0.,0.,0.,0.]),repeats=1)
        self.assertEqual(len(result['rows']),4)
        self.assertTrue(all(r['same_complete_protein'] and r['masked_targets_unchanged'] for r in result['rows']))
        self.assertTrue(all(torch.isfinite(torch.tensor(r['target_ce_delta'])) for r in result['rows']))
        self.assertGreater(sum(r['response_energy'] for r in result['rows']),0.)

    def test_identical_protein_probe_reports_actual_changes(self):
        self.check_probe('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_identical_protein_probe_cuda(self):
        self.check_probe('cuda')


if __name__=='__main__':
    unittest.main()
