import json
import tempfile
import time
import unittest
from pathlib import Path
import torch
from .b6_reproducibility import frozen_config
from .dual_space_ablation import CountedScorer, optimize_batch
from .model import CodonVMAE
from .test_b6_reproducibility import LinearScorer


class RandomStartChecks(unittest.TestCase):
    def test_real_randomness_shared_start_and_staged_freeze(self):
        torch.set_num_threads(2); torch.manual_seed(58)
        model=CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=32,latent_dim=4,dropout=0,max_codons=16).eval().requires_grad_(False)
        records=[dict(sequence_id='toy',n_codons=4,cds_dna='ATGGCTGCCTAA')]
        _,config=frozen_config(Path('deliverables/b0_b5_frozen_20261020'),['B4','B5','B6'])
        config.update(steps=4,tuning=True,random_start_rms=.05)
        initials=[]
        with tempfile.TemporaryDirectory() as directory:
            for index,seed in enumerate([71,71,72]):
                config['random_start_seed']=seed
                out=Path(directory)/str(index);out.mkdir()
                scorer=CountedScorer(LinearScorer())
                optimize_batch(model,scorer,records,torch.ones(64),config,out,time.monotonic())
                initials.append(json.loads((out/'random_initial.jsonl').read_text()))
                self.assertEqual(scorer.forward_sequences,6+3*4*3)
                steps=[json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
                for arm in ['B4','B6']:
                    direct=[r for r in steps if r['arm']==arm and r['phase']=='direct']
                    self.assertEqual(len({r['R_z'] for r in direct}),1)
                    self.assertTrue(all(abs(r['actual_residual_rms']**2-config['direct_scale']**2*r['R_delta'])<1e-6 for r in direct))
        self.assertEqual(initials[0],initials[1])
        self.assertNotEqual(initials[0]['latent_sha256'],initials[2]['latent_sha256'])
        self.assertAlmostEqual(initials[0]['latent_l2'],.05**2,places=7)


if __name__=='__main__':unittest.main()
