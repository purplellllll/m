import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
import torch
from .b6_reproducibility import frozen_config
from .dual_space_ablation import CountedScorer, optimize_batch, phase
from .model import CodonVMAE


class LinearScorer(torch.nn.Module):
    def forward(self, probabilities, tokens):
        return (probabilities * torch.arange(64, device=probabilities.device)).sum((1, 2)) / 100


class B6Checks(unittest.TestCase):
    def test_frozen_protocol_and_stages(self):
        reference = Path('deliverables/b0_b5_frozen_20261020')
        original, config = frozen_config(reference, ['B6'])
        self.assertEqual(config['regularization'], original['regularization'])
        self.assertEqual(config['arm_lr_multipliers']['B6'], original['arm_lr_multipliers']['B4'])
        self.assertEqual([phase('B6', s, 80) for s in (1, 40, 41, 80)], ['latent','latent','direct','direct'])

    def test_old_paths_unchanged_and_zero_regularized_b6_equals_b4(self):
        torch.set_num_threads(2)
        torch.manual_seed(58)
        model = CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
                          ff_dim=32,latent_dim=4,dropout=0,max_codons=16).eval().requires_grad_(False)
        records = [dict(sequence_id='toy', n_codons=4, cds_dna='ATGGCTGCCTAA')]
        original, config = frozen_config(Path('deliverables/b0_b5_frozen_20261020'), ['B4'])
        config.update(steps=4,tuning=True,lr=.05)
        config['regularization'] = dict(latent_l2=0.,direct_l2=0.,decoder_kl=0.)
        frozen_source = Path('deliverables/b0_b5_frozen_20261020/source/latent_decoder/dual_space_ablation.py')
        spec = importlib.util.spec_from_file_location('latent_decoder._old_ablation', frozen_source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ['old','new','b6']]
            for p in paths:p.mkdir()
            all_config = dict(config)
            all_config.pop('run_arms')
            before = old.optimize_batch(model,CountedScorer(LinearScorer()),records,torch.ones(64),all_config,paths[0],time.monotonic())
            after = optimize_batch(model,CountedScorer(LinearScorer()),records,torch.ones(64),all_config,paths[1],time.monotonic())
            for a,b in zip(before,after):
                for key in ['arm','initial','best','final','best_hard_gain','feasible_steps']:
                    self.assertEqual(a[key],b[key])
            config['run_arms']=['B6']
            config['arm_lr_multipliers']['B6']=config['arm_lr_multipliers']['B4']
            b6 = optimize_batch(model,CountedScorer(LinearScorer()),records,torch.ones(64),config,paths[2],time.monotonic())[-1]
            b4 = next(r for r in after if r['arm']=='B4')
            for key in ['initial','best','final','best_hard_gain','feasible_steps']:
                self.assertEqual(b4[key],b6[key])
            steps=[json.loads(s) for s in (paths[2]/'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len({r['latent_l2'] for r in steps if r['step']>2}),1)


if __name__=='__main__':unittest.main()
