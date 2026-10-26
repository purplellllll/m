import json
import tempfile
import time
import unittest
from pathlib import Path

import torch

from .b6_reproducibility import frozen_config
from .dual_space_ablation import CountedScorer, optimize_batch
from .length_diagnostics import EncoderPool, functional_probe, gradient_summary, length_group, select_records
from .model import CodonVMAE, collate_records
from .test_b6_reproducibility import LinearScorer
from .run_length_diagnostics import SEEDS, update_report


class LengthDiagnosticsTests(unittest.TestCase):
    def test_length_edges_and_deterministic_selection(self):
        self.assertEqual([length_group(n) for n in (300,301,699,700)], ['short','medium','medium','long'])
        records = [dict(sequence_id=str(i),cluster_id=str(i),n_codons=n) for i,n in enumerate((301,201,302,600,701,1001))]
        self.assertEqual(select_records(records,1),select_records(list(reversed(records)),1))

    def test_variance_pool_excludes_pad_and_controls_aa(self):
        tokens = collate_records([dict(cds_dna='ATGGCTGCCTAA'),dict(cds_dna='ATGGCTTAA')])
        mu = torch.zeros(2,4,2)
        mu[tokens == 64] = 1000
        mu[tokens == tokens[0,0],0] = 1
        pool = EncoderPool(2);pool.add(mu,torch.zeros_like(mu),tokens)
        result = pool.summary()
        self.assertEqual(result['valid_tokens'],7)
        self.assertEqual(result['pooled']['active_dimensions']['0.01'],1)
        self.assertEqual(result['within_amino_acid']['active_dimensions']['0.01'],0)
        self.assertEqual(gradient_summary(torch.zeros(4,2))['dims_for_99pct_gradient_energy'],0)
        self.assertEqual(gradient_summary(torch.tensor([[0.,1.],[0.,1.]]))['dims_for_99pct_gradient_energy'],1)

    def test_logging_keeps_search_identical_and_scores_unfiltered(self):
        torch.set_num_threads(2);torch.manual_seed(58)
        model = CodonVMAE(d_model=16,n_heads=2,encoder_layers=1,decoder_layers=1,
            ff_dim=32,latent_dim=4,dropout=0,max_codons=16).eval().requires_grad_(False)
        records = [dict(sequence_id='toy',n_codons=4,cds_dna='ATGGCTGCCTAA')]
        _, config = frozen_config(Path('deliverables/b0_b5_frozen_20261020'),['B5'])
        config.update(steps=4,tuning=True,random_start_rms=.05,random_start_seed=71,
                      diagnostic_mu_rms=1.,max_mutable_hamming=0.)
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False,True):
                out = Path(directory)/str(enabled);out.mkdir()
                config['length_diagnostics'] = enabled
                scorer = CountedScorer(LinearScorer())
                results.append(optimize_batch(model,scorer,records,torch.ones(64),config,out,time.monotonic()))
                self.assertEqual(scorer.forward_sequences,6+4*3)
            for a,b in zip(*results):
                for key in ('arm','initial','best','final','best_hard_gain','feasible_steps'):
                    self.assertEqual(a[key],b[key])
            rows = [json.loads(s) for s in (out/'length_search_steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['actual_z_update_rms'] > 0 for r in rows))
            summary = json.loads((out/'length_search_summary.jsonl').read_text())
            self.assertGreaterEqual(summary['filter_loss'],-1e-7)
            probe = functional_probe(model,CountedScorer(LinearScorer()),records[0],'cpu')
            self.assertEqual(probe['interventions']['zero_high_variance_dims']['channels']+
                             probe['interventions']['zero_low_variance_dims']['channels'],4)
            pool = EncoderPool(4)
            tokens = collate_records(records)
            with torch.no_grad(): mu, logvar = model.encode(tokens,tokens)
            pool.add(mu,logvar,tokens)
            (out/'encoder_summary.json').write_text(json.dumps({g:pool.summary() for g in ('short','medium','long')}))
            (out/'functional_probes.jsonl').write_text(json.dumps(probe)+'\n')
            update_report(out)
            for seed in SEEDS:
                with (out/'search_steps.jsonl').open('a') as f:
                    f.writelines(json.dumps(dict(seed=seed,**r))+'\n' for r in rows)
                with (out/'search_summary.jsonl').open('a') as f:
                    f.write(json.dumps(dict(summary,seed=seed))+'\n')
            update_report(out)
            aggregate = json.loads((out/'summary.json').read_text())
            self.assertEqual(aggregate['search']['short']['complete_cds'],1)


if __name__ == '__main__':
    unittest.main()
