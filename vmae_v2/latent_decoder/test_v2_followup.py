"""Checks for fair masks/objectives, leakage exclusion, scaling and final gates."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import torch

from .model import CodonVMAE, collate_records, vae_loss
from .v2_followup import (BASE, CHANGES, ScaledVMAE, acceptance, aggregate_rows, combine,
                          evaluate, fixed_mask, load_model, safe_reference, training_loss, train_trial, wait_ready)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.config = dict(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                           ff_dim=32, latent_dim=8, dropout=0., max_codons=900, max_positions=900)
        self.records = [dict(sequence_id=g, n_codons=n+2, cds_dna='ATG'+'GCT'*n+'TAA')
                        for g, n in [('short', 40), ('medium', 320), ('long', 700)]]

    def test_scaling_preserves_starting_function_and_survives_reload(self):
        original = CodonVMAE(**self.config).eval()
        with tempfile.TemporaryDirectory() as directory:
            initial, saved = Path(directory)/'initial.pt', Path(directory)/'scaled.pt'
            torch.save(dict(model=original.state_dict(), config=dict(model=self.config)), initial)
            recipe = dict(BASE, gain=2.)
            scaled, _ = load_model(initial, recipe, initialize=True, device='cpu')
            scaled.eval()
            torch.save(dict(model=scaled.state_dict(), config=dict(model=self.config), recipe=recipe), saved)
            restored, _ = load_model(saved, device='cpu')
            restored.eval()
            with torch.no_grad():
                tokens = collate_records(self.records[:1])
                expected = original(tokens, tokens, sample=False)[0]
                self.assertTrue(torch.allclose(expected, scaled(tokens, tokens, sample=False)[0], atol=1e-6))
                self.assertTrue(torch.allclose(expected, restored(tokens, tokens, sample=False)[0], atol=1e-6))

    def test_paired_masks_and_regularizer_cannot_redefine_evaluation_loss(self):
        model = ScaledVMAE(**self.config)
        tokens = collate_records(self.records[:1])
        inputs, selected = fixed_mask(tokens, 'same', 173)
        other, masked = fixed_mask(tokens, 'same', 173)
        self.assertTrue(torch.equal(inputs, other) and torch.equal(selected, masked))
        logits, mu, logvar = model(inputs, tokens, sample=False)
        ordinary, _ = vae_loss(logits, tokens, mu, logvar, selected, beta=BASE['beta'])
        changed, _ = training_loss(logits, tokens, mu, logvar, selected, BASE)
        self.assertTrue(torch.allclose(ordinary, changed, atol=1e-6))
        result = evaluate(model, self.records, probes=True)
        for metrics in result['by_length'].values():
            self.assertAlmostEqual(metrics['loss'], metrics['ce']+metrics['full_ce']+.001*metrics['kl'])
            self.assertEqual(len(metrics['latent']['encoder']['pooled']['variance_by_dim']), 8)
            self.assertEqual(len(metrics['latent']['interventions']), 1)
        aggregate = aggregate_rows(result['paired_rows'])
        self.assertEqual(result['macro'], aggregate['macro'])

    def test_reference_excludes_whole_homology_groups_and_validation(self):
        rows = [dict(sequence_id=str(i), split=s) for i, s in enumerate(('train', 'test', 'val', 'test', 'test'))]
        manifest = dict(assignments={str(i): dict(group=g) for i, g in enumerate((0, 0, 1, 1, 2))})
        self.assertEqual(safe_reference(rows, manifest), {'4'})

    def test_gate_rejects_one_bad_length_and_combine_uses_only_screen(self):
        def metrics(ce, loss):
            return dict(macro=dict(ce=ce, loss=loss),
                        by_length={g: dict(ce=ce, loss=loss) for g in ('short', 'medium', 'long')})
        reference, better = metrics(1., 2.), metrics(.9, 1.9)
        self.assertTrue(acceptance(better, reference)['passed'])
        better['by_length']['long']['ce'] = 1.01
        self.assertFalse(acceptance(better, reference)['passed'])
        screen = {n: dict(best=dict(metrics=metrics(1., 2.))) for n in CHANGES}
        screen['long_fast']['best']['metrics']['macro']['loss'] = 1.8
        screen['kl_low']['best']['metrics']['macro']['loss'] = 1.7
        selected = combine(screen)
        self.assertEqual(selected['profile'], 'long_fast')
        self.assertEqual(selected['beta'], .0003)
        self.assertEqual(selected['gain'], 1.)

    def test_queue_requires_all_fifteen_completions_and_successful_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'deployment').mkdir()
            status = root/'deployment'/'status.json'
            status.write_text(json.dumps(dict(stage='completed')))
            markers = [root/f'fold_{f}'/p/'complete.json'
                       for f in range(5) for p in ('uniform', 'short_fast', 'long_fast')]
            for marker in markers[:-1]:
                marker.parent.mkdir(parents=True)
                marker.write_text('{}')
            with patch('latent_decoder.v2_followup.time.sleep', side_effect=RuntimeError('still waiting')):
                with self.assertRaisesRegex(RuntimeError, 'still waiting'):
                    wait_ready(root)
            markers[-1].parent.mkdir(parents=True)
            markers[-1].write_text('{}')
            wait_ready(root)
            status.write_text(json.dumps(dict(stage='failed')))
            with self.assertRaisesRegex(RuntimeError, 'Upstream training failed'):
                wait_ready(root)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA resume smoke test runs on the server')
    def test_cuda_worker_checkpoint_resume_and_complete(self):
        model = CodonVMAE(**self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'folds.json').write_text('{}')
            initial = root/'initial.pt'
            torch.save(dict(model=model.state_dict(), config=dict(model=self.config)), initial)
            split = dict(train=self.records, val=self.records)
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('simulated interruption')
                return evaluate(*args, **kwargs)
            with patch('latent_decoder.v2_followup.corpus', return_value=split), \
                 patch('latent_decoder.v2_followup.pick', side_effect=lambda records, *_: records):
                with patch('latent_decoder.v2_followup.evaluate', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        train_trial(root, initial, root/'trial', 0, BASE, 'final', 2)
                state = torch.load(root/'trial'/'last.pt', map_location='cpu', weights_only=False)
                self.assertEqual(state['epoch'], 1)
                del state
                done = train_trial(root, initial, root/'trial', 0, BASE, 'final', 2)
                self.assertTrue((root/'trial'/'best.pt').exists())
                self.assertFalse((root/'trial'/'last.pt').exists())
                self.assertIn(done['best']['epoch'], (1, 2))
                self.assertEqual(done, train_trial(root, initial, root/'trial', 0, BASE, 'final', 2))


if __name__ == '__main__':
    unittest.main()
