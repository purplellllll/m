"""CPU-only checks; RNAfold is mocked explicitly, never represented as a real fold."""
import hashlib
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from latent_decoder.analyze_random_validation import sequence_metrics
from latent_decoder.tokenizer import AA, CODONS, PAD, TOKEN, encode_cds
from tools import vmae_multiobjective_core_20261026 as core


def tokens(dna):
    return torch.tensor([encode_cds(dna)], dtype=torch.long)


def fake_fold(dna, cache):
    key = hashlib.sha256(dna.encode()).hexdigest()
    per_nt = -.1 if dna[3:6] == 'TTC' else -.3
    if key not in cache:
        cache[key] = dict(sequence_sha256=key, mfe=per_nt * len(dna), mfe_per_nt=per_nt,
                          viennarna_version='2.7.2', temperature_celsius=37., dangles=2)
    return dict(cache[key])


class MultiobjectiveTests(unittest.TestCase):
    def setUp(self):
        self.weights = {c: .1 + .9 * (i + 1) / 64 for i, c in enumerate(CODONS)}

    def test_hard_properties_match_existing_metrics_with_padding(self):
        originals = ['ATGTTTGCTTAA', 'ATGTAA']
        candidates = ['ATGTTCGCCTAA', 'ATGTAA']
        target = torch.full((2, 5), PAD, dtype=torch.long)
        hard = torch.zeros((2, 5, 64), dtype=torch.float64)
        for i, (original, candidate) in enumerate(zip(originals, candidates)):
            source, proposed = encode_cds(original), encode_cds(candidate)
            target[i, :len(source)] = torch.tensor(source)
            hard[i, :len(source)] = F.one_hot(torch.tensor(proposed), 64).double()
        # Nonzero PAD distributions must never contribute to a property or its gradient.
        hard[target == PAD] = 1 / 64
        result = core.property_penalties(hard, target, self.weights)
        for i, (original, candidate) in enumerate(zip(originals, candidates)):
            old, new = sequence_metrics(original, self.weights), sequence_metrics(candidate, self.weights)
            self.assertAlmostEqual(result['gc_fraction'][i].item(), new['gc_fraction'], places=14)
            self.assertAlmostEqual(result['log_cai'][i].item(), math.log(new['train_reference_cai']), places=14)
            expected_gc = max(0, abs(new['gc_fraction'] - old['gc_fraction']) - .05)**2 / .05**2
            expected_cai = max(0, math.log(old['train_reference_cai']) - math.log(new['train_reference_cai']))**2 / .05**2
            self.assertAlmostEqual(result['gc_penalty'][i].item(), expected_gc, places=12)
            self.assertAlmostEqual(result['cai_penalty'][i].item(), expected_cai, places=12)
        self.assertTrue(all(value.shape == (2,) for value in result.values()))

    def test_start_in_cai_stop_excluded_and_gc_includes_stop(self):
        target = tokens('ATGTAA')
        hard = F.one_hot(target, 64).double()
        self.weights['ATG'] = .25
        for codon in ('TAA', 'TAG', 'TGA'):
            self.weights[codon] = 0
        result = core.property_penalties(hard, target, self.weights)
        self.assertAlmostEqual(result['log_cai'].item(), math.log(.25))
        self.assertAlmostEqual(result['gc_fraction'].item(), 1 / 6)

    def test_straight_through_gradients_and_pad_zero(self):
        target = torch.tensor([encode_cds('ATGTTTTAA') + [PAD]])
        logits = torch.zeros((1, 4, 64), dtype=torch.float64, requires_grad=True)
        soft = logits.softmax(-1)
        candidate = F.one_hot(torch.tensor([[TOKEN['ATG'], TOKEN['TTC'], TOKEN['TAA'], 0]]), 64).double()
        st = candidate + soft - soft.detach()
        self.weights['TTT'], self.weights['TTC'] = 1., .1
        result = core.property_penalties(st, target, self.weights)
        (result['gc_penalty'] + result['cai_penalty']).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[:, :3].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 3].abs().sum().item(), 0)

    def test_invalid_probabilities_and_padding_rejected(self):
        target = tokens('ATGTTTTAA')
        p = F.one_hot(target, 64).float()
        for bad in (p * .5, p * float('nan'), p - .1):
            with self.assertRaises(ValueError): core.property_penalties(bad, target, self.weights)
        broken = torch.tensor([[TOKEN['ATG'], PAD, TOKEN['TAA']]])
        with self.assertRaises(ValueError): core.property_penalties(p, broken, self.weights)
        self.weights['ATG'] = 0
        with self.assertRaises(ValueError): core.property_penalties(p, target, self.weights)

    def test_mfe_negative_sign_and_finite_boundary(self):
        self.assertAlmostEqual(core.mfe_penalty(-.1, -.2), 4.)
        self.assertEqual(core.mfe_penalty(-.3, -.2), 0.)
        self.assertEqual(core.mfe_penalty(-.2, -.2), 0.)
        with self.assertRaises(ValueError): core.mfe_penalty(float('nan'), -.2)

    def test_exact_fold_protocol_cache_and_immutability(self):
        calls = []
        def compound(dna, settings):
            calls.append((dna, settings.temperature, settings.dangles))
            return types.SimpleNamespace(mfe=lambda: ('.' * len(dna), -1.5))
        fake_rna = types.SimpleNamespace(__version__='2.7.2',
            md=lambda: types.SimpleNamespace(temperature=99, dangles=0), fold_compound=compound)
        cache = {}
        with patch.dict(sys.modules, {'RNA': fake_rna}):
            first = core.exact_fold('ATGTTTTAA', cache)
            first['mfe'] = 999
            second = core.exact_fold('ATGTTTTAA', cache)
        self.assertEqual(calls, [('AUGUUUUAA', 37., 2)])
        self.assertEqual(second['mfe'], -1.5)
        self.assertAlmostEqual(second['mfe_per_nt'], -1.5 / 9)
        key = next(iter(cache))
        for field, invalid in [('viennarna_version', '2.6.0'), ('temperature_celsius', 25),
                               ('dangles', 0), ('sequence_sha256', 'wrong'), ('mfe_per_nt', 100)]:
            original = cache[key][field]
            cache[key][field] = invalid
            with self.assertRaises(ValueError): core.exact_fold('ATGTTTTAA', cache)
            cache[key][field] = original
        fake_rna.__version__ = '2.6.0'
        with patch.dict(sys.modules, {'RNA': fake_rna}), self.assertRaises(RuntimeError):
            core.exact_fold('ATGTTCTAA', cache)

    def test_scst_gradient_sign_and_stop_lock(self):
        target = tokens('ATGTTTTAA')
        for worse in (True, False):
            logits = torch.zeros((1, 3, 64), dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                logits[0, 1, TOKEN['TTT']] = math.log(3)
                logits[0, -1, TOKEN['TAG']] = 8
            p = logits.softmax(-1)
            sampled = torch.tensor([TOKEN['ATG'], TOKEN['TTC'], TOKEN['TAA']])
            def fold(dna, cache):
                row = fake_fold(dna, cache)
                if not worse:
                    row['mfe_per_nt'] = -.4 - row['mfe_per_nt']
                    row['mfe'] = row['mfe_per_nt'] * len(dna)
                    cache[row['sequence_sha256']] = dict(row)
                return row
            with patch.object(core, 'exact_fold', fold), patch('torch.distributions.Categorical.sample', return_value=sampled):
                loss, diag = core.scst_loss(p, target, -.3, {})
            loss.backward()
            self.assertEqual(diag['sample_dna'], 'ATGTTCTAA')
            self.assertEqual(diag['greedy_dna'], 'ATGTTTTAA')
            self.assertEqual(diag['fold_calls'], 2)
            self.assertEqual(diag['mutable_sense_positions'], 1)
            self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
            self.assertEqual(logits.grad[0, -1].abs().sum().item(), 0)
            self.assertGreater(logits.grad[0, 1, TOKEN['TTC']].item() * (1 if worse else -1), 0)
            self.assertEqual([AA[i] for i in encode_cds(diag['sample_dna'])], [AA[i] for i in target[0]])

    def test_scst_deterministic_met_and_cache(self):
        target = tokens('ATGATGTAA')
        p = torch.full((3, 64), 1 / 64, dtype=torch.float64, requires_grad=True)
        cache = {}
        with patch.object(core, 'exact_fold', fake_fold):
            loss, diag = core.scst_loss(p, target[0], {'mfe_per_nt': -.3}, cache)
            again, second = core.scst_loss(p, target[0], -.3, cache)
        loss.backward()
        self.assertEqual(diag['sample_dna'], 'ATGATGTAA')
        self.assertEqual(diag['fold_calls'], 1)
        self.assertEqual(second['fold_calls'], 0)
        self.assertEqual(diag['advantage'], 0)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(p.grad.abs().sum().item(), 0)

    def test_scst_rejects_pad_batch_and_absent_synonym_mass(self):
        target = tokens('ATGTTTTAA')
        p = F.one_hot(target, 64).float()
        with self.assertRaises(ValueError): core.scst_loss(p.repeat(2, 1, 1), target.repeat(2, 1), -.3, {})
        padded = torch.cat([target, torch.tensor([[PAD]])], -1)
        with self.assertRaises(ValueError): core.scst_loss(torch.cat([p, p[:, :1]], 1), padded, -.3, {})
        wrong = p.clone();wrong[0, 1] = F.one_hot(torch.tensor(TOKEN['GCC']), 64)
        with self.assertRaises(ValueError): core.scst_loss(wrong, target, -.3, {})


if __name__ == '__main__':
    unittest.main()
