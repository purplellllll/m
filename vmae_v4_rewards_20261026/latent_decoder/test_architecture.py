"""Architecture contract checks; synthetic inputs do not certify research quality."""
import json
from pathlib import Path
import unittest

import torch

from .model import (AA_IDS, MASK, PAD, TOKEN, CodonVMAE, codon_probabilities,
                    collate_records, corrupt, mutable_positions, nucleotide_probabilities, vae_loss)


class ArchitectureChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.tokens = collate_records([{"cds_dna": "ATGGCTGCTTTATTCTAA"},
                                       {"cds_dna": "ATGGCTTTCTAG"}])
        self.model = CodonVMAE(d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1,
                               ff_dim=64, latent_dim=8, dropout=0, max_codons=32).eval()

    @torch.no_grad()
    def test_research_and_mvp_shapes_at_max_length(self):
        for filename in ("config.json", "config_mvp.json"):
            with self.subTest(config=filename):
                config = json.loads(Path(__file__).with_name(filename).read_text())["model"]
                model = CodonVMAE(**config).eval()
                sequence = "ATG" + "GCT" * (config["max_codons"] - 2) + "TAA"
                tokens = collate_records([{"cds_dna": sequence}])
                seen = {}
                def capture(name):
                    def hook(module, inputs, output):
                        seen[name] = tuple(output.shape)
                    return hook
                handles = [module.register_forward_hook(capture(name)) for name, module in
                           (("encoder", model.encoder), ("decoder", model.decoder))]
                logits, mu, logvar = model(tokens, tokens)
                for handle in handles:
                    handle.remove()
                self.assertEqual(seen["encoder"], (1, config["max_codons"], config["d_model"]))
                self.assertEqual(seen["decoder"], seen["encoder"])
                self.assertEqual(mu.shape, (1, config["max_codons"], config["latent_dim"]))
                self.assertEqual(mu.shape, logvar.shape)
                self.assertEqual(logits.shape, (1, config["max_codons"], 64))
                self.assertTrue(all(torch.isfinite(x).all() for x in (logits, mu, logvar)))
                short = self.tokens[:1]
                expected = model(short, short, sample=False)[0]
                padded = torch.nn.functional.pad(short, (0, 3), value=PAD)
                actual = model(padded, padded, sample=False)[0][:, :short.size(1)]
                self.assertTrue(torch.allclose(expected, actual, atol=2e-5))
                overlong = torch.nn.functional.pad(tokens, (0, 1), value=PAD)
                with self.assertRaisesRegex(ValueError, "truncation"):
                    model(overlong, overlong)

    def test_rope_relative_positions_and_bidirectional_attention(self):
        attention = self.model.encoder.layers[0].attention
        q = torch.randn(1, 4, 8, 8)
        k = torch.randn_like(q)
        rotated_q = attention.rotate(q)
        self.assertTrue(torch.allclose(q.square().sum(-1), rotated_q.square().sum(-1), atol=2e-6))
        score = (rotated_q[:, :, 1] * attention.rotate(k)[:, :, 4]).sum(-1)
        shifted_q = torch.cat((torch.zeros_like(q[:, :, :2]), q[:, :, :-2]), 2)
        shifted_k = torch.cat((torch.zeros_like(k[:, :, :2]), k[:, :, :-2]), 2)
        shifted_score = (attention.rotate(shifted_q)[:, :, 3] * attention.rotate(shifted_k)[:, :, 6]).sum(-1)
        self.assertTrue(torch.allclose(score, shifted_score, atol=2e-6))
        hidden = torch.randn(1, 6, 32, requires_grad=True)
        attention(hidden, torch.zeros(1, 6, dtype=torch.bool))[0, 0].square().sum().backward()
        self.assertGreater(hidden.grad[0, -1].abs().sum().item(), 0)

    @torch.no_grad()
    def test_explicit_aa_condition_without_codon_answer_leak(self):
        synonymous = self.tokens.clone()
        synonymous[synonymous == TOKEN["GCT"]] = TOKEN["GCC"]
        corrupted, _ = corrupt(self.tokens, 1.0)
        original = self.model.encode(corrupted, self.tokens)
        replaced = self.model.encode(corrupted, synonymous)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, replaced)))
        z = torch.randn(*self.tokens.shape, 8)
        logits = self.model.decode(z, self.tokens)
        self.assertTrue(torch.equal(logits, self.model.decode(z, synonymous)))
        changed_aa = self.tokens.clone()
        changed_aa[0, 1] = TOKEN["AAA"]
        self.assertFalse(torch.allclose(logits, self.model.decode(z, changed_aa)))
        self.assertTrue(torch.isfinite(logits).all())

    def test_frozen_dual_space_gradients_and_degenerations(self):
        self.model.requires_grad_(False)
        z = torch.randn(*self.tokens.shape, 8, requires_grad=True)
        logits = self.model.decode(z, self.tokens)
        delta = torch.randn_like(logits, requires_grad=True)
        probs = codon_probabilities(logits, self.tokens, delta=delta, alpha=0.5)
        nt = nucleotide_probabilities(probs)
        (nt * torch.randn_like(nt)).sum().backward()
        mutable = mutable_positions(self.tokens)
        for gradient in (z.grad, delta.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[mutable].abs().sum(-1) > 0).all())
            self.assertEqual(gradient[self.tokens == PAD].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        latent = codon_probabilities(logits, self.tokens)
        self.assertTrue(torch.equal(latent, codon_probabilities(logits, self.tokens, delta=delta, alpha=0)))
        self.assertTrue(torch.equal(latent, codon_probabilities(logits, self.tokens, delta=torch.zeros_like(delta))))
        direct = codon_probabilities(torch.zeros_like(logits), self.tokens, delta=delta.detach())
        self.assertTrue(torch.equal(direct, codon_probabilities(delta.detach(), self.tokens)))
        shared = delta.detach()[0]
        self.assertTrue(torch.equal(codon_probabilities(logits, self.tokens, delta=shared),
                                    codon_probabilities(logits + shared, self.tokens)))
        # Arbitrarily large illegal residuals still cannot change the target protein.
        forced = codon_probabilities(logits, self.tokens, delta=torch.randn_like(logits) * 1e6)
        valid = self.tokens != PAD
        self.assertTrue(torch.equal(AA_IDS[forced.argmax(-1)][valid], AA_IDS[self.tokens][valid]))
        self.assertEqual(forced[~valid].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(forced).all())

    def test_kl_valid_length_normalization_and_clean_loss(self):
        logits = torch.zeros(*self.tokens.shape, 64, requires_grad=True)
        mu = torch.ones(*self.tokens.shape, 8)
        mu[self.tokens == PAD] = 100
        loss, terms = vae_loss(logits, self.tokens, mu, torch.zeros_like(mu),
                               torch.zeros_like(self.tokens, dtype=torch.bool), beta=0.1, full_weight=0)
        self.assertAlmostEqual(loss.item(), 0.4, places=6)
        self.assertEqual(terms["kl_per_token"].item(), 4)
        self.assertEqual(terms["kl_per_dimension"].item(), 0.5)
        self.assertEqual(terms["masked_ce"].item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_rows_and_shapes_rejected(self):
        for bad in (torch.full_like(self.tokens, PAD), torch.full_like(self.tokens, MASK),
                    torch.full_like(self.tokens, -1), self.tokens.float(), self.tokens[:, :0]):
            with self.subTest(tokens=bad), self.assertRaises(ValueError):
                self.model(bad, bad)
            with self.assertRaises(ValueError):
                codon_probabilities(torch.zeros(*bad.shape, 64), bad)
        with self.assertRaises(ValueError):
            self.model.decode(torch.randn(*self.tokens.shape, 7), self.tokens)
        with self.assertRaises(ValueError):
            codon_probabilities(torch.zeros(*self.tokens.shape, 64), self.tokens, temperature=float("nan"))
        with self.assertRaises(ValueError):
            codon_probabilities(torch.zeros(*self.tokens.shape, 64), self.tokens, delta=torch.zeros(64))


if __name__ == "__main__":
    unittest.main()
