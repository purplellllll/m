"""Synthetic software checks only; no research dataset is used for training."""
import ast
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .model import (AA, CODONS, MASK, PAD, TOKEN, CodonVMAE, codon_probabilities,
                    collate_records, corrupt, encode_cds, mutable_positions,
                    nucleotide_probabilities, vae_loss)
from .train import evaluate, load_data, preflight, sha256
from . import train


class DecoderChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        self.records = [{"cds_dna": "ATGGCTGCCGCGTTCTTATGGAAATAG"},
                        {"cds_dna": "ATGGCCGCATTTCCTTAA"}]
        self.tokens = collate_records(self.records)
        self.model = CodonVMAE(d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1,
                               ff_dim=64, latent_dim=8, dropout=0, max_codons=32).eval()

    def test_baseline_vocabulary_and_code(self):
        original = json.loads(Path(__file__).with_name("baseline_codon_order.json").read_text())
        source = Path(__file__).resolve().parents[1] / "dataset.py"
        if source.exists():
            tree = ast.parse(source.read_text(encoding="utf-8"))
            node = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "my_vocab" for t in n.targets))
            self.assertEqual(original, ast.literal_eval(node.value))
        self.assertTrue(all(original[codon] == index for index, codon in enumerate(CODONS)))
        self.assertEqual(len(CODONS), 64)
        self.assertEqual(len(AA), 64)
        self.assertEqual(AA[TOKEN["ATG"]], "M")
        self.assertEqual(AA[TOKEN["TGG"]], "W")
        self.assertEqual({CODONS[i] for i, a in enumerate(AA) if a == "*"}, {"TAA", "TAG", "TGA"})

    def test_bad_cds_rejected(self):
        for sequence in ["ATGA", "ATGNNNTAA", "ATGTAGGCTTAA", "GCTGCTTAA", "ATGGCT"]:
            with self.assertRaises(ValueError):
                encode_cds(sequence)

    def test_mask_and_mapping_and_gradient(self):
        corrupted, selected = corrupt(self.tokens, 1.0)
        self.assertTrue(torch.equal(selected, mutable_positions(self.tokens)))
        self.assertTrue(torch.all(corrupted[selected] == MASK))
        logits, mu, logvar = self.model(corrupted, self.tokens)
        probs = codon_probabilities(logits, self.tokens, temperature=0.1)
        valid = self.tokens < PAD
        self.assertTrue(torch.allclose(probs.sum(-1)[valid], torch.ones_like(probs.sum(-1)[valid])))
        self.assertEqual(probs[~valid].abs().sum().item(), 0)
        predicted = probs.argmax(-1)
        for row, col in valid.nonzero().tolist():
            self.assertEqual(AA[predicted[row, col]], AA[self.tokens[row, col]])
        onehot = torch.nn.functional.one_hot(self.tokens.clamp_max(63), 64).float() * valid.unsqueeze(-1)
        nt = nucleotide_probabilities(onehot)
        for row, record in enumerate(self.records):
            length = len(record["cds_dna"])
            decoded = "".join("ACGT"[i] for i in nt[row, :length].argmax(-1).tolist())
            self.assertEqual(decoded, record["cds_dna"])
        loss, terms = vae_loss(logits, self.tokens, mu, logvar, selected, beta=0.001)
        loss.backward()
        for module in [self.model.encoder, self.model.mu, self.model.logvar, self.model.decoder]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(torch.isfinite(x) for x in terms.values()))
        z = mu.detach().requires_grad_(True)
        nt = nucleotide_probabilities(codon_probabilities(self.model.decode(z, self.tokens), self.tokens))
        (nt * torch.linspace(0, 1, nt.numel()).reshape_as(nt)).sum().backward()
        self.assertGreater(z.grad[valid].abs().sum().item(), 0)
        self.assertEqual(z.grad[~valid].abs().sum().item(), 0)

    def test_padding_and_serialization(self):
        tokens = self.tokens[:1]
        extra = torch.nn.functional.pad(tokens, (0, 3), value=PAD)
        a = self.model(tokens, tokens, sample=False)[0]
        b = self.model(extra, extra, sample=False)[0][:, :tokens.size(1)]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.pt"
            torch.save(self.model.state_dict(), path)
            self.model.load_state_dict(torch.load(path, weights_only=True))
        c = self.model(tokens, tokens, sample=False)[0]
        self.assertTrue(torch.equal(a, c))

    def test_preflight_preserves_weights_and_rng(self):
        records = [{**r, "n_codons": len(r["cds_dna"])//3} for r in self.records]
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = preflight(self.model, records, torch.device("cpu"), 2, 0.15)
        self.assertEqual(result["event"], "preflight_pass")
        self.assertEqual(result["max_codons"], max(r["n_codons"] for r in records))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in before.items()))

    def test_evaluation_and_leakage_guard(self):
        records = [{**r, "n_codons": len(r["cds_dna"])//3} for r in self.records]
        result = evaluate(self.model, records, torch.ones(64), torch.device("cpu"), 2, 1.0)
        self.assertEqual(result["protein_error_rate"], 0)
        self.assertGreater(result["masked_sites"], 0)
        self.assertFalse(result["decoder_pilot_gate"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/"audit.json").write_text(json.dumps({"source_verified": True,"homology_certified": False}))
            with self.assertRaisesRegex(ValueError, "clustering"):
                load_data(path)
            repeated = {"cds_dna": self.records[0]["cds_dna"], "n_codons": 9,
                        "gene_ids": ["gene1"], "cluster_id": "cluster1"}
            repeated["protein_sequence"] = "".join(AA[i] for i in encode_cds(repeated["cds_dna"])[:-1])
            (path/"corpus.jsonl").write_text("\n".join(json.dumps({**repeated,"split": split})
                                                     for split in ["train","val","test"]))
            (path/"audit.json").write_text(json.dumps({"source_verified": True, "homology_certified": False,
                "corpus_sha256": sha256(path/"corpus.jsonl")}))
            with self.assertRaisesRegex(ValueError, "Duplicate CDS"):
                load_data(path, True)
            with (path/"corpus.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(path, True)

    def test_synthetic_checkpoint_resume(self):
        config = json.loads((Path(__file__).with_name("config.json")).read_text())
        config["model"] = dict(d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1,
                               ff_dim=64, latent_dim=8, dropout=0.1, max_codons=32)
        config["training"].update(epochs=2, batch_size=1, gradient_accumulation=2,
                                  mask_probability=1.0, clean_batch_probability=0.0)
        records = []
        for index, (sequence, split) in enumerate([
                (self.records[0]["cds_dna"], "train"), (self.records[1]["cds_dna"], "train"),
                ("ATGGCTGCCAAACTGTAA", "val"), ("ATGCCAGCTGAACTGTAG", "test")]):
            records.append(dict(cds_dna=sequence, n_codons=len(sequence)//3, split=split,
                protein_sequence="".join(AA[i] for i in encode_cds(sequence)[:-1]),
                gene_ids=[str(index)], cluster_id=str(index)))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"config.json").write_text(json.dumps(config))
            (root/"corpus.jsonl").write_text("\n".join(map(json.dumps, records)))
            (root/"audit.json").write_text(json.dumps(dict(source_verified=True,
                homology_certified=False, corpus_sha256=sha256(root/"corpus.jsonl"))))

            def run(output, *extra):
                args = ["train", "--data", str(root), "--config", str(root/"config.json"),
                        "--output", str(root/output), "--device", "cpu", "--allow-development-split", *extra]
                with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            run("reference")
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt("Synthetic interruption")
                return evaluate(*args, **kwargs)

            with patch.object(train, "evaluate", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run("resumed")
            run("resumed", "--resume", str(root/"resumed"/"last.pt"))
            reference = torch.load(root/"reference"/"last.pt", weights_only=False)
            resumed = torch.load(root/"resumed"/"last.pt", weights_only=False)
            self.assertEqual(reference["step"], resumed["step"])
            for key in reference["model"]:
                self.assertTrue(torch.equal(reference["model"][key], resumed["model"][key]), key)
            self.assertEqual(len((root/"resumed"/"metrics.jsonl").read_text().splitlines()), 2)
            reference_metrics = [json.loads(line) for line in (root/"reference"/"metrics.jsonl").read_text().splitlines()]
            resumed_metrics = [json.loads(line) for line in (root/"resumed"/"metrics.jsonl").read_text().splitlines()]
            for a, b in zip(reference_metrics, resumed_metrics):
                for key in ("train_loss", "train_full_ce", "train_masked_ce", "train_kl_per_token", "val_loss_fixed_beta"):
                    self.assertEqual(a[key], b[key], key)
            self.assertEqual((root/"reference"/"steps.jsonl").read_text(), (root/"resumed"/"steps.jsonl").read_text())
            run("test.json", "--evaluate", str(root/"resumed"/"best.pt"), "--split", "test")
            self.assertEqual(json.loads((root/"test.json").read_text())["split"], "test")
            with self.assertRaises(FileExistsError):
                run("resumed")
            calls = 0
            with patch.object(train, "evaluate", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run("extended")
            first = torch.load(root/"extended"/"last.pt", weights_only=False)
            shutil.copytree(root/"extended", root/"extension_reference")
            run("extension_reference", "--resume", str(root/"extension_reference"/"last.pt"), "--extend-epochs", "4")
            calls = 0
            with patch.object(train, "evaluate", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run("extended", "--resume", str(root/"extended"/"last.pt"), "--extend-epochs", "4")
            run("extended", "--config", str(root/"extended"/"config.json"),
                "--resume", str(root/"extended"/"last.pt"))
            expected = torch.load(root/"extension_reference"/"last.pt", weights_only=False)
            extended = torch.load(root/"extended"/"last.pt", weights_only=False)
            self.assertEqual(extended["config"]["training"]["epochs"], 4)
            self.assertEqual(extended["next_epoch"], 4)
            self.assertEqual(extended["step"], 4)
            self.assertEqual(extended["lr_extension"]["start_step"], first["step"])
            self.assertEqual(extended["kl_warmup_steps"], 0.4)
            for key in expected["model"]:
                self.assertTrue(torch.equal(expected["model"][key], extended["model"][key]), key)
            steps = [json.loads(s) for s in (root/"extended"/"steps.jsonl").read_text().splitlines()]
            self.assertEqual(steps[1]["lr"], first["optimizer"]["param_groups"][0]["lr"])
            self.assertGreater(steps[1]["lr"], steps[-1]["lr"])
            self.assertEqual(len((root/"extended"/"metrics.jsonl").read_text().splitlines()), 4)
            self.assertEqual((root/"extended"/"steps.jsonl").read_text(), (root/"extension_reference"/"steps.jsonl").read_text())
            previous_config = (root/"resumed"/"config.json").read_text()
            with self.assertRaisesRegex(ValueError, "reached zero"):
                run("resumed", "--resume", str(root/"resumed"/"last.pt"), "--extend-epochs", "4")
            self.assertEqual((root/"resumed"/"config.json").read_text(), previous_config)

    def test_extended_schedule_continuity(self):
        factor = train.learning_rate_factor
        extension = dict(start_step=200, end_step=500, start_factor=0.03)
        self.assertEqual(factor(200, "warmup_cosine", 50, 500, extension), 0.03)
        self.assertEqual(factor(500, "warmup_cosine", 50, 500, extension), 0.)
        values = [factor(s, "warmup_cosine", 50, 500, extension) for s in range(200, 501)]
        self.assertTrue(all(a >= b >= 0 for a, b in zip(values, values[1:])))


if __name__ == "__main__":
    unittest.main()
