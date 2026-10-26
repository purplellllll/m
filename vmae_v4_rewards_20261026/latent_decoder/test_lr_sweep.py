"""Small CPU experiments validate logging and orchestration, not research performance."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from . import lr_sweep, train
from .model import AA, CodonVMAE, encode_cds


class SweepChecks(unittest.TestCase):
    def test_three_fixed_rates_five_epochs_and_reports(self):
        torch.set_num_threads(2)
        initial = []
        def create_model(**kwargs):
            model = CodonVMAE(**kwargs)
            initial.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
            return model
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = []
            for i, (sequence, split) in enumerate((("ATGGCTGCTTTATTCTAA", "train"),
                    ("ATGGCTTTCTAG", "train"), ("ATGCCAGCTGAACTGTAG", "val"), ("ATGGCTGCCAAACTGTAA", "test"))):
                ids = encode_cds(sequence)
                records.append(dict(cds_dna=sequence, n_codons=len(ids), split=split,
                    protein_sequence="".join(AA[t] for t in ids[:-1]), gene_ids=[str(i)], cluster_id=str(i)))
            (root / "corpus.jsonl").write_text("\n".join(map(json.dumps, records)))
            (root / "audit.json").write_text(json.dumps(dict(source_verified=True,
                homology_certified=False, corpus_sha256=train.sha256(root / "corpus.jsonl"))))
            for label in lr_sweep.RATES:
                cfg = json.loads((lr_sweep.CONFIGS / ("lr_" + label + ".json")).read_text())
                cfg["model"] = dict(d_model=16, n_heads=2, encoder_layers=1, decoder_layers=1,
                                     ff_dim=32, latent_dim=4, dropout=0.1, max_codons=32)
                cfg["training"].update(batch_size=1, gradient_accumulation=2, mask_probability=1.)
                config_path = root / (label + ".json")
                config_path.write_text(json.dumps(cfg))
                run = root / ("lr_" + label)
                argv = ["train", "--data", str(root), "--allow-development-split", "--device", "cpu",
                        "--config", str(config_path), "--output", str(run)]
                with patch("sys.argv", argv), patch.object(train, "CodonVMAE", side_effect=create_model), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                metrics = lr_sweep.read_rows(run / "metrics.jsonl")
                steps = lr_sweep.read_rows(run / "steps.jsonl")
                self.assertEqual([m["epoch"] for m in metrics], [1, 2, 3, 4, 5])
                self.assertTrue(all(m["learning_rate"] == float(label) for m in metrics))
                self.assertTrue(all(s["lr"] == float(label) for s in steps))
                for s in steps:
                    self.assertAlmostEqual(s["loss"], s["full_ce"] + s["masked_ce"] + s["weighted_kl"], places=5)
                    n = s["clean_batches"] + s["masked_batches"]
                    self.assertGreater(n, 0)
                    self.assertEqual(s["masked_batch_fraction"], s["masked_batches"] / n)
                    combined = sum(s[key] * s[count] for key, count in
                                   (("clean_mode_loss", "clean_batches"), ("masked_mode_loss", "masked_batches"))
                                   if s[count]) / n
                    self.assertAlmostEqual(s["loss"], combined, places=6)
                self.assertTrue(all(m["val_loss_fixed_beta"] >= 0 for m in metrics))
                for m in metrics:
                    expected = m["masked_input_full_ce"] + m["masked_mutable_ce"] + cfg["training"]["beta_kl"] * m["masked_input_kl_per_token"]
                    self.assertAlmostEqual(m["val_loss_fixed_beta"], expected)
                (root / ("status_" + label + ".json")).write_text(json.dumps({"status": "completed"}))
            self.assertTrue(all(torch.equal(initial[0], weights) for weights in initial[1:]))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = lr_sweep.summarize(root)
            self.assertTrue(all(s["recorded_epochs"] == 5 for s in summary.values()))
            with (root / "comparison.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 15)
            self.assertGreater((root / "loss_comparison.png").stat().st_size, 1000)
            self.assertGreater((root / "loss_comparison.svg").stat().st_size, 1000)

    def test_failure_does_not_skip_other_rates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "sweep"
            commands = []
            def fake_process(command, **kwargs):
                commands.append(command)
                run = Path(command[command.index("--output") + 1])
                run.mkdir()
                failed = len(commands) == 1
                if not failed:
                    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"epoch": e}) for e in range(1, 6)))
                return Mock(stdout=io.StringIO("synthetic log\n"), wait=Mock(return_value=1 if failed else 0))
            with patch("sys.argv", ["sweep", "--output", str(root)]), patch.object(train, "load_data", return_value=({"train": [1], "val": [1], "test": [1]}, {})), patch.object(train, "sha256", return_value="test-hash"), patch.object(lr_sweep.subprocess, "Popen", side_effect=fake_process), patch.object(lr_sweep, "summarize"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "Some runs failed"):
                    lr_sweep.main()
            self.assertEqual(len(commands), 3)
            for i, label in enumerate(lr_sweep.RATES):
                status = json.loads((root / ("status_" + label + ".json")).read_text())
                self.assertEqual(status["status"], "failed" if i == 0 else "completed")


if __name__ == "__main__":
    unittest.main()
