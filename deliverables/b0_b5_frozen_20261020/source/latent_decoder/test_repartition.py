"""Pure-stdlib synthetic checks; no real-data clustering or model training."""
import json
from pathlib import Path
import tempfile
import unittest

from .prepare_data import digest_file, translate
from .repartition import load_prepared, repartition


class RepartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "input"
        self.data.mkdir()
        self.records = []
        for identifier, cds, gene, split in (
                ("a", "ATGGCTTAA", "g1", "test"),
                ("b", "ATGGATTAA", "g2", "train"),
                ("c", "ATGGAATAA", "g2", "train"),
                ("d", "ATGGCCTAA", "g3", "test"),
                ("e", "ATGTTTTAA", "g4", "val")):
            self.records.append({"sequence_id": identifier, "cds_dna": cds,
                "protein_sequence": translate(cds)[0], "gene_ids": [gene],
                "transcript_ids": ["tx_" + identifier], "n_codons": 3,
                "split": split, "cluster_id": "old_" + gene, "source": "synthetic-test"})
        self.write_input()
        self.clusters = self.base / "clusters.tsv"
        self.clusters.write_text("a\ta\na\tb\nc\tc\nd\td\ne\te\n", encoding="utf-8")

    def write_input(self):
        (self.data / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")
        (self.data / "proteins.fasta").write_text("".join(">" + r["sequence_id"] + "\n" + r["protein_sequence"] + "\n" for r in self.records), encoding="utf-8")
        counts = {s: sum(r["split"] == s for r in self.records) for s in ("train", "val", "test")}
        self.audit = {"source_verified": True, "homology_certified": False,
            "corpus_sha256": digest_file(self.data / "corpus.jsonl"),
            "proteins_sha256": digest_file(self.data / "proteins.fasta"),
            "counts": counts, "min_codons": 1, "max_codons": 10,
            "unique_cds": len(self.records), "source_metadata": "preserved"}
        (self.data / "audit.json").write_text(json.dumps(self.audit), encoding="utf-8")

    def test_test_protection_propagates_through_cluster_gene_and_protein(self):
        output = self.base / "result"
        audit = repartition(self.data, self.clusters, output)
        records, _ = load_prepared(output)
        splits = {r["sequence_id"]: r["split"] for r in records}
        self.assertTrue(all(splits[s] == "test" for s in "abcd"))
        self.assertEqual(audit["original_audit"], self.audit)
        self.assertFalse(audit["homology_search_audited"])
        self.assertEqual(audit["test_records_added_by_components"], 2)
        for split in ("train", "val", "test"):
            self.assertTrue((output / (split + "_proteins.fasta")).exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            repartition(self.data, self.clusters, output)

    def test_missing_cluster_member_rejected(self):
        self.clusters.write_text("a\ta\nb\tb\nc\tc\nd\td\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cover every sequence"):
            repartition(self.data, self.clusters, self.base / "result")
        self.assertFalse((self.base / "result").exists())

    def test_duplicate_member_rejected_even_if_same_cluster(self):
        with self.clusters.open("a", encoding="utf-8") as handle:
            handle.write("a\tb\n")
        with self.assertRaisesRegex(ValueError, "Duplicate cluster member"):
            repartition(self.data, self.clusters, self.base / "result")

    def test_input_tampering_rejected(self):
        for filename in ("corpus.jsonl", "proteins.fasta"):
            with self.subTest(filename=filename):
                self.write_input()
                with (self.data / filename).open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    repartition(self.data, self.clusters, self.base / "result")

    def test_invalid_record_rejected_despite_updated_checksums(self):
        self.records[0]["protein_sequence"] = "WRONG"
        self.write_input()
        with self.assertRaisesRegex(ValueError, "Invalid CDS/protein"):
            repartition(self.data, self.clusters, self.base / "result")


if __name__ == "__main__":
    unittest.main()
