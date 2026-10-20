"""Small synthetic checks for frozen split propagation and shared tokenization."""
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .prepare_combined import SPLITS, annotation_records, assign_splits, identity_keys, verify, write_gzip_lines, write_split
from .prepare_data import digest_file, translate
from .tokenizer import decode_cds, encode_cds, specification


class CombinedChecks(unittest.TestCase):
    def test_transitive_holdout_and_versionless_transcripts(self):
        records = [dict(sequence_id=str(i), gene_ids=[g], transcript_ids=[t], protein_sequence=p)
                   for i, (g, t, p) in enumerate([
                       ('g1', 't1.1', 'MA'), ('g1', 't2.1', 'MB'),
                       ('g2', 't3.1', 'MB'), ('g3', 't3.2', 'MC'),
                       ('g4', 't4.1', 'MD'), ('g5', 't5.1', 'ME')])]
        anchors = {('gene', 'g1'): 'train', ('gene', 'g4'): 'val'}
        assign_splits(records, anchors, {'g2'})
        self.assertEqual([r['split'] for r in records[:5]], ['test'] * 4 + ['val'])
        self.assertEqual(len({r['cluster_id'] for r in records[:4]}), 1)
        expected = {r['sequence_id']: (r['split'], r['cluster_id']) for r in records}
        assign_splits(list(reversed(records)), anchors, {'g2'})
        self.assertEqual(expected, {r['sequence_id']: (r['split'], r['cluster_id']) for r in records})
        self.assertIn(('transcript', 't3'), identity_keys(records[3]))

    def test_token_roundtrip_and_special_tokens_rejected(self):
        self.assertEqual(decode_cds(encode_cds('auggaauaa'), rna=True), 'AUGGAAUAA')
        for bad in ('ATGN AATAA', 'ATGGAAT', 'ATGTAGGAATAA', 'TTGGAATAA'):
            with self.assertRaises(ValueError):
                encode_cds(bad)
        for bad in ([11, 64, 18], [11, 65, 18], [11, -1, 18], [11, True, 18]):
            with self.assertRaises(ValueError):
                decode_cds(bad)

    def test_gtf_filters_and_reproducible_gzip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'annotation.gz'
            rows = []
            for chrom, tx, kind, tag in [('chr1', 't1', 'protein_coding', ''),
                    ('chrM', 't2', 'protein_coding', ''),
                    ('chr1', 't3', 'nonsense_mediated_decay', ''),
                    ('chr1', 't4', 'protein_coding', 'tag "cds_end_NF";')]:
                rows.append(f'{chrom}\tX\ttranscript\t1\t9\t.\t+\t.\tgene_id "g.1"; transcript_id "{tx}"; transcript_type "{kind}"; {tag}')
            write_gzip_lines(path, rows)
            annotation = annotation_records(path)
            self.assertEqual(annotation['t1'], ('g', None))
            self.assertEqual(annotation['t2'][1], 'mitochondrial_code')
            self.assertEqual(annotation['t3'][1], 'not_protein_coding')
            self.assertEqual(annotation['t4'][1], 'incomplete_cds_annotation')
            other = root / 'other.gz'
            write_gzip_lines(other, rows)
            self.assertEqual(path.read_bytes(), other.read_bytes())
            with gzip.open(path, 'rt') as handle:
                self.assertEqual(handle.read().splitlines(), rows)
            shards = write_split(root / 'shards', [{'i': i} for i in range(5001)])
            self.assertEqual([p.name for p in shards], ['part-00000.jsonl.gz', 'part-00001.jsonl.gz'])
            decoded = []
            sizes = []
            for shard in shards:
                with gzip.open(shard, 'rt') as handle:
                    batch = [json.loads(line)['i'] for line in handle]
                decoded.extend(batch)
                sizes.append(len(batch))
            self.assertEqual(sizes, [5000, 1])
            self.assertEqual(decoded, list(range(5001)))

    def test_frozen_dataset_integrity_and_training_loader(self):
        from .train import load_data
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assignments = ['sequence_id\tsplit\tcomponent_id']
            paths = []
            for split, codon in zip(SPLITS, ('GCT', 'GAA', 'TTT')):
                cds = 'ATG' + codon * 28 + 'TAA'
                record = dict(sequence_id='cds_' + hashlib.sha256(cds.encode()).hexdigest(),
                    cds_dna=cds, protein_sequence=translate(cds)[0], n_codons=30,
                    codon_ids=encode_cds(cds), gene_ids=[split], transcript_ids=[split + '.1'],
                    sources=['GENCODE_v28'], source_transcript_ids={'GENCODE_v28': [split + '.1']},
                    split=split, cluster_id=split)
                (root / split).mkdir()
                path = root / split / 'part-00000.jsonl.gz'
                write_gzip_lines(path, [json.dumps(record)])
                paths.append(path)
                assignments.append('\t'.join((record['sequence_id'], split, split)))
            token_path, assignment_path = root / 'tokenizer.json', root / 'split_assignments.tsv.gz'
            token_path.write_text(json.dumps(specification()), encoding='utf-8')
            write_gzip_lines(assignment_path, assignments)
            paths.extend((token_path, assignment_path))
            manifest = dict(source_verified=True, homology_certified=False, official_test_genes=['test'],
                counts={s: 1 for s in SPLITS}, unique_cds=3,
                split_files={s: [s + '/part-00000.jsonl.gz'] for s in SPLITS},
                files_sha256={p.relative_to(root).as_posix(): digest_file(p) for p in paths})
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual({s: len(v) for s, v in verify(root)[0].items()}, manifest['counts'])
            with self.assertRaisesRegex(ValueError, 'clustering'):
                load_data(root)
            self.assertEqual(len(load_data(root, allow_development=True)[0]['train']), 1)
            with paths[0].open('ab') as handle:
                handle.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
