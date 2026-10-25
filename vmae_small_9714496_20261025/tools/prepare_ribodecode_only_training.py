"""Export only verified official RiboDecode CDS; no extra GENCODE sequences."""
import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

from latent_decoder.prepare_data import EXPECTED_MD5, digest_file, translate


def official_matches(record, originals):
    matches = [tx for tx in record['transcript_ids']
               if originals.get(tx) == record['cds_dna'][:-3]]
    if not matches:
        raise ValueError('CDS is absent from official RiboDecode: ' + record['sequence_id'])
    protein, reason = translate(record['cds_dna'])
    if reason or protein != record['protein_sequence'].rstrip('*'):
        raise ValueError('Invalid verified CDS: ' + record['sequence_id'])
    return matches


def main():
    root = Path('runs/ribodecode_only_vmae_20261023')
    output = root/'data'
    output.mkdir(parents=True, exist_ok=False)
    archive = Path('data/official/dataset_for_translation_model.zip')
    corpus = Path('data/latent_decoder/corpus.jsonl')
    audit = json.loads(corpus.with_name('audit.json').read_text(encoding='utf-8'))
    assert digest_file(archive, 'md5') == EXPECTED_MD5
    assert digest_file(corpus) == audit['corpus_sha256']
    with zipfile.ZipFile(archive) as z:
        originals = {r['transcript_id']: r['cds_sequence'].strip().upper()
                     for r in csv.DictReader(io.TextIOWrapper(
                         z.open('human_coding_transcript_sequences.csv'), encoding='utf-8-sig'))}
    records = sorted(map(json.loads, corpus.open(encoding='utf-8')), key=lambda r: r['sequence_id'])
    assert len(records) == len({r['sequence_id'] for r in records}) == 32796
    lengths = Counter()
    with gzip.open(output/'corpus.jsonl.gz', 'wt', encoding='utf-8') as target, (
            output/'proteins.fasta').open('w', encoding='ascii') as proteins:
        for row in records:
            row['official_matching_transcripts'] = official_matches(row, originals)
            row['training_source'] = 'RiboDecode_figshare_v3_only'
            target.write(json.dumps(row, ensure_ascii=True)+'\n')
            proteins.write('>'+row['sequence_id']+'\n'+row['protein_sequence'].rstrip('*')+'\n')
            aa = row['n_codons']-1
            lengths['short' if aa <= 300 else 'medium' if aa < 700 else 'long'] += 1
    report = dict(records=len(records), length_counts=dict(lengths), official_zip_md5=EXPECTED_MD5,
        prepared_corpus_sha256=digest_file(corpus), corpus_sha256=digest_file(output/'corpus.jsonl.gz'),
        proteins_sha256=digest_file(output/'proteins.fasta'),
        sequence_policy='Every coding base exactly equals an official RiboDecode transcript. '
            'Only the independently annotation-verified terminal stop is restored; no external coding sequences.',
        selection='Existing author-major-isoform and official-gene-set filtered corpus; '
            'ACGT, ATG, intact CDS, no internal stop, 30..1500 codons including stop; deduplicated.',
        extra_gencode_cds=0, optimized_sequences=0, expression_labels_used=False,
        folds='New five-fold grouped split after all-vs-all protein homology search; random initialization only.')
    (output/'provenance.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
