"""Freeze GENCODE v28 + RiboDecode CDS into dataset/{train,val,test}."""
import argparse
import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .prepare_data import Components, EXPECTED_MD5, SEQUENCE_MEMBER, digest_file, reference_cds, rows, translate
from .repartition import load_prepared
from .tokenizer import AA, CODONS, decode_cds, encode_cds, specification

SPLITS = ("train", "val", "test")
SEED = "ds-ribodecode-gencode28-v1"


def identity_keys(record):
    return ([('protein', record['protein_sequence'])]
            + [('gene', g.split('.')[0]) for g in record['gene_ids']]
            + [('transcript', t.split('.')[0]) for t in record['transcript_ids']])


def assign_splits(records, anchors, official_test):
    components, owner = Components(len(records)), {}
    for i, record in enumerate(records):
        for key in identity_keys(record):
            if key in owner:
                components.union(i, owner[key])
            else:
                owner[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[components.root(i)].append(i)
    for indices in groups.values():
        component = "component_" + hashlib.sha256("\n".join(sorted(
            records[i]['sequence_id'] for i in indices)).encode()).hexdigest()
        prior = {anchors[k] for i in indices for k in identity_keys(records[i]) if k in anchors}
        genes = {g for i in indices for g in records[i]['gene_ids']}
        if 'test' in prior or genes & official_test:
            split = 'test'
        elif 'val' in prior:
            split = 'val'
        elif 'train' in prior:
            split = 'train'
        else:
            bucket = int(hashlib.sha256((SEED + ':' + component).encode()).hexdigest(), 16) % 10
            split = 'test' if bucket == 0 else 'val' if bucket == 1 else 'train'
        for i in indices:
            records[i].update(split=split, cluster_id=component)
    return len(groups)


def annotation_records(path):
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9:
                raise ValueError('Invalid GTF row')
            if fields[2] != 'transcript':
                continue
            attrs = re.findall(r'(\w+) "([^"]*)";', fields[8])
            values = dict(attrs)
            tx = values['transcript_id']
            if tx in result:
                raise ValueError('Duplicate GTF transcript: ' + tx)
            tags = {v for k, v in attrs if k == 'tag'}
            reason = ('mitochondrial_code' if fields[0] == 'chrM' else
                      'not_protein_coding' if values.get('transcript_type') != 'protein_coding' else
                      'incomplete_cds_annotation' if tags & {'cds_start_NF', 'cds_end_NF'} else None)
            result[tx] = (values['gene_id'].split('.')[0], reason)
    return result


def write_gzip_lines(path, lines):
    # Fix gzip timestamp and omit filename so repeated builds have identical bytes.
    with path.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        for line in lines:
            zipped.write((line + '\n').encode('utf-8'))


def write_split(directory, records):
    directory.mkdir()
    paths = []
    for start in range(0, len(records), 5000):
        path = directory / ('part-%05d.jsonl.gz' % (start // 5000))
        write_gzip_lines(path, (json.dumps(r, ensure_ascii=False, separators=(',', ':'))
                                for r in records[start:start + 5000]))
        paths.append(path)
    return paths


def build(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory; frozen datasets are never overwritten')
    old, old_audit = load_prepared(root / 'data/latent_decoder')
    source = root / 'data/official/dataset_for_translation_model.zip'
    reference_path = root / 'data/reference/gencode_v28/gencode.v28.pc_transcripts.fa.gz'
    gtf_path = reference_path.with_name('gencode.v28.annotation.gtf.gz')
    if source.stat().st_size != 121954830 or digest_file(source, 'md5') != EXPECTED_MD5:
        raise ValueError('RiboDecode source size/MD5 mismatch')
    if digest_file(reference_path) != old_audit['stop_reference']['sha256']:
        raise ValueError('GENCODE FASTA differs from the audited v28 reference')
    reference, annotation = reference_cds(reference_path), annotation_records(gtf_path)
    counts = {s: Counter() for s in ('GENCODE_v28', 'RiboDecode_figshare_v3')}
    records = {}

    def add(source_name, tx, gene, cds):
        count = counts[source_name]
        count['input_rows'] += 1
        if tx not in annotation:
            count['excluded_missing_gtf_transcript'] += 1
            return
        annotated_gene, reason = annotation[tx]
        if gene != annotated_gene:
            raise ValueError('Transcript/gene mismatch: ' + tx)
        if reason:
            count['excluded_' + reason] += 1
            return
        protein, reason = translate(cds)
        if reason:
            count['excluded_' + reason] += 1
            return
        if not 30 <= len(cds) // 3 <= 1500:
            count['excluded_length_out_of_range'] += 1
            return
        ids = encode_cds(cds)
        if decode_cds(ids) != cds or ''.join(AA[i] for i in ids[:-1]) != protein:
            raise ValueError('Tokenizer/genetic-code mismatch')
        count['accepted_rows'] += 1
        if cds not in records:
            records[cds] = dict(sequence_id='cds_' + hashlib.sha256(cds.encode()).hexdigest(),
                cds_dna=cds, protein_sequence=protein, n_codons=len(ids), codon_ids=ids,
                transcript_ids=set(), gene_ids=set(), source_transcript_ids=defaultdict(set))
        record = records[cds]
        record['transcript_ids'].add(tx)
        record['gene_ids'].add(gene)
        record['source_transcript_ids'][source_name].add(tx)

    for tx, cds in reference.items():
        gene = annotation[tx][0] if tx in annotation else ''
        add('GENCODE_v28', tx, gene, cds)
    with zipfile.ZipFile(source) as archive:
        official_train, official_test = set(), set()
        for row in rows(archive, 'genes.csv'):
            if row.get('Training_set'):
                official_train.add(row['Training_set'].split('.')[0])
            if row.get('Test_set'):
                official_test.add(row['Test_set'].split('.')[0])
        if not official_train or not official_test or official_train & official_test:
            raise ValueError('Invalid official gene split')
        official_genes = official_train | official_test
        major = {r['isoform_id'] for r in rows(archive, 'mjIsoform_320.csv')}
        count = counts['RiboDecode_figshare_v3']
        for row in rows(archive, SEQUENCE_MEMBER):
            tx, gene = row['transcript_id'], row['gene_id'].split('.')[0]
            if tx not in major or gene not in official_genes:
                count['input_rows'] += 1
                count['excluded_not_author_major_isoform' if tx not in major else 'excluded_not_official_gene'] += 1
                continue
            cds = row['cds_sequence'].upper().replace('U', 'T')
            ref = reference.get(tx, '')
            if cds[-3:] not in {'TAA', 'TAG', 'TGA'} and ref[-3:] in {'TAA', 'TAG', 'TGA'} and cds == ref[:-3]:
                cds = ref
                count['reference_verified_stop_restorations_before_filter'] += 1
            add('RiboDecode_figshare_v3', tx, gene, cds)
    data = sorted(records.values(), key=lambda r: r['sequence_id'])
    if not data:
        raise ValueError('Empty corpus')
    for record in data:
        for key in ('transcript_ids', 'gene_ids'):
            record[key] = sorted(record[key])
        record['source_transcript_ids'] = {k: sorted(v) for k, v in sorted(record['source_transcript_ids'].items())}
        record['sources'] = sorted(record['source_transcript_ids'])
        record['source'] = ';'.join(record['sources'])
    anchors = {}
    for record in old:
        for key in identity_keys(record):
            if key in anchors and anchors[key] != record['split']:
                raise ValueError('Old split identity conflict')
            anchors[key] = record['split']
    component_count = assign_splits(data, anchors, official_test)
    split_by_id = {r['sequence_id']: r['split'] for r in data}
    migrations = Counter(r['split'] + '->' + split_by_id[r['sequence_id']]
                         for r in old if r['sequence_id'] in split_by_id)
    output.mkdir(parents=True)
    files, split_files = {}, {}
    for split in SPLITS:
        paths = write_split(output / split, [r for r in data if r['split'] == split])
        split_files[split] = [p.relative_to(output).as_posix() for p in paths]
        files.update({p.relative_to(output).as_posix(): digest_file(p) for p in paths})
    assignment_path = output / 'split_assignments.tsv.gz'
    write_gzip_lines(assignment_path, iter(['sequence_id\tsplit\tcomponent_id'] + [
        '\t'.join((r['sequence_id'], r['split'], r['cluster_id'])) for r in data]))
    files[assignment_path.name] = digest_file(assignment_path)
    token_path = output / 'tokenizer.json'
    token_path.write_text(json.dumps(specification(), indent=2) + '\n', encoding='utf-8', newline='\n')
    files[token_path.name] = digest_file(token_path)
    stats = {}
    for split in SPLITS:
        selected = [r for r in data if r['split'] == split]
        stats[split] = dict(cds=len(selected), genes=len({g for r in selected for g in r['gene_ids']}),
            proteins=len({r['protein_sequence'] for r in selected}),
            source_membership=dict(Counter(s for r in selected for s in r['sources'])))
    manifest = dict(dataset_version=SEED, frozen_date='2026-10-17', source_verified=True,
        homology_certified=False, homology_search_audited=False,
        split_status='frozen-gene-transcript-exact-protein-components',
        split_policy='Union gene, versionless transcript and exact protein. Official test or old test -> test; old val -> val; old train -> train. New components: SHA256(seed:component_id) mod 10: 0=test, 1=val, 2..9=train. No token-level random split.',
        seed=SEED, min_codons=30, max_codons=1500, n_codons_includes_stop=True,
        filters=['nuclear protein_coding transcript', 'no cds_start_NF/cds_end_NF',
                 'ACGT only', 'ATG start', 'terminal stop; no internal stop', '90..4500 nt; no truncation'],
        sources=[dict(name='GENCODE_v28', url='https://www.gencodegenes.org/human/release_28.html',
                      fasta_sha256=digest_file(reference_path), gtf_sha256=digest_file(gtf_path)),
                 dict(name='RiboDecode_figshare_v3', author='Yupeng Li', doi='10.6084/m9.figshare.28916288.v3',
                      license='CC BY 4.0', zip_md5=EXPECTED_MD5, zip_sha256=digest_file(source))],
        source_processing={k: dict(v) for k, v in counts.items()},
        unique_cds=len(data), unique_proteins=len({r['protein_sequence'] for r in data}),
        duplicate_accepted_rows_collapsed=sum(c['accepted_rows'] for c in counts.values()) - len(data),
        cross_source_shared_cds=sum(len(r['sources']) == 2 for r in data), components=component_count,
        counts={s: stats[s]['cds'] for s in SPLITS}, split_statistics=stats,
        previous_corpus_sha256=old_audit['corpus_sha256'],
        previous_audit_sha256=digest_file(root / 'data/latent_decoder/audit.json'),
        previous_records_retained=sum(migrations.values()), previous_record_split_transitions=dict(migrations),
        previous_records_excluded=len(old) - sum(migrations.values()),
        official_test_genes=sorted(official_test), files_sha256=files, split_files=split_files,
        optimization_sequences_used_for_training=False, expression_labels_used=False,
        limitations=['No near-homology clustering or sensitive cross-split protein search.',
                     'Old checkpoints trained on a different corpus/split are not formal evaluation checkpoints for this dataset.'])
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    verify(output)
    return manifest


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['split_files']) != set(SPLITS):
        raise ValueError('Invalid split file map')
    expected_files = {'tokenizer.json', 'split_assignments.tsv.gz'}
    for split in SPLITS:
        names = manifest['split_files'][split]
        if not names or names != [split + '/part-%05d.jsonl.gz' % i for i in range(len(names))]:
            raise ValueError('Invalid shard names/order')
        expected_files.update(names)
    if set(manifest['files_sha256']) != expected_files:
        raise ValueError('Unexpected/missing manifest files')
    for name, expected in manifest['files_sha256'].items():
        if digest_file(directory / name) != expected:
            raise ValueError('Checksum mismatch: ' + name)
    if json.loads((directory / 'tokenizer.json').read_text(encoding='utf-8')) != specification():
        raise ValueError('Tokenizer specification mismatch')
    legacy = json.loads(Path(__file__).with_name('baseline_codon_order.json').read_text(encoding='utf-8'))
    if any(legacy[c] != i for i, c in enumerate(CODONS)):
        raise ValueError('RiboDecode codon IDs changed')
    splits, seen_ids, seen_cds, seen_keys, assignments = {s: [] for s in SPLITS}, set(), set(), {}, {}
    official_test = set(manifest['official_test_genes'])
    for split in SPLITS:
        for r in iter_split(directory, split, manifest):
            cds, ids = r['cds_dna'], r['codon_ids']
            if r['split'] != split or r['sequence_id'] != 'cds_' + hashlib.sha256(cds.encode()).hexdigest():
                raise ValueError('Split/sequence ID mismatch')
            if ids != encode_cds(cds) or decode_cds(ids) != cds or r['n_codons'] != len(ids) or not 30 <= len(ids) <= 1500:
                raise ValueError('Token round-trip/length mismatch')
            if r['protein_sequence'] != ''.join(AA[i] for i in ids[:-1]):
                raise ValueError('Protein mismatch')
            if r['sequence_id'] in seen_ids or cds in seen_cds:
                raise ValueError('Duplicate CDS/ID')
            if not r['gene_ids'] or not r['transcript_ids'] or not r['sources']:
                raise ValueError('Missing provenance')
            if sorted(r['source_transcript_ids']) != r['sources'] or sorted({t for v in r['source_transcript_ids'].values() for t in v}) != r['transcript_ids']:
                raise ValueError('Source/transcript provenance mismatch')
            for key in identity_keys(r) + [('component', r['cluster_id'])]:
                if key in seen_keys and seen_keys[key] != split:
                    raise ValueError('Cross-split identity leakage: ' + str(key))
                seen_keys[key] = split
            if split != 'test' and set(r['gene_ids']) & official_test:
                raise ValueError('Official test gene leaked')
            seen_ids.add(r['sequence_id'])
            seen_cds.add(cds)
            assignments[r['sequence_id']] = (split, r['cluster_id'])
            splits[split].append(r)
        if not splits[split] or len(splits[split]) != manifest['counts'][split]:
            raise ValueError('Empty split/count mismatch')
    with gzip.open(directory / 'split_assignments.tsv.gz', 'rt', encoding='utf-8') as handle:
        rows_read = list(csv.DictReader(handle, delimiter='\t'))
    actual = {r['sequence_id']: (r['split'], r['component_id']) for r in rows_read}
    if len(rows_read) != len(actual) or actual != assignments or len(seen_ids) != manifest['unique_cds']:
        raise ValueError('Frozen assignment mismatch')
    return splits, manifest


def iter_split(directory, split, manifest=None):
    directory = Path(directory)
    if manifest is None:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for name in manifest['split_files'][split]:
        if not re.fullmatch(re.escape(split) + r'/part-\d{5}\.jsonl\.gz', name):
            raise ValueError('Invalid shard path')
        with gzip.open(directory / name, 'rt', encoding='utf-8') as handle:
            for line in handle:
                yield json.loads(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='Root containing the pinned raw inputs and old corpus')
    parser.add_argument('--output', default='dataset')
    parser.add_argument('--verify', help='Verify an existing frozen dataset instead of building')
    args = parser.parse_args()
    result = verify(args.verify)[1] if args.verify else build(args.root, args.output)
    print(json.dumps({k: result[k] for k in ('counts', 'unique_cds', 'unique_proteins', 'cross_source_shared_cds', 'homology_certified')}, indent=2))
