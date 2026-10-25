"""Verified corpus export and length-stratified, homology-grouped outer folds."""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .budget_js_extension import save_json
from .length_diagnostics import length_group
from .prepare_data import Components
from .train import load_data, sha256


def read_corpus(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def export(data, output):
    output.mkdir(parents=True, exist_ok=False)
    splits, _ = load_data(data, allow_development=True)
    records = sorted([r for rows in splits.values() for r in rows], key=lambda r: r['sequence_id'])
    with gzip.open(output/'corpus.jsonl.gz', 'wt', encoding='utf-8') as handle, (output/'proteins.fasta').open('w', encoding='ascii') as proteins:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=True)+'\n')
            proteins.write('>'+r['sequence_id']+'\n'+r['protein_sequence'].rstrip('*')+'\n')
    save_json(output/'export.json', dict(records=len(records), original_manifest_sha256=sha256(data/'manifest.json'),
        files_sha256={name: sha256(output/name) for name in ('corpus.jsonl.gz', 'proteins.fasta')},
        policy='Merge old train/val/test for new cross-validation; never warm-start outer models from old weights'))


def assign_folds(records, hits, folds=5, seed=20261022):
    from sklearn.model_selection import StratifiedGroupKFold
    ids = {r['sequence_id']: i for i, r in enumerate(records)}
    assert len(ids) == len(records)
    components, owners = Components(len(records)), {}
    for i, r in enumerate(records):
        for key in [('old_group', r['cluster_id'])]:
            if key in owners:
                components.union(i, owners[key])
            owners[key] = i
    edges = []
    with hits.open(encoding='utf-8') as handle:
        for line in handle:
            fields = line.rstrip().split('\t')
            if len(fields) != 5:
                raise ValueError('Expected query,target,fident,qcov,tcov')
            query, target, identity, qcov, tcov = fields
            if query not in ids or target not in ids:
                raise ValueError('Unknown sequence in homology hits')
            if min(float(qcov), float(tcov)) < .8-1e-6 or float(identity) < .3-1e-6:
                continue
            a, b = ids[query], ids[target]
            components.union(a, b)
            edges.append((a, b))
    groups = np.array([components.root(i) for i in range(len(records))])
    labels = np.array([length_group(r['n_codons']-1) for r in records])
    assignments = np.full(len(records), -1, dtype=int)
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(outer.split(np.zeros(len(records)), labels, groups)):
        assignments[test] = fold
    assert (assignments >= 0).all()
    assert all(assignments[a] == assignments[b] for a, b in edges)
    mapping = {r['sequence_id']: dict(outer_fold=int(assignments[i]), group=int(groups[i])) for i, r in enumerate(records)}
    inner = {}
    for fold in range(folds):
        remaining = np.flatnonzero(assignments != fold)
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed+fold+1)
        train_index, val_index = next(splitter.split(np.zeros(len(remaining)), labels[remaining], groups[remaining]))
        train, val = remaining[train_index], remaining[val_index]
        test = np.flatnonzero(assignments == fold)
        for a, b in ((train, val), (train, test), (val, test)):
            assert not set(groups[a]) & set(groups[b])
        inner[str(fold)] = dict(train=[records[i]['sequence_id'] for i in train],
            val=[records[i]['sequence_id'] for i in val], test=[records[i]['sequence_id'] for i in test],
            length_counts={name: dict(Counter(labels[index])) for name, index in [('train', train), ('val', val), ('test', test)]})
    return dict(k=folds, seed=seed, assignments=mapping, splits=inner,
        groups=len(set(groups)), largest_group=int(max(Counter(groups).values())), hit_edges=len(edges),
        homology_policy='Union all detected MMseqs edges identity>=.30 AND query/target coverage>=.80, plus original gene/transcript/exact-protein groups; sensitivity-limited, not proof of absence of all homology')


def pick(records, per_group, salt):
    result, used = [], set()
    for group in ('short', 'medium', 'long'):
        for r in sorted(records, key=lambda r: hashlib.sha256((salt+r['sequence_id']).encode()).hexdigest()):
            if length_group(r['n_codons']-1) != group or r['v2_group'] in used:
                continue
            result.append(r)
            used.add(r['v2_group'])
            if sum(length_group(x['n_codons']-1) == group for x in result) == per_group:
                break
        if sum(length_group(x['n_codons']-1) == group for x in result) != per_group:
            raise ValueError('Insufficient independent groups for '+group)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--data', default='dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--hits')
    args = parser.parse_args()
    output = Path(args.output)
    if args.export:
        export(Path(args.data), output)
        return
    data = Path(args.data)
    manifest = json.loads((data/'export.json').read_text())
    assert sha256(data/'corpus.jsonl.gz') == manifest['files_sha256']['corpus.jsonl.gz']
    records = read_corpus(data/'corpus.jsonl.gz')
    self_hits, query_counts = set(), Counter()
    with Path(args.hits).open(encoding='utf-8') as handle:
        for line in handle:
            query, target, identity, qcov, tcov = line.rstrip().split('\t')
            query_counts[query] += 1
            if query == target and float(identity) >= .999:
                self_hits.add(query)
    # Low-complexity proteins need not yield an exact self hit under alignment filtering.
    missing_self = sorted({r['sequence_id'] for r in records}-self_hits)
    folds = assign_folds(records, Path(args.hits))
    folds.update(queries_with_self_hits=len(self_hits), max_reported_hits_per_query=max(query_counts.values()),
                 queries_at_hit_cap=sum(n >= 10000 for n in query_counts.values()),
                 missing_exact_self_hits=missing_self,
                 queries_without_hits=sorted({r['sequence_id'] for r in records}-set(query_counts)))
    folds.update(corpus_sha256=sha256(data/'corpus.jsonl.gz'), hits_sha256=sha256(Path(args.hits)))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'folds.json', folds)
    print(json.dumps({k: folds[k] for k in ('k', 'groups', 'largest_group', 'hit_edges')}), flush=True)


if __name__ == '__main__':
    main()
