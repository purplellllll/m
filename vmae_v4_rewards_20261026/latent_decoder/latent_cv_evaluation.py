"""Locked unseen-group cohorts and paired fivefold score aggregation."""
import gzip
import hashlib
import json

import numpy as np


GROUPS = ('short', 'medium', 'long')
METRICS = ('ce', 'full_ce', 'kl', 'fixed_loss')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def outer_cohorts(data):
    manifest = json.loads((data/'folds.json').read_text())
    assert digest(data/'corpus.jsonl.gz') == manifest['corpus_sha256']
    assignment = manifest['assignments']
    development = {assignment[i]['group'] for f in ('0', '1') for i in manifest['splits'][f]['val']}
    bad, metadata = set(), []
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            group = assignment[row['sequence_id']]['group']
            if row['split'] != 'test':
                bad.add(group)
            metadata.append((row['sequence_id'], group))
    fold_for = {i: fold for fold, split in manifest['splits'].items() for i in split['test']}
    assert len(fold_for) == len(metadata)
    eligible = {i for i, group in metadata if group not in bad|development}
    cohorts = {fold: [] for fold in manifest['splits']}
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            if row['sequence_id'] in eligible:
                row['v2_group'] = assignment[row['sequence_id']]['group']
                cohorts[fold_for[row['sequence_id']]].append(row)
    for records in cohorts.values():
        records.sort(key=lambda r: r['sequence_id'])
    group_folds = {}
    for fold, records in cohorts.items():
        for record in records:
            group = record['v2_group']
            assert group_folds.setdefault(group, fold) == fold
            assert group not in bad|development and record['split'] == 'test'
    return cohorts


def aggregate_scores(results, metadata, resamples=5000, seed=20261023):
    """Equal folds and lengths; cluster-bootstrap within each fold, conditional on fitted models."""
    identities = sorted(metadata)
    assert identities and 'V1' in results
    index = {identity: i for i, identity in enumerate(identities)}
    arrays = {}
    for name, rows in results.items():
        values = np.zeros((len(identities), len(METRICS)))
        masks = {identity: [] for identity in identities}
        for row in rows:
            identity = row['sequence_id']
            assert row['fold'] == metadata[identity]['fold'] and row['group'] == metadata[identity]['length_group']
            values[index[identity]] += [row[key] for key in METRICS]
            masks[identity].append(row['mask_seed'])
        assert all(sorted(seeds) == [175, 176] for seeds in masks.values()), 'Require exactly both masks per CDS/model'
        arrays[name] = values/2
    folds = sorted({m['fold'] for m in metadata.values()})
    bins = {(fold, group): np.array([metadata[i]['fold'] == fold and metadata[i]['length_group'] == group
        for i in identities]) for fold in folds for group in GROUPS}
    assert all(mask.any() for mask in bins.values()), 'Every reported fold must contain every length bin'
    scores = {}
    for name, values in arrays.items():
        per_fold = {fold: {group: values[bins[fold, group]].mean(0) for group in GROUPS} for fold in folds}
        primary = np.mean([v for groups in per_fold.values() for v in groups.values()], axis=0)
        scores[name] = dict(primary=dict(zip(METRICS, map(float, primary))),
            pooled=dict(zip(METRICS, map(float, values.mean(0)))),
            by_fold={fold: dict(macro=dict(zip(METRICS, map(float, np.mean(list(groups.values()), axis=0)))),
                by_length={g: dict(zip(METRICS, map(float, value))) for g, value in groups.items()})
                for fold, groups in per_fold.items()})
    pairs = [(name, 'V1') for name in arrays if name != 'V1']
    pairs += [(a, b) for a, b in (('candidate', 'control'), ('compact', 'candidate'), ('compact', 'control'))
        if a in arrays and b in arrays]
    differences = np.stack([arrays[a]-arrays[b] for a, b in pairs]) if pairs else np.empty((0, len(identities), len(METRICS)))
    cluster_data = []
    for fold in folds:
        clusters = sorted({metadata[i]['homology_group'] for i in identities if metadata[i]['fold'] == fold})
        members = np.stack([[metadata[i]['fold'] == fold and metadata[i]['homology_group'] == cluster
            for i in identities] for cluster in clusters]).astype(float)
        count = np.stack([members@bins[fold, group].astype(float) for group in GROUPS], axis=1)
        sums = np.stack([np.einsum('cn,pnk->cpk', members*bins[fold, group], differences) for group in GROUPS], axis=1)
        cluster_data.append((count, sums))
    rng, draws = np.random.default_rng(seed), []
    for _ in range(resamples):
        parts = []
        for counts, sums in cluster_data:
            for attempt in range(1000):
                weights = np.bincount(rng.integers(0, len(counts), len(counts)), minlength=len(counts))
                denominator = weights@counts
                if (denominator > 0).all():
                    break
            else:
                raise ValueError('Cannot resample all length bins; inspect group coverage')
            parts.append(np.einsum('c,cgpk->gpk', weights, sums)/denominator[:, None, None])
        if len(parts) == len(folds):
            draws.append(np.concatenate(parts, axis=0).mean(0))
    comparisons = {}
    if pairs and resamples:
        assert len(draws) >= resamples*.95
        lower, upper = np.quantile(draws, [.025, .975], axis=0)
        for j, (a, b) in enumerate(pairs):
            point = np.mean([differences[j, mask].mean(0) for mask in bins.values()], axis=0)
            comparisons[a+'_minus_'+b] = {key: dict(delta=float(point[k]), interval95=[float(lower[j,k]), float(upper[j,k])])
                for k, key in enumerate(METRICS)}
    return dict(scores=scores, comparisons=comparisons, records=len(identities), folds=len(folds),
        homology_groups=len({m['homology_group'] for m in metadata.values()}),
        resamples=resamples, interpretation='Fixed fitted-model paired homology bootstrap within folds, conditional on each fold draw containing all3 length bins; does not estimate all training-seed or architecture-selection uncertainty.')
