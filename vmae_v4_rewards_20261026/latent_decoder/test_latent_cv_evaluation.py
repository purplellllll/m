import gzip
import json
from pathlib import Path
import tempfile
import unittest

from .latent_cv_evaluation import aggregate_scores, digest, outer_cohorts


class CVEvaluationTests(unittest.TestCase):
    def test_cohort_excludes_original_training_groups_and_historical_development_groups(self):
        records = [dict(sequence_id=i, split=old) for i, old in [('train','train'),('train_relative','test'),
            ('dev','test'),('safe0','test'),('safe1','test')]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with gzip.open(root/'corpus.jsonl.gz', 'wt') as stream:
                for row in records:
                    stream.write(json.dumps(row)+'\n')
            manifest = dict(corpus_sha256=digest(root/'corpus.jsonl.gz'), assignments={
                r['sequence_id']: dict(group='bad' if r['sequence_id'].startswith('train') else r['sequence_id']) for r in records},
                splits={'0': dict(val=['dev'], test=['train','train_relative','safe0']),
                    '1': dict(val=[], test=['dev','safe1'])})
            (root/'folds.json').write_text(json.dumps(manifest))
            cohorts = outer_cohorts(root)
            self.assertEqual({f:[r['sequence_id'] for r in rows] for f,rows in cohorts.items()}, {'0':['safe0'],'1':['safe1']})

    def test_equal_length_fold_aggregation_pairing_and_duplicate_mask_rejection(self):
        metadata, base, candidate = {}, [], []
        for fold in range(5):
            for j, group in enumerate(('short','medium','long')):
                for k in range(1 if group == 'long' else 3):
                    identity = f'{fold}:{group}:{k}'
                    metadata[identity] = dict(fold=fold, length_group=group, homology_group=f'{fold}:{j}:{k}')
                    for mask in (175,176):
                        row = dict(sequence_id=identity, fold=fold, group=group, mask_seed=mask,
                            ce=float(j+1),full_ce=.1,kl=3.,fixed_loss=float(j+1)+.103)
                        base.append(row)
                        candidate.append(dict(row,ce=row['ce']-.2,fixed_loss=row['fixed_loss']-.2))
        result = aggregate_scores(dict(V1=base,candidate=candidate), metadata, resamples=100)
        self.assertAlmostEqual(result['scores']['V1']['primary']['ce'], 2.)
        ci = result['comparisons']['candidate_minus_V1']['ce']
        self.assertAlmostEqual(ci['delta'], -.2)
        for value in ci['interval95']:
            self.assertAlmostEqual(value,-.2)
        with self.assertRaises(AssertionError):
            aggregate_scores(dict(V1=base,candidate=candidate+[candidate[0]]), metadata,resamples=0)


if __name__ == '__main__':
    unittest.main()
