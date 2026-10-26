import gzip
import json
from pathlib import Path
import tempfile
import unittest

from .latent_fullfold_recovery_gpu import fold_data
from .train import sha256


class FullfoldIsolationTest(unittest.TestCase):
    def test_original_training_contamination_excludes_entire_validation_group(self):
        records = [dict(sequence_id='new_train', split='train'),
            dict(sequence_id='safe_val', split='test'),
            dict(sequence_id='old_train_val', split='train'),
            dict(sequence_id='old_test_same_group', split='test'),
            dict(sequence_id='outer', split='test')]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with gzip.open(root/'corpus.jsonl.gz', 'wt') as stream:
                for record in records:
                    stream.write(json.dumps(record)+'\n')
            manifest = dict(corpus_sha256=sha256(root/'corpus.jsonl.gz'),
                splits={'0': dict(train=['new_train'], val=['safe_val','old_train_val','old_test_same_group'], test=['outer'])},
                assignments={r['sequence_id']: dict(group='contaminated' if 'old_' in r['sequence_id'] else r['sequence_id']) for r in records})
            (root/'folds.json').write_text(json.dumps(manifest))
            result = fold_data(root, 0)
            self.assertEqual([r['sequence_id'] for r in result['train']], ['new_train'])
            self.assertEqual([r['sequence_id'] for r in result['val']], ['safe_val'])
            self.assertNotIn('outer', {r['sequence_id'] for rows in result.values() for r in rows})


if __name__ == '__main__':
    unittest.main()
