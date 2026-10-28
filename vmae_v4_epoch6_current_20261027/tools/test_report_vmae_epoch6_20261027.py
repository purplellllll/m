"""Check matched identities and validation exclusion in the current-version report."""
import json
from pathlib import Path
import tempfile
import unittest

from tools import report_vmae_epoch6_20261027 as report


class ReportTest(unittest.TestCase):
    def test_only_common_test_identities_are_paired(self):
        old_base, old_root = report.BASE, report.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                report.BASE = Path(directory)
                report.ROOT = report.BASE/'current'
                report.ROOT.mkdir()
                (report.ROOT/'test_panel.json').write_text(json.dumps([{'sequence_id': 'a'}, {'sequence_id': 'b'}]))
                for folder, identity, method, value in [
                    ('vmae_tiny_30cds_20261025', 'a', 'small', 1),
                    ('vmae_tiny_30cds_20261025', 'b', 'small', 9),
                    ('current', 'a', 'v4_epoch6', 3),
                    ('current', 'validation_only', 'v4_epoch6', 100),
                ]:
                    target = report.BASE/folder/'metrics'/identity/(method+'.json')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(dict(sequence_id=identity, method=method, **{k: value for k in report.KEYS})))
                report.report()
                result = json.loads((report.ROOT/'all_model_summary.json').read_text())
                self.assertEqual(result['methods']['V4_epoch6_current']['n'], 1)
                pair = result['paired_epoch6_minus_baseline']['V3_2.5M_Hard']
                self.assertEqual(pair['sequence_ids'], ['a'])
                self.assertEqual(pair['mean_delta']['seq2ribo_score'], 2)
        finally:
            report.BASE, report.ROOT = old_base, old_root


if __name__ == '__main__':
    unittest.main()
