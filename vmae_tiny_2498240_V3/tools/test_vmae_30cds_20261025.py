import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from tools.run_vmae_30cds_20261025 import select_panel, length_bin
from tools.run_vmae_30cds_20261025 import archive_native


class PanelChecks(unittest.TestCase):
    def test_distinct_groups_and_input_order_independence(self):
        rows=[dict(sequence_id=f'{group}-{i}',v2_group=f'{group}-{i}',n_codons=n)
              for group,n in [('s',101),('m',501),('l',901)] for i in range(12)]
        selected=select_panel(rows)
        self.assertEqual(selected,select_panel(list(reversed(rows))))
        self.assertEqual(len({r['v2_group'] for r in selected}),30)
        self.assertEqual({g:sum(length_bin(r)==g for r in selected) for g in ('short','medium','long')},dict(short=10,medium=10,long=10))

    def test_shortage_does_not_silently_reuse_groups(self):
        rows=[dict(sequence_id=f'{n}-{i}',v2_group='shared',n_codons=n) for n in (101,501,901) for i in range(10)]
        with self.assertRaises(AssertionError):select_panel(rows)

    def test_archive_verified_before_workspace_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            output=root/'native/cds_example';output.mkdir(parents=True)
            (output/'completion_receipt.json').write_text(json.dumps(dict(full_default=True)))
            (output/'results.txt').write_text('a saved result\n')
            with patch('tools.run_vmae_30cds_20261025.OUT',root):archive_native(output)
            self.assertFalse(output.exists())
            receipt=json.loads((root/'native_archives/cds_example.tar.receipt.json').read_text())
            self.assertTrue(receipt['all_members_verified'])
            self.assertEqual(set(receipt['members']),{'completion_receipt.json','results.txt'})

    def test_archive_refuses_unexpected_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            output=root/'unrelated';output.mkdir()
            with patch('tools.run_vmae_30cds_20261025.OUT',root):
                with self.assertRaises(AssertionError):archive_native(output)
            self.assertTrue(output.exists())


if __name__=='__main__':unittest.main()
