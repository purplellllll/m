import tempfile
from pathlib import Path
import unittest
import torch

from .three_cds_pipeline import validate_dna
from tools.run_three_cds_native import copy_runtime
from tools.windows_torch_io import enable_unicode_save


class PipelineTests(unittest.TestCase):
    def test_old_torch_can_save_to_unicode_directories(self):
        original=torch.save
        try:
            enable_unicode_save()
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'中文路径'
                path.mkdir()
                torch.save(dict(value=torch.tensor([3.])),path/'state.pt')
                self.assertEqual(float(torch.load(path/'state.pt')['value']),3.)
        finally:
            torch.save=original

    def test_synonymous_edits_and_native_stop_changes_are_reported(self):
        value=validate_dna('ATGGCTTAA','ATGGCCTAG')
        self.assertEqual(value['changed_codons'],2)
        self.assertEqual(value['changed_nucleotides'],2)
        self.assertFalse(value['stop_preserved'])
        self.assertTrue(value['protein_preserved'])
        with self.assertRaises(AssertionError):
            validate_dna('ATGGCTTAA','ATGGTTTAA')

    def test_only_immutable_weights_share_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source'
            source.mkdir()
            (source/'best_model.p').write_bytes(b'weights')
            (source/'model_config.py').write_text('original')
            copy_runtime(source,root/'runtime')
            (root/'runtime/model_config.py').write_text('changed')
            self.assertEqual((source/'model_config.py').read_text(),'original')
            self.assertTrue((source/'best_model.p').samefile(root/'runtime/best_model.p'))


if __name__=='__main__':
    unittest.main()
