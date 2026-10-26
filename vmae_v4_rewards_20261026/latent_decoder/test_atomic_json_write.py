import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .budget_js_extension import save_json


class AtomicWriteChecks(unittest.TestCase):
    def test_transient_reader_lock_keeps_atomic_update(self):
        with tempfile.TemporaryDirectory(dir='.runtime') as directory:
            assert Path(directory).resolve().is_relative_to(Path.cwd().resolve())
            path=Path(directory)/'state.json';save_json(path,dict(old=True))
            original=Path.replace;attempts=[]
            def sometimes_locked(source,target):
                attempts.append(source)
                if len(attempts)<3:raise PermissionError('reader still holds target')
                return original(source,target)
            with patch.object(Path,'replace',sometimes_locked),patch('latent_decoder.budget_js_extension.time.sleep') as sleep:
                save_json(path,dict(new=True))
            self.assertEqual(json.loads(path.read_text()),dict(new=True))
            self.assertFalse(path.with_suffix('.tmp').exists())
            self.assertEqual(len(attempts),3);self.assertEqual(sleep.call_count,2)

    def test_persistent_lock_raises_and_preserves_old_file(self):
        with tempfile.TemporaryDirectory(dir='.runtime') as directory:
            assert Path(directory).resolve().is_relative_to(Path.cwd().resolve())
            path=Path(directory)/'state.json';save_json(path,dict(old=True))
            with patch.object(Path,'replace',side_effect=PermissionError('permanent')) as replace,patch('latent_decoder.budget_js_extension.time.sleep'):
                with self.assertRaises(PermissionError):save_json(path,dict(new=True))
            self.assertEqual(replace.call_count,20)
            self.assertEqual(json.loads(path.read_text()),dict(old=True))

    def test_invalid_json_does_not_overwrite_reference(self):
        with tempfile.TemporaryDirectory(dir='.runtime') as directory:
            assert Path(directory).resolve().is_relative_to(Path.cwd().resolve())
            path=Path(directory)/'state.json';save_json(path,dict(old=True))
            with self.assertRaises(ValueError):save_json(path,dict(value=float('nan')))
            self.assertEqual(json.loads(path.read_text()),dict(old=True))


if __name__=='__main__':unittest.main()
