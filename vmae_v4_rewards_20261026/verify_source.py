"""Verify the frozen V4 source and run receipts using only the standard library."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or unsafe snapshot path: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Snapshot SHA mismatch: '+name)
        if path.suffix=='.py':
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
    plan=json.loads((ROOT/'evidence/plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        assert manifest['files_sha256'][name]==expected,name
    assert plan['version']=='V4' and plan['test_used'] is False
    assert plan['checkpoint_sha256']=='6592a07293da25b18f9ed71584e48dc4da23fa121b4f735f10d26ee7da26c6af'
    for name in ('preflight.json','search_preflight.json'):
        assert json.loads((ROOT/'evidence'/name).read_text())['passed']
    update=json.loads((ROOT/'evidence/first_update.json').read_text())
    assert update['model_hash_before']!=update['model_hash_after']
    assert update['scorer_unchanged'] and update['source_checkpoint_unchanged']
    print(json.dumps(dict(files_verified=len(manifest['files_sha256']),
        version='V4',real_parameter_update_verified=True,training_started_by_this_check=False)))


if __name__=='__main__':
    main()
