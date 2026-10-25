"""Check the frozen source, optional fixed data, and optional trusted checkpoint."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise ValueError('Missing or changed snapshot file: '+name)
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
    card = json.loads((ROOT/'MODEL_CARD.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT/'latent_decoder/config_lsco_lr_50ep.json').read_text(encoding='utf-8'))
    assert config['model'] == card['model']
    assert config['model']['d_model'] == 256 and config['model']['ff_dim'] == 1024
    assert config['model']['latent_dim'] == 256
    receipt = dict(source_files_checked=len(manifest['files_sha256']), source_passed=True,
                   data_checked=False, checkpoint_checked=False, training_started=False)
    assets = json.loads((ROOT/'ASSETS_REQUIRED.json').read_text(encoding='utf-8'))
    if args.data:
        for name in ('corpus.jsonl.gz', 'folds.json'):
            if sha(args.data/name) != assets['training_data'][name]['sha256']:
                raise ValueError('Fixed data SHA mismatch: '+name)
        folds = json.loads((args.data/'folds.json').read_text(encoding='utf-8'))
        split = folds['splits']['0']
        assert {name: len(split[name]) for name in ('train', 'val', 'test')} == card['split_counts']
        groups = {name: {folds['assignments'][i]['group'] for i in split[name]}
                  for name in ('train', 'val', 'test')}
        assert all(not groups[a] & groups[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')))
        receipt['data_checked'] = True
    if args.checkpoint:
        if sha(args.checkpoint) != card['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA mismatch')
        from latent_decoder.latent_information_probe import load_model
        model = load_model(args.checkpoint, 'cpu')
        count = sum(p.numel() for p in model.parameters())
        assert count == card['parameter_count'] == 9714496
        receipt.update(checkpoint_checked=True, parameter_count=count)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
