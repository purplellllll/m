"""Portable entry point. Uses the bundled dataset and final folds; no remote connection."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = ('uniform', 'short_fast', 'long_fast')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check_files():
    manifest = json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed package file: '+name)
    print('Package checksums: PASS (%d files)' % len(manifest['files_sha256']), flush=True)


def check_data():
    folds = json.loads((ROOT/'data/folds.json').read_text(encoding='utf-8'))
    assert digest(ROOT/'data/corpus.jsonl.gz') == folds['corpus_sha256']
    records = {}
    with gzip.open(ROOT/'data/corpus.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            identity = record['sequence_id']
            assert identity not in records
            assert len(record['cds_dna']) == 3*record['n_codons']
            records[identity] = record['n_codons']
    assert len(records) == 44001 and set(records) == set(folds['assignments'])
    all_test = []
    for fold, split in folds['splits'].items():
        sets = [set(split[name]) for name in ('train', 'val', 'test')]
        assert all(len(split[name]) == len(ids) for name, ids in zip(('train', 'val', 'test'), sets))
        assert set.union(*sets) == set(records)
        groups = [{folds['assignments'][i]['group'] for i in ids} for ids in sets]
        assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        assert all(folds['assignments'][i]['outer_fold'] == int(fold) for i in split['test'])
        all_test.extend(split['test'])
        print('Fold %s: train=%d val=%d test=%d' % (fold, *map(len, sets)), flush=True)
    assert len(all_test) == len(set(all_test)) == len(records)
    print('Dataset and group separation: PASS (44,001 CDS; five folds)', flush=True)


def check_gpu():
    import torch
    if torch.__version__ != '2.0.0+cu118':
        raise RuntimeError('This frozen package expects torch 2.0.0+cu118; see README_TRAINING_ZH.md. Found '+torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available to this Python environment')
    device = torch.cuda.get_device_properties(0)
    print(json.dumps(dict(torch=torch.__version__, gpu=device.name, total_memory_gib=device.total_memory/2**30)), flush=True)
    # Exercise a kernel so an unsupported GPU architecture fails before a long run.
    value = torch.ones((32, 32), device='cuda')
    assert float((value@value).sum()) == 32768.


def invoke(module, *arguments):
    subprocess.run([sys.executable, '-u', '-m', module, *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'benchmark', 'train', 'train-all', 'full', 'status'))
    parser.add_argument('--fold', type=int, choices=range(5), default=0, help='0..4; fold 0 is the first fold')
    parser.add_argument('--profile', choices=PROFILES, default='uniform')
    parser.add_argument('--output', default='results', help='Relative to package root, or an absolute path')
    parser.add_argument('--gpu', action='store_true', help='Also check CUDA when using check')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.output).resolve()
    if args.command == 'status':
        print('Output:', output)
        files = [output/'status.json', output/'failure.json', output/'complete.json']
        files += sorted(output.glob('fold_*/*/progress.json'))
        for path in files:
            if path.is_file():
                print(str(path.relative_to(output)), path.read_text(encoding='utf-8'))
        print('Training runs completed:', len(list(output.glob('fold_*/*/complete.json'))), '/ 15')
        print('Outer folds completed:', len(list(output.glob('fold_*/complete.json'))), '/ 5')
        return
    check_files()
    if args.command == 'check':
        check_data()
        if args.gpu:
            check_gpu()
        return
    check_gpu()
    if args.command == 'benchmark':
        invoke('latent_decoder.v2_train', '--data', ROOT/'data', '--output', output/'hardware_benchmark', '--fold', args.fold, '--benchmark')
    elif args.command == 'full':
        invoke('latent_decoder.v2_experiment', '--data', ROOT/'data', '--output', output)
    else:
        for fold in range(5) if args.command == 'train-all' else [args.fold]:
            for profile in PROFILES if args.command == 'train-all' else [args.profile]:
                invoke('latent_decoder.v2_train', '--data', ROOT/'data', '--output', output/f'fold_{fold}'/profile,
                       '--fold', fold, '--profile', profile)


if __name__ == '__main__':
    main()
