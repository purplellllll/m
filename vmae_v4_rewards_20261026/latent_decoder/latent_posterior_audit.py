"""Compare mean decoding and actual posterior samples on an existing inner-val probe."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS, ce_metrics
from .latent_information_probe import load_model
from .latent_recovery_gpu import batch_mask
from .length_diagnostics import length_group
from .model import PAD
from .train import sha256

DRAWS = (20261030, 20261031, 20261032, 20261033)
METRICS = ('ce', 'full_ce', 'kl', 'fixed_loss')


def posterior_noise(identity, seed, shape):
    number = int(hashlib.sha256(f'posterior:{identity}:{seed}'.encode()).hexdigest()[:15], 16)
    return torch.randn(shape, generator=torch.Generator().manual_seed(number))


@torch.no_grad()
def audit(model, records, mask_seed=175, draws=DRAWS):
    assert draws and len(set(draws)) == len(draws)
    device = next(model.parameters()).device
    rows = []
    for record in records:
        tokens, inputs, selected = batch_mask([record], mask_seed, device)
        assert selected.any(), 'The fixed probe must have hidden mutable targets'
        mu, lv = model.encode(inputs, tokens)
        modes = [('mean', None)] + [('posterior', seed) for seed in draws]
        for mode, seed in modes:
            z = mu
            if seed is not None:
                noise = posterior_noise(record['sequence_id'], seed, mu.shape).to(device)
                z = mu + noise * (.5 * lv).exp()
            z = z * (tokens < PAD)[..., None]
            metrics = ce_metrics(model.decode(z, tokens), tokens, mu, lv, selected)
            assert all(torch.isfinite(torch.tensor(metrics[key])) for key in METRICS)
            rows.append(dict(sequence_id=record['sequence_id'],
                group=length_group(record['n_codons'] - 1), mode=mode, draw_seed=seed,
                mask_seed=mask_seed, **metrics))
    by_length = {}
    for group in GROUPS:
        if not any(row['group'] == group for row in rows):
            continue
        by_length[group] = {}
        for mode in ('mean', 'posterior'):
            chosen = [r for r in rows if r['group'] == group and r['mode'] == mode]
            by_length[group][mode] = {key: statistics.mean(r[key] for r in chosen) for key in METRICS}
    macro = {mode: {key: statistics.mean(g[mode][key] for g in by_length.values())
                   for key in METRICS} for mode in ('mean', 'posterior')}
    return dict(rows=rows, by_length=by_length, macro=macro,
        posterior_minus_mean={key: macro['posterior'][key] - macro['mean'][key] for key in METRICS},
        interpretation='Fixed-mask inner-val diagnostic; posterior decoding is forced even for MAE arms. Not V1 acceptance or outer-test evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = json.loads(args.reference_probe.read_text())['plan']
    manifest = json.loads((args.data / 'folds.json').read_text())
    assert not reference['outer_test_used']
    assert sha256(args.checkpoint) == reference['checkpoint_sha256']
    assert sha256(args.data / 'folds.json') == reference['folds_sha256']
    assert sha256(args.data / 'corpus.jsonl.gz') == reference['corpus_sha256']
    assert sha256(Path(__file__).with_name('latent_recovery_gpu.py')) == reference['core_source_sha256']
    identities = reference['records']
    assert len(identities) == len(set(identities)) == 12
    assert set(identities) <= set(manifest['splits'][str(reference['fold'])]['val'])
    with gzip.open(args.data / 'corpus.jsonl.gz', 'rt') as stream:
        records = {r['sequence_id']: r for r in map(json.loads, stream) if r['sequence_id'] in identities}
    assert set(records) == set(identities)
    records = [records[identity] for identity in identities]
    assert all(sum(length_group(r['n_codons'] - 1) == group for r in records) == 4 for group in GROUPS)
    model = load_model(args.checkpoint, args.device)
    # shortcut: common noise is paired only among equal 256-D models; compressed models need retained-index mapping.
    assert model.latent_dim == 256, 'Cross-capacity noise pairing is outside this audit'
    result = audit(model, records, reference['mask_seed'])
    result['plan'] = dict(checkpoint_sha256=reference['checkpoint_sha256'],
        reference_probe_sha256=sha256(args.reference_probe), source_sha256=sha256(Path(__file__)),
        core_source_sha256=reference['core_source_sha256'], corpus_sha256=reference['corpus_sha256'],
        folds_sha256=reference['folds_sha256'], fold=reference['fold'], records=identities,
        device=args.device, dtype='float32', draws=list(DRAWS), mask_seed=reference['mask_seed'],
        outer_test_used=False, training_performed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), macro=result['macro'])), flush=True)


if __name__ == '__main__':
    main()
