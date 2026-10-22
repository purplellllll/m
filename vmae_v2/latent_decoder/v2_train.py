"""Length-conditioned V2 training; independent initialization and held-out selection per fold."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .budget_js_extension import save_json
from .latent_optimization import write_rows
from .length_diagnostics import length_group
from .model import CodonVMAE, collate_records, corrupt, vae_loss
from .train import codon_frequency, evaluate, learning_rate_factor, sha256
from .v2_data import pick, read_corpus

PROFILES = dict(uniform=(1., 1., 1.), short_fast=(2., 1., .5), long_fast=(.5, 1., 2.))
GROUPS = ('short', 'medium', 'long')


def windows(records, seed, size=32):
    rng = random.Random(seed)
    result = []
    for group in GROUPS:
        indices = [i for i, r in enumerate(records) if length_group(r['n_codons']-1) == group]
        rng.shuffle(indices)
        for start in range(0, len(indices), size*16):
            bucket = sorted(indices[start:start+size*16], key=lambda i: records[i]['n_codons'])
            result.extend((group, bucket[j:j+size]) for j in range(0, len(bucket), size))
    rng.shuffle(result)
    return result


def corpus(data, fold):
    manifest = json.loads((data/'folds.json').read_text())
    assert sha256(data/'corpus.jsonl.gz') == manifest['corpus_sha256']
    records = read_corpus(data/'corpus.jsonl.gz')
    lookup = {r['sequence_id']: dict(r, v2_group=manifest['assignments'][r['sequence_id']]['group']) for r in records}
    return {name: [lookup[i] for i in ids] for name, ids in manifest['splits'][str(fold)].items() if name != 'length_counts'}


def atomic_torch(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='uniform')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--microbatch', type=int, choices=(2, 4, 8), default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = 20261022 + args.fold
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = corpus(args.data, args.fold)
    config = json.loads(Path('latent_decoder/config_lsco_lr_50ep.json').read_text())
    microbatch = args.microbatch
    config['training'].update(seed=seed, batch_size=microbatch, gradient_accumulation=32//microbatch, amp=True,
                              patience=8, minimum_epochs=10, profile=args.profile)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(config=config, fold=args.fold, folds_sha256=sha256(args.data/'folds.json'),
                source_sha256={name: sha256(Path(__file__).with_name(name)) for name in ('v2_train.py', 'v2_data.py', 'model.py', 'train.py')},
                torch_version=torch.__version__, precision='FP16 autocast with FP32 loss and dynamic GradScaler')
    if (args.output/'plan.json').exists():
        assert json.loads((args.output/'plan.json').read_text()) == plan, 'Resume configuration/source mismatch'
    else:
        save_json(args.output/'plan.json', plan)
    model = CodonVMAE(**config['model']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    if args.benchmark:
        measurements = []
        for group in GROUPS:
            candidates = sorted([r for r in data['train'] if length_group(r['n_codons']-1) == group], key=lambda r: r['n_codons'])
            for label, rows in [('median', candidates[len(candidates)//2:len(candidates)//2+microbatch]), ('longest', candidates[-microbatch:])]:
                tokens = collate_records(rows).cuda()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for repeat in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    corrupted, selected = corrupt(tokens, .15)
                    with torch.cuda.amp.autocast():
                        logits, mu, logvar = model(corrupted, tokens)
                    loss, _ = vae_loss(logits.float(), tokens, mu.float(), logvar.float(), selected, beta=.001)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    assert torch.isfinite(loss) and norm > 0
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    elapsed.append(time.monotonic()-start)
                measurements.append(dict(group=group, selection=label, codons=tokens.shape[1], seconds=sum(elapsed[1:])/2,
                                         peak_bytes=torch.cuda.max_memory_allocated(), loss=float(loss)))
                print(json.dumps(measurements[-1]), flush=True)
        save_json(args.output/'benchmark.json', dict(measurements=measurements, train_counts={g: sum(length_group(r['n_codons']-1)==g for r in data['train']) for g in GROUPS}))
        return
    counts = codon_frequency(data['train'])
    validation = pick(data['val'], 128, 'checkpoint-validation')
    num_windows = len(windows(data['train'], seed))
    total_steps, warmup = num_windows*50, max(1, int(num_windows*5))
    start_epoch, step, best, stale = 0, 0, float('inf'), 0
    if (args.output/'last.pt').exists():
        state = torch.load(args.output/'last.pt', map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, step, best, stale = state['epoch'], state['step'], state['best'], state['stale']
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    if (args.output/'complete.json').exists():
        return
    if start_epoch >= 50 or (start_epoch >= 10 and stale >= 8):
        save_json(args.output/'complete.json', dict(epochs=start_epoch, best_objective=best, best_sha256=sha256(args.output/'best.pt')))
        return
    for epoch in range(start_epoch, 50):
        model.train()
        started = time.monotonic()
        sums = {g: dict(loss=0., samples=0, steps=0, last_lr=0.) for g in GROUPS}
        skipped = 0
        for group, indices in windows(data['train'], seed+epoch):
            optimizer.zero_grad(set_to_none=True)
            lr = 1e-4*PROFILES[args.profile][GROUPS.index(group)]*learning_rate_factor(step, 'warmup_cosine', warmup, total_steps)
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = lr
            beta = .001*min(1., (step+1)/max(1, total_steps*.2))
            for offset in range(0, len(indices), microbatch):
                rows = [data['train'][i] for i in indices[offset:offset+microbatch]]
                tokens = collate_records(rows).cuda()
                corrupted, selected = corrupt(tokens, .15)
                with torch.cuda.amp.autocast():
                    logits, mu, logvar = model(corrupted, tokens)
                loss, _ = vae_loss(logits.float(), tokens, mu.float(), logvar.float(), selected, beta=beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite V2 loss')
                scaler.scale(loss*len(rows)/len(indices)).backward()
                sums[group]['loss'] += float(loss)*len(rows)
                sums[group]['samples'] += len(rows)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < old_scale
            step += 1
            sums[group]['steps'] += 1
            sums[group]['last_lr'] = lr
            if step % 25 == 0:
                save_json(args.output/'progress.json', dict(epoch=epoch+1, step=step, group=group, lr=lr, seconds=time.monotonic()-started, skipped_updates=skipped))
                print(json.dumps(dict(event='train', epoch=epoch+1, step=step, lr=lr, group=group)), flush=True)
        assert skipped < max(10, num_windows*.05), 'Too many AMP skipped updates'
        metrics = {g: evaluate(model, [r for r in validation if length_group(r['n_codons']-1)==g], counts, torch.device('cuda'), 2) for g in GROUPS}
        objective = sum(m['masked_input_full_ce']+m['masked_mutable_ce']+.001*m['masked_input_kl_per_token'] for m in metrics.values())/3
        improved = objective < best
        best, stale = (objective, 0) if improved else (best, stale+1)
        state = dict(model=model.state_dict(), config=config, epoch=epoch+1, step=step, best=best, stale=stale,
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_torch(args.output/'last.pt', state)
        if improved:
            atomic_torch(args.output/'best.pt', dict(model=model.state_dict(), config=config, epoch=epoch+1, objective=objective))
        row = dict(epoch=epoch+1, validation=metrics, objective=objective, improved=improved, training=sums,
                   seconds=time.monotonic()-started, skipped_updates=skipped)
        save_json(args.output/f'epoch_{epoch+1:02d}.json', row)
        print(json.dumps(dict(event='epoch', epoch=epoch+1, objective=objective, seconds=row['seconds'])), flush=True)
        if epoch+1 >= 10 and stale >= 8:
            break
    save_json(args.output/'complete.json', dict(epochs=epoch+1, best_objective=best, best_sha256=sha256(args.output/'best.pt')))


if __name__ == '__main__':
    main()
