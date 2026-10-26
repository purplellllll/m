"""Matched-exposure batch/LR diagnosis; select on the complete safe inner-val panel."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS, emit
from .latent_recovery_gpu import BASE, batch_mask, evaluate, geometry, initialize, panels, reconstruction_objective, training_batches
from .paper_guided_latent_cpu import load_checkpoint
from .train import sha256
from .v2_data import read_corpus
from .v2_train import PROFILES, atomic_torch


def accumulation_windows(batches, accumulation):
    assert accumulation > 0
    if accumulation == 1:
        return [(group, [(index, rows)]) for index, (group, rows) in enumerate(batches, 1)]
    grouped = {g: [(index, rows) for index, (group, rows) in enumerate(batches, 1) if group == g] for g in GROUPS}
    assert len({len(rows) for rows in grouped.values()}) == 1
    assert len(grouped['short']) % accumulation == 0
    return [(group, grouped[group][start:start+accumulation])
            for start in range(0, len(grouped['short']), accumulation) for group in GROUPS]


def lr_factor(index, updates):
    warmup = max(1, round(updates * 24 / 1152))
    return min(1., (index+1)/warmup) * (.25+.75*(1+math.cos(math.pi*index/max(1, updates-1)))/2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--validation-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--accumulation', type=int, choices=(1, 16), required=True)
    parser.add_argument('--microsteps', type=int, default=1152)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--profile', choices=PROFILES, default='uniform')
    args = parser.parse_args()
    assert math.isfinite(args.lr) and args.lr > 0
    assert args.microsteps > 0 and args.microsteps % (3*args.accumulation) == 0
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    selected = panels(args.data, 0, 1024, 64)
    reference = json.loads(args.validation_reference.read_text())['plan']
    assert not reference['outer_test_used']
    assert reference['corpus_sha256'] == sha256(args.data/'corpus.jsonl.gz')
    assert reference['folds_sha256'] == sha256(args.data/'folds.json')
    identities = set(reference['records'])
    manifest = json.loads((args.data/'folds.json').read_text())
    assert identities <= set(manifest['splits']['0']['val']) and len(identities) == 154
    validation = [dict(r, v2_group=manifest['assignments'][r['sequence_id']]['group'])
                  for r in read_corpus(args.data/'corpus.jsonl.gz') if r['sequence_id'] in identities]
    assert len(validation) == 154
    batches = training_batches(selected['train'], args.seed, args.microsteps)
    windows = accumulation_windows(batches, args.accumulation)
    plan = dict(source_sha256=sha256(Path(__file__)), core_sha256=sha256(Path(__file__).with_name('latent_recovery_gpu.py')),
        initial_sha256=sha256(args.initial), validation_reference_sha256=sha256(args.validation_reference),
        seed=args.seed, lr=args.lr, profile=args.profile, accumulation=args.accumulation,
        microbatch=2, microsteps=args.microsteps, optimizer_updates=len(windows), cds_exposures=2*args.microsteps,
        train_ids=[r['sequence_id'] for r in selected['train']],
        validation_ids=sorted(identities), validation_masks=[175,176], outer_test_used=False,
        selection='Minimum fixed loss on all154 old-test-safe INNER-val CDS, reused development data',
        schedule='Same relative warmup/cosine in optimizer-update units; effective batch changes update count and weight-decay applications',
        length_batching='Accumulate homogeneous length groups; preserve original sequence exposure/mask/noise index across accumulation settings')
    if (args.output/'plan.json').exists():
        assert json.loads((args.output/'plan.json').read_text()) == plan
    else:
        save_json(args.output/'plan.json', plan)
    if (args.output/'complete.json').exists():
        return
    model, config = initialize(args.initial, BASE)
    model.eval().requires_grad_(True)
    encoder = [p for n,p in model.named_parameters() if not n.startswith(('decoder.','from_latent.','output.'))]
    decoder = [p for n,p in model.named_parameters() if n.startswith(('decoder.','from_latent.','output.'))]
    optimizer = torch.optim.AdamW([dict(params=encoder), dict(params=decoder)], lr=args.lr, weight_decay=.01)
    start, best, history = 0, float('inf'), []
    if (args.output/'last.pt').exists():
        state = load_checkpoint(args.output/'last.pt')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start, best, history = state['update'], state['best'], state['history']
        del state
    else:
        metrics = evaluate(model, validation)
        best = metrics['macro']['fixed_loss']
        history = [dict(update=0, microstep=0, **metrics['macro'])]
        atomic_torch(args.output/'best.pt', dict(model=model.state_dict(), config=config, recipe=BASE, step=0, microstep=0))
        save_json(args.output/'initial_inner.json', metrics)
    clock = time.monotonic()
    for index in range(start, len(windows)):
        group, microbatches = windows[index]
        rate = args.lr * PROFILES[args.profile][GROUPS.index(group)] * lr_factor(index, len(windows))
        for pg in optimizer.param_groups:
            pg['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        terms_sum = Counter()
        for offset, (noise_index, records) in enumerate(microbatches):
            assert len(records) == 2
            microstep = index*args.accumulation + offset + 1
            torch.manual_seed(args.seed + noise_index*1000)
            tokens, inputs, hidden = batch_mask(records, args.seed+noise_index, 'cuda')
            loss, terms, _, _ = reconstruction_objective(model, tokens, inputs, hidden, BASE,
                args.seed+noise_index*1000+7, True)
            assert torch.isfinite(loss)
            (loss/len(microbatches)).backward()
            terms_sum.update(loss=float(loss), ce=float(terms['masked_ce']), kl=float(terms['kl_per_token']))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if microstep % 48 == 0:
            emit('batch_recovery_train', update=index+1, updates=len(windows), microstep=microstep,
                effective_batch=2*args.accumulation, group=group, lr=rate, gradient_norm=float(norm),
                **{k:v/len(microbatches) for k,v in terms_sum.items()})
        if microstep % 144 == 0 or index+1 == len(windows):
            metrics = evaluate(model, validation)
            history.append(dict(update=index+1, microstep=microstep, **metrics['macro']))
            if metrics['macro']['fixed_loss'] < best:
                best = metrics['macro']['fixed_loss']
                atomic_torch(args.output/'best.pt', dict(model=model.state_dict(), config=config, recipe=BASE,
                    step=index+1, microstep=microstep))
            atomic_torch(args.output/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                update=index+1, best=best, history=history))
            emit('batch_recovery_inner', update=index+1, microstep=microstep, **metrics['macro'])
    state = load_checkpoint(args.output/'best.pt')
    model.load_state_dict(state['model'], strict=True)
    chosen_update, chosen_microstep = state['step'], state['microstep']
    del state
    metrics = evaluate(model, validation)
    metrics['geometry'] = geometry(model, validation, limit=8)[0]
    save_json(args.output/'selected_inner.json', metrics)
    save_json(args.output/'complete.json', dict(selected_update=chosen_update, selected_microstep=chosen_microstep,
        metrics=metrics['macro'], history=history, seconds=time.monotonic()-clock,
        best_sha256=sha256(args.output/'best.pt'), outer_test_used=False, goal_complete=False))


if __name__ == '__main__':
    main()
