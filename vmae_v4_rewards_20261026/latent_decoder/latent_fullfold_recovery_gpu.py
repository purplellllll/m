"""One full training pass per homology fold, from V1 with explicitly safe validation."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import torch

from .budget_js_extension import save_json
from .latent_collapse_cpu import GROUPS, emit
from .latent_correlated_projection import ProjectedVMAE, project_state, projected_kl
from .latent_correlated_projection_gpu import evaluate_projected
from .latent_information_probe import load_model
from .latent_objective_recovery_gpu import lr_factor
from .latent_pca_initialization import compress_pca
from .latent_pca_recovery_gpu import calibration
from .latent_recovery_gpu import BASE, batch_mask, evaluate, geometry, initialize
from .latent_training_objectives import reconstruction_objective
from .model import PAD, vae_loss
from .paper_guided_latent_cpu import load_checkpoint
from .train import sha256
from .v2_data import pick
from .v2_train import PROFILES, atomic_torch, windows


ARMS = dict(control=(256, .001, 0.), candidate=(256, .003, .5), compact=(16, .003, .5))


class FullfoldProjected(ProjectedVMAE):
    def encode_features(self, inputs, targets):
        captured = []
        hook = self.encoder.layers[len(self.encoder.layers)//2-1].register_forward_hook(
            lambda module, args, result: captured.append(result))
        try:
            mu, lv = self.encode(inputs, targets)
        finally:
            hook.remove()
        return mu, lv, captured[0]


def fold_data(data, fold):
    manifest = json.loads((data/'folds.json').read_text())
    assert sha256(data/'corpus.jsonl.gz') == manifest['corpus_sha256']
    split = manifest['splits'][str(fold)]
    wanted = {identity: panel for panel in ('train', 'val') for identity in split[panel]}
    bad, rows = set(), dict(train=[], val=[])
    with gzip.open(data/'corpus.jsonl.gz', 'rt') as stream:
        for record in map(json.loads, stream):
            identity = record['sequence_id']
            group = manifest['assignments'][identity]['group']
            if record['split'] != 'test':
                bad.add(group)
            if identity in wanted:
                rows[wanted[identity]].append(dict(record, v2_group=group))
    rows['val'] = [r for r in rows['val'] if r['v2_group'] not in bad]
    assert len(rows['train']) == len(split['train']) and rows['val']
    assert not {r['v2_group'] for r in rows['train']} & {r['v2_group'] for r in rows['val']}
    return rows


def compact_objective(model, tokens, inputs, selected, recipe, noise_seed):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        mu, lv = model.distribution(inputs, tokens)
        mu, lv = mu.float(), lv.float()
        generator = torch.Generator(device='cuda').manual_seed(noise_seed)
        noise = torch.randn(lv.shape, device='cuda', generator=generator)
        logits = model.decode(model.sample(mu, lv, noise), tokens)
    zero = torch.zeros_like(mu)
    rec, terms = vae_loss(logits.float(), tokens, zero, zero, selected, beta=0.)
    if recipe['mean_weight']:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            mean_logits = model.decode(mu, tokens)
        mean_rec, _ = vae_loss(mean_logits.float(), tokens, zero, zero, selected, beta=0.)
        rec = (rec + recipe['mean_weight']*mean_rec)/(1+recipe['mean_weight'])
    # Full covariance KL stays FP32; marginal diagonal KL would be incorrect here.
    penalty = projected_kl(mu, lv, model.noise_projection)[tokens<PAD].mean()
    terms['kl_per_token'] = penalty.detach()
    return rec + recipe['training_beta']*penalty, terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = root/'source/data'
    selected = fold_data(data, args.fold)
    output = root/'fullfold_recovery'/('smoke' if args.smoke else f'fold_{args.fold}')/args.arm
    output.mkdir(parents=True, exist_ok=True)
    seed = 20261101+args.fold
    schedule = windows(selected['train'], seed, size=32)
    indices = [index for _, chosen in schedule for index in chosen]
    assert sorted(indices) == list(range(len(selected['train']))), 'Every training CDS exactly once'
    if args.smoke:
        schedule = schedule[:3]
    dimension, beta, mean = ARMS[args.arm]
    recipe = dict(BASE, training_beta=beta, mean_weight=mean)
    learning_rate = 1e-5*2304/len(selected['train'])
    dependencies = {name: sha256(Path(__file__).with_name(name)) for name in (
        'latent_training_objectives.py', 'latent_correlated_projection.py', 'latent_correlated_projection_gpu.py',
        'latent_pca_initialization.py', 'latent_pca_recovery_gpu.py', 'latent_recovery_gpu.py',
        'latent_objective_recovery_gpu.py', 'v2_train.py', 'model.py')}
    plan = dict(source_sha256=sha256(Path(__file__)), dependencies=dependencies, fold=args.fold, arm=args.arm,
        seed=seed, recipe=recipe, dimension=dimension, learning_rate=learning_rate, length_profile='long_fast',
        rate_heuristic='base1e-5*2304/Ntrain: match approximate integrated pilot update rate across a full pass, not a theoretical optimum',
        initial_sha256=sha256(root/'V1_epoch50_model_only.pt'), folds_sha256=sha256(data/'folds.json'),
        training_ids=[r['sequence_id'] for r in selected['train']], validation_ids=[r['sequence_id'] for r in selected['val']],
        optimizer_updates=len(schedule), effective_batch=32, microbatch=2, full_passes=1, smoke=args.smoke,
        validation='All inner-val groups entirely absent from original V1 train/val; masks175/176',
        pretrained_cv='Common V1 initialization; report outer only on entirely V1-unseen groups, not all original outer CDS',
        selection='Prefer minimum fixed loss among CE<V1 and loss<V1 inner checkpoints, else minimum fixed loss; initial permitted',
        dropout=False, covariance='full projected covariance forcompact; learned diagonal otherwise',
        outer_test_used=False, goal_complete=False)
    plan = json.loads(json.dumps(plan))
    if (output/'plan.json').exists():
        assert json.loads((output/'plan.json').read_text()) == plan
    else:
        save_json(output/'plan.json', plan)
    if (output/'complete.json').exists():
        return
    if not (output/'V1_inner.json').exists():
        baseline = load_model(root/'V1_epoch50_model_only.pt', 'cuda')
        save_json(output/'V1_inner.json', evaluate(baseline, selected['val']))
        del baseline
        torch.cuda.empty_cache()
    reference = json.loads((output/'V1_inner.json').read_text())['macro']
    if args.arm == 'compact':
        initial = output/'initial.pt'
        if not initial.exists():
            teacher = load_model(root/'V1_epoch50_model_only.pt', 'cuda')
            records = pick(selected['train'], 8, 'capacity-train-ranking-20261023')
            values = calibration(teacher, records)
            state = load_checkpoint(root/'V1_epoch50_model_only.pt')
            state['recipe'] = dict(BASE)
            _, report = compress_pca(state, *values, dimension)
            compressed = project_state(state, report)
            compressed['recipe'] = recipe
            atomic_torch(initial, compressed)
            save_json(output/'initialization.json', dict(report=report, records=[r['sequence_id'] for r in records],
                initial_sha256=sha256(initial)))
            del teacher, state, compressed, values
            torch.cuda.empty_cache()
        state = load_checkpoint(initial)
        model = FullfoldProjected(noise_dim=state['noise_dim'], **state['config']['model']).cuda()
        model.load_state_dict(state['model'], strict=True)
        config = state['config']
        metadata = dict(architecture='projected_full_covariance', noise_dim=state['noise_dim'])
        del state
        evaluate_model = evaluate_projected
    else:
        model, config = initialize(root/'V1_epoch50_model_only.pt', recipe)
        metadata = {}
        evaluate_model = evaluate
    model.recipe = recipe
    model.eval().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    history, start, best, joint = [], 0, float('inf'), float('inf')
    if (output/'last.pt').exists():
        state = load_checkpoint(output/'last.pt')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start, best, joint, history = state['update'], state['best'], state['joint'], state['history']
        del state
    def checkpoint(update, metrics):
        nonlocal best, joint
        macro = metrics['macro']
        history.append(dict(update=update, **macro))
        image = dict(model=model.state_dict(), config=config, recipe=recipe, step=update, **metadata)
        if macro['fixed_loss'] < best:
            best = macro['fixed_loss']
            atomic_torch(output/'best.pt', image)
        if macro['ce'] < reference['ce'] and macro['fixed_loss'] < reference['fixed_loss'] and macro['fixed_loss'] < joint:
            joint = macro['fixed_loss']
            atomic_torch(output/'best_joint.pt', image)
        save_json(output/f'inner_{update:04d}.json', metrics)
        atomic_torch(output/'last.pt', dict(**image, optimizer=optimizer.state_dict(), update=update,
            best=best, joint=joint, history=history))
        emit('fullfold_inner', fold=args.fold, arm=args.arm, update=update, updates=len(schedule), **macro)
    clock = time.monotonic()
    if not history:
        checkpoint(0, evaluate_model(model, selected['val']))
    for index in range(start, len(schedule)):
        group, chosen = schedule[index]
        rate = learning_rate*PROFILES['long_fast'][GROUPS.index(group)]*lr_factor(index, len(schedule))
        for pg in optimizer.param_groups:
            pg['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        terms_sum = Counter()
        for offset in range(0, len(chosen), 2):
            rows = [selected['train'][i] for i in chosen[offset:offset+2]]
            number = seed+(index*16+offset//2)*1000
            tokens, inputs, hidden = batch_mask(rows, number, 'cuda')
            if args.arm == 'compact':
                loss, terms = compact_objective(model, tokens, inputs, hidden, recipe, number+7)
            else:
                loss, terms, _, _ = reconstruction_objective(model, tokens, inputs, hidden, recipe, number+7, True)
            assert torch.isfinite(loss)
            fraction = len(rows)/len(chosen)
            (loss*fraction).backward()
            terms_sum.update(loss=float(loss)*fraction, ce=float(terms['masked_ce'])*fraction,
                kl=float(terms['kl_per_token'])*fraction)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (index+1) % 24 == 0 or index+1 == len(schedule):
            emit('fullfold_train', fold=args.fold, arm=args.arm, update=index+1, updates=len(schedule),
                group=group, lr=rate, gradient_norm=float(norm), **terms_sum)
        if (index+1) % 72 == 0 or index+1 == len(schedule):
            checkpoint(index+1, evaluate_model(model, selected['val']))
    chosen = output/('best_joint.pt' if joint < float('inf') else 'best.pt')
    state = load_checkpoint(chosen)
    model.load_state_dict(state['model'], strict=True)
    result = evaluate_model(model, selected['val'])
    result['geometry'] = geometry(model, selected['val'], limit=8)[0]
    save_json(output/'selected_inner.json', result)
    save_json(output/'complete.json', dict(fold=args.fold, arm=args.arm, selected_step=state['step'],
        selected_path=str(chosen), selected_sha256=sha256(chosen), metrics=result['macro'], reference=reference,
        has_joint=joint < float('inf'), history=history, seconds=time.monotonic()-clock,
        full_training_cds=len(selected['train']), used_cds=sum(len(chosen) for _, chosen in schedule),
        outer_test_used=False, goal_complete=False))


if __name__ == '__main__':
    main()
