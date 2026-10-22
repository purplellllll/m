"""Queued, inner-selected V2 fine-tuning; fixed-objective, paired V1 acceptance."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from .budget_js_extension import save_json
from .evaluate_prior import shuffle_latents
from .length_diagnostics import EncoderPool, gradient_summary, length_group
from .model import CodonVMAE, PAD, collate_records, corrupt, synonymous_logits, vae_loss
from .train import learning_rate_factor, sha256
from .v2_data import pick, read_corpus
from .v2_train import GROUPS, PROFILES, atomic_torch, corpus, windows

V1_SHA = '8009efdc3fc3ee61c82925b02feba600be3d6409f923838a612c26eb4af3d673'
OLD_DATA_SHA = 'b11cd373b8fb8f5f7b79fe93089312b73142266cf3893fe270c7eee1fa9d77de'
BASE = dict(gain=1., lr=5e-5, profile='uniform', beta=.001, weight_decay=.01, free_bits=0.)
CHANGES = dict(control={}, gain_half=dict(gain=.5), gain_double=dict(gain=2.),
               lr_half=dict(lr=2.5e-5), lr_double=dict(lr=1e-4),
               short_fast=dict(profile='short_fast'), long_fast=dict(profile='long_fast'),
               kl_low=dict(beta=.0003), kl_high=dict(beta=.003),
               decay_low=dict(weight_decay=.001), free_bits=dict(free_bits=.02))


class ScaledVMAE(CodonVMAE):
    def __init__(self, gain=1., **kwargs):
        super().__init__(**kwargs)
        self.gain = gain

    def decode(self, z, target_tokens):
        return super().decode(self.gain*z, target_tokens)


def load_model(path, recipe=None, initialize=False, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    recipe = recipe or state.get('recipe', BASE)
    model = ScaledVMAE(gain=recipe['gain'], **state['config']['model'])
    model.load_state_dict(state['model'])
    if initialize:
        # Preserve initial logits; gain changes the subsequent parameterization, not the starting function.
        with torch.no_grad():
            model.from_latent.weight.div_(recipe['gain']/state.get('recipe', BASE)['gain'])
    return model.to(device), state['config']


def fixed_mask(tokens, identity, seed):
    number = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:15], 16)
    return corrupt(tokens, .15, torch.Generator(device=tokens.device).manual_seed(number))


def metrics_summary(by_group):
    return {name: sum(by_group[g][name] for g in GROUPS)/3
            for name in ('ce', 'full_ce', 'kl', 'loss')}


@torch.no_grad()
def evaluate(model, records, seeds=(173,), probes=False):
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    sums = {g: dict(ce=0., full_ce=0., kl=0., masked_sites=0, valid_sites=0, sequences=0)
            for g in GROUPS}
    pools = {g: EncoderPool(model.latent_dim) for g in GROUPS}
    probe_rows, rows = {g: [] for g in GROUPS}, []
    for record in sorted(records, key=lambda r: r['sequence_id']):
        group = length_group(record['n_codons']-1)
        tokens = collate_records([record]).to(device)
        valid = tokens < PAD
        for seed in seeds:
            inputs, selected = fixed_mask(tokens, record['sequence_id'], seed)
            logits, mu, logvar = model(inputs, tokens, sample=False)
            ce = F.cross_entropy(synonymous_logits(logits, tokens).transpose(1, 2),
                                 tokens.clamp_max(63), reduction='none')
            kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1)
            row = dict(sequence_id=record['sequence_id'], group=group, mask_seed=seed,
                       ce=float(ce[selected].sum()), full_ce=float(ce[valid].sum()),
                       kl=float(kl[valid].sum()), masked_sites=int(selected.sum()),
                       valid_sites=int(valid.sum()), sequences=1)
            rows.append(row)
            for key in sums[group]:
                sums[group][key] += row[key]
            if probes and seed == seeds[0]:
                pools[group].add(mu, logvar, tokens)
                if len(probe_rows[group]) < 16 and selected.any():
                    baseline = float(ce[selected].mean())
                    high = mu[valid].var(0, unbiased=False) > .01
                    changed = dict(zero=torch.zeros_like(mu),
                        shuffled=shuffle_latents(mu, tokens, selected,
                            torch.Generator(device=device).manual_seed(9473)))
                    for name, dimensions in [('zero_high', high), ('zero_low', ~high)]:
                        changed[name] = mu.masked_fill(dimensions[None, None, :], 0.)
                    intervention = {}
                    for name, z in changed.items():
                        altered = F.cross_entropy(synonymous_logits(model.decode(z, tokens), tokens).transpose(1, 2),
                                                  tokens.clamp_max(63), reduction='none')
                        intervention[name+'_ce_increase'] = float(altered[selected].mean())-baseline
                    with torch.enable_grad():
                        z = mu.detach().requires_grad_()
                        nll = F.cross_entropy(synonymous_logits(model.decode(z, tokens), tokens).transpose(1, 2),
                                              tokens.clamp_max(63), reduction='none')[selected].mean()
                        gradient, = torch.autograd.grad(nll, z)
                    probe_rows[group].append(dict(sequence_id=record['sequence_id'],
                        high_variance_dimensions=int(high.sum()), gradient=gradient_summary(gradient[valid]),
                        **intervention))
    metrics = {}
    for group, s in sums.items():
        if not s['masked_sites'] or not s['valid_sites']:
            raise ValueError('Empty evaluation length group: '+group)
        ce, full, kl = s['ce']/s['masked_sites'], s['full_ce']/s['valid_sites'], s['kl']/s['valid_sites']
        metrics[group] = dict(ce=ce, full_ce=full, kl=kl, loss=full+ce+.001*kl,
                              masked_sites=s['masked_sites'], valid_sites=s['valid_sites'],
                              sequences=s['sequences']//len(seeds))
        if probes:
            metrics[group]['latent'] = dict(encoder=pools[group].summary(), interventions=probe_rows[group])
    return dict(by_length=metrics, macro=metrics_summary(metrics), paired_rows=rows,
                mask_seeds=list(seeds), loss_definition='full_ce + masked_mutable_ce + 0.001 * raw_kl_per_token')


def training_loss(logits, tokens, mu, logvar, selected, recipe):
    loss, terms = vae_loss(logits.float(), tokens, mu.float(), logvar.float(), selected, beta=0.)
    kl = -.5*(1+logvar.float()-mu.float().square()-logvar.float().exp())
    regularizer = kl[tokens < PAD].mean(0).clamp_min(recipe['free_bits']).sum()
    return loss+recipe['beta']*regularizer, terms


def train_trial(data_dir, initial, output, fold, recipe, stage, epochs):
    if (set(recipe) != set(BASE) or recipe['profile'] not in PROFILES
            or any(not math.isfinite(recipe[k]) or recipe[k] <= 0
                   for k in ('gain', 'lr', 'beta', 'weight_decay'))
            or not math.isfinite(recipe['free_bits']) or recipe['free_bits'] < 0):
        raise ValueError('Invalid follow-up recipe')
    output.mkdir(parents=True, exist_ok=True)
    source = sha256(Path(__file__))
    plan = dict(recipe=recipe, fold=fold, stage=stage, epochs=epochs, source_sha256=source,
                initial_sha256=sha256(initial), folds_sha256=sha256(data_dir/'folds.json'))
    if (output/'plan.json').exists():
        assert json.loads((output/'plan.json').read_text()) == plan, 'Changed follow-up trial on resume'
    else:
        save_json(output/'plan.json', plan)
    if (output/'complete.json').exists():
        (output/'last.pt').unlink(missing_ok=True)
        return json.loads((output/'complete.json').read_text())
    split = corpus(data_dir, fold)
    records = pick(split['train'], 512, 'followup-screen-train') if stage == 'screen' else split['train']
    validation = pick(split['val'], 32 if stage == 'screen' else 128, 'followup-validation')
    seed = 20261025+fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, config = load_model(initial, recipe, initialize=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    start, step, best = 0, 0, None
    if (output/'last.pt').exists():
        state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, step, best = state['epoch'], state['step'], state['best']
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    n_windows = len(windows(records, seed))
    for epoch in range(start, epochs):
        model.train().requires_grad_(True)
        sums, skipped = dict(loss=0., ce=0., full_ce=0., samples=0), 0
        started = time.monotonic()
        for group, indices in windows(records, seed+epoch):
            optimizer.zero_grad(set_to_none=True)
            lr = recipe['lr']*PROFILES[recipe['profile']][GROUPS.index(group)]*learning_rate_factor(
                step, 'warmup_cosine', max(1, n_windows//2), epochs*n_windows)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            for offset in range(0, len(indices), 8):
                batch = [records[i] for i in indices[offset:offset+8]]
                tokens = collate_records(batch).cuda()
                inputs, selected = corrupt(tokens, .15)
                with torch.cuda.amp.autocast():
                    logits, mu, logvar = model(inputs, tokens)
                loss, terms = training_loss(logits, tokens, mu, logvar, selected, recipe)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite follow-up training loss')
                scaler.scale(loss*len(batch)/len(indices)).backward()
                for key, value in [('loss', loss), ('ce', terms['masked_ce']), ('full_ce', terms['reconstruction_ce'])]:
                    sums[key] += float(value)*len(batch)
                sums['samples'] += len(batch)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < old_scale
            step += 1
            if step % 25 == 0:
                progress = dict(event='followup_train', fold=fold, trial=output.name, stage=stage,
                    epoch=epoch+1, step=step, group=group, lr=lr,
                    **{'train_'+k: sums[k]/sums['samples'] for k in ('loss', 'ce', 'full_ce')})
                save_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        assert skipped < max(10, n_windows*.05), 'Too many AMP skipped updates'
        metrics = evaluate(model, validation)
        improved = best is None or metrics['macro']['loss'] < best['metrics']['macro']['loss']
        if improved:
            best = dict(epoch=epoch+1, metrics=metrics)
            if stage == 'final':
                atomic_torch(output/'best.pt', dict(model=model.state_dict(), config=config, recipe=recipe,
                                                    epoch=epoch+1, objective=metrics['macro']['loss']))
        row = dict(epoch=epoch+1, validation=metrics, training=sums, skipped_updates=skipped,
                   seconds=time.monotonic()-started)
        save_json(output/f'epoch_{epoch+1:02d}.json', row)
        atomic_torch(output/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), epoch=epoch+1, step=step, best=best,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        print(json.dumps(dict(event='followup_epoch', fold=fold, trial=output.name, stage=stage,
                             epoch=epoch+1, **metrics['macro'])), flush=True)
    result = dict(recipe=recipe, stage=stage, best=best, initial_sha256=plan['initial_sha256'])
    if stage == 'final':
        result['best_sha256'] = sha256(output/'best.pt')
    save_json(output/'complete.json', result)
    (output/'last.pt').unlink()
    return result


def ranked(metrics, baseline):
    better = metrics['macro']['ce'] < baseline['macro']['ce'] and metrics['macro']['loss'] < baseline['macro']['loss']
    return (not better, metrics['macro']['loss'], metrics['macro']['ce'])


def combine(screen):
    result = dict(BASE)
    families = [('gain_half', 'gain_double'), ('lr_half', 'lr_double', 'short_fast', 'long_fast'),
                ('kl_low', 'kl_high'), ('decay_low',), ('free_bits',)]
    for family in families:
        name = min(('control',)+family, key=lambda n: screen[n]['best']['metrics']['macro']['loss'])
        result.update(CHANGES[name])
    return result


def safe_reference(records, manifest):
    bad = {manifest['assignments'][r['sequence_id']]['group'] for r in records if r['split'] != 'test'}
    return {r['sequence_id'] for r in records if r['split'] == 'test'
            and manifest['assignments'][r['sequence_id']]['group'] not in bad}


def acceptance(candidate, reference):
    comparison = {}
    for group in (*GROUPS, 'macro'):
        a = candidate['macro'] if group == 'macro' else candidate['by_length'][group]
        b = reference['macro'] if group == 'macro' else reference['by_length'][group]
        comparison[group] = {key+'_delta': a[key]-b[key] for key in ('ce', 'loss')}
    return dict(passed=all(v < 0 for row in comparison.values() for v in row.values()),
                deltas=comparison, criterion='CE and fixed loss strictly lower in every length bin and macro')


def source_plan(data, prior, v1):
    export = json.loads((data/'export.json').read_text())
    assert export['original_manifest_sha256'] == OLD_DATA_SHA
    manifest = json.loads((data/'folds.json').read_text())
    assert sha256(data/'corpus.jsonl.gz') == manifest['corpus_sha256'] == export['files_sha256']['corpus.jsonl.gz']
    assert sha256(v1) == V1_SHA, 'Wrong V1 baseline checkpoint'
    state = torch.load(v1, map_location='cpu', weights_only=False)
    assert state['corpus_sha256'] == OLD_DATA_SHA and state['next_epoch'] == 50
    del state
    dependencies = ('v2_train.py', 'v2_data.py', 'model.py', 'train.py',
                    'length_diagnostics.py', 'evaluate_prior.py', 'tokenizer.py')
    return dict(source_sha256=sha256(Path(__file__)), data_sha256=sha256(data/'folds.json'),
        corpus_sha256=sha256(data/'corpus.jsonl.gz'), v1_sha256=V1_SHA,
        dependency_sha256={name: sha256(Path(__file__).with_name(name)) for name in dependencies},
        torch_version=torch.__version__,
        prior_results=str(prior.resolve()), recipes={k: dict(BASE, **v) for k, v in CHANGES.items()},
        screen=dict(epochs=2, train_per_length=512, val_per_length=32),
        final=dict(epochs=8, val_per_length=128, maximum_candidates_per_fold=4),
        evaluation=dict(mask_seeds=[173, 174, 175], fixed_kl_weight=.001,
            v1_reference='epoch50; original-test groups disjoint from original train and val',
            limitation='Original test set has been used in previous exploratory studies; this is not a new external validation.'),
        folds=list(range(5)))


def run(args):
    plan = source_plan(args.data, args.prior, args.v1)
    assert json.loads((args.output/'queued_plan.json').read_text()) == plan, 'Queued source/data changed while waiting'
    if (args.output/'plan.json').exists():
        assert json.loads((args.output/'plan.json').read_text()) == plan
    else:
        save_json(args.output/'plan.json', plan)
    selections = []
    for fold in range(5):
        out = args.output/f'fold_{fold}'
        out.mkdir(exist_ok=True)
        if (out/'selection.json').exists():
            selections.append(json.loads((out/'selection.json').read_text()))
            continue
        if shutil.disk_usage(args.output).free < 5*1024**3:
            raise OSError('Need at least 5 GiB free for follow-up working checkpoints; existing training files are retained')
        split = corpus(args.data, fold)
        val = pick(split['val'], 128, 'followup-validation')
        controls = {}
        for profile in PROFILES:
            checkpoint = args.prior/f'fold_{fold}'/profile/'best.pt'
            done = json.loads(checkpoint.with_name('complete.json').read_text())
            assert sha256(checkpoint) == done['best_sha256']
            path = out/f'initial_{profile}.json'
            if not path.exists():
                model, _ = load_model(checkpoint)
                save_json(path, evaluate(model, val))
                del model
                torch.cuda.empty_cache()
            controls[profile] = dict(metrics=json.loads(path.read_text()), checkpoint=str(checkpoint.resolve()),
                                     sha256=done['best_sha256'], origin='existing_'+profile)
        initial = args.prior/f'fold_{fold}'/'uniform'/'best.pt'
        screen = {}
        for name, changes in CHANGES.items():
            command = [sys.executable, '-m', 'latent_decoder.v2_followup', 'worker', '--data', str(args.data),
                '--initial', str(initial), '--output', str(out/'screen'/name), '--fold', str(fold),
                '--recipe', json.dumps(dict(BASE, **changes)), '--stage', 'screen', '--epochs', '2']
            subprocess.run(command, check=True)
            screen[name] = json.loads((out/'screen'/name/'complete.json').read_text())
        top = sorted((n for n in screen if n != 'control'),
                     key=lambda n: ranked(screen[n]['best']['metrics'], screen['control']['best']['metrics']))[:2]
        candidates = dict(control=dict(BASE), **{name: dict(BASE, **CHANGES[name]) for name in top})
        combined = combine(screen)
        if combined not in candidates.values():
            candidates['combined'] = combined
        save_json(out/'final_candidates.json', candidates)
        baseline = controls['uniform']['metrics']
        champion = min(controls.values(), key=lambda c: ranked(c['metrics'], baseline))
        leaderboard = list(controls.values())
        for name, recipe in candidates.items():
            trial = out/'final'/name
            receipt = trial/'promotion.json'
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                champion = saved['champion']
                leaderboard.append(saved['candidate'])
                continue
            command = [sys.executable, '-m', 'latent_decoder.v2_followup', 'worker', '--data', str(args.data),
                '--initial', str(initial), '--output', str(trial), '--fold', str(fold),
                '--recipe', json.dumps(recipe), '--stage', 'final', '--epochs', '8']
            subprocess.run(command, check=True)
            result = json.loads((trial/'complete.json').read_text())
            candidate = dict(metrics=result['best']['metrics'], sha256=result['best_sha256'],
                             checkpoint=str((trial/'best.pt').resolve()), origin=name, recipe=recipe)
            if ranked(candidate['metrics'], baseline) < ranked(champion['metrics'], baseline):
                target = out/f'selected_{name}.pt'
                if target.exists():
                    assert sha256(target) == result['best_sha256']
                else:
                    temporary = target.with_suffix('.tmp')
                    shutil.copyfile(trial/'best.pt', temporary)
                    temporary.replace(target)
                old = Path(champion['checkpoint'])
                candidate['checkpoint'] = str(target.resolve())
                champion = candidate
                save_json(receipt, dict(champion=champion, candidate=candidate))
                if old.parent.resolve() == out.resolve() and old.name.startswith('selected_') and old != target:
                    old.unlink(missing_ok=True)
            else:
                save_json(receipt, dict(champion=champion, candidate=candidate))
            (trial/'best.pt').unlink(missing_ok=True)
            leaderboard.append(candidate)
        selection = dict(fold=fold, **champion, leaderboard=leaderboard,
            selection_scope='Inner validation only; no outer or V1 metrics used',
            learning_rate_comparison={p: controls[p]['metrics']['by_length'] for p in PROFILES})
        save_json(out/'selection.json', selection)
        selections.append(selection)
    # Lock every fold's selection before opening any held-out evaluation.
    save_json(args.output/'selections_locked.json', selections)
    manifest = json.loads((args.data/'folds.json').read_text())
    records = read_corpus(args.data/'corpus.jsonl.gz')
    safe = safe_reference(records, manifest)
    cohort = [r for r in records if r['sequence_id'] in safe]
    save_json(args.output/'v1_comparison_cohort.json', dict(ids=sorted(safe), count=len(safe),
        group_policy='Exclude every V2 homology group containing any original train or val record'))
    totals = {'candidate': [], 'V1': [], 'uniform': [], 'candidate_outer': []}
    for selection in selections:
        fold, out = selection['fold'], args.output/f"fold_{selection['fold']}"
        ids = set(manifest['splits'][str(fold)]['test'])
        test = [r for r in records if r['sequence_id'] in ids]
        reference_test = [r for r in cohort if r['sequence_id'] in ids]
        for arm, checkpoint, chosen in [('candidate', Path(selection['checkpoint']), reference_test),
                ('V1', args.v1, reference_test), ('uniform', args.prior/f'fold_{fold}'/'uniform'/'best.pt', test),
                ('candidate_outer', Path(selection['checkpoint']), test)]:
            target = out/f'evaluation_{arm}.json'
            if not target.exists():
                expected = selection['sha256'] if arm.startswith('candidate') else (
                    V1_SHA if arm == 'V1' else json.loads(checkpoint.with_name('complete.json').read_text())['best_sha256'])
                assert sha256(checkpoint) == expected
                model, _ = load_model(checkpoint)
                result = evaluate(model, chosen, seeds=(173, 174, 175), probes=arm in ('candidate', 'V1'))
                result['checkpoint_sha256'] = expected
                save_json(target, result)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            totals[arm].extend(json.loads(target.read_text())['paired_rows'])
    aggregate = {arm: aggregate_rows(rows) for arm, rows in totals.items()}
    assert [(r['sequence_id'], r['mask_seed']) for r in totals['candidate']] == [
        (r['sequence_id'], r['mask_seed']) for r in totals['V1']]
    summary = dict(acceptance_vs_v1=acceptance(aggregate['candidate'], aggregate['V1']), metrics=aggregate,
        v2_outer_comparison=acceptance(aggregate['candidate_outer'], aggregate['uniform']),
        latent_comparison={arm: latent_summary(args.output, arm) for arm in ('candidate', 'V1')},
        v1_cohort_cds=len(safe), selections=[dict(fold=s['fold'], origin=s['origin'], recipe=s.get('recipe')) for s in selections],
        limitation=plan['evaluation']['limitation'],
        latent_note='AU and gradient participation are diagnostics; neither proves 256 independent useful dimensions.')
    save_json(args.output/'summary.json', summary)
    save_json(args.output/'complete.json', dict(summary_sha256=sha256(args.output/'summary.json'),
                                              accepted=summary['acceptance_vs_v1']['passed']))
    print(json.dumps(summary), flush=True)


def aggregate_rows(rows):
    result = {}
    for g in GROUPS:
        current = [r for r in rows if r['group'] == g]
        masked, valid = sum(r['masked_sites'] for r in current), sum(r['valid_sites'] for r in current)
        if not masked or not valid:
            raise ValueError('Missing held-out length bin: '+g)
        ce = sum(r['ce'] for r in current)/masked
        full = sum(r['full_ce'] for r in current)/valid
        kl = sum(r['kl'] for r in current)/valid
        result[g] = dict(ce=ce, full_ce=full, kl=kl, loss=full+ce+.001*kl,
                         sequences=len({r['sequence_id'] for r in current}))
    return dict(by_length=result, macro=metrics_summary(result))


def latent_summary(output, arm):
    results = [json.loads((output/f'fold_{f}'/f'evaluation_{arm}.json').read_text()) for f in range(5)]
    summary = {}
    for group in GROUPS:
        latent = [r['by_length'][group]['latent'] for r in results]
        probes = [p for r in latent for p in r['interventions']]
        summary[group] = dict(
            mean_fold_au_001=sum(r['encoder']['pooled']['active_dimensions']['0.01'] for r in latent)/5,
            mean_fold_within_aa_au_001=sum(r['encoder']['within_amino_acid']['active_dimensions']['0.01'] for r in latent)/5,
            mean_fold_variance_participation=sum(r['encoder']['pooled']['variance_energy_participation'] for r in latent)/5,
            mean_fold_mu_rms=sum(r['encoder']['sequence_mean']['mu_rms'] for r in latent)/5,
            mean_probe_gradient_dimensions_99pct=sum(p['gradient']['dims_for_99pct_gradient_energy'] for p in probes)/len(probes),
            intervention_ce_increase={key: sum(p[key] for p in probes)/len(probes)
                for key in ('zero_ce_increase', 'shuffled_ce_increase', 'zero_high_ce_increase', 'zero_low_ce_increase')},
            functional_probe_cds=len(probes))
    return summary


def wait_ready(prior):
    done = [prior/f'fold_{f}'/p/'complete.json' for f in range(5) for p in PROFILES]
    status = prior/'deployment'/'status.json'
    while True:
        phase = json.loads(status.read_text()) if status.exists() else {}
        if phase.get('stage') == 'failed':
            raise RuntimeError('Upstream training failed; follow-up will not start')
        count = sum(p.exists() for p in done)
        if count == 15 and phase.get('stage') == 'completed':
            return
        print(json.dumps(dict(event='waiting_for_upstream', completed=count, total=15, stage=phase.get('stage'))), flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('queue', 'worker', 'check'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--v1', type=Path)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--recipe')
    parser.add_argument('--stage', choices=('screen', 'final'))
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'worker':
        return train_trial(args.data, args.initial, args.output, args.fold,
                           json.loads(args.recipe), args.stage, args.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'check':
        plan = source_plan(args.data, args.prior, args.v1)
        records = read_corpus(args.data/'corpus.jsonl.gz')
        manifest = json.loads((args.data/'folds.json').read_text())
        safe = safe_reference(records, manifest)
        counts = {str(f): {g: sum(r['sequence_id'] in safe and
            manifest['assignments'][r['sequence_id']]['outer_fold'] == f and length_group(r['n_codons']-1) == g
            for r in records) for g in GROUPS} for f in range(5)}
        assert all(n > 0 for c in counts.values() for n in c.values())
        lookup = {r['sequence_id']: dict(r, v2_group=manifest['assignments'][r['sequence_id']]['group']) for r in records}
        for split in manifest['splits'].values():
            pick([lookup[i] for i in split['train']], 512, 'followup-screen-train')
            pick([lookup[i] for i in split['val']], 128, 'followup-validation')
        save_json(args.output/'preflight.json', dict(plan=plan, reference_counts=counts, safe_cds=len(safe)))
        print(json.dumps(dict(event='preflight_pass', safe_cds=len(safe), reference_counts=counts)), flush=True)
        return
    import fcntl
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queued_plan = source_plan(args.data, args.prior, args.v1)
        if (args.output/'queued_plan.json').exists():
            assert json.loads((args.output/'queued_plan.json').read_text()) == queued_plan
        else:
            save_json(args.output/'queued_plan.json', queued_plan)
        save_json(args.output/'queue_status.json', dict(stage='waiting', pid=os.getpid()))
        try:
            wait_ready(args.prior)
            with (args.prior/'deployment'/'launch.lock').open('a') as upstream:
                fcntl.flock(upstream, fcntl.LOCK_EX)
                save_json(args.output/'queue_status.json', dict(stage='running', pid=os.getpid()))
                run(args)
            save_json(args.output/'queue_status.json', dict(stage='completed', pid=os.getpid()))
        except Exception:
            save_json(args.output/'queue_status.json', dict(stage='failed', pid=os.getpid(), error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
