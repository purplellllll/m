"""Resumable five-fold V2 experiment: train, inner selection, untouched outer evaluation."""
import argparse
import itertools
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .budget_js_extension import save_json
from .dual_space_ablation import CountedScorer
from .latent_optimization import TranslationObjective, state_hash
from .length_diagnostics import EncoderPool, length_group
from .model import CodonVMAE, collate_records
from .train import batches, codon_frequency, evaluate, sha256
from .v2_data import pick
from .v2_search import native, search
from .v2_train import GROUPS, PROFILES, corpus

COEFFICIENTS = list(itertools.product((.5, 1., 2.), repeat=2))
SEEDS = (20261023, 20261024)


def select(rows):
    choices = {}
    for profile in PROFILES:
        choices[profile] = {}
        for group in GROUPS:
            candidates = []
            for pair in COEFFICIENTS:
                current = [r for r in rows if r['profile']==profile and r['group']==group and r['coefficients']==list(pair)]
                assert len(current) == 9, 'Each choice requires 3 CDS x 3 budgets'
                candidates.append((sum(r['gain'] for r in current)/len(current), pair))
            gain, pair = max(candidates, key=lambda item: (item[0], -sum(abs(x-1.) for x in item[1])))
            choices[profile][group] = dict(coefficients=list(pair), mean_gain=gain)
    profile = max(PROFILES, key=lambda p: sum(choices[p][g]['mean_gain'] for g in GROUPS)/3)
    return dict(profile=profile, by_length=choices[profile], all_profiles=choices,
                criterion='Equal-weight length groups and 5/10/20% legal Hard Gain, JS<=.05; inner data only')


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False).cuda()
    model.load_state_dict(checkpoint['model'])
    return model


def cached(path, function, metadata):
    if path.exists():
        row = json.loads(path.read_text())
        assert all(row[k] == v for k, v in metadata.items())
        return row
    row = dict(function(), **metadata)
    save_json(path, row)
    print(json.dumps(dict(event='search_complete', file=str(path), gain=row['gain'])), flush=True)
    return row


def summarize(output):
    rows = [json.loads(p.read_text()) for p in output.glob('fold_*/outer/*.json')]
    summary = []
    for budget in (5, 10, 20):
        for group in GROUPS:
            units = {}
            for row in rows:
                if row['budget_percent'] > budget or row['group'] != group:
                    continue
                key = row['fold'], row['sequence_id'], row['seed'], row['arm']
                units[key] = max(units.get(key, -float('inf')), row['best']['hard_score'])
            per_cds = []
            completed = {(r['fold'], r['sequence_id'], r['seed'], r['arm'], r['budget_percent']) for r in rows}
            for fold, identity in sorted({k[:2] for k in units}):
                arms = ('selected_js05', 'selected_relaxed', 'uniform_default', 'RiboDecode')
                if not all((fold, identity, seed, arm, b) in completed for seed in SEEDS for arm in arms for b in (5, 10, 20) if b <= budget):
                    continue
                means = {arm: float(np.mean([units[fold, identity, seed, arm] for seed in SEEDS])) for arm in arms}
                baseline = next(r['baseline'] for r in rows if r['sequence_id']==identity)
                per_cds.append(dict(fold=fold, sequence_id=identity, baseline=baseline, **means))
            if per_cds:
                differences = np.array([r['selected_js05']-r['RiboDecode'] for r in per_cds])
                rng = np.random.default_rng(20261022)
                draws = differences[rng.integers(len(differences), size=(5000, len(differences)))].mean(1)
                fold_differences = {str(f): float(np.mean([r['selected_js05']-r['RiboDecode'] for r in per_cds if r['fold']==f])) for f in sorted({r['fold'] for r in per_cds})}
                summary.append(dict(budget_percent=budget, group=group, n_cds=len(per_cds),
                    means={a: float(np.mean([r[a] for r in per_cds])) for a in arms},
                    mean_gains={a: float(np.mean([r[a]-r['baseline'] for r in per_cds])) for a in arms},
                    paired_difference=float(differences.mean()), exploratory_group_bootstrap_ci95=np.quantile(draws, [.025, .975]).tolist(),
                    fold_differences=fold_differences, fold_difference_std=float(np.std(list(fold_differences.values()))),
                    per_cds=per_cds))
    save_json(output/'summary.json', dict(completed_searches=len(rows), comparisons=summary,
        caution='CDS are distinct homology groups; seeds averaged within CDS. Bootstrap does not capture all shared-training uncertainty; inspect fold variability. Partial cohorts are not final.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = dict(folds=5, profiles=PROFILES, coefficients=COEFFICIENTS, seeds=SEEDS, inner_per_length=3, outer_per_length=12,
        budgets=[5, 10, 20], primary_js=.05, relaxed_js={5:.05, 10:.1, 20:.3}, steps=80, scorer_queries_per_trial=160,
        folds_sha256=sha256(args.data/'folds.json'), scorer_sha256=sha256(Path('score_model/best_model.p')),
        sources={p.name:sha256(p) for p in Path('latent_decoder').glob('*.py')},
        selection='Five outer group folds, each with 10% inner group holdout. All training starts from scratch. One profile per fold, coefficients per length. No outer tuning.',
        limits='Frozen scorer surrogate, not biological validation. Native has no JS constraint; both use identical codon budget and score-call budget. Original fallback permitted. Budget summaries admit lower-budget winners equally for all arms.')
    plan = json.loads(json.dumps(plan))
    if (args.output/'plan.json').exists():
        assert json.loads((args.output/'plan.json').read_text()) == plan, 'Experiment changed on resume'
    else:
        save_json(args.output/'plan.json', plan)
    for fold in range(5):
        root = args.output/f'fold_{fold}'
        root.mkdir(exist_ok=True)
        if (root/'complete.json').exists():
            continue
        for profile in PROFILES:
            save_json(args.output/'status.json', dict(stage='training', fold=fold, profile=profile, updated=time.time()))
            subprocess.run([sys.executable, '-m', 'latent_decoder.v2_train', '--data', str(args.data), '--output', str(root/profile), '--fold', str(fold), '--profile', profile], check=True)
        data = corpus(args.data, fold)
        inner, outer = pick(data['val'], 3, 'search-inner'), pick(data['test'], 12, 'search-outer')
        cohort = dict(inner=[r['sequence_id'] for r in inner], outer=[r['sequence_id'] for r in outer])
        save_json(root/'cohort.json', cohort)
        for phase in ('inner', 'outer'):
            (root/phase).mkdir(exist_ok=True)
        scorer = CountedScorer(TranslationObjective('score_model')).eval().requires_grad_(False).cuda()
        scorer_hash = state_hash(scorer)
        inner_rows = []
        for profile in PROFILES:
            model = load_model(root/profile/'best.pt')
            model_hash = state_hash(model)
            checkpoint_hash = sha256(root/profile/'best.pt')
            save_json(args.output/'status.json', dict(stage='inner_search', fold=fold, profile=profile, updated=time.time()))
            for record in inner:
                tokens = collate_records([record]).cuda()
                for index, pair in enumerate(COEFFICIENTS):
                    for budget in (5, 10, 20):
                        metadata = dict(fold=fold, profile=profile, group=length_group(record['n_codons']-1), checkpoint_sha256=checkpoint_hash)
                        metadata.update(sequence_id=record['sequence_id'], seed=SEEDS[0], budget_percent=budget, coefficients=list(pair), js_limit=.05)
                        path = root/'inner'/f'{profile}_{record["sequence_id"]}_{index}_{budget}.json'
                        inner_rows.append(cached(path, lambda: search(model, scorer, tokens, record['sequence_id'], SEEDS[0], budget, pair), metadata))
            assert state_hash(model) == model_hash
            del model
            torch.cuda.empty_cache()
        selected = select(inner_rows)
        save_json(root/'selection.json', selected)
        for arm, profile in [('selected_js05', selected['profile']), ('selected_relaxed', selected['profile']), ('uniform_default', 'uniform'), ('RiboDecode', None)]:
            save_json(args.output/'status.json', dict(stage='outer_search', fold=fold, arm=arm, updated=time.time()))
            model = load_model(root/profile/'best.pt') if profile else None
            model_hash = state_hash(model) if model is not None else None
            checkpoint_hash = sha256(root/profile/'best.pt') if profile else None
            for record in outer:
                group = length_group(record['n_codons']-1)
                pair = selected['by_length'][group]['coefficients'] if arm.startswith('selected') else (1., 1.)
                tokens = collate_records([record]).cuda()
                for seed in SEEDS:
                    for budget in (5, 10, 20):
                        threshold = {5:.05, 10:.1, 20:.3}[budget] if arm=='selected_relaxed' else .05
                        path = root/'outer'/f'{arm}_{record["sequence_id"]}_{seed}_{budget}.json'
                        metadata = dict(fold=fold, arm=arm, group=group)
                        metadata.update(sequence_id=record['sequence_id'], seed=seed, budget_percent=budget, checkpoint_sha256=checkpoint_hash)
                        cached(path, lambda: native(scorer, tokens, record['sequence_id'], seed, budget) if arm=='RiboDecode' else search(model, scorer, tokens, record['sequence_id'], seed, budget, pair, threshold), metadata)
                summarize(args.output)
            if model is not None:
                assert state_hash(model) == model_hash
            del model
            torch.cuda.empty_cache()
        # All outer sequences enter encoder diagnostics; search only uses the score-blind subset.
        model = load_model(root/selected['profile']/'best.pt')
        pools = {g: EncoderPool(256) for g in GROUPS}
        with torch.no_grad():
            for group in GROUPS:
                for tokens in batches([r for r in data['test'] if length_group(r['n_codons']-1)==group], 2):
                    tokens = tokens.cuda()
                    mu, logvar = model.encode(tokens, tokens)
                    for index in range(len(tokens)):
                        pools[group].add(mu[index:index+1], logvar[index:index+1], tokens[index:index+1])
        save_json(root/'outer_encoder.json', {g:p.summary() for g,p in pools.items()})
        assert state_hash(scorer) == scorer_hash
        save_json(root/'complete.json', dict(completed=True, selected=selected, timestamp=time.time()))
        del model, scorer, data
        torch.cuda.empty_cache()
    summarize(args.output)
    save_json(args.output/'complete.json', dict(completed=True, folds=5, timestamp=time.time()))
    save_json(args.output/'status.json', dict(stage='complete', updated=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output')+1])
            output.mkdir(parents=True, exist_ok=True)
            save_json(output/'failure.json', dict(traceback=traceback.format_exc(), timestamp=time.time()))
            save_json(output/'status.json', dict(stage='failed', updated=time.time()))
        raise
