"""Fixed-budget B5-JS extension, threshold pilots and causal latent probes."""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .budget_comparison import ROOT, project_budget
from .dual_space_ablation import CountedScorer, masks, project_joint, regularizers
from .js_constraint import JSConstraint, mutable_js
from .latent_optimization import TranslationObjective, design_probabilities, state_hash, write_rows
from .length_diagnostics import gradient_summary, length_group
from .model import CODONS, CodonVMAE, collate_records
from .train import sha256


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows readers can briefly hold a handle that prevents atomic replacement.
            if attempt == 19:
                raise
            time.sleep(.05)


def task_key(task):
    return f"{task['sequence_id']}_{task['seed']}_{task['budget_percent']}_{task['js_limit']}"


def paired_stats(values):
    values = np.asarray(values, dtype=float)
    draws = values[np.random.default_rng(20261021).integers(len(values), size=(5000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), wins=int((values > 0).sum()), n_cds=len(values),
                exploratory_ci95=np.quantile(draws, [.025, .975]).tolist() if len(values) > 1 else None)


def summarize(rows, baseline):
    comparisons = []
    for budget in (5, 10, 20):
        for archive in ('best', 'best_budget_only'):
            per_cds = []
            for identity in sorted({r['sequence_id'] for r in rows}):
                current = [r for r in rows if r['sequence_id'] == identity and r['js_limit'] == .05
                           and r['budget_percent'] <= budget]
                # Only publish a CDS when every predeclared budget and seed is complete.
                expected = 3 * sum(b <= budget for b in (5, 10, 20))
                if len(current) != expected:
                    continue
                scores = dict(B5=[], RiboDecode=[], B5_JS=[])
                for seed in sorted({r['seed'] for r in current}):
                    scores['B5_JS'].append(max(r[archive]['hard_score'] for r in current if r['seed'] == seed))
                    for arm in ('B5', 'RiboDecode'):
                        scores[arm].append(max(r['best']['hard_score'] for r in baseline
                            if r['sequence_id'] == identity and r['seed'] == seed
                            and r['budget_percent'] <= budget and r['arm'] == arm))
                per_cds.append(dict(sequence_id=identity, **{k: float(np.mean(v)) for k, v in scores.items()}))
            if per_cds:
                comparisons.append(dict(budget_percent=budget, archive=archive, complete_cds=len(per_cds),
                    means={k: float(np.mean([r[k] for r in per_cds])) for k in ('B5', 'RiboDecode', 'B5_JS')},
                    js_minus_native=paired_stats([r['B5_JS']-r['RiboDecode'] for r in per_cds]),
                    js_minus_b5=paired_stats([r['B5_JS']-r['B5'] for r in per_cds]), per_cds=per_cds))
    pilots = []
    for r in rows:
        if r['js_limit'] == .05:
            continue
        control = next(t for t in rows if t['sequence_id'] == r['sequence_id'] and t['seed'] == r['seed']
                       and t['budget_percent'] == r['budget_percent'] and t['js_limit'] == .05)
        pilots.append(dict(sequence_id=r['sequence_id'], n_aa=r['n_aa'], js_limit=r['js_limit'],
            control_gain=control['gain'], gain=r['gain'], gain_delta=r['gain']-control['gain'],
            best_js=r['best']['soft_js'], changed_codons=r['best']['changed_codons'],
            filter_gap=r['filter_gap'], control_filter_gap=control['filter_gap'],
            common_js05_gain=r['best_js05']['hard_score']-r['baseline']))
    return dict(completed=len(rows), planned=78, comparisons=comparisons, threshold_pilots=pilots)


def latent_probe(model, scorer, tokens, mu, sigma, reference, latent, direct, initial_latent,
                 config, limit, js_limit, high_dims):
    """Reset channel groups to the matched random start; retain the fitted direct branch."""
    active, support = masks(tokens)
    high_dims = high_dims.to(tokens.device)
    def probabilities(v, d):
        return design_probabilities(model.decode(mu + sigma * config['latent_scale'] * v, tokens)
            + config['direct_scale'] * d * support, tokens)
    v = latent.detach().clone().requires_grad_()
    p = probabilities(v, direct)
    hard = project_budget(p, tokens, limit)
    score = scorer(hard + p - p.detach(), tokens).mean()
    grad = torch.autograd.grad(score, v)[0][active]
    displacement = (latent - initial_latent)[active].double()
    covariance = displacement.T @ displacement / len(displacement)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eig.sum()
    eigen_rank = float(total.square() / eig.square().sum()) if total > 0 else 0.
    random_dims = torch.zeros(256, dtype=torch.bool)
    random_dims[torch.randperm(256, generator=torch.Generator().manual_seed(20261021))[:int(high_dims.sum())]] = True
    random_dims = random_dims.to(tokens.device)
    interventions = []
    with torch.no_grad():
        for name, channels in [('reset_high21', high_dims), ('reset_low235', ~high_dims),
                               ('reset_random21', random_dims), ('reset_all_latent', torch.ones(256, dtype=torch.bool, device=tokens.device)),
                               ('remove_direct', torch.zeros(256, dtype=torch.bool, device=tokens.device))]:
            altered = latent.clone()
            altered[..., channels] = initial_latent[..., channels]
            q = probabilities(altered, torch.zeros_like(direct) if name == 'remove_direct' else direct)
            selected = project_budget(q, tokens, limit)
            value = float(scorer(selected, tokens))
            js = float(mutable_js(q, reference, active))
            interventions.append(dict(name=name, dimensions=int(channels.sum()), hard_score=value,
                score_drop=float(score.detach())-value, soft_js=js, feasible=js <= js_limit+1e-7,
                changed_from_fitted=int((selected.argmax(-1) != hard.argmax(-1)).sum()),
                changed_from_original=int((selected.argmax(-1) != tokens).sum()),
                probability_rms_shift=float((q-p.detach())[active].square().mean().sqrt())))
    return dict(fitted_hard_score=float(score.detach()), score_gradient=gradient_summary(grad),
        search_displacement=gradient_summary(displacement), displacement_spectral_participation=eigen_rank,
        high21_score_gradient_energy_fraction=float(grad[:, high_dims].square().sum()/grad.square().sum().clamp_min(1e-30)),
        high21_displacement_energy_fraction=float(displacement[:, high_dims].square().sum()/displacement.square().sum().clamp_min(1e-30)),
        interventions=interventions, diagnostic_scorer_queries=6,
        caveat='Post-hoc necessity at a fitted point, not a retrained/reoptimized reduced-dimension capacity test; direct branch can compensate')


def run_trial(model, scorer, tokens, mu, sigma, reference, config, task, baseline, run, high_dims, probe):
    active, support = masks(tokens)
    identity, seed = task['sequence_id'], task['seed']
    limit = tokens.shape[1] * task['budget_percent'] // 100
    key = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'big') % (2**63-1)
    noise = torch.randn(mu.shape[1:], generator=torch.Generator().manual_seed(key)).to(tokens.device)[None] * active[..., None]
    latent = (noise / (noise.square().sum() / (active.sum()*noise.shape[-1])).sqrt() * .05).requires_grad_()
    initial_latent = latent.detach().clone()
    direct = torch.zeros((*tokens.shape, 64), device=tokens.device, requires_grad=True)
    lr = config['lr'] / math.sqrt(2) * config['arm_lr_multipliers']['B5']
    optimizer = torch.optim.Adam([dict(params=[latent], lr=lr), dict(params=[direct], lr=lr*config['direct_lr_factor'])])
    controller = JSConstraint(reference, active, task['js_limit'], 1.)
    def probabilities():
        decoder = model.decode(mu + sigma * config['latent_scale'] * latent, tokens)
        p = design_probabilities(decoder + config['direct_scale'] * direct * support, tokens)
        terms = regularizers(p, design_probabilities(decoder, tokens), latent, direct, active, support)
        return p, sum(config['regularization'][k] * value.mean() for k, value in terms.items())
    original = dict(step=0, hard_score=baseline, cds_dna=''.join(CODONS[t] for t in tokens[0].tolist()),
                    changed_codons=0, soft_js=0., feasible=True, original_fallback=True)
    best = best_budget = best_js05 = original
    best_state = (initial_latent.clone(), direct.detach().clone())
    start = time.monotonic()
    before = scorer.forward_sequences
    feasible_steps = 0
    for step in range(1, 81):
        optimizer.zero_grad(set_to_none=True)
        p, penalty = probabilities()
        hard = project_budget(p, tokens, limit)
        score = scorer(hard+p-p.detach(), tokens).mean()
        loss = -score + penalty + controller.penalty(p).mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([latent, direct], 1., error_if_nonfinite=True)
        optimizer.step()
        project_joint(latent, direct, active, support, component_radius=16.)
        with torch.no_grad():
            p, _ = probabilities()
            hard = project_budget(p, tokens, limit)
            value = float(scorer(hard, tokens))
            js = float(mutable_js(p, reference, active))
            controller.update(p)
            selected = hard.argmax(-1)
            candidate = dict(step=step, hard_score=value, cds_dna=''.join(CODONS[t] for t in selected[0].tolist()),
                changed_codons=int((selected != tokens).sum()), soft_js=js, feasible=js <= task['js_limit']+1e-7)
        write_rows(run/'steps.jsonl', [dict(**candidate, **controller.metrics(0))])
        feasible_steps += candidate['feasible']
        if value > best_budget['hard_score']:
            best_budget = candidate
        if js <= .05+1e-7 and value > best_js05['hard_score']:
            best_js05 = candidate
        if candidate['feasible'] and value > best['hard_score']:
            best = candidate
            best_state = latent.detach().clone(), direct.detach().clone()
        if step % 20 == 0:
            print(json.dumps(dict(event='progress', key=task_key(task), step=step,
                gain=best['hard_score']-baseline, seconds=time.monotonic()-start)), flush=True)
    assert scorer.forward_sequences-before == 160
    assert best['soft_js'] <= task['js_limit']+1e-7 and best['changed_codons'] <= limit
    result = dict(**task, n_aa=tokens.shape[1]-1, baseline=baseline, limit=limit, best=best,
        best_budget_only=best_budget, best_js05=best_js05, gain=best['hard_score']-baseline,
        filter_gap=best_budget['hard_score']-best['hard_score'], feasible_step_fraction=feasible_steps/80,
        scorer_queries=160, backwards=80, initial_latent_sha256=hashlib.sha256(initial_latent.cpu().numpy().tobytes()).hexdigest(),
        search_seconds=time.monotonic()-start, run_path=str(run.resolve()))
    torch.save(dict(latent=best_state[0], direct=best_state[1], initial_latent=initial_latent), run/'best_state.pt')
    if probe:
        if best['step'] == 0:
            result['latent_probe'] = dict(skipped='Original fallback won; no fitted improvement to ablate', diagnostic_scorer_queries=0)
        else:
            result['latent_probe'] = latent_probe(model, scorer, tokens, mu, sigma, reference, *best_state,
                initial_latent, config, limit, task['js_limit'], high_dims)
            assert abs(result['latent_probe']['fitted_hard_score']-best['hard_score']) < 1e-6
    result['artifact_sha256'] = {name: sha256(run/name) for name in ('steps.jsonl', 'best_state.pt')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prior = ROOT/'runs/budget_comparison_20261021_v2'
    old_plan = json.loads((prior/'plan.json').read_text())
    assert json.loads((prior/'completion_receipt.json').read_text())['completed']
    baseline_rows = read_rows(prior/'results.jsonl')
    assert len(baseline_rows) == 144
    config = old_plan['frozen_b5_config']
    checkpoint_path = ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint_path) == old_plan['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p') == old_plan['scorer_sha256']
    encoder_path = ROOT/'runs/length_diagnostics_20261021/encoder_summary.json'
    variance = torch.tensor(json.loads(encoder_path.read_text())['all_test']['pooled']['variance_by_dim'])
    high_dims = variance > .01
    assert len(high_dims) == 256 and int(high_dims.sum()) == 21
    records, seeds = old_plan['records'], old_plan['seeds']
    pilots = [next(r for r in records if length_group(r['n_codons']-1) == group) for group in ('short', 'medium', 'long')]
    tasks = [dict(sequence_id=r['sequence_id'], seed=seeds[0], budget_percent=20, js_limit=threshold)
             for r in pilots for threshold in (.05, .10, .20)]
    tasks += [task for r in records for seed in seeds for budget in (5, 10, 20)
              if (task := dict(sequence_id=r['sequence_id'], seed=seed, budget_percent=budget, js_limit=.05)) not in tasks]
    assert len(tasks) == 78
    sources = ['budget_js_extension.py', 'budget_comparison.py', 'js_constraint.py', 'dual_space_ablation.py',
               'length_diagnostics.py', 'latent_optimization.py', 'model.py', 'tokenizer.py']
    plan = dict(tasks=tasks, baseline_plan=old_plan, config=config, rho=1., device=args.device,
        torch_version=torch.__version__, tf32=False,
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in sources},
        baseline_sha256={str(prior/name): sha256(prior/name) for name in ('plan.json', 'results.jsonl', 'completion_receipt.json')},
        encoder_summary_sha256=sha256(encoder_path), high_variance_dims=high_dims.nonzero().flatten().tolist(),
        protocol='Same 8 CDS, 3 seeds, all-codon budget denominator, 80 updates/160 search calls; no GC filter. B5-JS has additional soft JS constraint vs unfiltered historical controls. Primary archive respects JS; budget-only archive explicitly secondary.',
        js_definition='Mean mutable-site JS between pre-budget soft distribution and clean decoder reference; unchanged from prior JS method. Not JS of the projected DNA.',
        probes='First seed of each CDS/budget at JS .05; frozen global 21 AU channels vs remaining235, matched random21, all latent and direct interventions; probe scores never enter search archive.',
        pilot_policy='First previously listed CDS per length bin, first seed, 20% budget; .05/.10/.20 fixed before results. Report own-bound and common-.05 results separately.',
        inference='Seeds averaged within CDS, then exploratory paired CDS bootstrap; reused development cohort, no confirmatory claim')
    if args.resume:
        assert json.loads((output/'plan.json').read_text()) == plan, 'Plan or code changed during resume'
    else:
        save_json(output/'plan.json', plan)
        (output/'source').mkdir()
        for name in sources:
            shutil.copyfile(Path(__file__).with_name(name), output/'source'/name)
    (output/'results').mkdir(exist_ok=True)
    rows = [json.loads(path.read_text()) for path in sorted((output/'results').glob('*.json'))]
    done = {task_key(row) for row in rows}
    assert len(done) == len(rows)
    for row in rows:
        assert {k: row[k] for k in tasks[0]} in tasks
        for name, digest in row['artifact_sha256'].items():
            assert sha256(Path(row['run_path'])/name) == digest
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    scorer = CountedScorer(TranslationObjective(ROOT/'score_model')).to(args.device).eval().requires_grad_(False)
    hashes = state_hash(model), state_hash(scorer)
    cache = {}
    for record in records:
        identity = record['sequence_id']
        initial = next(r['initial']['cds_dna'] for r in read_rows(ROOT/'results_eight_cds_comparison_20261020/candidates.jsonl') if r['sequence_id'] == identity)
        tokens = collate_records([dict(cds_dna=initial)]).to(args.device)
        with torch.no_grad():
            mu, logvar = model.encode(tokens, tokens)
            reference = design_probabilities(model.decode(mu, tokens), tokens)
            score = float(scorer(F.one_hot(tokens, 64).float(), tokens))
        assert all(abs(r['baseline']-score) < 1e-5 for r in baseline_rows if r['sequence_id'] == identity)
        cache[identity] = tokens, mu, (.5*logvar).exp(), reference, score
    start, newly_completed, diagnostic_queries = time.monotonic(), 0, 0
    for task in tasks[:9] if args.pilot_only else tasks:
        if task_key(task) in done:
            continue
        tokens, mu, sigma, reference, baseline = cache[task['sequence_id']]
        torch.manual_seed(task['seed'])
        run = output/'runs'/task_key(task)/str(time.time_ns())
        run.mkdir(parents=True)
        print(json.dumps(dict(event='trial_start', completed=len(rows), total=len(tasks), **task)), flush=True)
        row = run_trial(model, scorer, tokens, mu, sigma, reference, config, task, baseline, run, high_dims,
                        probe=task['js_limit'] == .05 and task['seed'] == seeds[0])
        diagnostic_queries += row.get('latent_probe', {}).get('diagnostic_scorer_queries', 0)
        save_json(output/'results'/f'{task_key(task)}.json', row)
        rows.append(row)
        newly_completed += 1
        save_json(output/'summary.json', summarize(rows, baseline_rows))
        print(json.dumps(dict(event='trial_complete', completed=len(rows), total=len(tasks),
            gain=row['gain'], filter_gap=row['filter_gap'], seconds=time.monotonic()-start)), flush=True)
    assert scorer.forward_sequences == 8 + newly_completed*160 + diagnostic_queries
    assert hashes == (state_hash(model), state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    for path, digest in plan['baseline_sha256'].items():
        assert sha256(Path(path)) == digest
    if args.pilot_only:
        save_json(output/'pilot_receipt.json', dict(completed=True, complete_runs=len(rows), weights_unchanged=True))
        return
    assert len(rows) == 78
    save_json(output/'completion_receipt.json', dict(completed=True, complete_runs=len(rows),
        seconds_this_process=time.monotonic()-start, this_process_scorer_queries=scorer.forward_sequences,
        search_queries=sum(r['scorer_queries'] for r in rows),
        diagnostic_queries=sum(r.get('latent_probe', {}).get('diagnostic_scorer_queries', 0) for r in rows),
        weights_unchanged=True, historical_baselines_unchanged=True,
        final_outputs_within_own_js_limit=True, thresholds=[.05, .10, .20], main_defaults_unchanged=True))


if __name__ == '__main__':
    main()
