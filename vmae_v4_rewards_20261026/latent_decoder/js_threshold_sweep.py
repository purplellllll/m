"""Screen JS thresholds, then replicate cohort-level choices with two other seeds."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .budget_js_extension import ROOT, paired_stats, read_rows, run_trial, save_json, task_key
from .dual_space_ablation import CountedScorer
from .latent_optimization import TranslationObjective, design_probabilities, state_hash
from .model import CodonVMAE, collate_records
from .train import sha256

GRID = [.075, .10, .15, .20, .30, .50]
LEVELS = [.05] + GRID + [.70]
BUDGETS = [10, 20]


def contrasts(rows, baseline, identities, seeds, budget, threshold):
    lookup = {(r['sequence_id'], r['seed'], r['budget_percent'], r['js_limit']): r for r in rows}
    controls = {(r['sequence_id'], r['seed'], r['budget_percent'], r['arm']): r for r in baseline}
    per_cds = []
    for identity in identities:
        keys = [(identity, seed, budget, threshold) for seed in seeds]
        if not all(key in lookup for key in keys):
            continue
        scores = {arm: float(np.mean([controls[identity, seed, budget, arm]['best']['hard_score'] for seed in seeds]))
                  for arm in ('B5', 'RiboDecode')}
        scores['B5_JS'] = float(np.mean([lookup[key]['best']['hard_score'] for key in keys]))
        per_cds.append(dict(sequence_id=identity, **scores))
    if not per_cds:
        return None
    return dict(budget_percent=budget, js_limit=threshold, n_cds=len(per_cds), seeds=seeds,
        means={key: float(np.mean([r[key] for r in per_cds])) for key in ('B5', 'RiboDecode', 'B5_JS')},
        vs_native=paired_stats([r['B5_JS']-r['RiboDecode'] for r in per_cds]),
        vs_b5=paired_stats([r['B5_JS']-r['B5'] for r in per_cds]), per_cds=per_cds)


def select_thresholds(screen, expected_cds):
    choices = {}
    for budget in BUDGETS:
        group = [r for r in screen if r['budget_percent'] == budget]
        assert sorted(r['js_limit'] for r in group) == LEVELS
        assert all(r['n_cds'] == expected_cds for r in group)
        crossings = [r for r in group if r['vs_native']['mean'] > 0]
        first = min((r['js_limit'] for r in crossings), default=None)
        best = max(group, key=lambda r: (r['means']['B5_JS'], -r['js_limit']))['js_limit']
        choices[str(budget)] = dict(first_screen_crossing=first, best_screen_threshold=best,
                                   validate=sorted({best} | ({first} if first is not None else set())))
    return choices


def report(output, rows, baseline, identities, seeds, selection=None):
    screen = [r for budget in BUDGETS for threshold in LEVELS
              if (r := contrasts(rows, baseline, identities, seeds[:1], budget, threshold))]
    validation, all_seeds = [], []
    if selection:
        for budget in BUDGETS:
            for threshold in selection[str(budget)]['validate']:
                for selected_seeds, destination in ((seeds[1:], validation), (seeds, all_seeds)):
                    r = contrasts(rows, baseline, identities, selected_seeds, budget, threshold)
                    if r:
                        destination.append(r)
    new_rows = [r for r in rows if r.get('sweep_stage')]
    summary = dict(new_completed=len(new_rows), screen=screen, selection=selection,
        validation_other_two_seeds=validation, selected_all_three_seeds=all_seeds,
        completed=(output/'completion_receipt.json').exists(),
        archive_policy='Each run uses only its own 80-step legal archive, no best-of-threshold or lower-budget pooling',
        interpretation='Exploratory reused development CDS; selecting on seed1 and replicating on seeds2/3 does not provide independent-CDS validation')
    save_json(output/'summary.json', summary)
    text = f'# JS 阈值扫描与复核\n\n状态：{"完成" if summary["completed"] else "运行中"}；本轮新完成 {len(new_rows)} 组。\n\n'
    text += '固定模型、学习率、缩放、rho=1、同义投影、全部密码子预算和80步/160次评分。各阈值独立选优，不从多个阈值挑每条序列最高分。JS=.70 是复用旧无JS B5的数学等价端点（自然对数JS最大ln2<.70），不代表新跑了一轮。\n\n'
    for title, table in [('首种子筛查', screen), ('另外两个种子复核', validation), ('选定阈值三种子汇总', all_seeds)]:
        text += f'## {title}\n\n| 预算 | JS阈值 | 完整CDS | B5-JS Score | 原版 Score | 与原版差值 | 领先CDS |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in table:
            text += f"| {r['budget_percent']}% | {r['js_limit']:.3f} | {r['n_cds']} | {r['means']['B5_JS']:.6f} | {r['means']['RiboDecode']:.6f} | {r['vs_native']['mean']:+.6f} | {r['vs_native']['wins']} |\n"
        text += '\n'
    if selection:
        text += '预定规则选出的复核阈值：`'+json.dumps(selection, ensure_ascii=False)+'`。\n\n'
    text += '筛查未满全部8条CDS时不选阈值。先用20261023筛查，固定每个预算的首个均值超过原版阈值和均值最佳阈值，再用20261024/25复核；两者相同只复核一次。若所有阈值均不超过，仍复核最佳者，明确报告未跨越。\n\n'
    text += '阈值影响搜索轨迹，分数不保证随阈值单调。首次跨越只指已测网格，不是连续空间最小阈值；复核种子并非新的独立序列。没有改动主配置或进行表达实验。\n'
    (output/'report.md').write_text(text, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(4)
    prior = ROOT/'runs/budget_js_extension_20261021'
    old = ROOT/'runs/budget_comparison_20261021_v2'
    for directory in (prior, old):
        assert json.loads((directory/'completion_receipt.json').read_text())['completed']
    prior_plan = json.loads((prior/'plan.json').read_text())
    config = prior_plan['config']
    records, seeds = prior_plan['baseline_plan']['records'], prior_plan['baseline_plan']['seeds']
    identities = [r['sequence_id'] for r in records]
    baseline = read_rows(old/'results.jsonl')
    old_paths = sorted((prior/'results').glob('*.json'))
    inherited = [json.loads(path.read_text()) for path in old_paths]
    for row in inherited:
        assert row['best']['soft_js'] <= row['js_limit']+1e-7
        for name, digest in row['artifact_sha256'].items():
            assert sha256(Path(row['run_path'])/name) == digest
    for name, digest in prior_plan['source_sha256'].items():
        assert sha256(Path(__file__).with_name(name)) == digest
    # With natural-log JS bounded by ln(2), .70 leaves both penalty and filter inactive.
    endpoints = [dict(sequence_id=r['sequence_id'], seed=r['seed'], budget_percent=r['budget_percent'],
        js_limit=.70, best=r['best'], origin='reused unconstrained B5; mathematically inactive JS endpoint')
        for r in baseline if r['arm'] == 'B5']
    screen_tasks = [dict(sequence_id=identity, seed=seeds[0], budget_percent=budget, js_limit=threshold)
                    for threshold in GRID for budget in BUDGETS for identity in identities]
    sources = list(prior_plan['source_sha256']) + ['js_threshold_sweep.py']
    prior_files = old_paths + [prior/'plan.json', prior/'completion_receipt.json', old/'plan.json', old/'results.jsonl', old/'completion_receipt.json']
    plan = dict(grid=GRID, budgets=BUDGETS, screen_tasks=screen_tasks, seeds=seeds, records=records,
        config=config, torch_version=torch.__version__, device='cpu',
        selection_rule='Per-budget first strictly positive mean versus native and best mean score, ties choose lower threshold; never select per sequence',
        replication='Other two existing seeds, followed by all-three-seed descriptive summary; no independent cohort claim',
        prior_files_sha256={str(path): sha256(path) for path in prior_files},
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in sources},
        endpoint='.70 is mathematically inactive; reuse exact original B5 records, not a new experiment',
        archive='Independent per-run legal best, no cross-threshold or lower-budget pooling',
        expected_new_screen_runs=90, maximum_new_validation_runs=64)
    if args.resume:
        assert json.loads((output/'plan.json').read_text()) == plan
    else:
        save_json(output/'plan.json', plan)
        (output/'source').mkdir()
        for name in sources:
            shutil.copyfile(Path(__file__).with_name(name), output/'source'/name)
    (output/'results').mkdir(exist_ok=True)
    new_rows = [json.loads(path.read_text()) for path in sorted((output/'results').glob('*.json'))]
    for row in new_rows:
        for name, digest in row['artifact_sha256'].items():
            assert sha256(Path(row['run_path'])/name) == digest
    rows = inherited + endpoints + new_rows
    done = {task_key(row) for row in rows}
    assert len(done) == len(rows)
    checkpoint_path = ROOT/'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint_path) == prior_plan['baseline_plan']['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p') == prior_plan['baseline_plan']['scorer_sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    scorer = CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False)
    hashes = state_hash(model), state_hash(scorer)
    original_rows = read_rows(ROOT/'results_eight_cds_comparison_20261020/candidates.jsonl')
    cache = {}
    for identity in identities:
        initial = next(r['initial']['cds_dna'] for r in original_rows if r['sequence_id'] == identity)
        tokens = collate_records([dict(cds_dna=initial)])
        with torch.no_grad():
            mu, logvar = model.encode(tokens, tokens)
            reference = design_probabilities(model.decode(mu, tokens), tokens)
            value = float(scorer(F.one_hot(tokens, 64).float(), tokens))
        assert all(abs(r['baseline']-value) < 1e-6 for r in baseline if r['sequence_id'] == identity)
        cache[identity] = tokens, mu, (.5*logvar).exp(), reference, value
    newly_completed = 0
    start = time.monotonic()
    def execute(tasks, stage, selection=None):
        nonlocal newly_completed
        for task in tasks:
            if task_key(task) in done:
                continue
            tokens, mu, sigma, reference, initial_score = cache[task['sequence_id']]
            run = output/'runs'/task_key(task)/str(time.time_ns())
            run.mkdir(parents=True)
            torch.manual_seed(task['seed'])
            print(json.dumps(dict(event='trial_start', stage=stage, new_completed=sum(bool(r.get('sweep_stage')) for r in rows), **task)), flush=True)
            row = run_trial(model, scorer, tokens, mu, sigma, reference, config, task, initial_score,
                            run, high_dims=None, probe=False)
            matched = next(r for r in inherited if r['sequence_id'] == task['sequence_id']
                           and r['seed'] == task['seed'] and r['budget_percent'] == task['budget_percent'] and r['js_limit'] == .05)
            assert row['initial_latent_sha256'] == matched['initial_latent_sha256']
            row['sweep_stage'] = stage
            save_json(output/'results'/f'{task_key(task)}.json', row)
            rows.append(row)
            done.add(task_key(task))
            newly_completed += 1
            report(output, rows, baseline, identities, seeds, selection)
            print(json.dumps(dict(event='trial_complete', stage=stage, gain=row['gain'],
                                  this_process_completed=newly_completed, seconds=time.monotonic()-start)), flush=True)
    report(output, rows, baseline, identities, seeds)
    execute(screen_tasks, 'screen')
    screening = report(output, rows, baseline, identities, seeds)['screen']
    selection = select_thresholds(screening, len(identities))
    if (output/'selection.json').exists():
        assert json.loads((output/'selection.json').read_text()) == selection
    else:
        save_json(output/'selection.json', selection)
    validation_tasks = [dict(sequence_id=identity, seed=seed, budget_percent=budget, js_limit=threshold)
        for budget in BUDGETS for threshold in selection[str(budget)]['validate'] for seed in seeds[1:] for identity in identities]
    print(json.dumps(dict(event='screen_complete', selection=selection,
        validation_new_runs=sum(task_key(task) not in done for task in validation_tasks))), flush=True)
    execute(validation_tasks, 'validation', selection)
    assert all(task_key(task) in done for task in screen_tasks+validation_tasks)
    assert scorer.forward_sequences == 8 + newly_completed*160
    assert hashes == (state_hash(model), state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    for path, digest in plan['prior_files_sha256'].items():
        assert sha256(Path(path)) == digest
    save_json(output/'completion_receipt.json', dict(completed=True,
        new_completed=sum(bool(r.get('sweep_stage')) for r in rows),
        this_process_scorer_queries=scorer.forward_sequences, seconds_this_process=time.monotonic()-start,
        weights_unchanged=True, historical_baselines_unchanged=True, matched_initial_latents=True,
        main_defaults_unchanged=True, selection=selection))
    report(output, rows, baseline, identities, seeds, selection)


if __name__ == '__main__':
    main()
