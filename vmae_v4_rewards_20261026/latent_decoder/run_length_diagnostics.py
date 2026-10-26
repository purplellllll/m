"""Fixed-test length diagnostics; never retune weights or search hyperparameters."""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .b6_reproducibility import frozen_config, verify_reference
from .dual_space_ablation import CountedScorer, optimize_batch
from .latent_optimization import TranslationObjective, state_hash, write_rows
from .length_diagnostics import EncoderPool, encoder_row, functional_probe, length_group, select_records
from .model import CodonVMAE, collate_records
from .train import codon_frequency, load_data, sha256

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20261026, 20261027, 20261028]


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def update_report(output):
    results = read_rows(output / 'search_summary.jsonl')
    steps = read_rows(output / 'search_steps.jsonl')
    encoded = json.loads((output / 'encoder_summary.json').read_text())
    probes = {r['sequence_id']: r for r in read_rows(output / 'functional_probes.jsonl')}
    summaries = {}
    per_cds = []
    for identity in sorted({r['sequence_id'] for r in results}):
        runs = [r for r in results if r['sequence_id'] == identity]
        if len(runs) != len(SEEDS):
            continue
        trajectory = [r for r in steps if r['sequence_id'] == identity]
        first_steps = [r for r in trajectory if r['step'] <= 20]
        mean = lambda xs: float(np.mean(xs))
        row = dict(sequence_id=identity, group=runs[0]['group'],
            raw_gain=mean([r['best_before_filter']['hard_score']-r['baseline'] for r in runs]),
            feasible_gain=mean([r['best_after_filter']['hard_score']-r['baseline'] for r in runs]),
            raw_score=mean([r['best_before_filter']['hard_score'] for r in runs]),
            filter_loss=mean([r['filter_loss'] for r in runs]),
            feasible_edit_fraction=mean([r['best_after_filter']['mutable_hamming'] for r in runs]),
            raw_edit_fraction=mean([r['best_before_filter']['mutable_hamming'] for r in runs]),
            loss_gradient_dims_99pct=mean([r['dims_for_99pct_total_loss_gradient_energy'] for r in runs]),
            score_gradient_dims_99pct=probes[identity]['score_gradient']['dims_for_99pct_gradient_energy'],
            reconstruction_gradient_dims_99pct=probes[identity]['reconstruction_gradient']['dims_for_99pct_gradient_energy'],
            **{k:mean([r[k] for r in first_steps]) for k in ('latent_gradient_rms','actual_z_update_rms',
                'standardized_update_rms','decoder_logit_update_rms','gradient_abs_le_1e_minus8_fraction','actual_direct_update_rms')},
            feasible_step_fraction=mean([r['feasible'] for r in trajectory]),
            **{'rejected_'+reason+'_fraction':mean([reason in r['rejected_by'] for r in trajectory]) for reason in ('edit','gc','js')})
        per_cds.append(row)
    for group in ('short','medium','long'):
        rows = [r for r in per_cds if r['group'] == group]
        if rows:
            summaries[group] = dict(complete_cds=len(rows), **{k:float(np.mean([r[k] for r in rows]))
                for k in rows[0] if k not in ('sequence_id','group')})
    contrasts = {}
    short = [r for r in per_cds if r['group'] == 'short']
    long = [r for r in per_cds if r['group'] == 'long']
    if short and long:
        rng = np.random.default_rng(20261021)
        for key in ('raw_gain','feasible_gain','actual_z_update_rms','standardized_update_rms',
                    'decoder_logit_update_rms','latent_gradient_rms','score_gradient_dims_99pct'):
            a = np.array([r[key] for r in short]); b = np.array([r[key] for r in long])
            draws = b[rng.integers(len(b), size=(5000,len(b)))].mean(1) - a[rng.integers(len(a), size=(5000,len(a)))].mean(1)
            contrasts[key] = dict(long_minus_short=float(b.mean()-a.mean()), exploratory_ci95=np.quantile(draws,[.025,.975]).tolist())
    summary = dict(search_runs_completed=len(results), search_runs_planned=72, encoder=encoded,
                   search=summaries, per_cds=per_cds, long_minus_short=contrasts,
                   interpretation='Exploratory descriptive diagnostics; no causal identification from length correlations; no retuning')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    table = []
    for group in ('short','medium','long'):
        enc = encoded[group]; seq = enc['sequence_mean']; s = summaries.get(group, {})
        fmt = lambda key: f'{s[key]:.6g}' if key in s else 'pending'
        table.append(f"| {group} | {enc['sequences']} | {seq['mu_rms']:.6g} | {seq['sigma_mean']:.6g} | {enc['pooled']['active_dimensions']['0.01']} | {s.get('complete_cds',0)} | {fmt('score_gradient_dims_99pct')} | {fmt('actual_z_update_rms')} | {fmt('raw_gain')} | {fmt('feasible_gain')} | {fmt('feasible_edit_fraction')} |")
    report = '# 冻结 B5 长度诊断\n\n'
    report += f'搜索已完成 {len(results)}/72 组。短 ≤300 aa；中 301–699 aa；长 ≥700 aa。编码器统计覆盖完整测试集，搜索每档 8 条、每条 3 个固定种子。\n\n'
    report += '| 档位 | 编码 CDS | μ RMS | σ均值 | AU>0.01 | 搜索完整CDS | 评分梯度99%能量维数 | 前20步Δz RMS | 筛前增益 | 筛后增益 | 筛后可变位修改率 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(table)
    report += '\n\nAU 是通道方差阈值计数，评分梯度能量维数是局部敏感度；两者均不等于可修改位置比例。搜索统计先在 CDS 内平均全部种子，再对 CDS 平均。评分为冻结内部评分器的离散 Hard Score；筛前仍保持同义性和固定起止密码子，只忽略修改率/GC/JS筛选，不使用 soft score 代替。\n'
    report += '\nsummary.json 包含每条CDS、方差阈值敏感性、同AA内方差、初始任务梯度/重建梯度、逐步实际更新、筛选损失及探索性长短差区间。functional_probes.jsonl 包含高/低方差通道分组置零的解码干预；该干预可能离开数据分布，不单独证明信息容量不足。\n'
    report += '\n解释顺序：梯度变小但实际Δz不变，不能归为有效步长稀释（Adam会归一化梯度）。σ或实际Δz及Decoder变化随长度降低才支持尺度问题；σ稳定但标准化更新变小，应检查梯度相对Adam epsilon及正则。筛前增益良好、筛后损失大，优先定位可行性筛选；筛前本身差还需区分表征和搜索，不能仅凭AU少就改结构。\n'
    report += '\n21/256 与约8%修改率没有数学对应：256是每个密码子位置的通道数，修改率是位置比例，且B5直接残差绕过潜变量。模型、学习率、缩放及正则均未调整；本次不根据测试集选择新超参数。未进行独立表达验证。\n'
    (output / 'report.md').write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    reference = ROOT / 'deliverables/b0_b5_frozen_20261020'
    manifest = verify_reference(reference)
    original, config = frozen_config(reference, ['B5'])
    checkpoint_path = ROOT / 'deliverables/vmae_50ep_last_epoch50_20261019.pt'
    assert sha256(checkpoint_path) == manifest['checkpoint_sha256']
    assert sha256(ROOT/'score_model/best_model.p') == manifest['scorer_sha256']
    assert sha256(ROOT/'dataset/manifest.json') == original['dataset_manifest_sha256']
    splits, _ = load_data(ROOT/'dataset', allow_development=True)
    selected = select_records(splits['test'])
    config.update(batch_size=1, random_start_rms=.05, timeout_seconds=7*24*3600, length_diagnostics=True)
    sources = [Path(__file__), Path(__file__).with_name('length_diagnostics.py'), Path(__file__).with_name('dual_space_ablation.py')]
    source_hashes = {p.name:sha256(p) for p in sources}
    plan = dict(version='length-diagnostics-20261021', origin_skill='academic-research-suite/experiment-agent',
        split='test', bins=dict(short='<=300 aa', medium='301..699 aa', long='>=700 aa'),
        seeds=SEEDS, search_records=[{k:r[k] for k in ('sequence_id','n_codons','cluster_id')} for r in selected],
        selection='8 unique components per bin, salted SHA256 rank; score-blind; no tuning',
        encoder_records=len(splits['test']), original_b5_config=original, actual_config=config,
        active_units='Var(mu) over valid codon tokens > .01; also .001/.1 and within-AA variance',
        functional_metrics='score/reconstruction gradient energy across channels and high/low-variance group zeroing',
        frozen_values=dict(latent_lr=original['lr']/math.sqrt(2)*original['arm_lr_multipliers']['B5'],
            direct_lr=original['lr']/math.sqrt(2)*original['arm_lr_multipliers']['B5']*original['direct_lr_factor'],
            latent_scale=original['latent_scale'], direct_scale=original['direct_scale']),
        diagnostics_no_search_changes=True, device=args.device, source_sha256=source_hashes,
        checkpoint_sha256=manifest['checkpoint_sha256'], scorer_sha256=manifest['scorer_sha256'],
        limitations=['existing test split, not a new confirmatory sample','homology separation not certified',
                     'local sensitivities/length associations do not uniquely identify bottlenecks','no model or hyperparameter selection in this run'])
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model = CodonVMAE(**checkpoint['config']['model']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model']); model.to(args.device)
    scorer = CountedScorer(TranslationObjective(ROOT/'score_model')).eval().requires_grad_(False).to(args.device)
    hashes = state_hash(model), state_hash(scorer)
    start = time.monotonic()
    pools = {g:EncoderPool(model.latent_dim) for g in ('short','medium','long')}
    pooled = EncoderPool(model.latent_dim)
    encoder_by_id = {}
    for index, record in enumerate(splits['test']):
        tokens = collate_records([record]).to(args.device)
        with torch.no_grad():
            mu, logvar = model.encode(tokens,tokens)
        group = length_group(record['n_codons']-1)
        pools[group].add(mu, logvar, tokens); pooled.add(mu, logvar, tokens)
        row = dict(sequence_id=record['sequence_id'], group=group, n_aa=record['n_codons']-1, **encoder_row(mu,logvar,tokens))
        encoder_by_id[record['sequence_id']] = row
        write_rows(output/'encoder_records.jsonl',[row])
        if (index+1)%100 == 0 or index+1 == len(splits['test']):
            result = {g:p.summary() for g,p in pools.items() if p.count}
            result['all_test'] = pooled.summary()
            (output/'encoder_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(dict(event='encoder_progress',completed=index+1,total=len(splits['test']),seconds=time.monotonic()-start)),flush=True)
    update_report(output)
    for index, record in enumerate(selected):
        probe = functional_probe(model, scorer, record, args.device)
        write_rows(output/'functional_probes.jsonl',[probe])
        print(json.dumps(dict(event='functional_probe',completed=index+1,total=len(selected),sequence_id=record['sequence_id'],
            variance_dims=probe['per_sequence_high_variance_dims'],score_gradient_dims=probe['score_gradient']['dims_for_99pct_gradient_energy'])),flush=True)
    counts = codon_frequency(splits['train']).to(args.device)
    for index, record in enumerate(selected):
        for seed in SEEDS:
            torch.manual_seed(seed)
            config.update(random_start_seed=seed, diagnostic_mu_rms=encoder_by_id[record['sequence_id']]['mu_rms'])
            run = output/'runs'/record['sequence_id']/str(seed); run.mkdir(parents=True,exist_ok=False)
            optimize_batch(model,scorer,[record],counts,config,run,start)
            write_rows(output/'search_summary.jsonl',read_rows(run/'length_search_summary.jsonl'))
            write_rows(output/'search_steps.jsonl',[dict(seed=seed, **r) for r in read_rows(run/'length_search_steps.jsonl')])
            update_report(output)
            print(json.dumps(dict(event='search_complete',completed=index*3+SEEDS.index(seed)+1,total=72,
                group=length_group(record['n_codons']-1),n_aa=record['n_codons']-1,seed=seed,seconds=time.monotonic()-start)),flush=True)
    assert hashes == (state_hash(model),state_hash(scorer))
    assert all(p.grad is None for p in list(model.parameters())+list(scorer.parameters()))
    assert scorer.forward_sequences == len(selected) + len(selected)*len(SEEDS)*(6+240)
    assert source_hashes == {p.name:sha256(p) for p in sources}
    verify_reference(reference)
    (output/'completion_receipt.json').write_text(json.dumps(dict(completed=True,model_weights_unchanged=True,
        scorer_weights_unchanged=True,hyperparameters_unchanged=True,scorer_forward_sequences=scorer.forward_sequences,
        seconds=time.monotonic()-start,python=platform.python_version(),torch=torch.__version__)),encoding='utf-8')


if __name__ == '__main__':
    main()
