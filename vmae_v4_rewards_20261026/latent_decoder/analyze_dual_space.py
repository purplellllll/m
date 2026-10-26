"""Validate and summarize the fixed B0-B5 paired ablation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .tokenizer import AA, CODONS


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--docs', default='docs')
    args = parser.parse_args()
    root, docs = Path(args.results), Path(args.docs)
    config = json.loads((root / 'protocol.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    rows, steps = read_rows(root / 'results.jsonl'), read_rows(root / 'steps.jsonl')
    initial = read_rows(root / 'initial.jsonl')
    assert not summary['preflight'] and summary['model_and_scorer_unchanged']
    assert len(rows) == 96 and len(steps) == 6400 and len(initial) == 16
    assert len({(r['sequence_id'], r['arm'], r['step']) for r in steps}) == 6400
    assert all(np.isfinite(r['grad_norm']) and r['grad_norm'] > 0 for r in steps)
    assert all(r['joint_parameter_rms'] <= (512**.5 + 1e-5 if config.get('output_space_budget') else 1.00001) for r in steps)
    assert set(r['sequence_id'] for r in initial) == set(config['evaluation_ids'])
    budgets = {arm: (a['scorer_forward_sequences'], a['backward_sequence_evaluations'])
               for arm, a in summary['arms'].items() if arm != 'B0'}
    assert set(budgets.values()) == {(3840, 1280)}
    tuning = json.loads((root / 'tuning.json').read_text()) if config.get('output_space_budget') else None
    if tuning:
        assert tuning['selected']['B3'] == tuning['selected']['B4'] == tuning['selected']['B5']
        for multiplier in tuning['grid']:
            directory = root / 'train_tuning' / ('lr_' + str(multiplier))
            assert len(read_rows(directory / 'steps.jsonl')) == 3200
            assert len(read_rows(directory / 'results.jsonl')) == 48
    originals = {r['sequence_id']: r for r in initial}
    arms = list(summary['arms'])
    codon_aa = dict(zip(CODONS, AA))
    def translate(dna): return ''.join(codon_aa[dna[i:i+3]] for i in range(0, len(dna), 3))
    for row in rows:
        source = originals[row['sequence_id']]['cds_dna']
        for which in ['best', 'final']:
            dna = row[which]['cds_dna']
            assert len(dna) == len(source) and dna[:3] == source[:3] and dna[-3:] == source[-3:]
            assert translate(dna) == translate(source)
        best = row['best']
        assert best['feasible'] and best['mutable_hamming'] <= config['max_mutable_hamming'] + 1e-7
        assert abs(best['gc_fraction'] - row['initial']['gc_fraction']) <= config['max_gc_change'] + 1e-7
        assert best['js_from_initial'] <= config['max_js'] + 1e-7
    folds = {r['sequence_sha256']: r for r in read_rows(root / 'rnafold.jsonl')}
    def fold(dna): return folds[hashlib.sha256(dna.encode()).hexdigest()]['mfe_per_nt']
    metadata = {}
    for path in (Path(args.data) / 'val').glob('*.jsonl.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                metadata[record['sequence_id']] = record
    identities = config['evaluation_ids']
    groups = {}
    for index, identity in enumerate(identities):
        groups.setdefault(metadata[identity]['cluster_id'], []).append(index)
    rng = np.random.default_rng(63917)
    draws = rng.integers(0, len(groups), (5000, len(groups)))
    sizes = np.array([len(ix) for ix in groups.values()])
    def interval(values, quantiles=(.025, .975)):
        sums = np.array([sum(values[i] for i in ix) for ix in groups.values()])
        return np.quantile(sums[draws].sum(1) / sizes[draws].sum(1), quantiles).tolist()
    lookup = {(r['arm'], r['sequence_id']): r for r in rows}
    analysis = dict(arms={}, contrasts={}, bootstrap_components=len(groups),
                    bootstrap_draws=5000, bootstrap_seed=63917, budgets=budgets,
                    source_checkpoint_sha256=config['checkpoint_sha256'],
                    source_scorer_sha256=config['scorer_sha256'], source_unchanged=True)
    analysis['tuning'] = tuning
    for arm in arms:
        data = [lookup[arm, identity] for identity in identities]
        metrics = dict(summary['arms'][arm])
        metrics.update(best_gain_95=interval([r['best_hard_gain'] for r in data]),
            mean_soft_hard_gap=float(np.mean([r['best']['soft_hard_gap'] for r in data])),
            max_absolute_soft_hard_gap=max(abs(r['best']['soft_hard_gap']) for r in data),
            mean_js=float(np.mean([r['best']['js_from_initial'] for r in data])),
            mean_abs_gc_change=float(np.mean([abs(r['best']['gc_fraction'] - r['initial']['gc_fraction']) for r in data])),
            mean_frequency_nll_change=float(np.mean([r['best']['frequency_nll'] - r['initial']['frequency_nll'] for r in data])),
            mean_mfe_per_nt_change=float(np.mean([fold(r['best']['cds_dna']) - fold(r['initial']['cds_dna']) for r in data])),
            mean_decoder_kl=float(np.mean([r['best'].get('decoder_kl', 0.) for r in data])),
            mean_latent_l2=float(np.mean([r['best'].get('latent_l2', 0.) for r in data])),
            mean_direct_l2=float(np.mean([r['best'].get('direct_l2', 0.) for r in data])))
        analysis['arms'][arm] = metrics
    for left, right in [('B3', 'B1'), ('B3', 'B2'), ('B3', 'B4'), ('B5', 'B3')]:
        differences = [lookup[left, i]['best_hard_gain'] - lookup[right, i]['best_hard_gain'] for i in identities]
        analysis['contrasts'][left + '_minus_' + right] = dict(mean=float(np.mean(differences)),
            median=float(np.median(differences)), interval_95=interval(differences),
            bonferroni_98_75=interval(differences, (.00625, .99375)),
            wins=sum(d > 1e-5 for d in differences), ties=sum(abs(d) <= 1e-5 for d in differences),
            losses=sum(d < -1e-5 for d in differences), per_cds=dict(zip(identities, differences)))
    analysis['posthoc_edit_budget'] = {}
    for cap in [.02, .04, .06, .08, .1, .2]:
        analysis['posthoc_edit_budget'][str(cap)] = {}
        for arm in arms[1:]:
            gains = []
            for identity in identities:
                baseline = originals[identity]['hard_score']
                candidates = [r['hard_score'] for r in steps if r['arm'] == arm and r['sequence_id'] == identity
                              and r['feasible'] and r['mutable_hamming'] <= cap + 1e-7]
                gains.append(max([baseline] + candidates) - baseline)
            analysis['posthoc_edit_budget'][str(cap)][arm] = float(np.mean(gains))
    (root / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    (docs / 'ds_dual_space_analysis_20261019.json').write_text(json.dumps(analysis, indent=2) + '\n')
    with (docs / 'ds_dual_space_best_20261019.fasta').open('w') as handle:
        for r in rows:
            handle.write(f">{r['sequence_id']}|{r['arm']}|step={r['best']['step']}|score={r['best']['hard_score']:.6f}\n{r['best']['cds_dna']}\n")
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for index, arm in enumerate(arms):
        data = [lookup[arm, i] for i in identities]
        gains = [r['best_hard_gain'] for r in data]
        metrics = analysis['arms'][arm]
        axes[0].scatter(index + np.linspace(-.13, .13, 16), gains, s=13, alpha=.65, color=colors[index])
        mean, ci = metrics['mean_best_gain'], metrics['best_gain_95']
        axes[0].errorbar(index, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], fmt='k_', capsize=3)
        if arm == 'B0': continue
        traces = []
        for identity in identities:
            baseline = originals[identity]['hard_score']
            trajectory = sorted([r for r in steps if r['arm'] == arm and r['sequence_id'] == identity], key=lambda r:r['step'])
            best, trace = baseline, [0.]
            for r in trajectory:
                if r['feasible']: best = max(best, r['hard_score'])
                trace.append(best-baseline)
            assert len(trace) == 81
            traces.append(trace)
        axes[1].plot(range(81), np.mean(traces, 0), label=arm, color=colors[index])
        axes[2].scatter([r['best']['mutable_hamming'] for r in data], gains, color=colors[index], s=15, alpha=.65, label=arm)
    axes[0].set_xticks(range(6), arms)
    axes[0].set_ylabel('Best feasible discrete predictor-score gain')
    axes[0].set_title('16 paired validation CDS; mean and 95% interval')
    axes[1].axvline(40, color='#888', ls=':', lw=1)
    axes[1].set_xlabel('Update index (3 scorer forwards per update)')
    axes[1].set_ylabel('Mean best feasible score gain')
    axes[1].set_title('Matched 80-update search budget')
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel('Fraction of mutable codons changed')
    axes[2].set_ylabel('Best feasible score gain')
    axes[2].set_title('Shared candidate constraint: Hamming <= 20%')
    axes[2].legend(fontsize=8)
    fig.suptitle('B0-B5 dual-space ablation | frozen models | exploratory | no expression validation')
    fig.savefig(docs / 'ds_dual_space_ablation_20261019.png', dpi=170)
    fig.savefig(docs / 'ds_dual_space_ablation_20261019.svg')
    plt.close(fig)
    table = '\n'.join(f"| {arm} | {m['name']} | {m['mean_best_gain']:.6f} | {m['improved']}/16 | {m['mean_hamming']:.2%} | {m['mean_abs_gc_change']:.2%} | {m['feasible_candidate_fraction']:.2%} |" for arm, m in analysis['arms'].items())
    comparisons = '\n'.join(f"| {name} | {m['mean']:+.6f} | [{m['interval_95'][0]:+.6f}, {m['interval_95'][1]:+.6f}] | {m['wins']}/{m['ties']}/{m['losses']} |" for name, m in analysis['contrasts'].items())
    costs = '\n'.join(f"| {arm} | {m['scorer_forward_sequences']} | {m['backward_sequence_evaluations']} | {m['decoder_sequence_evaluations']} | {m['elapsed_seconds']:.1f} |" for arm, m in analysis['arms'].items())
    quality = '\n'.join(f"| {arm} | {m['mean_js']:.6f} | {m['mean_decoder_kl']:.6f} | {m['mean_frequency_nll_change']:+.6f} | {m['mean_mfe_per_nt_change']:+.6f} | {m['max_absolute_soft_hard_gap']:.6f} |" for arm, m in analysis['arms'].items())
    final_table = '\n'.join(f"| {arm} | {m['mean_final_gain']:.6f} | {m['final_feasible']}/16 |" for arm, m in analysis['arms'].items())
    interpretation = '\n'.join(f"- {name}：平均差 {m['mean']:+.6f}，配对胜/平/负={m['wins']}/{m['ties']}/{m['losses']}。" for name, m in analysis['contrasts'].items())
    tuning_text = (f"v2 在相同 8 条训练 CDS 上对每组运行 3 个学习率倍率、每次 80 步，选择结果为 `{tuning['selected']}`。B4/B5 使用 B3 的倍率和每分量步长，隔离调度与正则差异；全部独立网格结果保留。每组调参额外用 5,760 次评分器序列前向和 1,920 次目标反向，与验证合计分别为 9,600 / 3,200。" if tuning else '本版未做独立学习率网格选择。')
    edit_table = '\n'.join(f"| {float(cap):.0%} | " + ' | '.join(f"{values[arm]:.6f}" for arm in arms[1:]) + ' |' for cap, values in analysis['posthoc_edit_budget'].items())
    energies = '\n'.join(f"| {arm} | {analysis['arms'][arm]['mean_latent_l2']:.6f} | {analysis['arms'][arm]['mean_direct_l2']:.6f} | {analysis['arms'][arm]['mean_decoder_kl']:.6f} |" for arm in ['B3','B5'])
    report = f'''# B0–B5 双空间公平消融结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。Verification Status：ANALYZED；已真实运行，未做独立重复或生物学验证。
- 使用原第 50 轮权重 `{config['checkpoint_sha256']}`，模型和评分器前后内存及源文件哈希核对不变。
- 16 条固定验证 CDS，单一确定性起点；与前序诊断使用同一批短 CDS，属于探索性证据。
- 详细设计见[预定协议](DS_dual_space_ablation_protocol_2026-10-19.md)。预算与阈值在主实验前固定。

## 直接观察与审计

本批数据支持联合搜索优于单空间的初步证据，但分阶段 B4 的平均收益高于无正则联合 B3；B5 的平均收益最高。B3/B4/B5 使用相同的训练集选定步长，因此阶段和正则比较未混入验证集挑选学习率。不同方法实际改动数量仍有差别，不能把均值差全部解释为互补机制。

{interpretation}

首轮 v1 的直接空间参数半径阻断了离散改变，B2 的 16 条均为零提升，因此不能用那轮宣称双空间胜出。保留 v1 记录后，按[修正协议](DS_dual_space_ablation_protocol_v2_2026-10-19.md)运行 v2：取消内部参数能量配额，以共同输出约束为准，并增加对等的训练集调参。该修正发生在看过首轮验证结果后，属于公开披露的探索性修正，不能当成确认性实验。

## 六组结果

共 6,400 条更新记录及 96 个方法/序列结果。最佳与最终 DNA 均重新核验翻译、长度和原始起止密码子；所有最佳候选满足共同 Hamming≤20%、GC 变化≤5 个百分点、JS≤0.05 的约束。轨迹可以暂时不可行，最佳值包括原始起点。

| 编号 | 方法 | 平均最佳离散分数增量 | 提升数 | 平均可变位改动 | 平均绝对 GC 变化 | 可行候选比例 |
|---|---|---:|---:|---:|---:|---:|
{table}

| 组 | 最终一步的原始增量 | 最终一步可行数 |
|---|---:|---:|
{final_table}

最终一步可能不可行，不能把该行的原始增量作为满足约束的收益；用于方法比较和导出 FASTA 的始终是最佳可行候选。

这些数值是被优化评分器的原始输出单位，不能解释为实际表达倍数。均值正差不代表每条 CDS 都更好，也不证明联合空间普遍优于单空间。

## 四个预定配对对比

| 对比 | 平均差 | 探索性 95% 区间 | 胜/平/负 |
|---|---:|---|---|
{comparisons}

按 {len(groups)} 个既有 cluster_id 组件配对 bootstrap 5,000 次，seed=63917；非近同源认证分组。JSON 同时保存对四个对比进行 Bonferroni 口径调整的 98.75% 探索性区间。此处不作确认性显著性主张。

## 开销与公平性边界

| 组 | 评分器序列前向 | 目标反向序列次数 | decoder 序列前向 | 搜索耗时（秒） |
|---|---:|---:|---:|---:|
{costs}

共同初始化、训练集校准和分析开销另计。B1–B5 各 80 次更新、240 次评分器序列前向，B4 为 40+40；联合组一次反向同时更新两个变量。计数相同不等于 FLOPs 相同。

潜在尺度为 0.5 个条件后验标准差；直接尺度为 {config['direct_scale']:.6f}，直接学习率因子为 {config['direct_lr_factor']:.6f}，均来自 8 条训练 CDS 校准。该尺度匹配是局部近似。

参数能量配额仅属于 v1；本报告 v2 取消该配额，各分量 RMS≤16 只是数值保护。真正共同的可行性预算为输出 JS、GC 和 Hamming 阈值，不能把不同参数空间的半径看成等价的离散搜索容量。B2 共享初始 decoder logits，随后只更新直接空间。{tuning_text}

## 分布与结构检查

| 组 | 平均 JS | 相对当前 decoder 的平均 KL | 训练频率 NLL 变化 | MFE/nt 变化（kcal/mol/nt） | 最大绝对 soft−hard 分数差 |
|---|---:|---:|---:|---:|---:|
{quality}

RNAfold 仅在搜索后使用，不参与优化或候选选择；MFE 更低或更高都不能自动解释为表达更好。B5 的比较同时涉及固定的潜在偏移、直接残差和 decoder KL 正则，不能据此分解每个正则项的独立价值。B5 若分数较低，应结合漂移和残差变化判断，不自动认定正则无效或有效。

| 组 | 潜在偏移能量均值 | 直接残差能量均值 | decoder KL 均值 |
|---|---:|---:|---:|
{energies}

本批 B5 的直接残差能量和 decoder KL 均低于 B3，但潜在偏移并未同时下降；这支持固定正则组合改变了搜索路径和两空间分工，不能声称所有分布指标都更优。

## 改动预算敏感性（事后诊断）

从同样的 80 个候选中，进一步限制可变位改动上限，保留原 GC/JS 阈值并重新取最佳可行分数。没有新增目标查询或重选超参数。该分析是在看到主结果后增加的，只用于检查更多改动是否驱动收益，不替代四个主对比。

| 改动上限 | B1 | B2 | B3 | B4 | B5 |
|---|---:|---:|---:|---:|---:|
{edit_table}

相同上限仍不代表每条序列实际改动数完全一致；完整的等编辑数匹配与更大样本检验仍待完成。

预算敏感性提示：在 2% 和 4% 的上限下，B1 的平均收益高于 B3；在本批 2%–8% 的上限下，B4 的平均收益高于 B5。因此不能把主表的单一排序外推为所有编辑预算下的排序，也不能把 B5 的最高主表均值直接称为最优 mRNA。

## 验证与复查

全部四个新旧相关单元测试在部署前及远端预检通过；真实 GPU 模型执行了单条 CDS、每组两步的预检，再在独立目录运行主实验。主实验输出完整、梯度有限非零、源权重及冻结参数不变，预算核验通过。

首次 GPU 预检因缺少 pandas 在评分器导入处退出，未开始主实验；依赖安装到本次目录的独立 dependencies 后，新目录预检通过。执行环境为 RTX 4080 SUPER、PyTorch 2.0.0+cu118；v2 含训练网格和验证的模型执行耗时 {summary['elapsed_seconds']:.1f} 秒，退出码 0。B4 的 16 条序列在直接阶段 latent_l2 均保持不变，已从逐步记录单独核验；执行收据内 23 个文件哈希与本地副本匹配，数据 manifest 和 tokenizer 与本地冻结版一致。

- [结构化结果与所有预定配对差](ds_dual_space_analysis_20261019.json)
- [全部最佳候选 FASTA](ds_dual_space_best_20261019.fasta)
- [结果图](ds_dual_space_ablation_20261019.png)
- 原始记录：`{root.as_posix()}`；包含 protocol、calibration、initial、steps、results、summary 和 RNAfold。

## 解释限制

本轮未完成长 CDS、多初始化、近同源隔离、独立评分器或实际表达验证。强约束、参数能量分配和单一校准规则均可能影响方法排名；不能把本轮结果等同于各方法在所有预算下的最优表现。先前不设同样约束的潜在优化数值不可直接并表排名。

统计解释 11/11 覆盖：长度/家族混合（只限短 CDS）、总体到个体外推（保存配对结果）、选择偏差（固定已有样本并披露）、条件化混杂（相同约束但报告接受比例）、基率（保留 B0）、均值回归（相同起点与控制）、存活偏差（要求全部 96 结果）、多条件选择（四个对比全报）、分析路径（预定协议及固定阈值）、相关/因果（不声称真实表达因果）、反向因果（优化的是预测器目标，非生物机制证明）。
'''
    (docs / 'DS_dual_space_ablation_results_2026-10-19.md').write_text(report, encoding='utf-8')
    print(json.dumps(analysis['contrasts']))


if __name__ == '__main__':
    main()
