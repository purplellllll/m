"""Per-CDS paired analysis and strict scientific-content replay checks for B6."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .b6_reproducibility import verify_reference
from .tokenizer import AA, CODONS


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def scientific(row):
    return {key: value for key, value in row.items() if key != 'elapsed_seconds'}


def compare(reference, actual, reference_steps, actual_steps):
    expected = {(r['arm'], r['sequence_id']): r for r in reference}
    obtained = {(r['arm'], r['sequence_id']): r for r in actual}
    assert set(expected) == set(obtained)
    score_errors = []
    for key, source in expected.items():
        current = obtained[key]
        score_errors += [abs(current[k]-source[k]) for k in ['best_hard_gain','final_hard_gain']]
        score_errors += [abs(current[which]['hard_score']-source[which]['hard_score']) for which in ['initial','best','final']]
    reference_steps = {(r['arm'],r['sequence_id'],r['step']):r for r in reference_steps}
    actual_steps = {(r['arm'],r['sequence_id'],r['step']):r for r in actual_steps}
    assert set(reference_steps) == set(actual_steps)
    metric_errors = []
    for key, source in reference_steps.items():
        current = actual_steps[key]
        metric_errors += [abs(current[k]-source[k]) for k in source if isinstance(source[k],(int,float)) and not isinstance(source[k],bool)]
    best_dna = sum(expected[k]['best']['cds_dna']==obtained[k]['best']['cds_dna'] for k in expected)
    final_dna = sum(expected[k]['final']['cds_dna']==obtained[k]['final']['cds_dna'] for k in expected)
    feasible_match = all(source['feasible']==actual_steps[key]['feasible'] for key, source in reference_steps.items())
    step_match = all(expected[k]['best']['step']==obtained[k]['best']['step'] for k in expected)
    exact_rows = sum(scientific(expected[k]) == scientific(obtained[k]) for k in expected)
    exact_steps = sum(source == actual_steps[key] for key, source in reference_steps.items())
    return dict(result_rows=len(expected), strict_result_rows=exact_rows, step_rows=len(reference_steps),
        strict_step_rows=exact_steps, best_dna_matches=best_dna, final_dna_matches=final_dna,
        all_feasibility_flags_match=feasible_match, all_best_steps_match=step_match,
        max_absolute_score_error=max(score_errors,default=0.),
        max_absolute_step_metric_error=max(metric_errors,default=0.),
        strict_match=exact_rows==len(expected) and exact_steps==len(reference_steps),
        score_tolerance_diagnostic=(max(score_errors,default=0.)<=1e-6 and best_dna==len(expected)
                                    and final_dna==len(expected) and feasible_match and step_match))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--results',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--docs',default='docs')
    args=parser.parse_args()
    reference,root,docs=Path(args.reference),Path(args.results),Path(args.docs)
    verify_reference(reference)
    config=json.loads((reference/'reference/protocol.json').read_text())
    identities=config['evaluation_ids']
    old=rows(reference/'reference/results.jsonl')
    old_steps=rows(reference/'reference/steps.jsonl')
    primary=root/'runs/primary_b6'
    b6=[r for r in rows(primary/'results.jsonl') if r['arm']=='B6']
    b6_steps=rows(primary/'steps.jsonl')
    assert len(old)==96 and len(b6)==16 and len(b6_steps)==1280
    assert set(r['sequence_id'] for r in b6)==set(identities)
    originals={r['sequence_id']:r for r in rows(reference/'reference/initial.jsonl')}
    lookup={(r['arm'],r['sequence_id']):r for r in old+b6}
    arms=['B0','B1','B2','B3','B4','B5','B6']
    repeats={}
    replay_rows={}
    environment=json.loads((reference/'reference_execution_receipt.json').read_text())
    for name,seed in [('primary_b6',20261020),('repeat_1',20261020),('repeat_2',20261021),('repeat_3',20261022)]:
        assert (root/'logs'/f'{name}.exitcode').read_text().strip()=='0'
        summary=json.loads((root/'runs'/name/'summary.json').read_text())
        assert summary['seed']==seed and summary['model_and_scorer_unchanged'] and summary['reference_unchanged']
        assert summary['freeze_manifest_sha256']==hashlib.sha256((reference/'freeze_manifest.json').read_bytes()).hexdigest()
        assert all(summary[key]==environment[key] for key in ['python','torch','cuda','gpu'])
        if name=='primary_b6':continue
        current=rows(root/'runs'/name/'results.jsonl')
        replay_rows[name]={(r['arm'],r['sequence_id']):r for r in current}
        current_steps=rows(root/'runs'/name/'steps.jsonl')
        assert len(current)==112 and len(current_steps)==7680
        repeats[name]=dict(seed=seed,
            baseline=compare(old,[r for r in current if r['arm']!='B6'],old_steps,[r for r in current_steps if r['arm']!='B6']),
            b6=compare(b6,[r for r in current if r['arm']=='B6'],b6_steps,[r for r in current_steps if r['arm']=='B6']))
    for identity in identities:
        trajectory=sorted([r for r in b6_steps if r['sequence_id']==identity],key=lambda r:r['step'])
        assert len(trajectory)==80 and sum(r['phase']=='latent' for r in trajectory)==40
        assert len({r['latent_l2'] for r in trajectory if r['phase']=='direct'})==1
        assert all(r['direct_l2']==0. for r in trajectory if r['phase']=='latent')
        assert all(np.isfinite(r['grad_norm']) and r['grad_norm']>0 for r in trajectory)
    metadata={}
    for path in (Path(args.data)/'val').glob('*.jsonl.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as handle:
            for line in handle:
                r=json.loads(line);metadata[r['sequence_id']]=r
    aa=dict(zip(CODONS,AA))
    def translate(dna):return ''.join(aa[dna[i:i+3]] for i in range(0,len(dna),3))
    for arm in arms:
        for identity in identities:
            record=lookup[arm,identity]
            source=originals[identity]['cds_dna']
            for which in ['best','final']:
                dna=record[which]['cds_dna']
                assert len(dna)==len(source) and dna[:3]==source[:3] and dna[-3:]==source[-3:]
                assert translate(dna)==translate(source)
            best=record['best']
            assert best['feasible'] and best['mutable_hamming']<=config['max_mutable_hamming']+1e-7
            assert abs(best['gc_fraction']-record['initial']['gc_fraction'])<=config['max_gc_change']+1e-7
            assert best['js_from_initial']<=config['max_js']+1e-7
            if arm!='B0':assert record['scorer_forward_sequences']==240 and record['backward_sequence_evaluations']==80
    folds={r['sequence_sha256']:r for r in rows(reference/'reference/rnafold.jsonl')+rows(primary/'rnafold.jsonl')}
    def mfe(dna):return folds[hashlib.sha256(dna.encode()).hexdigest()]['mfe_per_nt']
    groups={}
    for index,identity in enumerate(identities):groups.setdefault(metadata[identity]['cluster_id'],[]).append(index)
    rng=np.random.default_rng(20261020)
    draws=rng.integers(0,len(groups),(5000,len(groups)))
    sizes=np.array([len(ix) for ix in groups.values()])
    def interval(values,q):
        sums=np.array([sum(values[i] for i in ix) for ix in groups.values()])
        return np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),q).tolist()
    analysis=dict(technical_repeats=repeats,primary_comparisons={},arms={},per_cds=[],
                  technical_repeats_are_not_new_samples=True,independent_cds=16,
                  bootstrap_components=len(groups),bootstrap_draws=5000,bootstrap_seed=20261020,
                  reference_manifest_sha256=hashlib.sha256((reference/'freeze_manifest.json').read_bytes()).hexdigest())
    strict=all(entry[part]['strict_match'] for entry in repeats.values() for part in ['baseline','b6'])
    analysis['technical_reproduction_strict']=strict
    analysis['technical_verdict']='REPRODUCIBLE' if strict else 'PARTIALLY_REPRODUCIBLE'
    analysis['strict_step_comparisons']=sum(entry[part]['strict_step_rows'] for entry in repeats.values() for part in ['baseline','b6'])
    analysis['max_reproduction_score_error']=max(entry[part]['max_absolute_score_error'] for entry in repeats.values() for part in ['baseline','b6'])
    for arm in arms:
        data=[lookup[arm,i] for i in identities]
        analysis['arms'][arm]=dict(mean_gain=float(np.mean([r['best_hard_gain'] for r in data])),
            median_gain=float(np.median([r['best_hard_gain'] for r in data])),improved=sum(r['best_hard_gain']>1e-5 for r in data),
            mean_hamming=float(np.mean([r['best']['mutable_hamming'] for r in data])),
            mean_abs_gc_change=float(np.mean([abs(r['best']['gc_fraction']-r['initial']['gc_fraction']) for r in data])),
            mean_js=float(np.mean([r['best']['js_from_initial'] for r in data])),
            mean_decoder_kl=float(np.mean([r['best'].get('decoder_kl',0.) for r in data])),
            mean_direct_l2=float(np.mean([r['best'].get('direct_l2',0.) for r in data])),
            mean_mfe_per_nt_change=float(np.mean([mfe(r['best']['cds_dna'])-mfe(r['initial']['cds_dna']) for r in data])),
            final_feasible=sum(r['final']['feasible'] for r in data))
    for right in ['B4','B5']:
        values=[lookup['B6',i]['best_hard_gain']-lookup[right,i]['best_hard_gain'] for i in identities]
        analysis['primary_comparisons']['B6_minus_'+right]=dict(mean=float(np.mean(values)),median=float(np.median(values)),
            interval_95=interval(values,[.025,.975]),bonferroni_97_5=interval(values,[.0125,.9875]),
            wins=sum(v>1e-5 for v in values),ties=sum(abs(v)<=1e-5 for v in values),losses=sum(v< -1e-5 for v in values))
    long_rows=[]
    for index,identity in enumerate(identities):
        row=dict(index=index+1,sequence_id=identity,n_codons=metadata[identity]['n_codons'],cluster_id=metadata[identity]['cluster_id'])
        for arm in arms:
            r=lookup[arm,identity]
            row[arm+'_gain']=r['best_hard_gain']
            long_rows.append(dict(index=index+1,sequence_id=identity,n_codons=row['n_codons'],arm=arm,
                initial_score=r['initial']['hard_score'],best_score=r['best']['hard_score'],best_gain=r['best_hard_gain'],
                best_step=r['best']['step'],final_score=r['final']['hard_score'],final_gain=r['final_hard_gain'],
                final_feasible=r['final']['feasible'],mutable_hamming=r['best']['mutable_hamming'],
                gc_change=r['best']['gc_fraction']-r['initial']['gc_fraction'],js=r['best']['js_from_initial'],
                decoder_kl=r['best'].get('decoder_kl',0.),direct_l2=r['best'].get('direct_l2',0.),
                soft_hard_gap=r['best']['soft_hard_gap'],mfe_per_nt_change=mfe(r['best']['cds_dna'])-mfe(r['initial']['cds_dna'])))
        row['B6_minus_B4']=row['B6_gain']-row['B4_gain']
        row['B6_minus_B5']=row['B6_gain']-row['B5_gain']
        row['B6_repeat_best_dna_matches']=sum(replay_rows[name]['B6',identity]['best']['cds_dna']==lookup['B6',identity]['best']['cds_dna'] for name in replay_rows)
        row['B6_repeat_strict_result_matches']=sum(scientific(replay_rows[name]['B6',identity])==scientific(lookup['B6',identity]) for name in replay_rows)
        row['B0_B5_all_exact_repeats']=sum(all(scientific(replay_rows[name][a,identity])==scientific(lookup[a,identity]) for a in arms[:-1]) for name in replay_rows)
        row['B6_max_repeat_gain_error']=max(abs(replay_rows[name]['B6',identity]['best_hard_gain']-lookup['B6',identity]['best_hard_gain']) for name in replay_rows)
        analysis['per_cds'].append(row)
    for filename,data in [('b6_per_cds_paired_20261020.csv',analysis['per_cds']),('b0_b6_per_cds_long_20261020.csv',long_rows)]:
        with (docs/filename).open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (docs/'b6_paired_reproducibility_20261020.json').write_text(json.dumps(analysis,indent=2)+'\n')
    with (docs/'b6_best_candidates_20261020.fasta').open('w') as handle:
        for identity in identities:
            r=lookup['B6',identity]
            handle.write(f">{identity}|B6|step={r['best']['step']}|score={r['best']['hard_score']:.6f}\n{r['best']['cds_dna']}\n")
    fig,axes=plt.subplots(1,2,figsize=(12,8),gridspec_kw=dict(width_ratios=[1.15,1]),constrained_layout=True)
    matrix=np.array([[lookup[a,i]['best_hard_gain'] for a in arms] for i in identities])
    plot=axes[0].imshow(matrix,aspect='auto',cmap='YlGnBu',vmin=0)
    axes[0].set_xticks(range(7),arms)
    axes[0].set_yticks(range(16),[f'{ix+1:02d}  {i[4:12]}' for ix,i in enumerate(identities)])
    axes[0].set_title('Per-CDS best feasible score gain')
    for y in range(16):
        for x in range(7):axes[0].text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',fontsize=7,
                                     color='white' if matrix[y,x]>matrix.max()*.55 else 'black')
    fig.colorbar(plot,ax=axes[0],shrink=.7,label='Frozen predictor raw score units')
    for right,color,offset in [('B4','#2878B5',-.12),('B5','#D46B3C',.12)]:
        values=np.array([r['B6_minus_'+right] for r in analysis['per_cds']])
        axes[1].scatter(values,np.arange(16)+offset,label='B6 - '+right,color=color,s=25)
        axes[1].hlines(np.arange(16)+offset,0,values,color=color,alpha=.35)
    axes[1].axvline(0,color='#555',lw=.8)
    axes[1].set_ylim(15.5,-.5)
    axes[1].set_yticks(range(16),[f'{i+1:02d}' for i in range(16)])
    axes[1].set_xlabel('Paired score-gain difference')
    axes[1].set_title('16 paired CDS; technical repeats are not extra samples')
    axes[1].legend()
    fig.suptitle('Frozen B0-B5 + B6 | no retuning | same 16 validation CDS')
    fig.savefig(docs/'b6_per_cds_paired_20261020.png',dpi=170)
    fig.savefig(docs/'b6_per_cds_paired_20261020.svg')
    plt.close(fig)
    arm_table='\n'.join(f"| {a} | {m['mean_gain']:.6f} | {m['improved']}/16 | {m['mean_hamming']:.2%} | {m['mean_decoder_kl']:.6f} | {m['final_feasible']}/16 |" for a,m in analysis['arms'].items())
    pairs='\n'.join(f"| {name} | {m['mean']:+.6f} | [{m['interval_95'][0]:+.6f}, {m['interval_95'][1]:+.6f}] | {m['wins']}/{m['ties']}/{m['losses']} |" for name,m in analysis['primary_comparisons'].items())
    per_cds='\n'.join(f"| {r['index']:02d} | `{r['sequence_id'][4:16]}` | {r['n_codons']} | {r['B4_gain']:.6f} | {r['B5_gain']:.6f} | {r['B6_gain']:.6f} | {r['B6_minus_B4']:+.6f} | {r['B6_minus_B5']:+.6f} | {r['B6_repeat_best_dna_matches']}/3 |" for r in analysis['per_cds'])
    repeat_table='\n'.join(f"| {name} | {part} | {m['strict_result_rows']}/{m['result_rows']} | {m['strict_step_rows']}/{m['step_rows']} | {m['best_dna_matches']}/{m['result_rows']} | {m['max_absolute_score_error']:.8g} | {m['score_tolerance_diagnostic']} |" for name,entry in repeats.items() for part,m in entry.items() if isinstance(m,dict))
    residual_reduction=1-analysis['arms']['B6']['mean_direct_l2']/analysis['arms']['B4']['mean_direct_l2']
    report=f'''# B6 补充、逐序列配对与技术复现

## Material Passport

- 日期：2026-10-20；Origin Skill：academic-research-suite / experiment-agent。
- 科研解释状态：ANALYZED；同环境技术复现状态：**{analysis['technical_verdict']}**，严格检查通过为 **{strict}**。该状态不表示独立生物学验证。
- 冻结清单 SHA256：`{analysis['reference_manifest_sha256']}`；未重新训练 VMAE、校准尺度或选择学习率。
- B0–B5 数值取自冻结原始结果，B6 只取第一次 primary_b6，不按重复运行中的最高值选择。

## 固定设计和结果

B6=40 步 Latent→冻结 z→40 步 Direct，两个阶段使用 B5 的原正则，沿用 B4/B5 步长与系数。共同约束仍为 Hamming≤20%、GC 变化≤5 个百分点、JS≤0.05；导出的是最高离散分数的可行候选。模型和评分器权重前后不变。详情见[执行前协议](B6_reproducibility_protocol_2026-10-20.md)。

| 组 | 平均最佳可行增量 | 提升数 | 平均可变位改动 | 平均 decoder KL | 最终一步可行数 |
|---|---:|---:|---:|---:|---:|
{arm_table}

B6 不是本批平均内部评分最高的组；不据此改变正则系数、步长或初始化。最终一步可能不可行，不能以其更高原始分数替代最佳可行候选。

## 两个主配对比较

| 对比 | 平均差 | 探索性 95% 区间 | 胜/平/负 |
|---|---:|---|---|
{pairs}

以 16 条 CDS、{len(groups)} 个冻结 cluster_id 组件进行 5,000 次配对 bootstrap，seed=20261020；JSON 同时给出两个主比较的 Bonferroni 口径 97.5% 区间。不将技术重复算作 48 或 64 条独立样本，不据小样本区间声称正式泛化。

两个主比较的 95% 区间均包含零，当前既没有 B6 超过 B4/B5 的明确证据，也不能据此证明等价或非劣。B6 相对 B4 的直接残差平均能量减少 {residual_reduction:.1%}，平均可变位改动为 {analysis['arms']['B6']['mean_hamming']:.2%}，低于 B4 的 {analysis['arms']['B4']['mean_hamming']:.2%}。这是更保守搜索的描述性证据，不等于表达效果更优。

逐序列差异仍须保留：第 02 条是两个比较中回落最大的样本，第 07/12 条则相对 B5 有较大增益。它们都在固定分析中保留，不删除回落样本，也不根据这些结果重调参数。

## 16 条逐序列结果

ID 使用 SHA 前 12 位，完整 ID 见 [16 行配对 CSV](b6_per_cds_paired_20261020.csv)；[112 行完整长表](b0_b6_per_cds_long_20261020.csv)另含全部七组、原始分数、阶段终点、改动/漂移和结构指标。

| 序号 | CDS ID 前缀 | codons | B4 增量 | B5 增量 | B6 增量 | B6−B4 | B6−B5 | B6 重复 DNA 一致 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{per_cds}

## 独立进程技术复现

primary_b6 之后，在三个新的 Python 进程重复 B0–B6，seed 元数据为 20261020/20261021/20261022。设计变量始终从 μ、零直接残差开始，模型 eval；seed 变化不构成新的随机起点。这检查计算路径是否重现，不检验新数据或随机起点鲁棒性。

严格检查忽略实际耗时，其余结果字段和逐步记录要求完全一致。另按预先声明的 1e-6 分数绝对容差、相同 DNA/可行性/最优步数给出诊断，不用容差掩盖严格检查的差异。

| 重复 | 对照来源 | 严格相同结果行 | 严格相同步骤行 | 最佳 DNA 一致 | 最大分数误差 | 容差诊断通过 |
|---|---|---:|---:|---:|---:|---|
{repeat_table}

baseline 对照为原冻结 B0–B5，b6 对照为第一次 B6。所有进程退出码为 0，检查原权重和参考文件哈希不变；对已有环境记录的 Python、PyTorch、CUDA 和 GPU 字段逐项核对一致。原始收据没有记录的后端标志不能追认为已核对，新运行已记录。

B6 的 16 条 CDS 均核验 Latent/Direct 各 40 步，Latent 阶段直接变量为零，Direct 阶段潜在变量能量保持不变。每方法每 CDS 80 次目标反向、240 次评分器序列前向，与冻结预算一致。独立进程复现开销另计，未给单次优化增加搜索预算。

## 复查文件与边界

- [结构化配对与复现结果](b6_paired_reproducibility_20261020.json)
- [逐序列图](b6_per_cds_paired_20261020.png)
- [B6 最佳候选 FASTA](b6_best_candidates_20261020.fasta)
- 原始输出：`{root.as_posix()}`；冻结参考：`{reference.as_posix()}`。

RNAfold 在搜索完成后计算，不参与候选选择，结构变化不等于表达效果。原 16 条都是此前使用过的短 CDS，近同源隔离未认证。当前证据区分技术复现、跨 CDS 一致性和独立外部验证：前两项可由本报告核查，后者尚未完成。不能以同一确定性路径的重复成功代替新的基因、不同起点、独立评价器或实际表达复现。

统计解释 11/11 覆盖：长度/家族混合（限制短 CDS）、总体外推（逐条全报）、选择偏差（不挑新高分样本）、条件化混杂（共同约束且保留不可行轨迹）、基率（B0 原序列）、均值回归（同起点固定参考）、存活偏差（要求全部进程和全部样本）、多重比较（两个主对比全报并给调整区间）、分析路径（固定协议且不再调参）、相关/因果（内部预测非表达因果）、反向因果（优化目标并非机制证明）。
'''
    (docs/'B6_paired_reproducibility_2026-10-20.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(strict_reproduction=strict,primary_comparisons=analysis['primary_comparisons'],B6=analysis['arms']['B6'])))


if __name__=='__main__':main()
