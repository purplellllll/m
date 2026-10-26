"""Exploratory length and post-treatment covariate audit of frozen candidates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def estimates(rows, control):
    length=np.array([float(r['n_codons']) for r in rows])
    y=np.array([float(r['B5_minus_'+control]) for r in rows])
    log_length=np.log2(length/512.)
    covariates=np.array([[float(r['B5_minus_'+control+'_'+key])/scale
        for key,scale in [('gc_fraction',.01),('rnafold_mfe_per_nt',.01),('mutable_hamming',.01)]] for r in rows])
    base=np.column_stack([np.ones(len(rows)),log_length])
    design=np.column_stack([base,covariates])
    if np.linalg.matrix_rank(design)!=5:raise ValueError('Rank deficient adjustment model')
    raw=np.linalg.lstsq(base,y,rcond=None)[0]
    adjusted=np.linalg.lstsq(design,y,rcond=None)[0]
    long=length>512
    if not long.any() or long.all():raise ValueError('Both length groups required')
    return dict(mean=float(y.mean()),short_mean=float(y[~long].mean()),long_mean=float(y[long].mean()),
        long_minus_short=float(y[long].mean()-y[~long].mean()),
        slope_per_length_doubling=float(raw[1]),adjusted_slope=float(adjusted[1]),
        adjusted_equal_covariates_mean=float(adjusted[0]+adjusted[1]*log_length.mean()))


def audit(rows,draws=5000):
    groups=sorted({r['cluster_id'] for r in rows})
    members=[[r for r in rows if r['cluster_id']==g] for g in groups]
    point={c:estimates(rows,c) for c in ['B4','B6']}
    samples={c:{k:[] for k in point[c]} for c in point}
    rng=np.random.default_rng(20261020)
    for _ in range(draws):
        boot=[r for i in rng.integers(0,len(groups),len(groups)) for r in members[i]]
        for c in point:
            values=estimates(boot,c)
            for k,v in values.items():samples[c][k].append(v)
    return {c:{k:dict(estimate=v,ci95=np.quantile(samples[c][k],[.025,.975]).tolist())
        for k,v in point[c].items()} for c in point}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results_b456_random_validation_20261020/analysis/paired_cds.csv')
    p.add_argument('--output',default='docs/B456_length_control_audit_20261020')
    args=p.parse_args();source=Path(args.input)
    with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==64
    result=dict(status='EXPLORATORY',cds=len(rows),clusters=len({r['cluster_id'] for r in rows}),
        long_definition='n_codons > 512; chosen after pilot, requires fresh confirmatory set',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        adjustment='paired score difference ~ 1 + log2(codons/512) + GC difference/0.01 + MFE-per-nt difference/0.01 + mutable-Hamming difference/0.01',
        caveat='GC/MFE/edit are post-treatment variables: adjustment is conditional association, not causal or independent validation',
        contrasts=audit(rows))
    prefix=Path(args.output)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2))
    table='\n'.join('| B5−{} | {} | {:+.6f} | [{:+.6f}, {:+.6f}] |'.format(c,k,v['estimate'],*v['ci95'])
        for c,values in result['contrasts'].items() for k,v in values.items())
    report='''# B4/B5/B6 长度与控制变量探索性审计

2026-10-20；Origin Skill: academic-research-suite / experiment-agent。状态 EXPLORATORY。使用固定 64 条 CDS，先平均三个 seed，按 63 个原组件进行 5,000 次 bootstrap。原实验与候选不修改。长序列定义为 >512 codon；该定义和本次模型在看到 pilot 后提出，不能包装为事前确认性检验。

| 配对 | 指标 | 估计值 | 探索性 95% 区间 |
|---|---|---:|---|
'''+table+'''

连续长度模型使用 log2(codons/512)，斜率为长度翻倍时配对优势的关联变化。long_minus_short 比较长短组的平均配对优势；它与“长组本身是否仍有正优势”是两个问题。两者均报告，不凭某组不显著、另一组显著断言有交互，也不将曲线可能非线性的结果强行解释为单调下降。

调整模型另含 B5 与对照的 GC 比例差、MFE/nt 差和可变位改动率差；等协变量均值将这三个差置零，并在观察到的长度分布上平均。它依赖线性模型和重叠假设。GC/MFE/修改率均由优化方法影响，属于处理后变量，控制它们可能产生选择偏差或阻断真实作用路径。因此调整后保留的差不能单独证明双空间的独立作用，更不能回答独立预测器能否复现。

正式验证须在未用于这些分析的新 CDS 上固定长度交互检验；候选须在查看独立评分前锁定。共同变化预算、控制指标匹配与未匹配全体结果都应报告；不能只报告匹配成功者。RiboDecode 同时用于优化和评分，本审计没有独立预测器结果，也不构成生物学验证。未做多重比较确认性推断。

统计解释核对：11/11 checked。保留各长度层与逐序列差异，不从群体均值推断每条序列；长度分层选择与处理后协变量的选择/碰撞偏差均限制外推；分类基率不适用；非极端分数抽样，保留全部 64 CDS 和三个 seed；报告所有模型与区间，不做探索后确认性挑选；不以内部评分关联推出表达因果或反向生物学机制。
'''
    prefix.with_suffix('.md').write_text(report,encoding='utf-8')
    print(json.dumps(result['contrasts']))


if __name__=='__main__':main()
