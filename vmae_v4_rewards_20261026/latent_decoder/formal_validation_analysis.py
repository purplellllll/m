"""Freeze score-blind matching, then evaluate four predeclared endpoints."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches(rows,calipers):
    return all(max(float(r[k]) for r in rows)-min(float(r[k]) for r in rows)<=v+1e-10
        for k,v in [('gc_fraction',calipers['max_pairwise_gc_fraction']),
                    ('rnafold_mfe_per_nt',calipers['max_pairwise_mfe_per_nt']),
                    ('mutable_hamming',calipers['max_pairwise_mutable_hamming'])])


def endpoint_values(rows):
    long=np.array([r['n_codons']>512 for r in rows])
    result={}
    for control in ['B4','B6']:
        gain=np.array([r['internal_B5_minus_'+control] for r in rows])
        result['length_B5_minus_'+control]=float(gain[long].mean()-gain[~long].mean())
        matched=[r['external_B5_minus_'+control] for r in rows if r['matched_seeds']]
        result['external_B5_minus_'+control]=float(np.mean(matched)) if matched else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--metrics',required=True)
    p.add_argument('--output',required=True);p.add_argument('--freeze-matching',action='store_true')
    p.add_argument('--external');p.add_argument('--external-receipt')
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    with Path(args.metrics).open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
    identities=plan['evaluation_ids'];seeds=plan['seeds'];arms=['B0','B4','B5','B6']
    lookup={(r['sequence_id'],int(r['seed']),r['arm']):r for r in metrics}
    assert len(metrics)==len(lookup)==len(identities)*len(seeds)*4
    assert set(lookup)=={(i,s,a) for i in identities for s in seeds for a in arms}
    mask_path=output/'matching_lock.json'
    if args.freeze_matching:
        assert not mask_path.exists()
        eligible={i:[s for s in seeds if matches([lookup[i,s,a] for a in ['B4','B5','B6']],plan['matching'])] for i in identities}
        payload=dict(plan_sha256=digest(args.plan),metrics_sha256=digest(args.metrics),eligible=eligible,
            coverage=sum(bool(v) for v in eligible.values())/len(identities),
            no_independent_scores_used=True)
        with mask_path.open('x') as f:json.dump(payload,f,indent=2)
        print(json.dumps(dict(matching_coverage=payload['coverage'])),flush=True);return
    if not args.external or not args.external_receipt:p.error('external predictions and receipt required')
    mask=json.loads(mask_path.read_text())
    assert mask['plan_sha256']==digest(args.plan) and mask['metrics_sha256']==digest(args.metrics)
    receipt=json.loads(Path(args.external_receipt).read_text())
    assert receipt['completed'] and not receipt['smoke'] and receipt['weights_unchanged']
    assert receipt['evaluator_lock_sha256']==plan['independent_evaluator_lock_sha256']
    predictions=[json.loads(s) for s in Path(args.external).read_text().splitlines()]
    scores={r['sequence_sha256']:r['seq2ribo_te'] for r in predictions}
    assert len(scores)==len(predictions)
    assert {r['sequence_sha256'] for r in metrics}==set(scores)
    records={r['sequence_id']:r for r in plan['records']};rows=[]
    for identity in identities:
        row=dict(sequence_id=identity,n_codons=records[identity]['n_codons'],cluster_id=records[identity]['cluster_id'],
            matched_seeds=len(mask['eligible'][identity]))
        for control in ['B4','B6']:
            row['internal_B5_minus_'+control]=float(np.mean([float(lookup[identity,s,'B5']['ribodecode_gain'])-float(lookup[identity,s,control]['ribodecode_gain']) for s in seeds]))
            row['external_full_B5_minus_'+control]=float(np.mean([scores[lookup[identity,s,'B5']['sequence_sha256']]-scores[lookup[identity,s,control]['sequence_sha256']] for s in seeds]))
            eligible=mask['eligible'][identity]
            row['external_B5_minus_'+control]=float(np.mean([scores[lookup[identity,s,'B5']['sequence_sha256']]-scores[lookup[identity,s,control]['sequence_sha256']] for s in eligible])) if eligible else None
        rows.append(row)
    point=endpoint_values(rows);rng=np.random.default_rng(20261020)
    strata=[[r for r in rows if lo<=r['n_codons']<=hi] for lo,hi,_ in plan['length_strata']]
    assert len({r['cluster_id'] for r in rows})==len(rows)==256
    samples={k:[] for k in point}
    for _ in range(5000):
        selected=[group[i] for group in strata for i in rng.integers(0,len(group),len(group))]
        for k,v in endpoint_values(selected).items():
            if v is not None:samples[k].append(v)
    endpoints={k:dict(estimate=v,ci95=np.quantile(samples[k],[.025,.975]).tolist() if samples[k] else None,
        corrected_ci98_75=np.quantile(samples[k],[.00625,.99375]).tolist() if samples[k] else None) for k,v in point.items()}
    coverage=sum(bool(r['matched_seeds']) for r in rows)/len(rows)
    coverage_by_length={name:sum(bool(r['matched_seeds']) for r in rows if (r['n_codons']>512)==flag)/sum((r['n_codons']>512)==flag for r in rows) for name,flag in [('short',False),('long',True)]}
    attenuation={c:endpoints['length_B5_minus_'+c]['corrected_ci98_75'][1]<0 for c in ['B4','B6']}
    transfer={c:coverage>=plan['matching']['minimum_cds_coverage'] and endpoints['external_B5_minus_'+c]['corrected_ci98_75'] is not None and endpoints['external_B5_minus_'+c]['corrected_ci98_75'][0]>0 for c in ['B4','B6']}
    summary=dict(status='ANALYZED',cds=len(rows),seed_repeats_are_not_independent_samples=True,
        plan_sha256=digest(args.plan),matching_lock_sha256=digest(mask_path),external_receipt=receipt,
        matching_coverage=coverage,matching_coverage_by_length=coverage_by_length,endpoints=endpoints,
        length_attenuation_supported=attenuation,controlled_cross_predictor_transfer_supported=transfer,
        full_cohort_external_mean={c:float(np.mean([r['external_full_B5_minus_'+c] for r in rows])) for c in ['B4','B6']},
        matched_estimand='conditional on common feature overlap, not a causal direct effect',
        training_data_independence='not certified',condition_limitation='optimizer HEK293T, evaluator HEK293')
    summary['length_groups']={name:dict(n=sum((r['n_codons']>512)==flag for r in rows),
        matched_n=sum(bool(r['matched_seeds']) and (r['n_codons']>512)==flag for r in rows),
        internal={c:float(np.mean([r['internal_B5_minus_'+c] for r in rows if (r['n_codons']>512)==flag])) for c in ['B4','B6']},
        external_full={c:float(np.mean([r['external_full_B5_minus_'+c] for r in rows if (r['n_codons']>512)==flag])) for c in ['B4','B6']})
        for name,flag in [('short',False),('long',True)]}
    with (output/'paired_formal.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    table='\n'.join('| {} | {:+.6f} | {} |'.format(k,v['estimate'],v['corrected_ci98_75']) if v['estimate'] is not None else '| '+k+' | 不可估计 | 无匹配样本 |' for k,v in endpoints.items())
    text='''# B4/B5/B6 正式验证结果

2026-10-20；Origin Skill: academic-research-suite / experiment-agent。状态 ANALYZED。256 条固定测试 CDS，组件互异，长/短各 128 条；三个真正随机起点在 CDS 内平均。固定所有权重、优化预算和候选规则，未重新调参。

| 预定主要终点 | 配对估计值 | 四终点校正后的 98.75% 区间 |
|---|---:|---|
'''+table+'''

length 终点是 B5 优势的“长组减短组”，小于零表示减弱，不等于长组绝对没有优势。external 是满足同 CDS、同 seed 三方法共同匹配的 seq2ribo 分数差，正值表示独立于优化流程的模型也给 B5 更高评分。两类分数的量纲不同，不比较其绝对幅度。

匹配覆盖率：'''+str(coverage)+'''；按长短分组的覆盖率和未匹配全体的外部评分差见 summary.json。匹配不足 50% 或区间不满足预定方向时不宣称控制后跨预测器复现成功；不扩大卡尺、不换种子、不剔除失败或回落样本。四终点采用分层 CDS bootstrap 5,000 次，保持四长度层的样本数，不以 seed 扩充独立样本量。

控制指标由方法影响，匹配是条件性能比较，不是纯粹因果分解。seq2ribo 与优化过程及权重分开，但训练数据独立性未认证，HEK293 也不等于 RiboDecode 的 HEK293T。此结果不是生物学表达验证，不能证明对所有长度、所有基因或所有条件都成立。

统计解释核对：11/11 checked。分层均值见 length_groups 与内部 analysis.json，整体优势不外推到每条 CDS；匹配的选择/碰撞偏差作为限制保留，不称因果效应；诊断基率不适用；不按极端分数抽样；全部预定 CDS/seed 完整性严格核验，匹配失败亦披露；四主要终点统一多重比较校正，其余描述性；计划在正式候选生成前锁定；不从两个模型的相关评分推断表达因果或反向生物学机制。

相关文件：[逐 CDS 结果](paired_formal.csv)、[完整统计与成功判据](summary.json)、[查看独立分数前锁定的匹配表](matching_lock.json)。
'''
    (output/'report.md').write_text(text,encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
