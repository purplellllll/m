"""Join existing generator baselines by exact case/sequence identity; no inference."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
WORK=OUT.parent.parent
OLD=WORK/'runs/seq2ribo_multicell_comparison_20261027'
KEYS=['ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_kcal_mol','mfe_per_nt']
METHODS=['original','V3_2.5M_Hard','own80','VMAE_9.7M_Hard','V3_same_reward','V4_epoch6_current',
         'codontransformer','fpp_codonbert','gemorna','lineardesign_l0','RiboDecode_native']
LABELS={'original':'原始CDS','V3_2.5M_Hard':'原V3／RiboDecode搜索','own80':'V3／自有评分器搜索',
    'VMAE_9.7M_Hard':'971万VMAE／RiboDecode搜索','V3_same_reward':'V3／多指标搜索',
    'V4_epoch6_current':'V4第6轮','codontransformer':'CodonTransformer','fpp_codonbert':'FPP-CodonBERT',
    'gemorna':'GEMORNA','lineardesign_l0':'LinearDesign λ=0','RiboDecode_native':'原生RiboDecode完整生成流程'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (OLD/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
    manifest=read(OLD/'candidate_manifest.json')
    registry={(r['sequence_id'],r['method']):r for r in manifest.values()}
    index={}
    for row in old:
        key=(row['sequence_id'],row['method']);assert key not in index
        entry=registry[key]
        assert hashlib.sha256(entry['cds_dna'].encode()).hexdigest()==entry['sequence_sha256']
        for metric in KEYS:assert abs(float(row[metric])-entry[metric])<1e-9
        index[key]=dict(sequence_id=row['sequence_id'],method=row['method'],**{k:float(row[k]) for k in KEYS})
    new=read(OUT/'comparison_metrics.json')
    for row in new:
        if row['method']=='V3_own_scorer_search':
            index[row['sequence_id'],'own80']=dict(sequence_id=row['sequence_id'],method='own80',**{k:row[k] for k in KEYS})
        else:
            method='original' if row['method']=='original' else 'V3_2.5M_Hard'
            reference=index[row['sequence_id'],method]
            assert row['dna']==registry[row['sequence_id'],method]['cds_dna']
            for metric in KEYS:assert abs(row[metric]-reference[metric])<1e-5
    ids={method:{identity for identity,m in index if m==method} for method in METHODS}
    all30=ids['original'];assert len(all30)==30
    assert len(ids['lineardesign_l0'])==27 and len(ids['RiboDecode_native'])==8
    assert ids['lineardesign_l0']<=all30 and ids['RiboDecode_native']<=all30
    cohorts=[('common30',all30,[m for m in METHODS if len(ids[m])==30]),
             ('common27_LinearDesign',ids['lineardesign_l0'],[m for m in METHODS if len(ids[m])>=27]),
             ('common8_nativeRiboDecode',ids['RiboDecode_native'],[m for m in METHODS if m!='lineardesign_l0'])]
    summaries=[];lines=['# 全部生成方法：同样本基线对照','',
        '更新日期：2026-11-01。本轮仅汇总既有结果及新完成的自有评分器搜索，没有重训或重新生成其他基线。',
        '按sequence_id逐条对齐，核验历史候选序列SHA、全部历史指标以及与新表中原始/V3端点的一致性。',
        'RiboDecode列是预测分数；“原生RiboDecode完整生成流程”行是生成方法，两者含义不同。',
        '','## 原完整30条面板','']
    for number,(cohort,selected,methods) in enumerate(cohorts):
        if number==1:lines+=['','## 加入LinearDesign后的共同27条','',
            '仅取LinearDesign已完成的同一27条，所有其他方法也重新按这27条汇总。不能把这里的27条均值与上表30条直接排名。','']
        if number==2:lines+=['','## 加入原生RiboDecode后的共同8条','',
            '仅取原生完整生成流程已完成的8条，其他方法同步取相同8条。原生与VMAE在线计算预算不同。','']
        lines+=['| 方法 | N | RiboDecode分数 | seq2ribo TE | GC% | CAI | MFE kcal/mol |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for method in methods:
            assert selected<=ids[method]
            rows=[index[identity,method] for identity in sorted(selected)]
            mean=dict(cohort=cohort,method=method,label=LABELS[method],n=len(rows),
                      **{k:statistics.mean(r[k] for r in rows) for k in KEYS})
            summaries.append(mean)
            lines.append(f"| {LABELS[method]} | {len(rows)} | {mean['ribodecode_score']:.4f} | {mean['seq2ribo_score']:.4f} | {mean['gc_percent']:.2f} | {mean['cai']:.4f} | {mean['mfe_kcal_mol']:.2f} |")
    lines+=['','## 历史消融附表（同30条，不作为当前主线）','',
        '| 方法 | N | RiboDecode分数 | seq2ribo TE | GC% | CAI | MFE kcal/mol |','|---|---:|---:|---:|---:|---:|---:|']
    for method in ['V4_epoch2','V4_epoch3']:
        rows=[v for (identity,m),v in index.items() if m==method];assert len(rows)==30
        mean={k:statistics.mean(r[k] for r in rows) for k in KEYS}
        lines.append(f"| {method} | 30 | {mean['ribodecode_score']:.4f} | {mean['seq2ribo_score']:.4f} | {mean['gc_percent']:.2f} | {mean['cai']:.4f} | {mean['mfe_kcal_mol']:.2f} |")
    lines+=['','## 解读与限制','',
        '- 原V3在完整30条上保留最高RiboDecode和seq2ribo均值；自有评分器搜索的独立分数没有超过原V3。',
        '- 同27条中，LinearDesign的MFE更负，但seq2ribo分数较低；不能把更负MFE直接称为更高表达。',
        '- 自有评分器原来在细菌mRFP标量上训练，本人类面板属于域外搜索；所有分数是模型预测，不是生成序列实测表达。',
        '- 各方法训练来源、优化目标与生成预算不同。本表是既有完整方法管线的结果比较，不是等训练/等算力架构消融。',
        '- 历史基线未计算自有评分器分数，所以不填充虚构的自有分数。这里统一展示已有的两套独立评分和理化指标。',
        '- 仅展示精确匹配的同一批样本；面板已反复用于开发，不能称新的独立确认。',
        '', '[全部分组均值CSV](all_baselines_group_means.csv) · [合并逐样本指标CSV](all_baselines_per_case.csv)']
    (OUT/'全部基线对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,rows in [('all_baselines_group_means.csv',summaries),('all_baselines_per_case.csv',list(index.values()))]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt=dict(status='passed',historical_rows=len(old),own_added_rows=30,cohort_counts={name:len(selected) for name,selected,_ in cohorts},
        historical_candidate_sha_verified=True,historical_metrics_verified=True,original_and_v3_endpoints_exactly_matched=True,
        sources_sha256={str(p.relative_to(WORK)):sha(p) for p in [OLD/'metrics.csv',OLD/'candidate_manifest.json',OUT/'comparison_metrics.json']},
        no_new_inference=True,no_imputed_own_scores_for_baselines=True)
    (OUT/'all_baselines_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
