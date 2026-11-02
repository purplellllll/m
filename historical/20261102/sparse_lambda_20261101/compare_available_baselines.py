"""Align existing baseline endpoints to the lambda panel; never impute predictions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
OLD=WORK/'runs/seq2ribo_multicell_comparison_20261027'
FIELDS=['ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_per_nt']
LABELS={'original':'原始CDS','V3_2.5M_Hard':'V3／RiboDecode目标搜索','VMAE_9.7M_Hard':'971万VMAE／RiboDecode目标搜索',
        'V3_same_reward':'V3历史多目标','V4_epoch6_current':'V4第6轮历史多目标','codontransformer':'CodonTransformer',
        'fpp_codonbert':'FPP-CodonBERT','gemorna':'GEMORNA','lineardesign_l0':'LinearDesign λ=0','RiboDecode_native':'原生RiboDecode完整生成流程'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    panel=read(ROOT/'lambda_search_results.json');selected={r['sequence_id'] for r in panel};assert len(selected)==6
    rows=list(csv.DictReader((OLD/'metrics.csv').open(encoding='utf-8-sig',newline='')))
    manifest={(r['sequence_id'],r['method']):r for r in read(OLD/'candidate_manifest.json').values()}
    for r in rows:
        m=manifest[r['sequence_id'],r['method']]
        assert hashlib.sha256(m['cds_dna'].encode()).hexdigest()==m['sequence_sha256']
        for k in FIELDS:assert abs(float(r[k])-m[k])<1e-9
    native={r['sequence_id'] for r in rows if r['method']=='RiboDecode_native'};assert len(native)==8
    cohorts=[('lambda_common6',selected,[m for m in LABELS if m!='RiboDecode_native']),
             ('native_common8',native,[m for m in LABELS if m!='lineardesign_l0'])]
    means=[];lines=['# lambda面板与已有生成基线','',
        '只汇总已有预测，不重新生成、不填补缺失指标。RiboDecode列是评分器输出，原生RiboDecode行是生成流程。分数不是实测相关系数，训练/搜索预算也不同。','']
    for cohort,ids,methods in cohorts:
        lines+=['## '+('与lambda扫描相同6条' if cohort=='lambda_common6' else '原生RiboDecode的共同8条（不同于lambda面板）'),'',
                '|方法|N|RiboDecode分数|seq2ribo TE分数|GC%|CAI|MFE/nt|','|---|---:|---:|---:|---:|---:|---:|']
        for method in methods:
            subset=[r for r in rows if r['method']==method and r['sequence_id'] in ids]
            assert len(subset)==len(ids)
            mean=dict(cohort=cohort,method=method,label=LABELS[method],n=len(ids),**{k:statistics.mean(float(r[k]) for r in subset) for k in FIELDS})
            means.append(mean)
            lines.append(f"|{mean['label']}|{mean['n']}|{mean['ribodecode_score']:.4f}|{mean['seq2ribo_score']:.4f}|{mean['gc_percent']:.2f}|{mean['cai']:.4f}|{mean['mfe_per_nt']:.4f}|")
    lines+=['','原生RiboDecode在lambda六条中仅有两条，因此没有放进六条主表。不同面板均值不可直接排名。其余lambda支线目前只有自有评分和性质指标，尚无完整RiboDecode/seq2ribo复评；历史外部基线也尚无完整自有评分，因此不补造这些列。LinearDesign的lambda是其自身参数，与本轮lambda含义不同。']
    (ROOT/'外部生成基线对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'external_baseline_summary.json').write_text(json.dumps(means,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'external_baseline_audit.json').write_text(json.dumps(dict(status='passed',historical_candidate_hashes_verified=True,metrics_verified=True,
        lambda_panel_ids=sorted(selected),native_panel_ids=sorted(native),native_overlap_with_lambda=len(native&selected),
        source_sha256={str(p):sha(p) for p in [OLD/'metrics.csv',OLD/'candidate_manifest.json',ROOT/'lambda_search_results.json']},no_new_inference=True),indent=2),encoding='utf-8')
    print('PASS: matched common6/common8 cohorts, candidate hashes and historical metrics')

if __name__=='__main__':main()
