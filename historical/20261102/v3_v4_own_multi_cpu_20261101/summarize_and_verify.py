"""Independently audit both reward searches and compare seven fixed pipelines."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
PREVIOUS=WORK/'runs/vmae_own_full30_cpu_20261031'
KEYS=['own_score','combined_reward','ribodecode_score','seq2ribo_score','gc_percent','cai','mfe_kcal_mol','mfe_per_nt']
CODONS='TTT TTC TTA TTG CTT CTC CTA CTG ATT ATC ATA ATG GTT GTC GTA GTG TAT TAC TAA TAG CAT CAC CAA CAG AAT AAC AAA AAG GAT GAC GAA GAG TCT TCC TCA TCG CCT CCC CCA CCG ACT ACC ACA ACG GCT GCC GCA GCG TGT TGC TGA TGG CGT CGC CGA CGG AGT AGC AGA AGG GGT GGC GGA GGG'.split()
AMINO='FFLLLLLLIIIMVVVVYY**HHQQNNKKDDEESSSSPPPPTTTTAAAACC*WRRRRSSRRGGGG'
CODE=dict(zip(CODONS,AMINO))
LABELS={'original':'原始CDS','V3_RiboDecode_search':'V3原RiboDecode搜索','V3_own_only':'V3仅自有分数',
    'v3_historical_multiobjective':'V3旧多指标','v3_own_multiobjective':'V3自有分数＋性质奖励',
    'v4_historical_multiobjective':'V4第6轮旧多指标','v4_own_multiobjective':'V4第6轮自有分数＋性质奖励'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def protein(dna):return ''.join(CODE[dna[i:i+3]] for i in range(0,len(dna),3))
def combined(value,reference):
    mfe=value.get('mfe_per_nt');rmfe=reference['mfe_per_nt']
    return value['own_score']+(value['gc_percent']-reference['gc_percent'])/5.+(value['cai']-reference['cai'])/.05+(rmfe-mfe)/.05
def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    all_rows=[];audits={};references={}
    for row in read(PREVIOUS/'comparison_metrics.json'):
        method='V3_own_only' if row['method']=='V3_own_scorer_search' else row['method']
        value=dict(sequence_id=row['sequence_id'],group=row['group'],method=method,
            own_score=row['own_score'],ribodecode_score=row['ribodecode_score'],seq2ribo_score=row['seq2ribo_score'],
            gc_percent=row['gc_percent'],cai=row['cai'],mfe_kcal_mol=row['mfe_kcal_mol'],mfe_per_nt=row['mfe_per_nt'])
        all_rows.append(value)
        if method=='original':references[row['sequence_id']]=value
    for version,stem in [('v3_multi','v3'),('v4_epoch6','v4')]:
        out=ROOT/version;results=read(out/'final_search_results.json');protocol=read(out/'protocol_locked.json')
        complete=read(out/'evaluation_complete.json');table=read(out/'comparison_metrics.json')
        assert complete['status']=='completed' and len(table)==90 and len(results)==30
        assert sha(out/'comparison_metrics.csv')==complete['metrics_sha256']
        assert sha(out/'evaluation_protocol_locked.json')==complete['protocol_sha256']
        assert sha(ROOT/'run_experiment.py')==protocol['driver_sha256']
        expected=WORK/('runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt' if stem=='v3' else 'deliverables/vmae_v4_epoch6_current_20261027/checkpoints/epoch6.pt')
        assert sha(expected)==protocol['checkpoint_sha256']
        lookup={(r['sequence_id'],r['method']):r for r in table};assert len(lookup)==90
        max_reward_error=0.;unique=set()
        for result in results:
            original=result['original'];identity=result['sequence_id']
            assert result['steps']==80 and len(result['trace'])==81
            assert [r['step'] for r in result['trace']]==list(range(81))
            for value in [original,result['old_v3']]+result['trace']:
                assert protein(value['dna'])==protein(original['dna'])
                assert value['dna'][:3]==original['dna'][:3] and value['dna'][-3:]==original['dna'][-3:]
                key=hashlib.sha256(value['dna'].encode()).hexdigest();assert key==value['sequence_sha256']
                source=Path(value.get('feature_cache_root',str(out)))
                assert sha(source/'features'/f'{key}.npz')==value['feature_sha256']
                error=abs(value['combined_reward']-combined(value,original));max_reward_error=max(max_reward_error,error)
                assert error<1e-10
                assert abs(statistics.mean(value['own_seed_scores'])-value['own_score'])<1e-12
                unique.add(key)
            winner=max([dict(original,step=-1)]+result['trace'],key=lambda v:v['combined_reward'])
            assert winner['sequence_sha256']==result['best']['sequence_sha256']
            assert abs(result['combined_gain_from_original']-(winner['combined_reward']-original['own_score']))<1e-10
            for field,method in [('original','original'),('old_v3','historical_multiobjective'),('best','own_multiobjective')]:
                row=lookup[identity,method];assert row['dna']==result[field]['dna'] and row['own_score']==result[field]['own_score']
                key=row['sequence_sha256']
                hard=read(out/'evaluation/ribodecode'/f'{key}.json');seq=read(out/'evaluation/seq2ribo'/f'{key}.json')
                assert row['ribodecode_score']==hard['ribodecode_score'] and row['seq2ribo_score']==seq['seq2ribo_score']
                assert abs(row['seq2ribo_score']-statistics.mean(seq['seq2ribo_by_seed'].values()))<1e-10
        for row in table:
            if row['method']=='original':continue
            all_rows.append(dict(sequence_id=row['sequence_id'],group=row['group'],method=stem+'_'+row['method'],
                own_score=row['own_score'],ribodecode_score=row['ribodecode_score'],seq2ribo_score=row['seq2ribo_score'],
                gc_percent=row['gc_percent'],cai=row['cai'],mfe_kcal_mol=row['mfe_kcal_mol'],mfe_per_nt=row['mfe_per_nt']))
        audits[version]=dict(status='passed',n_cases=30,trajectory_records=2490,unique_sequences=len(unique),max_reward_error=max_reward_error,
            checkpoint_unchanged=True,combined_winner_verified=True,protein_start_stop_verified=True,
            feature_and_evaluation_hashes_verified=True,all_seed_score_means_verified=True)
    for row in all_rows:row['combined_reward']=combined(row,references[row['sequence_id']])
    assert len(all_rows)==210 and len({(r['sequence_id'],r['method']) for r in all_rows})==210
    summaries=[]
    for method in LABELS:
        rows=[r for r in all_rows if r['method']==method];assert len(rows)==30
        summaries.append(dict(method=method,label=LABELS[method],n=30,**{k:statistics.mean(r[k] for r in rows) for k in KEYS}))
    csv_write(ROOT/'comparison_metrics.csv',all_rows);csv_write(ROOT/'summary.csv',summaries)
    save(ROOT/'comparison_summary.json',dict(summary=summaries,version_audits=audits))
    lines=['# V3多指标与V4第6轮：改用自有评分器后的完整30条比较','',
        '固定原30条人类CDS面板，两个版本各80轮，保留原性质奖励，仅替换表达评分项。以下为全部30条均值。',
        '','| 方法 | 自有评分 | 自有＋性质总奖励 | RiboDecode | seq2ribo TE | GC% | CAI | MFE kcal/mol |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['label']} | {row['own_score']:.4f} | {row['combined_reward']:.4f} | {row['ribodecode_score']:.4f} | {row['seq2ribo_score']:.4f} | {row['gc_percent']:.2f} | {row['cai']:.4f} | {row['mfe_kcal_mol']:.2f} |")
    lines+=['','## 解释边界','',
        '- V3多指标不是另一份权重；V4第6轮是多指标微调后的另一份固定权重。架构相同，权重与先验不同。',
        '- 总奖励为自有分数＋ΔGC/0.05＋ΔCAI/0.05−ΔMFE每nt/0.05；性质奖励没有上限，总奖励上升不保证自有分数上升。',
        '- 旧方法使用RiboDecode梯度／SCST搜索；本轮使用黑盒潜空间搜索，外循环相同但计算预算与优化器不同。',
        '- 自有标量头训练于细菌mRFP，在人类CDS上属域外使用；新序列没有实测表达。',
        '- 搜索采用与评分器训练特征一致的ViennaRNA2.7.0，独立性质展示统一2.7.2；原90端点预检MFE差为0，不把这当作所有候选能量必然相同。',
        '- 独立RiboDecode和seq2ribo只在选择锁定后复评，不用于挑选新候选。既有面板重复开发，不称独立确认。',
        '', '[逐条完整指标](comparison_metrics.csv) · [均值CSV](summary.csv)']
    (ROOT/'完整结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(ROOT/'independent_audit.json',dict(status='passed',versions=audits,total_completed_cases=60,all_summary_rows_n30=True,
        no_retraining=True,no_measured_expression_claim=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Original','V3 Hard','V3 own only','V3 old multi','V3 own+props','V4 old multi','V4 own+props']
    metrics=[('own_score','Own scorer'),('combined_reward','Own + property reward'),('ribodecode_score','RiboDecode'),
             ('seq2ribo_score','seq2ribo HEK293 TE'),('cai','CAI'),('mfe_kcal_mol','MFE (kcal/mol)')]
    colors=['#9ca3af','#3b82f6','#a78bfa','#6b7280','#f59e0b','#94a3b8','#ef4444']
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for ax,(key,title) in zip(axes.flat,metrics):
        values=[r[key] for r in summaries];bars=ax.barh(names,values,color=colors)
        ax.invert_yaxis();ax.set_title(title);ax.axvline(0,color='#555',lw=.6)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8);ax.margins(x=.18);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fixed30 cases: V3 / V4 epoch6, own scorer + unchanged property rewards')
    fig.text(.5,.02,'Human CDS are out of domain for mRFP-trained scorer. Predictions only; unequal optimizers/query budgets.\nUnbounded GC/CAI/MFE rewards can dominate scalar-score changes.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(ROOT/'comparison.png',dpi=170);plt.close(fig)
    save(ROOT/'completion_receipt.json',dict(status='completed',versions=2,cases_each=30,steps_each=80,
        summary_sha256=sha(ROOT/'summary.csv'),metrics_sha256=sha(ROOT/'comparison_metrics.csv'),audit_sha256=sha(ROOT/'independent_audit.json')))
    print('BOTH VERSIONS COMPLETE: 60 cases, independent audits, table, CSV and figure',flush=True)


if __name__=='__main__':main()
