"""Separate native scalar tasks from matched position-profile comparisons."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'runs/scorer_benchmark_breakdown_20261031'
PUBLIC=WORK/'deliverables/public_riboseq_validation_20261028'
SEEDS=[30101,30102,30103]
LABELS={'bilstm_context_x2':'BiLSTM序列2倍候选','gnn_transformer_polish':'sTASEP＋GNN→Transformer',
        'parallel_transformer':'GNN与Transformer并行候选','seq2ribo':'seq2ribo同数据重训','ribogl':'RiboGL同数据重训'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan


def interval(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    samples=np.random.default_rng(20261031).integers(0,len(values),(2000,len(values)))
    lo,hi=np.percentile(values[samples].mean(1),[2.5,97.5]);return float(lo),float(hi)


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    assert sha(OUT/'common_panel_predictions.npz')==verification['predictions_sha256']
    table=pd.read_csv(OUT/'common_panel_per_transcript.csv');errors=[]
    with zipfile.ZipFile(OUT/'common_panel_predictions.npz') as z:
        arrays={name[:-4]:np.load(io.BytesIO(z.read(name)),allow_pickle=False) for name in z.namelist()}
    rows=json.loads((OUT/'transfer/panel.json').read_text())['records']
    for model,frame in table.groupby('model',sort=False):
        assert frame.gene.tolist()==[r['gene'] for r in rows]
        for j,row in enumerate(frame.itertuples()):
            mask=arrays[f'observed__{j}'];inside=mask.copy();inside[:5]=False;inside[-5:]=False
            y=arrays[f'label__{j}'];q=arrays[f'{model}__{j}']
            for column,m in [('pcc',mask),('interior_pcc',inside)]:
                value=correlation(y[m],q[m]);saved=getattr(row,column)
                assert np.isfinite(value)==np.isfinite(saved)
                if np.isfinite(value):errors.append(abs(value-saved));assert errors[-1]<1e-7
            if model in LABELS:
                expected=np.mean([arrays[f'{model}_{s}__{j}'] for s in SEEDS],0)
                np.testing.assert_array_equal(expected,q)
    # Recompute all native human scalar metrics from saved predictions.
    human=pd.read_csv(PUBLIC/'predictions.csv');native=[]
    for sample,g in human.groupby('sample'):
        assert set(g.gene_split)=={'test'}
        eligible=g[g.te_eligible]
        for target,predictor,frame in [('RPF','predicted_log_rpf',g),('TE','fixed_count_score',eligible)]:
            y=frame['log_rpf' if target=='RPF' else 'log2_te'];q=frame[predictor]
            native.append(dict(benchmark='RiboDecode',sample=sample,cell=g.cell.iloc[0],study=g.study.iloc[0],target=target,
                n=len(frame),pearson=correlation(y,q),ours_pearson=None,comparison='unavailable: no matching human RPF/TE head'))
    seq=pd.read_csv(PUBLIC/'seq2ribo_calibration/predictions.csv')
    genes_by_cell=[set(g.gene_id) for _,g in seq.groupby('cell')]
    assert all(genes==genes_by_cell[0] for genes in genes_by_cell)
    for cell,g in seq.groupby('cell'):
        assert g.gene_id.nunique()==len(g)==50
        native.append(dict(benchmark='seq2ribo',sample=cell,cell=cell,study='public human TE supplementary table',target='TE',
            n=len(g),pearson=correlation(g.observed,g.prediction),ours_pearson=None,
            comparison='unavailable: descriptive calibration; training overlap uncertified'))
    native=pd.DataFrame(native);native.to_csv(OUT/'native_scalar_benchmarks.csv',index=False)
    original=pd.read_csv(PUBLIC/'summary.csv').set_index('sample')
    for r in native[native.benchmark=='RiboDecode'].itertuples():
        col='rpf_pearson' if r.target=='RPF' else 'sequence_te_pearson'
        assert abs(r.pearson-original.loc[r.sample,col])<1e-10
    seqmetrics=pd.read_csv(PUBLIC/'seq2ribo_calibration/calibration_metrics.csv').query("predictor=='prediction'").set_index('cell')
    for r in native[native.benchmark=='seq2ribo'].itertuples():assert abs(r.pearson-seqmetrics.loc[r.cell,'pearson'])<1e-10
    effects=[];summaries=[]
    strata={'all':lambda f:np.ones(len(f),bool),
        'length_80_139':lambda f:(f.n_codons>=80)&(f.n_codons<140),
        'length_140_199':lambda f:(f.n_codons>=140)&(f.n_codons<200),
        'length_200_250':lambda f:(f.n_codons>=200)&(f.n_codons<=250),
        'observed_lt95pct':lambda f:f.observed_fraction<.95,
        'observed_ge95pct':lambda f:f.observed_fraction>=.95}
    for label,condition in strata.items():
        frame=table[condition(table)]
        for model in LABELS:
            group=frame[frame.model==model]
            assert len(group)>0
            lo,hi=interval(group.pcc.to_numpy())
            ilo,ihi=interval(group.interior_pcc.to_numpy())
            seedmeans=[frame[frame.model==f'{model}_{s}'].pcc.mean() for s in SEEDS]
            summaries.append(dict(stratum=label,model=model,n=len(group),pcc=group.pcc.mean(),ci_low=lo,ci_high=hi,
                interior_pcc=group.interior_pcc.mean(),interior_ci_low=ilo,interior_ci_high=ihi,
                seed_pcc_mean=np.mean(seedmeans),seed_pcc_sd=np.std(seedmeans,ddof=1)))
        for candidate in ['bilstm_context_x2','gnn_transformer_polish','parallel_transformer']:
            for reference in ['seq2ribo','ribogl']:
                for metric in ['pcc','interior_pcc']:
                    a=frame[frame.model==candidate].set_index('gene')[metric]
                    b=frame[frame.model==reference].set_index('gene')[metric]
                    paired=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna();d=paired.a-paired.b
                    lo,hi=interval(d.to_numpy());seed_delta=[]
                    for s in SEEDS:
                        x=frame[frame.model==f'{candidate}_{s}'].set_index('gene')[metric]
                        y=frame[frame.model==f'{reference}_{s}'].set_index('gene')[metric]
                        seed_delta.append(float((x-y).mean()))
                    effects.append(dict(stratum=label,metric=metric,model=candidate,reference=reference,n=len(d),delta=d.mean(),ci_low=lo,ci_high=hi,
                        gene_wins=int((d>0).sum()),positive_seeds=sum(x>0 for x in seed_delta),
                        **{f'seed_delta_{s}':x for s,x in zip(SEEDS,seed_delta)}))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'profile_summary.csv',index=False)
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'profile_paired_differences.csv',index=False)
    selected=summary[summary.stratum=='all'].set_index('model')
    lines=['# RiboDecode与seq2ribo评测集拆分核验','', '## Material Passport','',
        '- Origin Skill: academic-research-suite/experiment-agent；2026-10-31。',
        '- Verification Status: 已复算已有标量预测，并新增冻结候选在共同位置面板的推理、独立指标复算。',
        '- 本轮没有训练新模型、没有改变运行中的扩大数据训练队列。','',
        '## 结论边界','',
        '目前不能宣称同时超越RiboDecode和seq2ribo。RiboDecode原评测是跨基因RPF/TE标量相关；我们当前自建模型是转录本内部位置分布预测，没有匹配的人类RPF/TE头和外部测量。',
        '在下列同一历史120小鼠基因的位置任务中，可以比较自建模型与seq2ribo官方架构共同数据重训；它不等于超越seq2ribo在人类TE原评测集上的泛化性能。RiboGL与RiboDecode是两个不同模型。',
        '早期V3在两个评分器上获得高生成序列分数，也不等于自建评分器的实测Pearson超过原评分器。',
        '新增同面板复核：BiLSTM序列2倍候选整体/内部位置PCC为0.4231/0.3568，seq2ribo重训为0.3040/0.2684；前者两个口径均更高。完整GNN→Transformer为0.3318/0.2561，整体较高、内部位置点估计反而更低，内部差值的95%配对区间仍跨0，未确认可靠劣势；也不能只报整体均值就声称位置表现全面超越。移除端部的相关性变化不是各位置贡献的可加分解，尚不能证明全部优势来自端部。','',
        '## RiboDecode：原有人类标量评测分研究列出','',
        '| 细胞/研究 | RPF N | RiboDecode RPF r | RNA基线 r | TE N | 固定RNA序列评分与TE r | 当前自建模型 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for sample in ['SRX384671','SRX474109','SRX021966','SRX1070446']:
        r=original.loc[sample]
        lines.append(f'| {r.cell}/{r.study} | {int(r.n)} | {r.rpf_pearson:.4f} | {r.rna_baseline_pearson:.4f} | {int(r.te_n)} | {r.sequence_te_pearson:.4f} | 未测对应RPF/TE输出 |')
    lines+=['',f'3942条是基因×样本记录，对应{human.gene_id.nunique()}个不同测试基因，不能视为3942个独立基因。所有行均为官方留出基因；HEK293T是已见环境，HeLa按官方元数据为留出环境。完整RPF输入真实RNA丰度及样本环境，目标log1p(5×RPKM)；固定RNA评分的TE标签log2(RPF/RNA)，要求两种RPKM均>=1。',
        '0.91–0.95的完整丰度相关包含RNA输入贡献，不能当作CDS-only序列评分准确率；不得直接与我们约0.3–0.4的位置相关相减。官方权重是否曾按测试表现选取仍未认证。','',
        '## seq2ribo：原有人类TE校准分细胞列出','',
        '| 细胞 | N（不同基因） | seq2ribo TE Pearson | 当前自建模型 | 是否可声称超越 |','|---|---:|---:|---|---|']
    for cell in ['hek293','lcl','rpe']:
        r=seqmetrics.loc[cell];lines.append(f'| {cell} | 50 | {r.pearson:.5f} | 未测匹配TE输出 | 否 |')
    lines+=['','同一50基因在三个环境重复测量；这批标签来源也是seq2ribo的TE训练数据来源，训练/测试身份清单未恢复，属于描述性校准。0.99不能当作已认证独立留出性能。iPSC未校准，不擅用其他细胞替换。','',
        '## 冻结候选：同一历史120基因上的新位置复核','',
        '三个当前候选在新推理前预先指定并锁定9份权重。BiLSTM序列2倍是此前验证选定候选；parallel是此前polish实验验证选定候选；完整polish是用户重点架构。没有按这批样本结果重新选择模型。',
        '三者与两套参考的训练400/验证80基因及CDS完全一致，物理拟合参数也逐项相同。参考是此前冻结的seq2ribo/RiboGL共同数据重训权重。全部方法每种子先按完整有效CDS预测总和归一化，再等权集成，统一去除不同种子输出总量的隐式加权。',
        '该120基因早已用于历史开发，本轮是诊断再评估，不是新的确认性测试。源为GSE73553小鼠肝脏80–250密码子筛选面板；没有近同源或跨研究隔离。','',
        '| 模型 | 平均位置PCC | 95%基因区间 | 内部位置PCC | 单种子均值±SD |','|---|---:|---|---:|---|']
    for model,label in LABELS.items():
        r=selected.loc[model];lines.append(f'| {label} | {r.pcc:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.interior_pcc:.4f} | {r.seed_pcc_mean:.4f}±{r.seed_pcc_sd:.4f} |')
    lines+=['','| 口径 | 自建候选减参考 | ΔPCC | 95%配对区间 | 逐基因胜出 | 正差种子 |','|---|---|---:|---|---:|---:|']
    for r in effects[effects.stratum=='all'].itertuples():
        lines.append(f'| {"整体" if r.metric=="pcc" else "内部位置"} | {LABELS[r.model]}减{LABELS[r.reference]} | {r.delta:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.gene_wins}/{r.n} | {r.positive_seeds}/3 |')
    lines+=['','这些PCC是每条转录本内预测与log1p归一化标签的相关，再按转录本等权平均；内部位置去首尾各5密码子。集成PCC不等于单种子PCC均值。',
        '统一尺度后可能与旧报告直接平均原始输出的集成值略有不同；不把不同集成口径混排。参数量、原生损失和训练批量仍不同，不是严格等容量优化预算比较。','',
        '## 按长度与覆盖拆分（探索性）','',
        '| 分层 | N | BiLSTM候选 | 完整GNN→Transformer | 并行Transformer | seq2ribo | RiboGL |','|---|---:|---:|---:|---:|---:|---:|']
    for stratum in strata:
        if stratum=='all':continue
        f=summary[summary.stratum==stratum].set_index('model');v=[f'{f.loc[a,"pcc"]:.4f}' for a in LABELS]
        lines.append(f'| {stratum} | {int(f.iloc[0].n)} | '+' | '.join(v)+' |')
    lines+=['','采用固定边界作事后探索性分层：80–139、140–199、200–250密码子；实测标签有限比例<95%或>=95%。没有前瞻性分层登记，也未按模型分数扫描最优阈值。这里只报位置PCC均值；每个配对区间和种子差另存profile_paired_differences.csv。全部子组报告，不能挑一个子组声称整体超越。',
        '2000次基因bootstrap及配对差未做多重比较校正；历史样本反复开发，区间仅用于描述已训练模型的测试基因重采样，不覆盖训练全过程不确定性。','',
        '## 正在扩大训练的数据','',
        '新训练使用5564训练/696验证/696最终测试；本轮没有打开其测试预测。尚无该队列完成权重，因此不能把旧候选的历史120成绩写成新训练成绩。新模型在696测试上的最终结论应在全部权重及验证候选锁定之后给出。',
        '若要在RiboDecode/seq2ribo原TE任务上直接比较，需要与真实TE/RNA-RPF配对标签训练和验证的标量输出，并对齐细胞、基因划分、RNA及环境输入条件。当前归一化位置损失不识别绝对总量，不能简单把预测位置求和冒充TE。','',
        '## 核验','',
        f'9份新推理冻结权重哈希通过，参考6份权重的保存预测及原面板哈希随输入包保留；9份当前权重未重训。服务器重复推理误差{verification["reproducibility_errors"]}。',
        f'本地用NumPy独立重算{len(errors)}个位置PCC/内部PCC、全部三种子集成，最大误差{max(errors):.3g}；4研究RPF/TE和3细胞TE标量值均从逐基因保存预测复算，与原汇总一致。',
        '11/11统计风险核查：报告长度/覆盖分层但不据此作因果；位置指标不外推总TE；筛选限制外推；未做碰撞变量因果调整；非分类无基准率判别；未按极端分数选样；全部组/种子报告；多重比较未校正；明确历史面板与训练身份不确定性；相关不作机制因果；不识别反向因果。','',
        '输入、权重和输出哈希记录见evaluation_locked.json、verification.json、local_audit.json及transfer/transfer_manifest.json。原始标量预测保留在deliverables/public_riboseq_validation_20261028。']
    (OUT/'评分器评测集拆分与超越核验.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'local_audit.json').write_text(json.dumps(dict(status='passed',position_values=len(errors),max_error=max(errors),
        native_scalar_rows=len(native),n_human_ribodecode_records=len(human),n_human_te_genes=50,
        inputs={str(p.relative_to(WORK)):sha(p) for p in [PUBLIC/'predictions.csv',PUBLIC/'seq2ribo_calibration/predictions.csv',OUT/'common_panel_predictions.npz']}),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');names=list(LABELS);xx=np.arange(len(names))
    for offset,column,low,high,color,label in [(-.13,'pcc','ci_low','ci_high','#397ba7','Full valid CDS'),
        (.13,'interior_pcc','interior_ci_low','interior_ci_high','#b54e42','Remove5 codons at each end')]:
        v=selected.loc[names,column].to_numpy();lo=selected.loc[names,low].to_numpy();hi=selected.loc[names,high].to_numpy()
        ax.errorbar(v,xx+offset,xerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=3,color=color,label=label)
    ax.set_yticks(xx,names);ax.invert_yaxis();ax.set_xlabel('Mean within-transcript PCC (log1p)')
    ax.set_title('Same historical120 mouse genes; all ensembles normalized per seed');ax.legend()
    fig.savefig(OUT/'profile_full_vs_interior.png',dpi=170);plt.close(fig)
    print(summary[summary.stratum=='all'].to_json(orient='records'));print(effects[effects.stratum=='all'].to_json(orient='records'))


if __name__=='__main__':main()
