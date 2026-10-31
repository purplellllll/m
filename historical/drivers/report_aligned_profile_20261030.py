"""Create a result report and scientific plot from completed, verified outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/aligned_profile_20261030'
NAMES={'full':'完整三路','without_physics':'去 sTASEP','without_gnn':'去 GNN','without_context':'去上下文',
       'transformer_only':'仅 Transformer','gnn_only':'仅 GNN','physics_only':'仅物理路','bilstm_context':'上下文改 BiLSTM'}


def main():
    metrics=pd.read_csv(OUT/'metrics.csv');difference=pd.read_csv(OUT/'paired_ablation_differences.csv')
    locked=json.loads((OUT/'checkpoints_locked.json').read_text())['models']
    summary=json.loads((OUT/'summary.json').read_text());verify=json.loads((OUT/'verification.json').read_text())
    assert verify['status']=='verified'
    items=[]
    for arm,name in NAMES.items():
        r=metrics[metrics.model==arm+'_ensemble'].iloc[0]
        seeds=metrics[metrics.model.isin([arm+'_'+str(s) for s in [30101,30102,30103]])]
        n=next(x['parameters'] for x in locked if x['arm']==arm)
        items.append(dict(arm=arm,name=name,r=r.mean_transcript_pcc_log,lo=r.ci_low,hi=r.ci_high,
            raw=r.mean_transcript_pcc_raw_normalized,interior=r.mean_interior_pcc_log,
            seed_sd=seeds.mean_transcript_pcc_log.std(ddof=1),parameters=n))
    per=pd.read_csv(OUT/'per_transcript_metrics.csv');secondary=[]
    rng=np.random.default_rng(301030);boot=rng.integers(0,120,(2000,120))
    for metric in ['pearson_raw_normalized','pearson_interior_log']:
        wide=per.pivot(index='gene',columns='model',values=metric)
        for arm in list(NAMES)[1:]:
            delta=(wide.full_ensemble-wide[arm+'_ensemble']).to_numpy()
            lo,hi=np.nanpercentile(np.nanmean(delta[boot],axis=1),[2.5,97.5])
            secondary.append(dict(metric=metric,comparison='full minus '+arm,delta_mean_pcc=float(np.nanmean(delta)),ci_low=lo,ci_high=hi))
    pd.DataFrame(secondary).to_csv(OUT/'secondary_ablation_differences.csv',index=False)
    lines=['# 三路位置预测与消融结果','', '## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent；实现沿用 ponytail。',
        '- Date: 2026-10-30；Verification Status: VERIFIED。',
        '- 范围：公开实测小鼠肝脏 Ribo-seq 的归一化位置标签；本轮不生成或优化序列。','',
        f'完成 8 组 × 3 种子，共 {summary["n_frozen_models"]} 份模型，运行 {summary["seconds"]/60:.1f} 分钟。',
        '训练 400、独立验证 80、测试 120 个基因；与之前 300 基因小试排除基因与相同蛋白重叠。','',
        '## 测试结果','',
        '下表为预先固定的三种子预测均值。主指标在每条转录本的有效位置上，对预测和 log1p 归一化实测标签计算 Pearson，随后按转录本等权平均。',
        'raw 列使用相同位置，在原归一化标签尺度和逆变换预测上计算；不能将这两列混排。','',
        '| 模型 | 主指标 PCC | 95%转录本 bootstrap 区间 | raw-normalized PCC | 去端5密码子 PCC | 单种子 PCC 标准差 | 参数量 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for x in items:
        lines.append(f'| {x["name"]} | {x["r"]:.4f} | [{x["lo"]:.4f}, {x["hi"]:.4f}] | {x["raw"]:.4f} | {x["interior"]:.4f} | {x["seed_sd"]:.4f} | {x["parameters"]:,} |')
    lines+=['','辅助结果：','', '| 方法 | 平均逐转录本 log1p PCC |','|---|---:|']
    for key,name in [('gnn_aux_ensemble','完整模型内 GNN 独立辅助分布'),('codon_train_mean','仅训练集密码子均值基线'),('stasep_unpolished','未精修 sTASEP 分布')]:
        r=metrics[metrics.model==key].iloc[0];lines.append(f'| {name} | {r.mean_transcript_pcc_log:.4f} |')
    lines+=['','## 消融差异','', '完整模型减去各对照；区间使用相同转录本做配对重采样，未经多重比较校正。','',
        '| 对照 | ΔPCC | 95%配对区间 |','|---|---:|---|']
    for row in difference.itertuples(index=False):
        arm=row.comparison.removeprefix('full minus ')
        lines.append(f'| {NAMES[arm]} | {row.delta_mean_pcc:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] |')
    full=items[0];best=max(items,key=lambda x:x['r'])
    lines+=['',f'本轮点估计最高的是“{best["name"]}”（{best["r"]:.4f}）；完整三路为 {full["r"]:.4f}。',
        '是否存在稳定增益须结合配对区间与随机种子变化，不能仅凭最高点估计断言模块有效或达到 SOTA。',
        '当前设置下，GNN 的移除对主指标影响最大；sTASEP 增益较小。BiLSTM 上下文对照高于当前小型 Transformer，结果不支持直接用后者替代前者。',
        f'去掉首尾各5个密码子后，完整模型 PCC={full["interior"]:.4f}，去上下文组={items[3]["interior"]:.4f}；',
        '上下文支在全位置主指标上的增益不能直接解释为内部位置预测的提升。次要指标的配对区间另存 secondary_ablation_differences.csv。','',
        '## 已完成的对齐与验证','',
        '- 密码子先由 RiboGL 的 ATCG 编号恢复 DNA，再转换为 seq2ribo 的 ACGU 编号；64 个密码子及 600 条序列均完成往返核验。',
        '- 核苷酸配对边映射到密码子节点；模拟、结构、标签和输出逐位置等长，缺失、实测零、padding 分开处理。',
        '- 三路位置 embedding 独立形成后拼接；GNN 有接受真实位置监督的辅助分布头，Transformer 学习序列上下文。',
        '- 使用原发布 sTASEP 模拟器，训练集非负停留代理拟合，4 次模拟；没有使用人类细胞预训练常数。该代理不等同于动力学参数恢复。',
        '- 24 份权重在任何测试预测前按验证损失冻结并记录 SHA256。',
        f'- 从已保存预测独立重算全部 {verify["n_models_scored"]} 组统计，Pearson 最大差 {verify["max_independent_pearson_error"]:.2g}。',
        f'- 重新加载完整组一个种子的权重，对全部 120 条测试转录本再次推理，预测最大差 {verify["max_reloaded_prediction_error"]:.2g}。',
        '- 将所有验证/测试标签替换为 999 后重新拟合物理代理，参数保持一致，核验这些标签未进入物理拟合。','',
        '## 解释边界','',
        '这是单研究、短 CDS、覆盖充分样本上的探索性结果，没有近同源隔离或人类独立外部验证。',
        'sTASEP 参数使用训练集的简化代理拟合，模拟次数较少；参数量和辅助监督随分支移除变化，消融不能证明单独机制因果。',
        '置信区间是给定训练模型后的转录本重采样，单种子变化另外列出；区间没有覆盖全部训练不确定性。',
        '上游密码子标签坐标被保留，未独立重做原始 footprint 的 A/P-site offset。',
        '**当前源表不提供绝对 RPF 和配对 RNA 丰度，故没有绝对转录本总量 PCC、TE PCC 或蛋白产量验证。**',
        '本轮与 RiboGL 小试在逐转录本 log1p PCC 颗粒度上对齐，但采用新样本划分；不是 RiboGL 或 seq2ribo 原文结果复现。','',
        '## 文件','',
        '- `实验协议.md`：训练前固定的协议与指标口径。',
        '- `alignment_audit.csv`、`graph_simulation_metrics.csv`：逐样本对齐及图/模拟统计。',
        '- `metrics.csv`、`per_transcript_metrics.csv`：全部种子、集成和基线结果。',
        '- `paired_ablation_differences.csv`：完整模型与消融的配对差区间。',
        '- `test_predictions.npz`：逐密码子预测、观测标签、掩码和 GNN 辅助分布。',
        '- `verification.json`：独立重算及重新加载推理凭据。',
        '- `completed_snapshot/`：冻结样本、全部权重、特征、代码与日志。',
        '- `profile_ablation.png`：主指标、配对差和固定首条测试转录本的位置分布图。','',
        '源数据：[Riboclette dataset](https://doi.org/10.5281/zenodo.17100113)；',
        '原模拟器：[Kingsford-Group/seq2ribo](https://github.com/Kingsford-Group/seq2ribo)。']
    (OUT/'测试结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,5.3),layout='constrained')
    labels=[x['arm'].replace('_',' ') for x in items];values=[x['r'] for x in items]
    err=np.array([[x['r']-x['lo'] for x in items],[x['hi']-x['r'] for x in items]])
    axes[0].barh(labels,values,xerr=err,color=['#bb453c']+['#447ba5']*7)
    axes[0].invert_yaxis();axes[0].set_xlabel('Mean within-transcript Pearson (log1p)');axes[0].set_title('120 held-out genes; 3-seed ensemble')
    delta=difference.delta_mean_pcc.to_numpy();err=np.vstack([delta-difference.ci_low.to_numpy(),difference.ci_high.to_numpy()-delta])
    axes[1].errorbar(delta,np.arange(7),xerr=err,fmt='o',color='#447ba5',capsize=3)
    axes[1].set_yticks(np.arange(7),[s.removeprefix('full minus ').replace('_',' ') for s in difference.comparison])
    axes[1].axvline(0,color='gray',linewidth=1);axes[1].invert_yaxis();axes[1].set_title('Full minus ablation: paired 95% CI');axes[1].set_xlabel('Difference in mean Pearson')
    panel=json.loads((OUT/'panel.json').read_text());row=next(r for r in panel['records'] if r['split']=='test')
    with np.load(OUT/'test_predictions.npz',allow_pickle=False) as p:
        y=p['observed_log__0'].copy();y[~p['observed_mask__0']]=np.nan
        axes[2].plot(y,color='black',linewidth=1,label='Measured Ribo-seq')
        axes[2].plot(p['full_ensemble__0'],linewidth=1,label='Full model')
        axes[2].plot(p['gnn_aux_ensemble__0'],linewidth=1,alpha=.8,label='GNN auxiliary')
    axes[2].set_title(row['transcript']+'\nFirst fixed test record; no outcome selection')
    axes[2].set_xlabel('CDS codon index (zero based)');axes[2].set_ylabel('log1p normalized occupancy');axes[2].legend(fontsize=8)
    fig.savefig(OUT/'profile_ablation.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(full=full,best=best,report=str(OUT/'测试结果.md')),ensure_ascii=False))


if __name__=='__main__':main()
