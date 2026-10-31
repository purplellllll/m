"""All-arm report, paired ablations and residual physics-path diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
ROOT=BASE if (BASE/'metrics.csv').exists() else BASE.parent/'runs/graph_polish_20261031/gpu_result'
SEEDS=[30101,30102,30103]
LABELS={'parallel':'并行对照','physics_polish':'物理提前交互','gnn_polish':'GNN后Transformer',
    'physics_gnn_polish':'物理＋GNN后Transformer','shuffled_physics':'早期物理位置打乱',
    'self_graph':'GNN仅自身变换','sequence_edges':'GNN仅序列邻接边','diagonal_attention':'仅自身注意力',
    'ref_bilstm':'前轮BiLSTM参考','ref_context_x2':'前轮序列2倍参考','ref_transformer':'前轮Transformer参考',
    'early_zero':'完整模型推理时早期物理置零','early_shuffle':'完整模型推理时早期物理打乱'}
PAIRS={
    'graph_early':('gnn_polish','parallel','早期GNN交互'),
    'physics_early':('physics_polish','parallel','早期物理交互'),
    'both_early':('physics_gnn_polish','parallel','物理与GNN共同提前交互'),
    'physics_given_graph':('physics_gnn_polish','gnn_polish','已有GNN时早期物理收益'),
    'graph_given_physics':('physics_gnn_polish','physics_polish','已有物理时早期GNN收益'),
    'alignment':('physics_gnn_polish','shuffled_physics','早期物理位置对齐'),
    'neighbors':('physics_gnn_polish','self_graph','全部邻居聚合'),
    'structure_edges':('physics_gnn_polish','sequence_edges','结构配对邻居'),
    'global_attention':('physics_gnn_polish','diagonal_attention','跨位置全局注意力'),
    'physics_path_zero':('physics_gnn_polish','early_zero','冻结完整模型早期物理路径'),
    'physics_path_shuffle':('physics_gnn_polish','early_shuffle','冻结完整模型早期物理对齐'),
}


def contrast(frame,weights,metric):
    table=pd.concat([frame[frame.model==name].set_index('gene')[metric].rename(name) for name in weights],axis=1).dropna()
    delta=sum(table[name].to_numpy()*weight for name,weight in weights.items())
    boot=np.random.default_rng(94031).integers(0,len(delta),(2000,len(delta)))
    lo,hi=np.percentile(delta[boot].mean(1),[2.5,97.5])
    return dict(delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),n=len(delta))


def main():
    verification=json.loads((ROOT/'verification.json').read_text());audit=json.loads((ROOT/'metric_recheck.json').read_text())
    assert verification['status']==audit['status']=='passed'
    choice=json.loads((ROOT/'validation_choice.json').read_text());selected=choice['selected_arm']
    records=json.loads((ROOT/'all_checkpoints_locked.json').read_text())['models']
    p=pd.read_csv(ROOT/'per_transcript.csv');m=pd.read_csv(ROOT/'metrics.csv');effects=[]
    comparisons={k:(dict([(a,1),(b,-1)]),label) for k,(a,b,label) in PAIRS.items()}
    comparisons['interaction']=(dict(physics_gnn_polish=1,gnn_polish=-1,physics_polish=-1,parallel=1),'早期物理与GNN交互差')
    comparisons['selected_vs_bilstm']=({selected:1,'ref_bilstm':-1},'验证所选候选比旧BiLSTM参考')
    for split in ['fresh_test','supplementary_test','validation']:
        frame=p[p.split==split]
        for metric in ['pcc','interior_pcc','physics_pcc','gnn_pcc','context_pcc']:
            for key,(weights,label) in comparisons.items():
                result=contrast(frame,weights,metric)
                seeds=[contrast(frame,{f'{name}_{s}':w for name,w in weights.items()},metric)['delta'] for s in SEEDS]
                effects.append(dict(split=split,metric=metric,comparison=key,label=label,**result,
                    positive_seeds=sum(v>0 for v in seeds),**{f'seed_{s}':v for s,v in zip(SEEDS,seeds)}))
    effects=pd.DataFrame(effects);effects.to_csv(ROOT/'paired_effects.csv',index=False)
    fresh=m[m.split=='fresh_test'].set_index('model');supp=m[m.split=='supplementary_test'].set_index('model')
    heads=p[p.split=='fresh_test'].groupby('model')[['physics_pcc','gnn_pcc','context_pcc']].mean()
    lines=['# GNN邻居聚合与Transformer全局修正实验','', '## Material Passport','',
        '- Origin Skill: academic-research-suite/experiment-agent；实现遵循ponytail。',
        '- Date: 2026-10-31；Verification Status: VERIFIED（代表权重重载及独立指标复算）。',
        '- Runtime: gpt-6.1-sol；8组×3种子24新权重，另9份旧参考。','',
        f'测试前按三个种子验证Pearson均值、限定四个非消融架构，锁定候选 **{selected}**。以下报告全部组，不按新测试重新挑选。','',
        '## 测量结论','',
        '本轮支持保留GNN邻居聚合，尚未支持当前串联polish替代并行融合或BiLSTM参考。完整模型主PCC0.34026，并行对照0.34044，旧BiLSTM参考在同一新测试上为0.43270。旧参考配方和参数量不同，该差距不是严格的单因素架构效应。','',
        '去掉两类邻居后，完整模型主PCC从0.34026降至0.27004；保留邻居的配对差+0.07022，95%探索性区间[+0.06012,+0.08053]，三个种子方向一致，补充120亦一致。这是早期与末端GNN两条路径合计的邻居作用，不应称为提前交互单独的增益。','',
        'GNN提前进入Transformer使上下文辅助头从0.26518提高到0.30601，但主头由0.34044轻降至0.34011；三个种子的主头差值均为负。说明图信息确实能改善上下文分支的单独预测，尚未转化为主融合提升。主头已有直接GNN路径，信息重复或融合方式限制是后续待检验的解释，尚非已证实原因。','',
        '完整模型比仅自身注意力模型低0.00143，95%配对区间[-0.00520,+0.00255]；三个种子差值均负。当前数据与配方下，没有证实跨位置全局注意力的额外收益；这不等于一般性否定Transformer。','',
        '仅保留序列邻接边时PCC0.34505，比完整图高0.00478；三个种子均高，但新主测试配对区间仍跨0，补充面板同方向。结构配对边的质量和聚合方式值得优先核查，尚不能认定这些边本身普遍有害。此组是解释性消融，不按测试最高分重新替换预选候选。','',
        'sTASEP与GNN一起提前融合后主PCC0.34026；真实物理对齐比训练时打乱仅高0.000016，区间跨0；冻结完整模型早期物理置零的主头差仅0.000149，区间跨0。早期物理路径的主指标敏感性很弱，没有额外收益证据；它不代表末端物理总贡献为零。','',
        '下一轮若继续此路线，可优先验证序列邻接为主的图、可靠性门控的结构边，以及能将上下文辅助收益传到主头的融合方式；对全局注意力和物理交互的进一步调整应在训练/验证集上完成，再选另一批新测试。400训练样本和本轮统一配方仍限制结论外推。','',
        '## 全新120基因','', '| 模型 | 集成PCC | 95%基因区间 | 内部位置PCC | 单种子PCC均值±SD | 补充120集成PCC |','|---|---:|---|---:|---|---:|']
    for arm,label in LABELS.items():
        r=fresh.loc[arm];v=fresh.loc[[f'{arm}_{s}' for s in SEEDS]].mean_pcc
        lines.append(f'| {label} | {r.mean_pcc:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.mean_interior_pcc:.4f} | {v.mean():.4f}±{v.std(ddof=1):.4f} | {supp.loc[arm,"mean_pcc"]:.4f} |')
    lines+=['','## 主测试配对消融','', '| 比较（前项减后项） | ΔPCC | 95%探索性区间 | 三种子差值 |','|---|---:|---|---|']
    primary=effects[(effects.split=='fresh_test')&(effects.metric=='pcc')]
    for r in primary.itertuples():
        values=', '.join(f'{getattr(r,f"seed_{s}"):+.5f}' for s in SEEDS)
        lines.append(f'| {r.label} | {r.delta:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {values} |')
    lines+=['','正差只表示该比较前项在此面板更高。去除邻居或注意力的组均重新训练；冻结模型物理置零/打乱只改变早期物理路径，末端物理及GNN辅助头保持。推理时干预也可能引入分布偏移，不能仅凭下降认定物理机制正确。','',
        '## 辅助头及参数','', '| 模型 | 物理头PCC | GNN头PCC | 上下文头PCC | 参数量 | 三种子输入残差门 |','|---|---:|---:|---:|---:|---|']
    for arm,label in LABELS.items():
        rs=[r for r in records if r['arm']==arm]
        if not rs:continue
        h=heads.loc[arm];gates='、'.join(f'{r["input_gate"]:.4f}' for r in rs) if rs[0]['kind']=='polish' else '—'
        lines.append(f'| {label} | {h.physics_pcc:.4f} | {h.gnn_pcc:.4f} | {h.context_pcc:.4f} | {rs[0]["parameters"]:,} | {gates} |')
    lines+=['','辅助头不是其他分支屏蔽后的主头，也不是主性能的可加分解。输入残差门是投影残差的整体幅度，并非三路生物学贡献。','',
        '## 数据、训练和解释范围','',
        '400训练、80验证；新120按基因/CDS/完全相同蛋白排除过去1500记录；补充120为前轮主测试。未做近同源聚类，仍是同一GSE73553小鼠肝脏的80–250密码子筛选面板。',
        '每种子预测先按有效CDS总量归一化再集成；PCC是各转录本内预测与log1p标签的位置相关，再等权平均。内部位置去首尾各5密码子；报告集成PCC及单种子均值，避免混淆。',
        '八新组初始化和总参数相同；禁用输入槽对应列仍保留，所以名义参数不等于有效容量严格相同。所有新组用相同lr3e-4、2轮warmup、cosine、最多30轮及patience7，随机初始化，无预训练。',
        '前轮BiLSTM/Transformer/序列2倍的学习率、训练轮数、输入投影及参数量不同；与旧参考差异不能单独归因连接方式。主要结论以本轮parallel及匹配消融为准。',
        'z=E+sigmoid(a)·LN(GELU(W[E,P,G]))，门初值0.1，Transformer做全局修正；原P/G仍直接进入主融合。早期物理比较只检验末端物理存在时的附加收益，不代表移除所有物理信息。',
        '邻居包括双向序列边和预测结构配对边；self_graph保留自身线性变换及结构节点特征，sequence_edges只去配对边；diagonal_attention保留逐位置FFN与自身注意力。',
        'sTASEP数值引擎不可微，停留代理参数只由原400训练样本拟合；不声称真实速率恢复、TE、总表达、蛋白产量或通用SOTA。',
        '固定长度相关随机置乱逐样本只排列有效物理位置；它检验空间对齐信息，不是生物机制因果随机化。',
        '架构参考：[GraphGPS原论文](https://arxiv.org/abs/2205.12454)；本实现采用局部聚合与全局注意力的组合思路，没有复现其完整模型或基准成绩。',
        '每个比较2000次配对基因bootstrap，均为探索性未校正区间；全部比较和种子报告。三个种子不覆盖全部训练不确定性，验证集反复开发，测试不是独立外部研究。','',
        '## 核验','',
        f'服务器核验{verification["n_records"]}输入和{verification["n_total_weights"]}权重，独立公式复算{verification["checked_metric_values"]}个值；代表权重重载最大差{max(verification["reload_max_errors"].values()):.3g}。',
        f'本地重新读保存的预测，独立中心化点积复算{audit["checked_metric_values"]}指标、全部集成和单位总量；最大误差{audit["max_error"]:.3g}；排除1500记录的主测试独立核对通过。',
        '11/11统计风险核查：长度/组成分层未控（聚合反转仍可能）；位置指标不外推总表达；质量筛选限制外推；未以碰撞变量作因果调整；非分类任务；',
        '未按极端预测选样；全部模型均报告；多重比较未校正；协议锁定但验证集重复开发；相关不作生物因果；不作反向因果判断。']
    (ROOT/'GNN_Transformer_polish实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=[a for a in LABELS if not a.startswith('early_')]
    fig,axes=plt.subplots(1,2,figsize=(16,8),layout='constrained')
    r=fresh.loc[arms];v=r.mean_pcc.to_numpy()
    axes[0].barh(arms,v,xerr=np.vstack([v-r.ci_low.to_numpy(),r.ci_high.to_numpy()-v]),
        color=['#b54e42' if a==selected else '#397ba7' for a in arms]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean within-transcript PCC (log1p)');axes[0].set_title('Fresh120; red selected before test')
    r=primary[~primary.comparison.isin(['interaction','selected_vs_bilstm'])];v=r.delta.to_numpy()
    axes[1].errorbar(v,np.arange(len(r)),xerr=np.vstack([v-r.ci_low.to_numpy(),r.ci_high.to_numpy()-v]),fmt='o',capsize=3)
    axes[1].set_yticks(range(len(r)),r.comparison);axes[1].invert_yaxis();axes[1].axvline(0,color='gray')
    axes[1].set_title('Paired effects; exploratory unadjusted95% CI')
    fig.savefig(ROOT/'graph_polish_results.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='reported',selected=selected,effects=len(effects))))


if __name__=='__main__':main()
