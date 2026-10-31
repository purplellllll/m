"""Compare gain retraining and context replacements on the frozen fresh panel."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from report_gnn_physics_recipe_20261030 import paired

ROOT=Path(__file__).resolve().parents[1]/'runs/fusion_encoders_20261031/gpu_result'
SEEDS=[30101,30102,30103]
LABELS={'bilstm_baseline':'BiLSTM基线','physics_x2':'物理分支2倍重训','gnn_x2':'GNN分支2倍重训','context_x2':'序列分支2倍重训',
    'learned_fusion':'可学习三路倍率','bilstm_ln':'BiLSTM＋LayerNorm','transformer':'Transformer','bert':'小型BERT（随机）','t5':'小型T5Encoder（随机）','frozen_bilstm':'旧冻结BiLSTM参考'}


def main():
    verification=json.loads((ROOT/'verification.json').read_text());assert verification['status']=='passed'
    metric_check=json.loads((ROOT/'metric_recheck.json').read_text());assert metric_check['status']=='passed'
    choice=json.loads((ROOT/'validation_choice.json').read_text());selected=choice['selected_arm']
    m=pd.read_csv(ROOT/'metrics.csv');p=pd.read_csv(ROOT/'per_transcript.csv').assign(mode='normal')
    records=json.loads((ROOT/'all_checkpoints_locked.json').read_text())['models'];effects=[]
    pairs=[(a,'bilstm_baseline') for a in LABELS if a not in ['bilstm_baseline','frozen_bilstm']]+[(selected,'frozen_bilstm')]
    for split in ['fresh_test','supplementary_test','validation']:
        frame=p[p.split==split]
        for metric in ['pcc','interior_pcc','physics_pcc','gnn_pcc','context_pcc']:
            for a,b in pairs:
                v=paired(frame,a,b,metric=metric)
                seeds=[paired(frame,f'{a}_{s}',f'{b}_{s}',metric=metric)['delta'] for s in SEEDS]
                effects.append(dict(split=split,metric=metric,model=a,reference=b,**v,positive_seeds=int(sum(x>0 for x in seeds)),
                    **{f'seed_{s}':v for s,v in zip(SEEDS,seeds)}))
    effects=pd.DataFrame(effects);effects.to_csv(ROOT/'paired_effects.csv',index=False)
    table=m[m.split=='fresh_test'].set_index('model');old=m[m.split=='supplementary_test'].set_index('model')
    primary=effects[(effects.split=='fresh_test')&(effects.metric=='pcc')]
    lines=['# 三路倍率重训与上下文替换：新服务器实验','', '## Material Passport','',
        '- Origin Skill: academic-research-suite/experiment-agent；代码沿用ponytail。',
        '- Date: 2026-10-31；Verification Status: VERIFIED。',
        '- A800；27份新权重＋3份旧冻结参考；本轮只作小规模架构与优化诊断。','',
        f'测试前按三种子验证PCC均值选定候选 **{selected}**；以下全组报告，不按新测试最高分替换候选。','',
        '## 本轮判断','',
        '优先保留BiLSTM。若下一轮只探索一个放大方向，应选序列上下文分支；当前没有充分证据把物理或GNN分支调大。',
        '序列2倍重训的新120基因PCC为0.42538，基线为0.42431；配对差+0.00107，95%探索性区间[-0.00083,+0.00297]，三个种子中两个改善。旧120补充面板为0.42059对0.41684，三个种子均改善。它是测试前验证集选出的候选，但新主测试尚不能确认可靠增益。',
        '可学习倍率在三个种子上均轻微压低物理/GNN、抬高序列，平均约[0.9722,0.9506,1.0772]；新测试主融合PCC却降至0.42346。这说明系数方向只能提供优化线索，不能替代留出性能，也不支持直接使用大倍率。',
        'Transformer/BERT/T5的主测试PCC为0.34576/0.32667/0.31110，三个种子及补充面板均下降；配对区间全部低于0。BiLSTM加输出LayerNorm也未改善新测试，因此不能只用末端尺度解释替换模型的差距。',
        '辅助头显示差距主要伴随上下文预测能力下降：BiLSTM上下文头0.41484，Transformer/BERT/T5为0.26567/0.21947/0.17953；这些模型的GNN头反而从0.27486升到0.28425/0.29136/0.29865。它支持后续优先检查上下文训练，但共同训练会耦合分支，不能据此认定唯一因果原因。',
        '下一轮建议先保留BiLSTM，仅在训练/验证集搜索序列小倍率与正则；若继续注意力编码器，应单独验证学习率、warmup和局部位置偏置，再另取新留出集。当前结果没有测试预训练BERT/T5，不能据此否定预训练路线。','',
        '## 全新120基因','', '| 模型 | 集成PCC | 95%基因区间 | 内部位置PCC | 单种子PCC均值±SD |','|---|---:|---|---:|---|']
    for arm,label in LABELS.items():
        r=table.loc[arm];v=table.loc[[f'{arm}_{s}' for s in SEEDS]].mean_pcc
        lines.append(f'| {label} | {r.mean_pcc:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.mean_interior_pcc:.4f} | {v.mean():.4f}±{v.std(ddof=1):.4f} |')
    lines+=['','## 各分支辅助头：新120基因','', '| 模型 | 物理头PCC | GNN头PCC | 上下文头PCC |','|---|---:|---:|---:|']
    heads=p[p.split=='fresh_test'].groupby('model')[['physics_pcc','gnn_pcc','context_pcc']].mean()
    for arm,label in LABELS.items():
        r=heads.loc[arm];lines.append(f'| {label} | {r.physics_pcc:.4f} | {r.gnn_pcc:.4f} | {r.context_pcc:.4f} |')
    lines+=['','辅助头为各分支的并行监督输出，不是将其他分支屏蔽后的主头输出，也不是分支贡献的可加分解。','',
        '## 对BiLSTM基线的配对差','', '| 改动 | ΔPCC | 95%探索性配对区间 | 三种子差值 |','|---|---:|---|---|']
    for r in primary.itertuples():
        deltas=', '.join(f'{getattr(r,f"seed_{s}"):+.5f}' for s in SEEDS)
        lines.append(f'| {LABELS[r.model]}减{LABELS[r.reference]} | {r.delta:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] | {deltas} |')
    lines+=['','## 训练出的倍率与参数','', '| 模型 | 三种子倍率[物理,GNN,序列] | 登记总参数 |','|---|---|---:|']
    for arm,label in LABELS.items():
        if arm=='frozen_bilstm':continue
        rs=[r for r in records if r['arm']==arm]
        vals='；'.join('['+', '.join(f'{g:.4f}' for g in r['gains'])+']' for r in rs)
        lines.append(f'| {label} | {vals} | {rs[0]["parameters"]:,} |')
    lines+=['',
        '固定倍率只作用于主融合输入，分支辅助头保持不变。3×softmax全局倍率仍可与融合第一层权重相互补偿；不能直接把倍率数值当作唯一重要性。',
        '判断是否应放大某支，应同时看新面板配对增益、三个种子方向和补充面板，而不是只看学到的系数或冻结模型屏蔽效应。','',
        '## 上一轮120基因补充面板','', '| 模型 | PCC |','|---|---:|']
    for arm,label in LABELS.items():lines.append(f'| {label} | {old.loc[arm,"mean_pcc"]:.4f} |')
    lines+=['','## 数据与解释范围','',
        '400训练/80验证，主测试新120排除以前1380条记录的基因/CDS/完全相同蛋白。没有近同源聚类；仍是同一小鼠肝脏GSE73553公开归一化Ribo-seq短CDS面板。',
        '新三路全部直接末端融合，未关闭物理槽，也未额外向GNN注入物理；这样隔离倍率与上下文替换。不要将本轮点估计与前轮不同面板或不同集成口径直接比较。',
        '每种子预测先按完整有效CDS总量归一化再等权集成。PCC为转录本内log1p标签位置相关，再按转录本等权平均；内部位置去首尾5密码子。',
        '三种子集成Pearson不是三个Pearson的平均；每种子数据另外列出，避免把多样性收益当作每个模型均改善。',
        'BERT/T5均为1层8头、随机初始化、密码子级小模型；没有加载大型或序列预训练权重。结果不能代表预训练模型的最优表现。',
        '参数近似匹配不代表计算量、梯度尺度和学习难度严格相等；同一lr0.001的初筛也不代表每个架构完成独立超参数寻优。BiLSTM+LayerNorm是输出尺度对照。',
        'BERT/T5接口依据：[BERT](https://huggingface.co/docs/transformers/model_doc/bert)、[T5Encoder](https://huggingface.co/docs/transformers/model_doc/t5)。',
        '2000次配对基因bootstrap未作多重比较校正，属于探索性结果；三个种子不覆盖全部训练不确定性；验证集已反复用于开发。不能据点估计宣称通用SOTA、TE或蛋白产量改善。','',
        '## 核验','',
        f'核验{verification["n_records"]}输入、{verification["n_total_weights"]}份权重；服务器保存预测复算主PCC最大误差{verification["max_pearson_error"]:.3g}。',
        f'另用中心化点积公式独立重算全部主/内部/三分支头指标，覆盖{metric_check["checked_metric_values"]}个值，最大误差{metric_check["max_error"]:.3g}。',
        f'五类代表模型在全新120基因重载最大预测差：{verification["reload_max_errors"]}。',
        '源码、输入与权重锁定记录保留；用户新服务器凭证没有写入脚本或日志。','',
        '统计风险11/11核查：未单独控制长度/组成分层；位置相关不外推表达总量；质量筛选限制外推；不作碰撞变量因果调整；连续回归不使用分类基准率；',
        '未按极端模型预测选样；全组全种子报告；多重比较未校正；候选在测试前锁定；相关不作生物学因果；不识别反向因果。']
    (ROOT/'新服务器实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained');arms=list(LABELS)
    r=table.loc[arms];v=r.mean_pcc.to_numpy()
    axes[0].barh(arms,v,xerr=np.vstack([v-r.ci_low.to_numpy(),r.ci_high.to_numpy()-v]),color=['#b54e42' if a==selected else '#397ba7' for a in arms]);axes[0].invert_yaxis()
    axes[0].set_title('Fresh120: red selected before test');axes[0].set_xlabel('Mean within-transcript PCC (log1p)')
    r=primary[primary.reference=='bilstm_baseline'];v=r.delta.to_numpy()
    axes[1].errorbar(v,np.arange(len(r)),xerr=np.vstack([v-r.ci_low.to_numpy(),r.ci_high.to_numpy()-v]),fmt='o',capsize=3)
    axes[1].set_yticks(range(len(r)),r.model);axes[1].invert_yaxis();axes[1].axvline(0,color='gray');axes[1].set_title('Minus baseline: paired unadjusted95% CI')
    fig.savefig(ROOT/'fusion_encoder_results.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='reported',selected=selected,effects=len(effects))))


if __name__=='__main__':main()
