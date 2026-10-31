"""Report only measured validation sensitivity; no unrun GPU score claims."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'runs/fusion_encoders_20261031'


def main():
    coarse=pd.read_csv(ROOT/'gain_scan_summary.csv');fine=pd.read_csv(ROOT/'gain_refinement/gain_scan_summary.csv')
    selected=json.loads((ROOT/'gain_refinement/gain_selection.json').read_text())
    preflight=json.loads((ROOT/'encoder_preflight.json').read_text())
    assert selected['n_test_evaluated']==0 and not preflight['gpu_training_started']
    base=float(coarse[coarse.candidate=='baseline'].mean_validation_pcc.iloc[0])
    combined=pd.concat([coarse.assign(stage='coarse'),fine.assign(stage='fine')],ignore_index=True)
    combined.to_csv(ROOT/'combined_gain_scan_summary.csv',index=False)
    names={'physics':'物理','gnn':'GNN','context':'BiLSTM序列'}
    lines=['# 三路放大与编码器替换：已完成的验证及待运行实验','',
        '## Material Passport','',
        '- Origin Skill: academic-research-suite/experiment-agent；代码沿用ponytail。',
        '- Date: 2026-10-31；Status: VALIDATION_SCAN_COMPLETED / GPU_TRAINING_BLOCKED。',
        '- SSH 183.147.142.40:31234多次拒绝连接；当前没有启动新的服务器训练。','',
        '## 当前能回答什么','',
        '**现有冻结模型不支持大幅放大任何一路。小幅调整只有很弱的验证信号，尚不能定为最终权重。编码器已完成结构适配与CPU单元检查，没有性能排名。**','',
        '这里使用原始三路直接融合的width424/native-loss/aux0.1权重，三个种子30101/2/3；不是上一轮关闭末端物理槽的模型。',
        '只读取80个验证基因，指标为每转录本log1p归一化位置Pearson，再按基因/种子等权平均；没有计算本轮新测试成绩。',
        '融合前分别将物理、GNN或序列向量乘以系数，其余两路保持1。编码器、辅助头及所有权重冻结。','',
        '## 直接放大2倍','', '| 改动 | 验证PCC | 相对1倍的差 | 95%探索性配对区间 | 正差种子数 |','|---|---:|---:|---|---:|',
        f'| 三路均1倍 | {base:.6f} | 0 | — | — |']
    for b,label in names.items():
        r=coarse[coarse.candidate==f'{b}_2'].iloc[0]
        lines.append(f'| 仅{label}2倍 | {r.mean_validation_pcc:.6f} | {r.delta_from_unit:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | {r.positive_seeds}/3 |')
    lines+=['','## 接近1倍的细化扫描','',
        '第一轮网格0/0.5/1/2/4/8没有发现放大收益；随后只根据验证集细化0.75/0.9/1/1.1/1.25/1.5。两阶段源码与结果分别保留，未覆盖初始记录。',
        '以下为每次单独调整一支的点估计最优值，不能直接合并成三路联合最优组合。','',
        '| 单独调整分支 | 点估计最优倍率 | 验证PCC | ΔPCC | 95%探索性配对区间 | 正差种子数 |','|---|---:|---:|---:|---|---:|']
    for b,label in names.items():
        choice=selected['branch_best'][b];r=fine[fine.candidate==choice['candidate']].iloc[0]
        lines.append(f'| {label} | {choice["gain"]:g} | {r.mean_validation_pcc:.6f} | {r.delta_from_unit:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | {r.positive_seeds}/3 |')
    lines+=['',
        '三个最优倍率的区间都跨0；物理/序列略增、GNN略减仅是验证候选，不能证明应该这样调整。',
        '两个阶段有多次比较，而且验证集已被前轮开发反复使用；所列2000次配对基因bootstrap区间未校正，不覆盖全部训练随机性。','',
        '## 屏蔽分支与重要性的区别','', '| 屏蔽一支 | 验证PCC | 相对基线差 |','|---|---:|---:|']
    for b,label in names.items():
        r=coarse[coarse.candidate==f'{b}_0'].iloc[0]
        lines.append(f'| {label}置零 | {r.mean_validation_pcc:.6f} | {r.delta_from_unit:+.6f} |')
    lines+=['',
        '冻结模型最依赖序列分支，但依赖程度大不等于应该继续放大。屏蔽也可能造成分布外输入，不能代替重新训练的消融。',
        '对α>0，W·h=(W/α)·(αh)，固定倍率可以被融合第一层吸收。但优化器、正则化和梯度尺度会使训练过程不同，因此仍要重训验证；倍率本身不是唯一、可直接解释的生物学重要性。',
        'branch_embedding_rms.csv同时记录输入RMS、投影RMS和权重范数；三者均受表征尺度影响，不单独作重要性排名。','',
        '## BiLSTM替代接口','',
        '五类模块均通过CPU小例的前向/反向、梯度、padding和样本隔离检查；424维模型已实例化并核验总参数量。尚未在目标GPU执行batch8预检或完整训练。',
        'BERT和T5使用Hugging Face正式模型类、共享64密码子加PAD的嵌入，通过inputs_embeds接入；BERT保留学习型绝对位置，T5只用编码器及相对位置机制，不再叠加正弦位置。',
        '接口依据：[BERT文档](https://huggingface.co/docs/transformers/model_doc/bert)、[T5 Encoder文档](https://huggingface.co/docs/transformers/model_doc/t5)。',
        '**本轮准备的是小规模、随机初始化的架构对照，没有加载BERT/T5预训练权重。结果将不能代表大规模预训练模型的最优能力。**','',
        '| 上下文模块 | 总登记参数量 | 前馈宽度 |','|---|---:|---:|']
    for r in preflight['measurements']:lines.append(f'| {r["encoder"]} | {r["parameters"]:,} | {r["ff"] if r["ff"] is not None else "—"} |')
    lines+=['',
        '各组宽度424；注意力模型1层8头，总登记参数差小于0.1%，不等于实际计算量或学习难度完全相同。BiLSTM+LayerNorm用于检验输出归一化本身的影响。',
        'CPU环境Torch2.14.0+cpu、Transformers5.7.0；目标GPU环境尚未核验版本兼容性。','',
        '## 已准备的后续训练','',
        '九组各三种子：原始BiLSTM；物理2倍；GNN2倍；序列2倍；可学习全局三路倍率；BiLSTM+LayerNorm；Transformer；BERT；T5Encoder。',
        '固定倍率只作用于主融合输入，不改变分支辅助监督。可学习倍率为3×softmax(logits)，初始各1；仍不能将其数值直接当作信息重要性。',
        '采用同一400训练/80验证、原生目标和aux0.1、AdamW lr0.001 wd0.01 clip1、batch8、最多18轮/耐心4、主native验证loss选权重。',
        '训练入口tools/train_fusion_encoders_20261031.py已编译检查，尚未进行实际数据整轮训练。冻结新留出面板、目标GPU依赖预检和最终留出比较尚待服务器恢复后完成。','',
        '## 核验与限制','',
        f'三个冻结检查点及80份验证特征均核对SHA；缓存投影与完整融合前向最大绝对预测差{selected["cached_projection_max_abs_error"]:.3g}。',
        'sTASEP仍为训练集拟合代理的四次模拟均值/波动；以上只证明既有模型的验证敏感性，不是SOTA、独立测试、TE或蛋白产量结论。',
        '统计风险11/11核查：未分层控制长度/组成；不外推总体表达；质量筛选限制适用范围；不作因果碰撞变量调整；连续回归不使用分类基准率；',
        '未按极端预测选样；全候选全种子报告；多重比较未校正；细化只用验证且尚无本轮测试；相关不作生物学因果；不识别反向因果。','',
        '## 文件','',
        '- gain_scan_summary.csv、gain_scan_per_transcript.csv：第一阶段完整扫描。',
        '- gain_refinement/：接近1倍的完整细化扫描与选择记录。',
        '- combined_gain_scan_summary.csv：两阶段合并结果，保留stage列。',
        '- branch_embedding_rms.csv、gain_local_derivatives.csv：表征尺度与局部斜率。',
        '- encoder_preflight.json：模型参数量及CPU测试范围。']
    (ROOT/'三路权重与编码器替换_阶段报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='reported',n_test_evaluated=0,gpu_training_started=False,preferred_single_branch_gains=selected['preferred_gains'])))


if __name__=='__main__':main()
