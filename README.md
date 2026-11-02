# RiboSeq scorer — human TE results and model handoff

2026-11-02研究归档。分支 **`results/human-te-scorer-20261102`** 从评分器快照分支派生；不覆盖`main`、`V3`、`V4`、`vmae-v2`或2026-10-31快照，不自动恢复训练。

**人源HEK293 CDS-only TE：7373训练／922验证／903测试。下表全部由同903条样本的本地实际预测计算，没有使用论文相关系数。**

| 模型 | 留出Pearson | 95%基因bootstrap区间 |
|---|---:|---|
| 自有两种子人源TE模型 | **0.5921** | [0.5457, 0.6329] |
| seq2ribo历史作者权重 | 0.5568 | [0.5060, 0.6031] |
| RiboDecode固定RNA代理 | 0.6275 | [0.5608, 0.6850] |

我们减seq2ribo：Δr=+0.0353，配对95%区间[-0.0033,+0.0742]；两项参照的配对区间均跨零，**没有证明稳定超越或SOTA**。我们的训练/验证与测试在基因、精确CDS和精确蛋白上隔离，尚未做近同源聚类。作者seq2ribo权重训练成员身份未认证；RiboDecode此项是固定RNA/环境分数与TE的代理相关，不是其原生RPF准确率。

## 模型与验证

验证选定`no_stats`：两层width192、方向性±1/±2/±4图邻接、结构边门控、sTASEP物理输入先融合至Transformer，context mean/max读出人源标量。无显式长度、GC/MFE/CAI或模拟全局统计输入。GC/MFE/CAI仍独立报告并可作为搜索奖励。物理参数仍为鼠肝代理拟合，不声称已经人源动力学校准。

**重要评估修正：**初版测试脚本漏载训练标量头，初版自有预测无效。这里公布的是载入原验证选定的两个完整权重后重算的结果；没有重训、换模型或按测试调参。验证批次重载误差≤2.4e-7，903行及三个Pearson独立复核通过。[修正记录](evidence/20261102/human_te_expanded_20261101/评估修正说明.md)保留错误和原生进程退出异常，不隐藏失败。

## 下载和使用

- [本次独立模型Release](https://github.com/purplellllll/m/releases/tag/human-te-scorer-20261102)
- [60份权重清单与SHA256](models/20261102/manifest.json)：4份扩大人源模型、12份人源pilot、40份历史mRFP读出、2份图骨干、2份VMAE生成器快照。只有清单`selected_paths`中的两个种子构成人源主模型；其他是研究对照。
- 不包含优化器/RNG、外部模型权重、原始序列数据集或第三方模拟源码。40份mRFP探针中，sequence_graph骨干依赖[原评分器Release](https://github.com/purplellllll/m/releases/tag/riboseq-scorer-snapshot-20261031)。

```bash
gh release download human-te-scorer-20261102 --repo purplellllll/m --pattern human-te-own-models-20261102.tar.gz
tar -xzf human-te-own-models-20261102.tar.gz
python scripts/verify_release_20261102.py --models-root models_20261102
python scripts/predict_human_te.py --models-root models_20261102 --features your_aligned_CDS_features.npz
```

推理默认CPU，自动选两个主模型并平均原标签尺度输出。NPZ需要`cod:[L]`、`physics:[L,2]`、`structure:[L,5]`、`edges:[2,E]`，其中密码子ID采用上游seq2ribo顺序；不得混用VMAE tokenizer或用零物理冒充原输入。CDS折叠/模拟必须采用记录的结构、参数和种子规则；完整第三方依赖需从原项目取得，见[输入合同](docs/HUMAN_TE_INFERENCE.md)与[第三方来源](third_party/README.md)。

`load_exported_checkpoint`采用`weights_only=True`、SHA校验和完整`strict=True`载入，包含训练后的标量头。`HumanTEScalar`直接输出训练目标尺度；历史`BranchProbe`输出仍需用其metadata中的`y_scale/y_mean`还原。

## 其他实验结果

| 实验 | 结果与边界 |
|---|---|
| 人源小pilot，256/96 | 两层集成r0.4379，四层0.4239；仅验证开发，不支持加深直接改善 |
| mRFP读出，217历史测试 | gated_graph+C r0.7521，三路拼接0.7234；C已混合物理/图，不能称纯Transformer优势 |
| 稀疏推理，219 mRFP验证 | 全注意力r0.7251；滑窗16/32/64/128均较低；未重新训练稀疏模型 |
| 长序列合成推理 | 8192密码子滑窗可运行；FP32 math后端下约0.070秒/86.6MiB，非训练显存或长CDS准确率证据 |
| λ扫描，2模型×5档×6条×20步 | 全部60组完成并补齐两外部评分；旧mRFP头的人源域外搜索，不是扩大人源头的结果 |
| 人源pilot直接搜索，3条×20步 | 自有分数上升，但seq2ribo均下降；不能认定表达改善 |
| 旧V3/V4完整30条 | 两轮搜索和独立评估完成；保留性质奖励、替换评分项未显示全面外部改善 |

- [结果总览与下一轮交接](docs/HANDOFF_20261102_ZH.md)
- [903条逐样本预测CSV](evidence/20261102/human_te_expanded_20261101/test_predictions.csv)
- [独立测试摘要与配对区间](evidence/20261102/human_te_expanded_20261101/test_summary.json)
- [λ完整外部评分表](evidence/20261102/sparse_lambda_20261101/lambda完整外部评分对比.md)
- [直接搜索对照](evidence/20261102/direct_vs_vmae_human_20261101/直接搜索对比.md)
- [旧V3/V4完整对照](evidence/20261102/v3_v4_own_multi_cpu_20261101/完整结果.md)

`historical/20261102/`是原实验源码溯源，保留工作区路径，不承诺fresh clone一键训练。其原`evaluate.py`包含已归档的漏载头错误；正确复算需使用同目录`recovery_20261102/correct_head_loading.py`及完整依赖。日常推理使用上面的新入口，不直接运行历史控制器。

原`docs/HANDOFF_ZH.md`和旧证据保留2026-10-31快照身份，当前状态以本README和新交接为准。LIVER旧696留出中120已在历史固定比较打开，余576未预测；本次没有再读取该测试。
