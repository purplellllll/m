# RiboSeq Scorer: sTASEP + GNN + Transformer

独立研究分支 `riboseq-gnn-transformer-20261031`，与本仓库VMAE生成模型分开。

**状态：按用户要求停止实验，先补机制和评测协议证据。没有证明下游任务SOTA。上传是保存与交接，不授权恢复训练。**

- [完整交接文档](docs/HANDOFF_ZH.md)：已经做了什么、停止时状态、证据缺口、未来路线和资产位置。
- [模型权重清单](models/manifest.json)：26份自有tensor-only权重及各自配置、SHA256、完成状态。
- [独立模型Release](https://github.com/purplellllll/m/releases/tag/riboseq-scorer-snapshot-20261031)：权重附件与导出核验回执。
- [证据缺口](evidence/scalar_tasks/停止记录与证据缺口.md)；[任务颗粒度核验](evidence/scorer_breakdown/评分器评测集拆分与超越核验.md)。

## 已有证据

| 实验 | 已有结果 | 结论边界 |
|---|---|---|
| 新LIVER方向性序列邻接 | 验证Pearson 0.4021→0.5173 | 相同696验证样本，两个配对种子；不是最终测试 |
| 再加结构门控 | 0.5173→0.5278 | 有初步正信号，第三种子未完成 |
| 再加上下文锚定 | 0.5278→0.5232 | 当前没有改善 |
| mRFP蛋白标量 | 219测试集成r=0.8510；去重217同为0.8510 | 同集seq2ribo原权重对照未完成；完整原划分尚未认证 |
| 人类TE/RPF标量 | 数据与模型接口已整理 | 尚无本模型完整任务测试结果 |

新v2使用6956条公开相关研究LIVER记录，5564/696/696按基因及精确CDS/蛋白去重划分；不是原RiboGL论文划分。最终696测试未打开。当前蛋白标量来自旧width424编码器，不能冒称新width192门控模型的下游成绩。

## 模型结构

三路信息分别嵌入：sTASEP模拟统计、序列/结构邻接GNN、上下文Transformer。新模型提供有方向±1/±2/±4邻接、结构边门控和上下文主头选项。标量头使用三路屏蔽均值/最大池化与非线性读出；RPF另有真实RNA和环境输入，TE另有UTR输入选项。

位置Pearson、跨转录本RPF/TE Pearson与蛋白表达相关分别报告。归一化位置预测求和不作为TE标签。门控数值不视为校准后的真实结构概率。

## 代码与权重核验

只做模型数值检查，不启动训练或生物数据评测：

```bash
python -m physics_scorer.test_reliable_graph_context
python -m physics_scorer.test_graph_polish_model
python scripts/test_scalar_heads.py
```

下载权重并按清单检查（Release附件约221MB）：

```bash
gh release download riboseq-scorer-snapshot-20261031 --repo purplellllll/m --pattern riboseq-scorer-weights-20261031.tar.gz
tar -xzf riboseq-scorer-weights-20261031.tar.gz
python scripts/verify_exported_models.py --models-root models
```

26份权重只包含本项目自有模型：13份v2最佳（其中1份未完成）及1份未完成trial的latest、6份蛋白标量、3份旧polish和3份旧BiLSTM参考。它们不是26个独立最终候选，也没有据测试成绩选出一个“最终SOTA模型”。原训练优化器/RNG留在服务器，Release导出不含这些状态。

`physics_scorer/checkpoint_io.py`提供`load_exported_checkpoint`，默认CPU、`weights_only=True`、严格载入，支持先验证SHA256。蛋白模型原始输出是训练标准化空间的标量，需要使用权重metadata中的`target_scale/target_mean`还原原标签尺度。

## 目录

- `physics_scorer/`：独立模型及数值检查，不含VMAE生成器。
- `scripts/`：可独立运行的检查入口。
- `historical/drivers/`：原实验驱动与准备/训练/报告源码。保留原绝对路径，**不承诺新机器上一键运行**；不要直接执行setup或控制器。
- `evidence/`：冻结记录、验证日志、数据/模型哈希及停止收据。大规模原始序列、测量矩阵和全部特征缓存未放Git。
- `docs/HANDOFF_ZH.md`：下一轮对话入口。
- `third_party/`：依赖来源及本项目的参考后端装配胶水；不重新分发第三方网络源码或权重。

## 环境与第三方依赖

模型核心使用PyTorch和NumPy；实验使用pandas/scipy/ViennaRNA/SwanLab等。完整原环境及基线隔离环境版本在交接文档中。BERT/T5编码器实验是随机初始化的小模型，需可选Transformers依赖。

seq2ribo有非商业研究许可和再分发限制；外部网络/模拟源码、权重和数据应从其原始项目依许可获取，见[第三方说明](third_party/README.md)。这里不对其重新授予许可；本分支也未替用户选择新的通用软件许可证。

原项目：[seq2ribo](https://github.com/Kingsford-Group/seq2ribo)、[RiboDecode](https://github.com/wangfanfff/RiboDecode)。实验背景：[seq2ribo论文](https://doi.org/10.1093/bioinformatics/btag296)、[RiboDecode论文](https://doi.org/10.1038/s41467-025-64894-x)。
