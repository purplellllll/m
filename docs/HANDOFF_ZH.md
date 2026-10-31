# RiboSeq专属评分器：sTASEP、GNN、Transformer与标量下游任务交接

更新日期：2026-10-31。本文是这条研究线下一轮对话的入口，优先于旧VMAE交接文件中的运行状态。**最新用户指令是停止实验，先补证据；上传GitHub及整理文档不代表恢复训练授权。**

## 1. 下一轮先读、目标与当前边界

用户目标：建立自己的Ribo-seq评分模型，以sTASEP物理信息、GNN局部/结构邻接和Transformer上下文形成表征；在与RiboDecode、seq2ribo匹配的真实标量下游任务上取得可靠优势，最终再考虑引导原VMAE潜空间搜索，并兼顾GC、MFE、CAI等性质。

目前完成了模型与任务接口、若干位置任务消融和一轮蛋白标量训练。**没有证明全部下游任务SOTA，也没有把新评分器接回VMAE做新的潜空间优化。** 原VMAE在其他评分器上得到高分与新评分器对实测标签预测准确，是两类不同证据。

下一轮建议开场：

> 请读取工作区根目录《交接文档_RiboSeq评分器_GNN_Transformer_2026-10-31.md》。这条线与VMAE分开。所有训练及评测已按用户要求停止，先不要启动服务器任务。先核对协议与已有证据，优先设计能证明方向性序列邻接、结构门控及sTASEP独立收益的小规模开发实验；原始划分、物理跨域和标量读出未对齐前，不恢复全量验证，不宣布SOTA。

先读文件：

1. `runs/scalar_downstream_20261031/停止记录与证据缺口.md`。
2. `runs/scalar_downstream_20261031/stopped_evidence_snapshot.json`：停止时12组完成的profile权重和6组标量权重元信息。
3. `runs/scalar_downstream_20261031/标量任务首轮实测与后续队列.md`，末尾的用户停止指令覆盖前文后台继续说明。
4. `runs/scorer_benchmark_breakdown_20261031/评分器评测集拆分与超越核验.md`：旧位置面板与标量口径的区别。
5. 新分支的 `README.md`、`models/manifest.json`、`evidence/` 和本地发布回执。

## 2. GitHub与VMAE的区分

- 仓库：`purplellllll/m`。
- 本线独立分支：`riboseq-gnn-transformer-20261031`。
- 本线独立Release标签：`riboseq-scorer-snapshot-20261031`，研究快照/预发布，非SOTA正式版本。
- 原`main`、`V3`、`V4`、`vmae-v2`保留。新分支工作树只呈现评分器代码、记录及交接，不把VMAE生成器混为评分模型；历史父提交保留来源关系，不合并回VMAE分支。
- 代码/模型清单放Git；大权重通过Release附件提供，不放Git大文件。实际发布commit、附件URL、SHA256和核验结果以本地 `runs/riboseq_github_publication_20261031/publication_receipt.json` 为准。
- 公共代码整理目录：`deliverables/riboseq_scorer_github_20261031`。权重导出是tensor-only，默认`torch.load(weights_only=True)`可读；不是带优化器/RNG的训练恢复包。

原第三方seq2ribo、RiboDecode模型权重不冒称自有模型上传；原数据与第三方源码按其来源、版本和许可单独恢复。seq2ribo许可有非商业研究及再分发限制，不能将其源码或权重重新标成我们的许可。

## 3. 已经完成的工作

### 3.1 数据、物理及图对齐

- 坐标固定为CDS第k个密码子对应核苷酸`[3k,3k+3)`；codon ID、物理特征、结构特征、边和标签逐项对齐。
- 用ViennaRNA完整CDS折叠，构建密码子级配对/角度/位置特征及结构配对边；序列邻接与结构邻接分开。
- 使用公开sTASEP模拟实现，旧模型输入为四次模拟均值/标准差的归一化特征。停留代理参数仅由相应训练集拟合，不称为真实动力学参数恢复。
- 核查了缺失标签、右侧padding、跨样本边、边界、三路梯度、权重重载及最长输入显存。
- RiboGL原始训练文件/原划分未恢复；用户明确选择同研究公开GSE73553 LIVER完整数据重新按基因划分。不能称原RiboGL论文成绩已复现。

### 3.2 三路融合、编码器与polish实验

- `FusionEncoderModel`比较physics/GNN/context分别放大、可学习增益、BiLSTM、Transformer，以及随机初始化的小BERT/T5。这里不是大规模预训练BERT/T5迁移。
- 完成27组融合/编码器权重训练，以及24组GraphPolishModel训练，均有三个种子与冻结记录。
- 旧400训练/80验证的共同历史120基因位置面板复核：BiLSTM context×2集成Pearson约0.4231，旧完整GNN→Transformer约0.3318，seq2ribo同数据重训约0.3040，RiboGL同数据重训约0.4574。
- 上述是每条转录本内部的log1p位置相关再平均，面板已用于历史开发，不是人类TE或蛋白表达独立验证。不能与论文的跨转录本相关直接比较，也不能将不同120面板/不同集成尺度数字混排。
- 旧polish实验显示邻居聚合有明显作用；全局注意力和早期物理注入尚无独立正收益证据。关闭早期物理注入仍保留直接物理分支，不能推断整个sTASEP模块无效。

### 3.3 扩大LIVER的v2模型

来源16256条，经过质量、完整CDS、基因/CDS/相同蛋白去重后6956条：5564训练、696验证、696最终测试。新测试排除以往1620条记录关联的基因/CDS/相同蛋白，尚未做近同源聚类，仍是单研究内部划分。

五组各三种子、最多3000优化器步；每100步完整验证：

| 组别 | 改动 | 已完成配对种子30101/30102的最佳验证位置Pearson均值 |
|---|---|---:|
| v2_control | 基础邻接、无结构门控、普通融合 | 0.4021 |
| sequence_graph | 有方向±1/±2/±4邻接 | 0.5173 |
| gated_graph | 再加学习的结构边门控 | 0.5278 |
| context_anchor | 再让Transformer主头作锚、融合残差修正 | 0.5232 |

旧GraphPolishModel width192是另一baseline，不将其与v2_control当作同架构单因素消融。v2系列注册参数相同，但关闭路径使有效参数不同，不能宣称完全等有效容量。

目前最有支持的是方向性序列邻接；门控有较小正向信号；context anchor在这两个种子上略降。全部只是验证集开发信号。停止时12组完成：baseline和v2_control各3种子、sequence_graph/gated_graph/context_anchor各2种子；sequence_graph_30103进行中，最后保存至1200步；后两组30103未开始。**最终696条测试未打开。**

### 3.4 标量下游接口与蛋白初轮

独立标量头从三路中间embedding做屏蔽均值/最大池化、归一化、可学习分支增益，再经非线性MLP预测；不将归一化位置输出求和冒充TE。

| 任务 | 输出与输入 | 当前状态 |
|---|---|---|
| RiboDecode RPF | `log1p(5×RPF_RPKM)`；CDS、实际RNA表达和作者顺序10552基因环境 | 数据已整理，尚未训练 |
| seq2ribo TE | Mean、HEK293、LCL、RPE-1分别单任务；CDS-only/含UTR分别比较 | 只准备了部分特征，尚未训练 |
| 蛋白表达 | mRFP实测表达标量 | 6组初轮完成，已打开锁定测试 |

蛋白数据1459条，原划分1021/219/219。统一大小写和T/U后有2个训练—测试重复CDS；我方训练剔除这2行，实际1019训练，另外报告217条无重复测试。seq2ribo公开的6条测试例与此数据的序列和标签精确匹配，但完整原始划分尚未完全认证。

蛋白初轮使用**此前width424 GraphPolishModel**三个已完成权重，未使用新width192门控模型。冻结编码器/端到端微调各三个种子、每组2000步，按三种子验证均值选中微调，打开测试前锁定。微调各种子最佳验证约0.8241/0.8551/0.8440；219测试集成Pearson0.85103177，去重217为0.85101303。单种子测试只预设报告，不能按测试重挑种子。

seq2ribo论文蛋白最高报告0.903；我们的同集原权重推理尚未完成。不能把不同完整划分身份未认证的两个数直接相减作公平差距或宣称超越。

TE完整标签10348条，Mean/HEK293/LCL/RPE-1映射已核对；现用RiboNN来源fold0测试、fold1验证、fold2–9训练（8240/1063/1045），**不是seq2ribo原始按A-site数分层80/10/10划分**。公开TE权重可能见过该共同测试集的标签，因此计划只从profile预训练权重重训TE头，不用现成TE头假装独立测试。原计划48组共同输入对比不等于原生配方复现。

RPF官方200训练环境、120测试环境、10552训练基因/1173测试基因已核对；当前规范90–4500nt CDS共同子集3245241行、36484条不同CDS，作者训练基因内再留验证。长序列处理/筛选与原文总体尚需逐项核对。

## 4. 停止时的状态与需要保留的东西

两个控制器及其计算子进程已退出，核查当时显卡没有计算任务。旧控制器PID8394和10781只作历史记录，不应在下一轮直接按旧PID操作。

- 扩大profile：12组完成，1组部分训练，2组未开始；`best.pt`、`latest.pt`、优化器和RNG保存于服务器；最终测试未打开。
- 蛋白：6组完成；按验证选定的3种子微调模型已测219/217测试，预测可复算。
- seq2ribo蛋白原权重对照：iPSC停在112/219进度，尚无完整指标。旧脚本只在一个细胞整批完成后落盘结果，112进度不等于已有112条可恢复的逐条预测。
- TE特征停在400/10348，逐序列缓存保留；RPF特征与训练未启动。
- 原队列最多126000步的时间估算与自动后续计划，**均被停止指令覆盖**。这不是训练失败后的自动重试场景。
- 导出、哈希、模型加载检查和上传可以进行；不据此启动新的生物数据验证或继续训练。

## 5. 五个证据缺口

1. 原始评测划分/筛选/输入/变换及原权重复现尚未完全对齐。共同划分重训练要诚实命名，不称原论文复现。
2. sTASEP的独立收益未证实。现参数来自小鼠训练的停留代理，人类TE及细菌mRFP没有目标域校准证据。双方同用鼠参数不能代表seq2ribo原生流程的最佳性能。
3. 位置表征是否能迁移到TE/RPF/蛋白仍需证明。高占据可来自停顿和拥堵，不能直接当作高产量。新标量头虽有真实监督和总量统计，尚未验证通量等物理量的增量信息。
4. 当前蛋白标量只是旧模型迁移初轮；不能用它代表新门控架构的最终能力。LIVER内部验证不能代替跨研究、跨细胞的人类标量评测。
5. 没有完成两套原模型在相同样本上的配对比较，也未覆盖RiboNN等强基线；尚无支持“全部任务SOTA”的统计证据。

## 6. 未来应先做什么

**第一步，文献/协议证据表。** 按任务列标签、物种/细胞、CDS/UTR/RNA环境输入、训练/验证/测试ID、序列筛选与长度处理、变换、权重来源及主指标。恢复或说明缺失的原始TE划分。区分RPF的RNA输入收益与真正序列增量。

**第二步，开发集上的小预算机制筛选。** 在预先固定的训练/验证样本和预算下，配对三个种子比较简单序列模型、方向性序列邻接GNN+Transformer、结构门控、真实物理、完整无物理、打乱物理、简单长度/GC/MFE统计代理。关闭整个物理分支与关闭早期注入必须分别命名。开发集试验只是选择假设，不作SOTA证明。

**第三步，标量读出筛选。** 比较编码表征池化、profile统计、pool+profile/通量/停顿信息；RPF必须有RNA/环境基线；TE分CDS-only与含UTR；物理参数只能在训练数据拟合。先证明收益稳定且超过简单代理，再决定昂贵模拟是否值得扩大。

**第四步，冻结后完整验证。** 冻结任务协议、候选、超参数、种子和主指标后，才做独立测试及跨研究/细胞确认，报告逐任务结果、单种子/集成及配对不确定性。已经打开的219/217蛋白测试不用于回调配方；未来确认性结论另设独立留出。保留尚未打开的LIVER696测试。

**第五步，潜空间接口。** 评分器经上述验证后再讨论连接VMAE。sTASEP及RNA折叠是离散/外部计算，不能假设可端到端反传；候选CDS改变后应重新计算对应图与物理信息或验证代理近似。高预测分不等于真实表达提升，GC/MFE/CAI约束也不能替代实测证据。

这些是建议路线，不是重启指令。用户仍需要决定何时恢复计算。

## 7. 代码与运行环境

核心模块位于`physics_scorer/`：

- `profile_model.py`、`model.py`：三路基础编码及图聚合。
- `fusion_encoder_model.py`：融合增益和编码器比较。
- `graph_polish_model.py`：旧width424早期融合polish。
- `reliable_graph_context.py`：新方向邻接、结构门控和上下文主头选项。
- `scalar_downstream.py`：独立标量监督、RNA环境及UTR输入。
- `checkpoint_io.py`：独立加载Release的tensor-only权重，不导入训练控制器。
- `seq2ribo_scalar_reference.py`：对第三方公开API的共同划分TE重训练适配，不含第三方网络源码。

原实验驱动保存在GitHub分支`historical/drivers/`，保留原绝对路径和源码身份；不是任意新机器上一键运行的CLI。不要直接运行控制器或setup脚本。迁移前需明确data-root、source-root和输出目录，保留原实验不覆盖。通用模型代码和导出权重可独立加载；`scripts/verify_exported_models.py`只做合成数值前向检查，不做生物数据测试或训练。

主实验：Python3.10、Torch2.7.1+cu118、numpy1.26.4、pandas2.2.3、scipy1.15.3、ViennaRNA2.7.0，SwanLab0.10.3；新profile环境复用旧ML依赖。BERT/T5比较使用transformers5.7.0。

seq2ribo基线隔离环境：transformers4.44.2、tokenizers0.19.1、huggingface-hub0.36.0、einops0.8.0、Mamba2.2.6.post3适配Torch2.7 ABI TRUE的CUDA wheel；已对未修改Mamba2.2.4参考定义核对，四层profile模型最大差2.384e-7。四细胞CDS-only/UTR任务头8条件前反向通过。这是有限范围数值检查，不替代全任务原生复现。

SwanLab项目：`https://swanlab.cn/@purplellllll/RiboSeq-GNN-Transformer-20261031`。原run可能因进程停止仍显示历史运行状态；以停止收据和现场进程为准。API密钥不写入文件，也不随GitHub发布。

## 8. 本地与服务器资产位置

Windows工作区：`F:\科研实验复现\新建文件夹`，PowerShell。根目录没有Git仓库，GitHub整理目录是单独快照，不能在整个工作区`git add .`。

| 资产 | 服务器目录 | 本地索引 |
|---|---|---|
| 旧融合/编码器 | `/root/fusion_encoders_20261031` | `runs/fusion_encoders_20261031/gpu_result/completed_snapshot/` |
| 旧polish消融 | `/root/graph_polish_20261031` | `runs/graph_polish_20261031/gpu_result/completed_snapshot/` |
| 扩大LIVER v2 | `/root/riboseq_graph_v2_20261031` | `runs/riboseq_graph_v2_20261031/` |
| 标量任务 | `/root/scalar_downstream_20261031` | `runs/scalar_downstream_20261031/` |
| 旧评分器同面板核验 | `/root/scorer_breakdown_20261031` | `runs/scorer_benchmark_breakdown_20261031/` |
| GitHub导出归档 | `/root/riboseq_github_snapshot_20261031` | `deliverables/riboseq_scorer_github_20261031/`及发布回执 |

可用解释器：`/root/riboseq_graph_v2_20261031/venv/bin/python`；基线隔离解释器`/root/scalar_downstream_20261031/venv/bin/python`。新对话先现场核查，不能按旧PID停止或删除其他项目进程；不要改动`ntl_8b_official_20261031`等无关任务。

数据来源与哈希：

- LIVER：Zenodo `10.5281/zenodo.17100113`，`data/processed/LIVER.csv`，SHA256 `70a6621d0779a0cef17ce49427560c0c3c1b3b237e005a1d1038f047aff8aa7f`。
- RiboDecode：`data/official/dataset_for_translation_model.zip`，SHA256 `2a365a5241f3b421a50c9fadb2d367b2076e8d68f530d3a87364f09af1d5f5a6`。
- 人类TE：`deliverables/public_riboseq_validation_20261028/human_TE_Supplementary_Table1.xlsx`，SHA256 `5a9dd751cb8ac96dbe364e249561cb54d4912c4170e7d659b9d465bd008e4c47`。
- mRFP：同组CodonMoE公开数据，SHA256 `ed32c1e547ca4db464ef873768fcc6b3f4738704981a88bd9b842bd4e630ed99`。
- seq2ribo源码原参考commit：`66c80c653e6db0d0b19a737f5d9fc8009db252b2`。公共分支只记录引用和哈希，不重新分发第三方受限源码/权重。

## 9. 恢复检查清单

先读最新用户指令与停止收据；核对GitHub分支/Release及附件SHA；确认模型类型、width、种子、标签和数据划分；查看缓存是否完整；将现有测试与开发集严格区分；记录仍缺原协议和物理校准证据。拿到用户恢复指令后，再设计独立的新输出目录和有限预算，不能删除旧launch状态后盲目重跑整个队列。

**本次GitHub上传是保存与交接，不是开始下一轮实验。**
