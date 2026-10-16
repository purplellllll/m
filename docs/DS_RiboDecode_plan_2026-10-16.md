# DS-RiboDecode：解码器准备与双空间实验方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-16
- Version Label: ds_ribodecode_plan_v1
- Verification Status: UNVERIFIED（研究假设与效果尚未验证；数据校验和软件测试结果另列）
- 当前范围：准备 CDS-VMAE 数据、代码、验收标准；规划双空间优化。服务器正式训练、性质预测器改造和双空间优化器实现属于后续阶段。

## 1. 结论与需要修改的定义

采用用户草案的 `H=D(z)+αΔ` 作为主线。研究问题是：在相同条件和计算预算下，受约束的直接残差是否补充了生成先验难以搜索的方向，并在未参与开发的评价器上保持收益。当前没有结果证明双空间一定更优，也不能预先承诺真实蛋白产量超过原版。

必须澄清四点：

1. **α不是两个空间的贡献百分比。** 令实际残差 r=αΔ，L2的有效系数为 λΔ/α²；温度还使概率空间的有效残差成为 αΔ/τ。不能同时任意改变三者再解释机制。
2. **预训练的是 encoder＋decoder。** 当前使用带已知氨基酸条件的小型 VMAE，不是独立训练一个无编码器的 decoder。其 latent 维度 d_z=64，Transformer hidden=256，二者不同；可优化的参数化几何比“降维”更准确。
3. **现有 RiboDecode 的前向使用 straight-through one-hot。** `train_optim_sim.py` 中 `onehot + soft - soft.detach()` 把硬序列送入翻译预测器，其翻译损失还包含到 `rpf_target` 的距离；不能把它描述成单纯最大化 soft translation score。因此需要原版和目标/输入匹配两个直接空间对照。
4. **当前准备的是 RiboDecode 兼容开发库。** 它不是完整 GENCODE 主库，也不是 MANE core。新版草案的完整 Human/MANE 主库是独立的数据扩展阶段，不能把已有32,796条重新命名后当作已经完成。

## 2. 已经准备的数据及其适用边界

来源：RiboDecode 官方 Figshare v3 的 `dataset_for_translation_model.zip`，MD5 `9e73fdb5b25aa063a8152b541ac302d5`，121,954,830 bytes。对应 GENCODE v28 参考 CDS 用于核对终止密码子。原始文件未改写，decoder不使用RNA/RPF数值标签或优化产生的序列。

| 项目 | 已核实结果 |
|---|---:|
| 官方序列表记录 | 97,824 |
| 通过筛选的转录本记录 | 39,163 |
| 去重后 CDS | 32,796 |
| 不同氨基酸序列 | 32,746 |
| 开发 train / val / test | 26,764 / 2,744 / 3,288 |
| CDS长度 | 90–4,500 nt；中位数1,032 nt；包含末端stop |
| 基因、转录本、完全相同蛋白跨split重叠 | 0 |
| 近同源聚类 | 尚未执行；当前 `homology_certified=false` |

官方CDS表中的正常末端stop未包含在CDS字段中。仅在 `official CDS == matched GENCODE CDS[:-3]` 且参考最后3nt是真实stop时恢复；不盲加TAA，不以另一个版本整段替换。恢复计数发生于后续长度/质量筛选之前，不能将47,631次恢复等同于最终CDS条数。

存储沿用baseline的DNA字母T，生物学RNA表示等价于T→U；64个codon ID严格沿用本地 `dataset.py`。这样避免同一数据出现两套ID。PAD=64、MASK=65仅用于VMAE，**VMAE的64号PAD与baseline的64号NNN语义不同**，不能直接传递token ID。

原版test genes及已有验证目标的同蛋白组件保持test。服务器用MMseqs2聚类后，把 gene＋exact-AA＋cluster 的传递组件作为不可拆单位，再重新划分；同源组件中任一成员属于原版test，就整组留test。

这里的“未见”首先只针对VMAE。原translation predictor若训练过测试蛋白的同源基因，不能将它称作全系统的未见同源蛋白评测；需另审计其训练集，或将结论限定为decoder泛化。

建议预注册聚类阈值：AA identity≥30%、双向覆盖≥80%、connected-component聚类，保留软件版本、完整命令、输入输出哈希。它界定的是所用算法/阈值下的隔离，不能证明完全不存在远缘同源。聚类后还应做 train↔val/test 的敏感搜索审计；有命中则合并后重新划分，不能只删掉不利测试样本。使用正式测试集前完成这一步。

`homology_certified=true` 在当前脚本中只表示提供了完整cluster TSV并按组件划分，**不是第三方认证或零同源泄漏证明**。不能用全singleton伪文件绕过检查。

### Human/MANE主库的后续冻结方案

GENCODE v28早于MANE集成，不能靠v28 GTF标签得到MANE核心。已另备固定版本MANE v1.4官方summary和版本完整的ENST交集清单；与当前库仅235条精确ID匹配（183/17/35）。此交集不代表完整MANE core，尚未证明MANE实际CDS与v28序列逐字等价。

正式Human主库应另定一个含MANE注释的GENCODE release，成套获取GTF＋pc_transcripts FASTA；先提取同版本MANE Select，再扩展经过同样质量规则的protein-coding转录本。完成CDS实序列核对、去重、所有测试目标隔离和聚类后冻结新manifest。不要仅去掉ENST版本号强行跨版本合并。数据量扩大后重训VMAE，不复用旧checkpoint作为同预算“从零训练”的对照。

当前开服务器可先跑兼容开发pilot，用来验证工程与潜在空间是否有信号；不能用该pilot替代完整Human/MANE主库实验。细节见 [数据审阅](ds_ribodecode_data_review.md)。

## 3. 准备好的VMAE配置

| 项目 | 当前可执行配置 |
|---|---|
| Encoder / decoder | 各4层双向Transformer；不是自回归decoder |
| hidden / heads / FFN | 256 / 8 / 1,024 |
| latent | 每个codon一个64维高斯后验，μ/logvar独立头 |
| 参数量 | 6,407,616 |
| 输入 | codon＋位置；encoder另加已知氨基酸embedding |
| 位置表示 | 固定sin/cos；不是ModernBERT的RoPE |
| 输出 | 每位置64 codon logits，外部施加硬同义mask |
| prior / masking | 标准高斯；15%可变sense-codon位置遮挡；25%训练batch不遮挡 |
| Optimizer | AdamW，lr=1e-4，weight decay=.01；10%warmup后cosine |
| batch | microbatch4，累积8；不同长度microbatch均权，不严格等价32条拼接后token均权 |
| 最大训练 | 40epochs；验证selection loss连续6轮不改善早停 |
| KL | β最终=.001；前20%优化步数线性升温；按有效token及latent维度平均 |
| 数值 | 默认FP32；AMP为单独配置选项，GPU行为尚未验证 |

损失明确为：

`L = CE_all-valid + 1.0 × CE_masked-mutable + β(t) KL_mean-token-and-dim`。

这把masked重建额外加权，区别于草案中未展开的单一重建项；clean batch用于对齐后续从完整CDS编码的用法。已知目标蛋白是条件信息，masked位置的AA不是待预测标签泄漏。训练频率基线也使用同样的AA条件。停止、Met/Trp及PAD不计入主要可变位点准确率。

以上是**本项目的小规模实现提案**，不是严格LSCO复现。用户提供PDF的LSCO主干为6＋6层ModernBERT、hidden768；其完整VAE损失细节未披露充分。本项目未取得或加载LSCO官方预训练权重。

## 4. 解码器验收：分四道门

阈值是预注册的工程起点，需要在开发集作一次可行性检查后冻结；不得看测试结果后降低门槛。

| 层级 | 指标与通过规则 | 当前状态 |
|---|---|---|
| A 数据/实现 | 序列校验、词表一致、PAD零概率/零梯度、有效概率和1、硬mask后100%蛋白保真；哈希匹配、跨split基因/AA零重叠 | 已有检查与合成测试；正式同源审计待服务器 |
| B 可训练与重建 | 未见基因的clean可变位点accuracy≥95%；masked CE相对训练集同义频率先验降低≥5% | `evaluate`自动报告；未训练，尚无结果 |
| B latent使用 | zero-z与**同AA family内部shuffle-z**的可变位点CE，均比clean CE至少高0.10 nats/token | 自动报告；用以排除“仅携带蛋白类型”的弱证据 |
| B KL诊断 | KL/token/dim、active dimensions（跨token μ方差>.01）、full-sequence exact-match均报告；active dimensions=0需排查 | 自动报告；非零KL本身不能证明无collapse |
| C 可优化性 | 固定decoder，z→概率→原预测器的梯度有限且非零；同一蛋白内部小扰动/插值产生连续变化；核对soft-hard差、unique候选、密码子分布 | 适配阶段验收，尚未实现完整性质模型链路 |
| D 下游效果 | 同预算、同初始分布、相同scorer及约束，纯latent至少有稳定改进；双空间超过最佳单空间及顺序法才支持互补主张 | 尚未运行 |

`decoder_pilot_gate=true`只对应B层的有限自动门槛及mask保真，不等于C/D通过。掩码后的蛋白一致性由规则保证，不能拿它证明模型学会了生物学。全长exact-match需要按长度报告，不要求所有长CDS完全逐密码子重建；例如位点准确率95%不能推出整条CDS95%准确。

建议C层用每个开发蛋白8个posterior样本与固定的扰动强度网格，检查概率JS距离、可变位点Hamming、GC/同义频率变化。插值只在**相同目标蛋白、相同长度**的两个编码之间做；不能混合不同蛋白后把硬mask修正误称平滑语义空间。多样性追求不过度坍缩，也不要求随机性越高越好；不把CAI、GC最大化作为验收目标。

checkpoint选择只使用val的 `clean_mutable_ce + masked_mutable_ce`。test仅在模型及规则冻结后显式执行一次，按长度、同源簇、与训练库相似度分层报告，并以蛋白/簇为统计单位给bootstrap区间。当前自动汇总主要为token加权；正式报告还需每蛋白宏平均，防止长蛋白支配结果。

## 5. 双空间定义与权重

令S_i为第i位合法同义密码子集合，k_i=|S_i|。在合法位置中心化残差：

`Δc[i,j] = Δ[i,j] − mean_{j∈S_i} Δ[i,j]`；非法位置、PAD及不可变位置设0。

`H=D_logit(z)+α(t)Δc`，`P=softmax((H+M)/τ(t))`。

`RN = Σ_valid ||z_i−z_ref,i||² / (L_valid d_z)`。

`RΔ = Σ_mutable Σ_{j∈S_i} Δc[i,j]² / max(1, Σ_mutable(k_i−1))`。

相较草案的L×64分母，这避免非法codon、PAD和不同蛋白的简并度稀释正则。固定原始终止codon作为第一版设计约束；VMAE预训练的stop重建可覆盖三个stop，但设计时统一固定，所有方法相同。

优化目标：

`max J = λT T̄(P,e,b) − λM MFĒ(P) − λN RN − λΔ RΔ`。

固定校准候选上的均值/标准差用于T、MFE缩放；不得随每轮候选重估。跨长度可采用MFE/nt进行校准，但不能把原版总MFE目标无说明地换掉。MFE越负越低，负号保证更低MFE提高该效用；这不是断言越稳定就越容易翻译。先定义可接受RNAfold区间/不劣于参考的容差，再报告多目标结果。

| 参数 | 作用 | 建议开发起点/范围（不是论文测得最优值） |
|---|---|---|
| α | 直接残差开放程度与logit尺度 | 先固定1；α=0为latent-only；后续日程0→1 |
| λΔ | 实际残差的约束强度，与α耦合 | α=1时先.1，单因素试.01/.1/1 |
| λN | latent偏离固定参考的约束 | 首轮0，随后试.001/.01/.1；不要和全部新项同时上线 |
| λT / λM | 翻译与MFE的目标权重 | 标准化后λT=1；λM=0/.1/1；无合格MFE代理时先λM=0并用RNAfold做离散约束 |
| τ | 分布熵与梯度 | 先固定.5或1验证；再比较.5→.1；.03作为敏感性项 |
| lr_z / lr_Δ | 两种变量的更新尺度 | 先1e-3 / 1e-3；z试1e-4，Δ试1e-2；记录更新RMS |
| β_VAE | 仅在预训练约束高斯后验 | .001；与上述设计阶段正则完全分开 |

α不设为可学习变量。监测合法中心化decoder logits和αΔ的RMS、两分支梯度和更新RMS。没有限制的Δ可以覆盖decoder贡献，“可达到更高内部得分”并不证明使用了先验。更完整推导见 [数学审阅](ds_ribodecode_math_review.md)。

LSCO论文两套参数单独记录，不能拼成默认：Appendix B的τ=3→.01、lr1e-4、Adam，与Appendix D对照中的τ=.5→.03、lr5e-3、AdamW，均为3,000步。本文较温和的日程是待验证提案。

## 6. 初始化、阶段与RiboDecode适配

1. 冻结encoder、decoder、translation和MFE模型的参数，设eval；只更新z、Δ。推理优化不能包在`no_grad`中，否则会切断输入梯度。z_ref=μ(c0,a)，固定detach，Δ0=0。
2. 将decoder重构的p0作为几何对照共同起点。direct-only初始logits复制同一decoder输出；latent与hybrid使用同一z0。原始CDS另保留在候选档案，避免重构变差后仍把搜索称作从原始序列出发。
3. 固定α、固定τ的B1/B3成功后，再加阶段调度。提案：前40%预算仅latent、τ=.5；中间40%逐步α从0开至1、τ仍=.5；最后20%固定α=1，τ从.5降至.1。三个阶段共享同一总预算，不能每阶段各跑一整套预算。
4. 每固定查询间隔保存受约束argmax候选，去重，验证蛋白一致，再用hard序列复算分数及RNAfold。保留历史可行候选，最后一步不自动等于最佳。

概率适配：本包已提供 `[B,L,64] → [B,3L,4]` 的可微映射，碱基顺序A/C/G/T。实际RiboDecode模型调用还需其原始长度处理、padding与`[B,4,T]`转置，不能省略或超长静默截断。MFE模型使用原始65通道，必须把64个有效codon概率追加全零NNN通道后按其接口转置；PAD不是可设计密码子。

固定环境e和mRNA abundance b，不把它们当可优化变量。现有接口的环境是细胞表达环境，不能泛称“物种标签”。分数空间与`expm1()/5`逆变换只执行一次。

当前已有MFE权重是特定蛋白的模型。VMAE预训练不会让MFE自动跨蛋白泛化；没有相应MFE代理的目标，先做translation-only pilot并用RNAfold核验，再独立安排MFE代理训练。原代码存在长序列分段MFE路径，必须区分分段标签与全长RNAfold，不能混作同一真值。

## 7. 最小实验矩阵与统计

| 组 | 必要性与用途 |
|---|---|
| B0-original | 保留原RiboDecode流程、目标、直通估计；回答相对原工具表现 |
| B0-matched | direct-only，与本版hybrid均使用soft概率前向，且同scorer、目标缩放、初始p0和温度；隔离几何效应 |
| B1 | latent-only，α=0 |
| B2 | hybrid，无残差正则；检查是否退化为直接搜索 |
| B3 | hybrid＋合法中心化残差L2；MVP主要对照 |
| S | latent→direct顺序优化；不同时维护两个空间的必要低成本对照 |
| 后续B4/B5/B6 | 分别加分布KL、阶段调度、原RiboDecode warm start；不一次加入 |

第一轮不重构translation predictor，避免把预测器收益误算为空间收益；不引入OAS、CodonBERT或10成员ensemble。若顺序法与联合方法同样好且更快，优先接受更简单的方法。

同时给等scorer查询数、等wall-time两套比较；记录decoder前向次数、梯度次数、RNAfold调用预算、峰值显存、唯一候选数和总耗时。Warm start的RiboDecode运行必须计入总成本。预训练成本单列，并报告按目标数量摊销后的成本；不能只报后续搜索快而隐藏预训练成本。

先用10个开发蛋白、3个随机种子检验工程和方差（建议规模，不是功效保证）；冻结超参数后再按长度/表达范围抽取独立测试蛋白。正式样本量根据开发阶段的蛋白间配对差异方差确定。随机种子是重复测量，蛋白/同源簇才是主要独立统计单位。

至少一套独立translation评价器只用于最后盲测，不参与梯度、超参选择或候选选择。若反复用它选方法，它就成为开发评价器，需另留盲测评价。不同架构但同训练标签不能等同于独立生物学证据。

主要比较为每蛋白在相同约束下的hard候选质量、可行率、独立评分改变量、Pareto前沿和耗时；以蛋白/同源簇做配对bootstrap区间。RNAfold只核验热力学模型中的结构能，不能验证真实表达或半衰期。PERSIST-seq已有构建体可验证评价器泛化与特征关联，不能赋给新序列实测标签。

“大于等于原版”分两层：保留原版可行候选可保证**同一确定性选择规则下**不丢掉其已获得分数，代价是要支付原版搜索预算；它不能保证盲测分数或真实产量不降低。研究层面预先定义非劣界限与配对区间，失败则不宣称改进。

decoder不使用R²验收。后续若比较MFE回归器，在相同held-out样本、相同RNAfold真值上计算 `ΔR²=(SSE_baseline−SSE_new)/SST`，同时报告每蛋白与总体结果；不能比较不同测试集的两个R²。

## 8. 交付和下一步

已准备：清洗开发库及来源文件、MANE版本交集资料、VMAE训练/评估/断点恢复、合成单测、服务器运行说明和打包材料。

软件验证：6项单测覆盖词表、非法CDS、mask/PAD/梯度、序列化、数据篡改与泄漏拒绝，以及合成数据中断恢复与连续运行逐参数一致；本地PyTorch 2.14 CPU和已有PyTorch 2.0.1 CPU均通过。完整32,796行也已通过训练入口的数据校验。合成烟测不产生科研结果或可用预训练权重。

服务器顺序：检查环境/文件哈希 → 聚类及划分审计 → 开发训练计时 → 冻结数据和配置 → 正式VMAE训练 → val门槛及latent诊断 → 适配性质预测器 → B0/B1/B2/B3/S → 最后盲测。

没有启动真实语料训练；当前无已训练decoder、无GPU耗时/显存实测、无双空间性能结论。具体命令见配套本地训练包中的 `latent_decoder/README.md`（本次文档提交未包含训练包）。

## 参考与证据位置

- 用户提供 `LOSC.pdf`，实际方法名为LSCO；18页，结构检查PASS，阅读正文及Appendix B/D；公开核对：[LSCO](https://arxiv.org/html/2610.03098v1)。
- [RiboDecode论文](https://doi.org/10.1038/s41467-025-64894-x)；[官方代码](https://github.com/wangfanfff/RiboDecode)；本地 `train_optim_sim.py` 是本次适配审计依据。
- [官方数据 v3](https://doi.org/10.6084/m9.figshare.28916288.v3)，数据使用及再分发保留其署名和CC BY 4.0信息；不把数据许可自动套用于所有软件。
- [GENCODE v28](https://www.gencodegenes.org/human/release_28.html)、[MANE官方介绍](https://www.ncbi.nlm.nih.gov/refseq/MANE/)，固定下载信息见 `data/reference/mane/`。
- [MMseqs2官方手册](https://github.com/soedinglab/MMseqs2/wiki)，聚类/identity/coverage参数语义以所安装版本为准。
