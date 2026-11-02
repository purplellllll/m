# 人源TE推理合同

主权重为models/20261102/manifest.json中selected_paths的两个no_stats种子。下载Release后，按清单SHA逐份加载；两个原尺度标量取算术平均。不要使用旧mRFP头、旧TaskScalarModel随机head或对归一化位置profile求和代替TE。

`HumanTEScalar`内部严格含backbone、head及训练统计缓冲。无统计配方忽略properties参数；其他研究变体要求三列原始[GC fraction, MFE/nt, CAI]，内部使用各自训练统计标准化。输出为原生TE数据提供的标签尺度，不宣称物理通量。未认证HEK293之外或80–1500密码子之外的校准。

每个密码子对应一个节点。输入NPZ：

|键|形状|内容|
|---|---|---|
|cod|L|seq2ribo的64密码子ID，不是VMAE/RiboDecode tokenizer顺序|
|physics|L×2|四次模拟的占据均值/标准差，除以该序列平均占据后log1p|
|structure|L×5|配对比例、角度分箱/3、归一化度、归一化远距度、相对位置|
|edges|2×E|不同密码子之间的预测结构接触，模型内补双向及去重|

非填充批次single-CDS不含padding。主CLI从NPZ建立全true有效掩码。外部调用支持右侧填充，PAD ID64，边索引须在同样本有效节点内；GNN自己计算方向性序列邻居。

特征需用原锁定的ViennaRNA2.7.0、seq2ribo几何/sTASEP源及physics_parameters_train_only.json。自有拟合的物理参数和CAI参考随models/20261102及Release提供，附独立SHA清单；主模型物理代理未做人源动力学重拟合。输入规范及SHA记录见evidence/20261102/human_te_expanded_20261101/protocol_locked.json；外部源码需按原项目许可取得，不随本包重新分发。

统一模拟seed：`int(SHA256('human_te:'+RNA_SHA256)[:8],16)`。RNA_SHA256对大写、T替换为U后的完整CDS计算；DNA_SHA256另算，不能混淆。历史NPZ的dna_sha256键实际是legacy RNA UID，已在发布合同中明确。可重复地重新折叠/模拟新序列，不能用原序列物理输入给改写序列评分。

原native cohort是转录本TE，准确性报告按903个基因的同一留出标签计算。原始序列/工作簿不放Git；逐样本预测与必要评估标签用于结果复核，保留GENCODE和研究数据来源。输入序列构建和测量来源见third_party/README.md。

没有训练自动启动、远程部署或外部消息发送。该发布不是药物/临床级性能或SOTA声明。
