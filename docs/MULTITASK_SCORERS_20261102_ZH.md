# 新多任务Transformer与sTASEP修复：实测验收报告

## Material Passport

- 日期：2026-11-02，Asia/Shanghai。
- 状态：模型实际训练、锁定测试、三实测端点分别验证；见verification.json。
- 任务：功能荧光表达、细胞mRNA半衰期、蛋白可溶性，TE只作辅助监督。
- 验收：不能用TE、MRL或模拟半衰期代替上述实测标签。缺同一样本细胞表达+HL、三标签联合面板，联合验收仍不可宣称通过。

## 已实施

共享两层width192 Transformer，密码子及氨基酸嵌入，5′/3′UTR独立卷积编码及完整组成，宿主显式编码，四个读出头（功能表达/HL/可溶性/辅助TE）。缺失标签mask，训练统计单独拟合，测试标签独立封存。人源旧TE预训练仅迁移密码子和context，所有新任务头训练。
两个阶段均序列对照/动力学两组×两种子×6000步，batch8，按任务轮转；验证集三个主要实测端点macro Pearson选择checkpoint，全部模型锁定后预测测试。完整保留负结果。旧实验未覆盖。

## 三端点测试

第一阶段使用明示neutral等待，不含同义密码子差异；它是必要对照，不能叫已标定物理。

|任务|模型|N|Pearson|95%条件区间|
|---|---|---:|---:|---|
|expression|sequence_ensemble|217|0.764155|[0.703531, 0.816544]|
|expression|dynamics_ensemble|217|0.767472|[0.708554, 0.819840]|
|half_life|sequence_ensemble|601|0.594083|[0.532749, 0.649123]|
|half_life|dynamics_ensemble|601|0.590893|[0.531038, 0.645618]|
|solubility|sequence_ensemble|370|0.714782|[0.658368, 0.764702]|
|solubility|dynamics_ensemble|370|0.713165|[0.654548, 0.764158]|

第二阶段独立代码/缓存/协议，使用公开真实RPF训练拟合的宿主专属相对占据参数，加RNAfold2.7.2 first150nt配对信息；两组均使用相同六项短窗口MFE特征及宿主context条件。不能将第二阶段与第一阶段差异全归因于物理；第二阶段内sequence/dynamics才是受控物理对照。

|任务|模型|N|Pearson|95%条件区间|
|---|---|---:|---:|---|
|expression|sequence_ensemble|217|0.871980|[0.832061, 0.903110]|
|expression|dynamics_ensemble|217|0.867530|[0.825020, 0.901172]|
|expression|seq2ribo_native_expression|217|0.903342|[0.871889, 0.927149]|
|expression|previous_own424|217|0.851013|[0.801788, 0.893022]|
|half_life|sequence_ensemble|601|0.573455|[0.510006, 0.629475]|
|half_life|dynamics_ensemble|601|0.574914|[0.512372, 0.630263]|
|half_life|previous_HL_auxiliary|601|0.519252|[0.460238, 0.576205]|
|half_life|seq2ribo_TE_cross_task|601|0.257075|[0.180142, 0.330550]|
|half_life|RiboDecode_TE_cross_task|601|0.299481|[0.226210, 0.374622]|
|solubility|sequence_ensemble|370|0.708881|[0.650066, 0.759779]|
|solubility|dynamics_ensemble|370|0.707576|[0.645733, 0.759750]|
|solubility|CDS_Ridge_auxiliary|370|0.680812|[0.626376, 0.729104]|
|solubility|AA_Ridge_auxiliary|370|0.654447|[0.597314, 0.705388]|

完整配对差值：

|任务|动力学集成减参照|Δr|95%条件区间|
|---|---|---:|---|
|expression|sequence_ensemble|-0.004450|[-0.015422, 0.005890]|
|expression|seq2ribo_native_expression|-0.035812|[-0.071930, -0.004385]|
|expression|previous_own424|0.016517|[-0.026624, 0.065205]|
|half_life|sequence_ensemble|0.001459|[-0.007538, 0.010342]|
|half_life|previous_HL_auxiliary|0.055662|[0.014418, 0.096420]|
|half_life|seq2ribo_TE_cross_task|0.317840|[0.235684, 0.399490]|
|half_life|RiboDecode_TE_cross_task|0.275433|[0.188045, 0.355269]|
|solubility|sequence_ensemble|-0.001305|[-0.012759, 0.010316]|
|solubility|CDS_Ridge_auxiliary|0.026764|[-0.006712, 0.060028]|
|solubility|AA_Ridge_auxiliary|0.053129|[0.015594, 0.092232]|

这些不同端点来自不同宿主/实验，不能将217条mRFP与601条人源HL逐行拼成联合标签。HL辅助模型及TEcross-task分数的监督权限不同，不能解释为公平架构比较。
mRFP为单蛋白同义变体，非跨蛋白；HL按基因/精确蛋白分离；eSOL为PURE无细胞体系、参考W3110CDS与ASKA标签未完全认证，188条原测量超过0–100未裁剪。以上全部为项目历史已打开面板，非新独立确认。

## sTASEP实际修复及剩余限制

1. 固定8192预热+4096观测步、三启动率(.003/.01/.03)、每率两轨迹，记录真实模拟启动/移动/排他阻挡/完成事件及占据分数；电流为完成数/步，不称真实秒通量。
2. 保留12维位置特征和36维摘要，不再仅保留逐转录本归一化mean/std；支持细菌table11七类合法起始，终止位置不占据，最后sense按自身等待完成。
3. 排他距离、进入/完成守恒、事件计数、同seed确定性、缺失标签梯度、padding、缓存来源和严格模型重载均有实际检查。v1不支持ATT/CTG时失败保留，v2修复未丢样本。
4. 不再把鼠肝等待参数用于human/ecoli。Ecoli来自Riboformer GSE119104高Mg flash-frozen observed strandedWIG，明确原3′端P-site14→A-site11假设；human来自GSE65778/GSM1606107官方.ribo真实5′read-start覆盖，固定28–30nt，A-site+15假设，P-site+12仅敏感性。不按heldout选择偏移。
5. 拟合排除当前任务val/test基因、精确DNA、精确蛋白，human另排除旧TEval/test基因。源数据再按固定gene/protein规则拆分，train-only非负拟合。参数仍是relative-occupancy统计代理，不可识别真实等待速率或启动率；5′150之外结构为0假设、角度未观测，非完整结构建模。
6. 人源源数据内部按gene拆分有2/4个精确蛋白跨train-val/train-test，已额外审计；因此其源测试仅gene独立、非protein独立。当前任务heldout蛋白在拟合前已排除。独立源的位置r较弱，应以最终三个端点比较判断信息效用。

## 开源配对数据检索

已下载PERSIST官方Source Data及补表，实际有24构建的细胞Nluc/Fluc表达和加速in-solution化学半衰期；它不等于细胞mRNA half-life，且5条CDS与完整转录本字段不一致，不强行修补。正文Fig2f细胞12h表达原始精确表未在官方31sheet/SD1列中找到。
新取得OPT/SoluProt严格同构建595条E/S等级配对，0–5等级而非功能浓度/可溶比例，也没有HL；未混进核心百分比头。mRFP10变体组分表下载未成功，记录公开访问失败，未把HTML当ZIP。详见前阶段label_audit.md及sources/paired_assays。

## 复现与边界

核心代码model.py/train.py/evaluate.py，参数human_parameters.json/ecoli_parameters.json，特征features.py/dynamics_v2.py。训练/测试逐行输入与hash、两个种子best.pt、逐样本预测和配对bootstrap均保存。
运行verify_results.py独立复算Pearson并核对所有模型SHA与实测标签；参数源协议及原始数据收据保留在上一阶段sources/profile_open。
未校正多重比较；bootstrap为固定模型条件区间，未包括项目多轮探索、近同源、训练随机性与参数推断不确定性。不把相关解释为因果，不将跨基因结果解释为同义变体效果，不按测试调整权重直到自有第一。
尚缺：独立新面板上的细胞功能表达+HL+可溶性联合实测验证、真实绝对动力学/启动率标定、完整3D结构输入。

## 配对化学稳定性外部诊断（不替代原验收）

24条中仅19条CDS与完整转录本字段精确一致；排除5条来源不一致记录，未根据结果挑选。表达头从细菌mRFP迁移到人细胞Nluc，HL头从细胞内转为加速化学降解，均为跨测定外推。
|指标|N|Pearson|95%条件区间|
|---|---:|---:|---|
|expression24h|19|0.154952|[-0.203334, 0.566479]|
|expression6h|19|-0.329068|[-0.612518, 0.048377]|
|chemical_HL|19|-0.062836|[-0.484438, 0.350411]|
|joint24h_chemicalHL|19|-0.132476|[-0.555979, 0.355071]|

该外部诊断失败，不能把化学半衰期或这里的负相关联合指标包装成细胞内联合验收成功。

## 明确结论

新多任务Transformer和宿主RPF代理版均已实际启用、训练、测试，sTASEP事件观测与宿主误用问题有可运行修复。
功能表达目前未超过seq2ribo，受控物理增益未成立，同样本细胞功能表达+HL+可溶性标签仍缺。因此用户原定验收尚未通过，不能仅凭HL提升宣布整体胜出。
后续有根据的优先项是：独立的人源功能表达监督、功能任务专用适配与初始化、更可靠的位点/启动率标定、同条件配对测定；继续盲增模拟次数或堆Transformer层数尚无收益证据。
推理入口：predict.py；示例输入inference_example.json、输出inference_example_prediction.json。输出保留任务训练宿主/测定范围警示，不是通用绝对表达/小时/体内可溶性预测。
