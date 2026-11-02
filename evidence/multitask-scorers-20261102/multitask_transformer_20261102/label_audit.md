# 多任务真实标签审计与验收边界

## Material Passport

- 日期：2026-11-02，Asia/Shanghai。
- 范围：功能蛋白实测表达、实测mRNA半衰期、实测可溶性；原始来源与可配对性核查。
- 状态：已取得额外NESG E/S公开表并完成严格join；mRFP十个变体组分原始表仍未取得。没有训练或修改任何旧模型。

## 当前能做什么

|来源|实测标签|宿主/体系|同样本配对|能证明的内容|
|---|---|---|---|---|
|mRFP Nieuwkoop 2023|1459条同义变体荧光功能表达|E.coli细胞|表达序列配对；无HL标签|可作功能荧光输出预测；不能复制为半衰期或可溶比例|
|Dieterich / Saluki原始来源|HEK293两重复半衰期，经原作者log10(HL+1)变换|人源HEK293细胞|可与同基因TE关联，但不是同一次功能蛋白测量|半衰期头实测验证；不是功能蛋白头验收|
|eSOL|可溶比例百分比、总产量；可推导可溶产量|E.coli蛋白在PURE无细胞体系|同条目的产量与可溶比例配对；无HL|可单独验证可溶性；不能当细胞内可溶性或酶活|
|OPT / Boel 2016|6348构建的E表达等级0–5|E.coli过表达|E与DNA配对，原附件无S/HL|有序表达等级的辅助任务；不是浓度或功能活性|
|新增SoluProt官方NESG|9703条exp/sol等级0–5及蛋白FASTA|E.coli异源表达|可与OPT部分精确ID配对|补充E/S辅助验证，但仍无HL，S不是百分比|

不同来源、宿主、构建体的标签不能按行号或蛋白名称跨表拼成三标签。多任务模型可使用缺失标签mask分别训练各头；这不等于拥有同样本三项联合真值。

## 新取得的NESG实测E/S面板

- 官方下载页：https://loschmidt.chemi.muni.cz/soluprot/?page=download
- 直接公开下载：https://loschmidt.chemi.muni.cz/soluprot/?f=nesg.zip
- ZIP通过文件签名与zipfile验证，1,339,711字节，SHA256 `ea26072f601fb91dd2eaa3a842cd9268ceef2a8139722e78de04ffef6d0db71e`。
- 解压得到 `sources/nesg/nesg.csv`（9703条、id/exp/sol）及 `nesg.fasta`。
- OPT6348与NESG按name/id精确join：778条；其中E一致718条。不能忽略60条表达等级不一致记录。
- 778条OPT翻译产物全部带`LEHHHHHH`尾标签；SoluProt FASTA未含该尾标签。原始全长蛋白精确相同数为0，不能声称全构建已精确认证。
- 显式去除上述固定尾标签后，640条protein core精确一致；再要求E一致，得到595条可佐证配对。
- 原始join存 `sources/opt_nesg_expression_solubility_join.csv`；严格595子集存 `sources/opt_nesg_E_S_corroborated.csv`；计数、来源不一致示例和限制见 `sources/nesg_join_audit.json`。
- 剩余差异不做模糊序列匹配，不自动替换标签。核心序列与E一致只能佐证同源测量身份，不等于原始构建条码已核实。
- 如果纳入训练，必须沿用OPT按蛋白SHA的划分，并与eSOL及其他任务的train/validation/test做跨任务蛋白泄漏检查。不得在此595条上另随机拆分后与原OPT测试混用。
- E、S均为有序胶图等级。可报告等级Pearson/Spearman及阈值分类；不能把`E*S`称为实测可溶产量。即使定义E*S为可用性代理，也不含半衰期或功能活性。

## mRFP十个变体可溶/不可溶数据追索

原文XML明确说明：十个覆盖荧光范围的变体做LC–MS/MS，两组分分别测LFQ，并对未裂解细胞混入不可溶沉淀做校正。功能荧光与蛋白丰度的文中相关性不能代替我们逐序列重新预测的相关性。

本地原文XML列出的补充文件为`gkad035_supplemental_files.zip`，预期1,538,334字节，MD5 `756488dd92de30a38bc743ff3ba66b40`。本轮公开入口尝试如下：

|入口|结果|处理|
|---|---|---|
|PMC instance/bin补充链接|HTTP200但只有1817字节text/html|不是ZIP，不保存为原始表|
|EuropePMC supplementaryFiles|ConnectionError|未取得数据|
|按XML云资源标识构造的公开CDN路径|HTTP404、text/html|不是原始表|
|OUP公开文章链接|HTTP403|不绕过访问控制|
|NCBI旧/新OA工具公开URL|HTTP404|不据此编造FTP文件|

`label_download_receipts.json`保留URL、状态、内容类型、字节及响应SHA。mew软件Zenodo ZIP已核对条目，仅含代码/模型和密码子表，没有proteomics/solubility原始表。不能将软件归档当组分实验结果。

未取得这十个变体的精确CDS身份、两组分校正LFQ和归一化分母，故未从图上拟造数值。即便后续取到这十条，也仍需要该构建体的实测mRNA half-life才能形成三标签联合面板；原文已知组分验证没有提供half-life测量。

## 验收建议：分别明确“单项通过”与“联合通过”

1. 三个任务头分别使用实测功能表达、实测HL、实测可溶比例；标签mask只影响相应损失，不用其他评分器预测补标签。
2. 在相同样本ID上比较新模型、seq2ribo与RiboDecode，保留宿主、输入上下文、校准训练标签和预算差异；输出Pearson、Spearman、条件区间及配对差值。不同任务不同测试集的r不能求成一个Pearson。
3. 若要验收同一样本联合标量，必须先取得相容条件下至少表达+HL双标签；可溶性加入则还需同一构建体相同条件下可溶比例。权重和变换预先固定，由训练集拟合均值尺度，不能测试调权。
4. 当前可报告三头的分别实测性能，但不得宣称同样本三标签联合通过。真正独立确认还需要新留出基因/近同源簇或新实验，既有903/227/217面板已多轮打开。
5. 新模拟器的电流、占据、阻挡等只作模型输入；不能作为功能蛋白、HL或可溶性的实测验收真值。

## 本轮未完成

未取得mRFP十个变体原始组分表；未发现同一构建体的功能表达+实测HL+实测可溶比例三标签公开面板。未对新595条NESG辅表训练模型或宣称通过验收。
