# 条件潜表示Diffusion对照实验

原目标：尝试用Diffusion替代无额外收益的图通路。最新多任务模型已经无GNN，因此本轮是在它上面新增条件去噪扩散分支，与无Diffusion的同预算profile-dynamics基线比较；不能称直接GNN/DDPM等价替换。

## 配方

相同数据/划分/输入、192维两层Transformer、两种子、每种子6000更新、batch8和验证集三主要端点macro Pearson选模。公共骨干初始权重逐张量相等，采样次序相同。
扩散16步线性beta从0.01至0.25，目标是detach后的context潜表示，条件是序列/AA/宿主/物理输入token；辅助epsilon MSE固定权重0.05，不使用表型标签作扩散目标。推理固定种子噪声和4步确定性DDIM(15,10,5,0)。
以0.1*tanh门控残差融合回原context，门控初始0；记录最终门控。参数量、FLOPs和推理时间增加，因此不声称完全计算预算匹配。没有生成新RNA，也没有把表达/HL预测当真实标签。

## 实测端点结果

|任务|模型|N|Pearson|95%条件区间|
|---|---|---:|---:|---|
|expression|Diffusion集成|217|0.868387|[0.826090, 0.901808]|
|expression|无Diffusion基线|217|0.867530|[0.825020, 0.901172]|
|half_life|Diffusion集成|601|0.572777|[0.510503, 0.628866]|
|half_life|无Diffusion基线|601|0.574914|[0.512372, 0.630263]|
|solubility|Diffusion集成|370|0.706528|[0.645640, 0.758916]|
|solubility|无Diffusion基线|370|0.707576|[0.645733, 0.759750]|

|任务|Diffusion减无Diffusion Δr|95%配对区间|
|---|---:|---|
|expression|0.000857|[-0.000985, 0.002852]|
|half_life|-0.002138|[-0.008260, 0.003498]|
|solubility|-0.001048|[-0.003007, 0.000922]|

## 解释边界

功能表达为217条Ecoli mRFP荧光；HL为601人源基因实测log10(HL+1)；可溶性为370条PURE蛋白测定。三者不是同批构建体，不能逐行拼接联合标签。测试历史已多次打开，本轮为探索，不是独立确认；区间未包括训练随机性、同源性或多重搜索。
门控非零或分支有梯度只证明函数依赖，不证明有稳定额外收益。仍需同时看三项和配对区间，不按测试选择有利种子或时间步。

训练协议comparison字段沿用旧模板，未运行sequence组；实际对照为上轮启用物理但无Diffusion的dynamics模型，evaluation通过锁定逐样本预测复用该基线，no_diffusion_ensemble标识为准。原定功能表达+细胞mRNA半衰期+可溶性验收尚未通过；Diffusion测试不能替代缺失的同样本实测配对数据。
原已发布多任务证据：[GitHub Release](https://github.com/purplellllll/m/releases/tag/multitask-scorers-20261102)。本轮Diffusion结果独立补充。
