# 物理入口修复与通路审计

日期：2026-10-31。基于发布提交23adedd7的本地开发分支`dev/physics-te-path-audit-20261031`。没有重新训练、没有计算生物数据预测指标。

## 修复范围

- `TaskScalarModel`读取`ReliableGraphContext.use_physics`，在池化前置零禁用物理表示，并把开关传至标量头。
- `ScalarHeads`禁用物理时同时屏蔽物理池化、LayerNorm偏置、物理分支增益和后三维模拟统计。两条剩余分支增益重新归一化，物理分支参数无梯度。
- `ReliableGraphContext`禁用物理时不计算/返回物理辅助头，避免辅助损失继续训练该头。所有参数保留以兼容旧state_dict。
- 默认启用物理时，两个主头模式的标量、profile及三个辅助输出与发布源码逐位相同；没有改变默认配方。

直接调用缓存`ScalarHeads`时必须显式设置`physics_enabled=False`；缓存应记录模型开关和模拟/结构来源，旧缓存不能仅因CDS相同就当作新参数缓存。旧GraphPolish的early_zero只关早期注入，不承诺完整无物理。

## 验证

`scripts/test_physics_ablation.py`覆盖TE/蛋白/RPF的输入不变性、模拟统计不变性、梯度为零、辅助物理损失移除、带偏置归一化、缓存入口及旧state_dict严格加载。原scalar_heads和reliable_graph_context检查通过。独立合成审计覆盖14种损失/锚定条件和10种标量输出通路，默认输出与git发布源比较逐位相同。

无物理时aux字典只含gnn/context；未来训练器需按实际头统计指标。旧`historical/drivers`作为来源记录保留，其evaluate硬编码三头，不是直接运行无物理实验的新入口。若为公平消融保持辅助头系数一致，应固定每个保留头的权重，避免从mean(3头)改成mean(2头)时暗中放大其权重。

## 头与embedding的真实关系

三路表示可被标量监督更新；三个位置辅助头与最终profile融合头未被标量读出引用，因此纯标量训练中没有这些头的梯度。这是当前任务接口的设计，不是断线。下游只池化embedding，不能声称已经利用profile输出或通量。context输入已包含图和物理表示，因此三路不是独立专家；context辅助分数也不能当作sequence-only对照。

非锚定主头的correction末层为零初始化：初始时main-only梯度只到correction末端，上游需由辅助损失带动；加入辅助损失后编码器都有梯度。此现象仅限初始化附近，不意味着训练后永久断路。锚定主头初始即能通过context主头更新编码器，但已有验证数据未显示其更好。

更进一步建议先做同一标量标签的单分支probe和条件去除，判断分支增量；再比较现有mean/max池化与增加profile形状统计；最后才评估样本相关融合。全局branch_logits加上可重缩放的MLP权重不能直接作为分支重要性证据。方案细节和本轮原生TE对齐结果保存在工作区`runs/scorer_path_alignment_20261031/`，大数据与第三方源不进Git。
