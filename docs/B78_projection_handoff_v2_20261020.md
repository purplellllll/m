# B7/B8 v2：Direct → Projection → Latent

状态：开发集交接比较配置已准备；尚未选择或冻结正式交接方案。模型、评分器权重均冻结，仅优化设计变量。

旧版“固定残差继续优化潜变量”已被替代。远端旧任务已停止，目录 `/root/autodl-tmp/ds_b78_reverse_validation_20261020` 中的结果保留为 legacy，不得与新版混合。

## 阶段和正则

1. 复用 Encoder 的 mu、sigma 和 CDS 特定共享随机起点 v，实际 z=mu+sigma*0.5*v，初始 v 的 RMS=.05。
2. Direct 40 步：固定初始 z，仅优化残差。B7 无正则；B8 仅使用原 B5 的 direct_l2 和 decoder_kl（系数均 .1）。
3. 交接目标为第一阶段**最佳合法**候选对应的连续分布，包含原始 CDS 和共享随机起点的兜底；不使用最后一步替代最佳点。
4. KL 投影：最小化 KL(P_direct || P_D(z)) + eta*R_z。tau=1、eta=.1、Adam lr=.01、40 次更新，按每条 CDS 的投影目标最小值保留解，不用翻译评分挑投影解。R_z 为相对原始 mu 的标准化潜变量 v 的均方，复用 sigma 和 latent_scale 定义。
5. 工程对照：对最佳 Direct 分布 argmax 的合法 CDS 重新编码，以 Encoder mu 为投影解；保留原始 CDS 的 mu/sigma 作为第二阶段正则参考。无投影梯度步骤。两种方法均受原坐标 RMS≤16 安全界限约束。
6. 交接后 delta **清零**，Adam 潜变量状态重置。Latent 40 步仅更新 z。B8 仅有原 B5 的 latent_l2（系数 .1）；残差项和最终分布对 Decoder 的 KL 自然为零。任何投影退化均不得恢复残差。
7. 保留原始、Direct、投影、Latent 的历史合法候选，最终取最高内部评分；最终候选可能来自 Direct，因此必须同时报告投影与第二阶段终点，不能将历史最优当作交接成功。

## 日志和预算

`handoff.jsonl`：逐 CDS/arm 的 before、after、after.feasible、delta_projection=F(after)-F(before)、direct_l2_after=0，以及逐步 KL/目标轨迹。评分差在不可行交接时同样保留。`steps.jsonl`：两阶段变量范数与评分；`results.jsonl`：最终端点、历史最优和预算。

每方法仍为 40+40 次搜索 backward、240 次搜索 scorer forward；额外 2 次交接 scorer forward。KL 增加 40 次 Decoder backward 和 42 次 Decoder forward；重新编码增加一次 Encoder forward 和 2 次 Decoder forward。投影耗时包含于搜索耗时，额外 backward 单独记录，不能声称与无投影 B4/B5/B6 的总计算成本相同。

## 开发集比较与冻结

先复用 `docs/b456_random_validation_plan_20261020.json` 的 64 条 val CDS 和 20261023/24/25 三个随机起点，比较两种交接方式，正式 256 条 test 不用于交接选择。

预先记录选择规则：先按交接后满足原约束的比例选择；比例相同时按逐 CDS、跨 arm/seed 平均的 delta_projection 选择；完全相同则选较便宜的重新编码。固定全部样本和 seed，不选最佳 seed，不删除退化样本。报告投影 KL、交接分数下降、最终潜变量端点、档案最优、长度分层和成本。开发集已有历史使用，本比较是探索性，不宣称全新无泄漏验证。

运行示例（输出目录必须不存在）：

```bash
python -m latent_decoder.paired_random_validation --arms B7 B8 --handoff-config latent_decoder/config_b78_projection_dev.json --plan docs/b456_random_validation_plan_20261020.json --checkpoint checkpoints/epoch50.pt --output runs/b78_kl_seed20261023 --seed 20261023
python -m latent_decoder.paired_random_validation --arms B7 B8 --handoff-config latent_decoder/config_b78_reencode_dev.json --plan docs/b456_random_validation_plan_20261020.json --checkpoint checkpoints/epoch50.pt --output runs/b78_reencode_seed20261023 --seed 20261023
```

对另外两个 seed 重复，各输出目录独立。比较完成后建立新的正式 B7/B8 计划，记录交接配置哈希、来源开发集结果和新增预算；运行器目前拒绝直接套用旧 B4/B5/B6 正式计划启动新版 B7/B8。旧正式结论与匹配掩码不得改写。独立评分使用 seq2ribo，而 RiboDecode 是内部优化评分器，两者不可混称。
