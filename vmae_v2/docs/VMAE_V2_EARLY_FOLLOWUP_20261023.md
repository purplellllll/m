# 按用户要求提前启动系数实验

2026-10-23 用户明确要求暂停基础训练并直接开启后续实验。
基础训练停止于最近完整 epoch 的 checkpoint；未完成的当轮更新不覆盖已保存状态。
原 15 次训练的结果、优化器状态、日志和旧排队计划保留，原训练不会自动恢复。

新的启动命令为 `tools/run_vmae_v2_followup_early_20261023.sh`。
`--available-now` 必须看到上游 `paused_by_user` 状态，且仍要获得同一 GPU 工作锁，
不能与原训练并行抢占同一资源。

本轮使用已完成 uniform 训练的第 0、1 折。第 0 折三个完整配置都可作为控制；
第 1 折只有已完成的 uniform 控制，不把未完成的 short_fast checkpoint 当成完整控制。
每折 11 组单因素筛查、2 epoch，接着最多 4 组完整训练集微调、8 epoch。
增益、学习率、KL、weight decay、free bits 参数范围及内部选择规则沿用原计划。
每 25 步输出训练 CE 和 loss；每轮输出固定公式的验证 CE、loss 和按长度统计的 masked μ AU。
该 masked AU 与原训练日志的 clean μ AU 口径不同，不直接比较其数值大小。

所有参数仍只由内部验证集选择。两折选择锁定后才计算外折测试结果。
V1 比较限制在对应两折、原测试集且同源组与原 train/val 隔离的 780 条 CDS，
不把两折先导结果表述成完整五折交叉验证。
summary.json、complete.json 都写出 `partial_cross_validation_pilot` 和
`full_five_fold_validation_complete=false`；通过标志只适用于本轮子集。
固定 loss 的 KL 评估权重仍为 0.001，不因训练正则权重改变而降低验收要求。

新结果目录：`/root/gpufree-data/vmae_v2_followup_early_20261023/results`。
新日志：`/root/gpufree-data/vmae_v2_followup_early_20261023/logs/followup.log`。
旧队列状态标记为已被本轮提前实验替代，保留旧 queued_plan.json 不混写新结果。
