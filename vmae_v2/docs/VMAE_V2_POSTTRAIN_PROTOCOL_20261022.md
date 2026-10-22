# VMAE V2 训练完成后的系数实验

## 自动衔接

入口为 `python -u -m latent_decoder.v2_followup queue`，服务器启动脚本为
`tools/run_vmae_v2_followup_20261022.sh`。等待当前五折 × 三个学习率配置全部写出
`complete.json`，且原部署状态为 `completed`；再取得原训练的进程锁后使用 GPU。
原训练失败时后续队列失败退出，不抢跑、不修改原训练源码、权重或日志。

结果目录：`/root/gpufree-data/vmae_v2_followup_20261022/results`。
日志：`/root/gpufree-data/vmae_v2_followup_20261022/logs/followup.log`。
`queue_status.json` 显示等待、运行、失败或完成。服务器重启后可重启同一脚本；
训练按最近完整 epoch 恢复，完成的试验不重复。源码、初始权重或数据改变会拒绝混用旧结果。

## 实验对象与参数

修改的是 VMAE 的训练/微调参数。只改变冻结模型搜索时的 Rz、搜索步长、JS 或
direct L2，并不会改善该模型的重建交叉熵。因此本轮保留原搜索代码，先验收模型本身。
以下系数不代表对原 B5-JS 的 JS≤0.05 或修改预算做了放宽。

短 ≤300 aa；中 301–699 aa；长 ≥700 aa，长度均不含终止密码子。
每折都从该折已经独立训练的 `uniform/best.pt` 开始，绝不从原 V1 权重初始化。
所有微调分支从同一权重、随机种子、样本顺序开始，AdamW 优化器重新初始化。

基线微调参数：潜变量到 decoder 的输入增益 1；基础 LR=5e-5；
短/中/长倍率=(1,1,1)；KL β=0.001；weight decay=0.01；free bits=0。

第一阶段每折 11 组单因素筛查：

| 变化项 | 设置 |
|---|---|
| control | 所有基线参数 |
| decoder 输入增益 | 0.5、2，相对基线 1 |
| 基础学习率 | 2.5e-5、1e-4，相对基线 5e-5 |
| 长度倍率 | (2,1,0.5)、(0.5,1,2)，相对 (1,1,1) |
| KL β | 0.0003、0.003，相对 0.001 |
| weight decay | 0.001，相对 0.01 |
| free bits | 每潜变量维度 0.02 nat，相对 0 |

free bits 对有效 token 的每维平均 KL 施加下限，再求和；低于下限的维度不再受到
额外压向先验的梯度。验收仍计算未截断的原始 KL，不享受训练 loss 口径变化的优惠。

调整输入增益时，初始 `from_latent.weight` 除以同一增益，保持起始输出函数相同。
这检验参数化对训练的影响；增益本身可被线性层吸收，不能独立证明信息容量增加。
新 checkpoint 包含 `recipe.gain`，推理必须由 `ScaledVMAE`/`load_model` 加载，
不能直接丢弃 gain 字段后按旧模型推理。

筛查固定抽取每档 512 个训练同源组、32 个内部验证同源组，训练 2 epoch。
每折取两个较好的非 control 设置，并按各参数族内部验证 loss 选取组合参数；
连同 control，最多 4 组从同一初始权重重新在完整该折训练集微调 8 epoch。
每轮使用固定的每档 128 个内部验证同源组选择 checkpoint。
前半 epoch warmup，之后 cosine；每组微调使用相同轮数与种子。
这是一轮有限预算的筛查，不保证找到全局最佳系数。

已有的 uniform、short_fast、long_fast 三组完整训练模型也在同一内部验证口径下比较，
用于判断短、长是否需要不同 LR。最终模型先优先满足内部验证宏平均 CE 和固定 loss
均低于 uniform，再按固定 loss、CE 排序。候选也包括这三组已有模型，避免微调退化后
强行替换模型。各折的参数可以不同；不能据一个折就宣布通用最优系数。

## 固定的验收口径

全部五折完成内部选择并锁定后，才打开外部测试评估，测试结果不反馈给参数筛选。
原 V1 对照固定为 epoch 50，SHA-256：
`8009efdc3fc3ee61c82925b02feba600be3d6409f923838a612c26eb4af3d673`。
其训练数据 manifest SHA-256 必须匹配 `b11cd373b8fb8f5f7b79fe93089312b73142266cf3893fe270c7eee1fa9d77de`。

V1 已见过原训练集，因此不能在整个新五折测试集上作无泄漏对照。
验收子集固定为原 test 中、所在 V2 同源组从未含有原 train/val 样本的 2,082 条 CDS。
短/中/长分别为 923/797/362 条，每条只由对应外折模型预测。
还将独立记录 V2 在完整外折测试集上的表现，并对比该折原 uniform 模型。

每条序列独立哈希生成固定掩码，mask seeds=173/174/175，mask probability=0.15。
两模型使用相同样本、掩码、posterior mean 和同义密码子支撑集：

- CE：被遮盖的可修改密码子位点上的平均交叉熵。
- full CE：同一次遮盖输入下所有有效 token 的平均交叉熵。
- 固定 loss = full CE + CE + **0.001 × 原始 KL/token**。
- 每档内部按位点加权，宏平均对短/中/长三档等权。
- 通过：三档以及宏平均的 CE 和固定 loss 都严格低于 V1；否则明确记录 `passed=false`。

训练日志里的 train_loss 使用该试验的正则参数，仅作优化监控，不作为跨设置验收依据。
每 25 个优化步同时打印 train_loss、train_ce、train_full_ce；每轮打印验证 CE、full CE、KL、loss。

这批原 test 之前已用于探索性研究，所以结果只能证明当前锁定比较子集上的改进，
不能视为新的外部盲测、独立生物学验证或真实翻译实验验证。单次严格均值优于也不是显著性结论。

## 隐空间诊断与输出

在 V1 比较子集上，按长度分别输出：编码器 μ 的 RMS/绝对均值、后验 σ；
AU 阈值 0.001/0.01/0.1；控制氨基酸类别后的方差；方差能量参与度。
每折每档固定前 16 条 CDS 额外记录：全部置零、同 AA 且同可见性范围内打乱、
高/低方差维度分别置零后的 masked CE 变化，以及重建梯度 99% 能量覆盖的维度数。

AU 和梯度参与度都不等价于独立有效信息维数。不会把“AU=256”作为训练成功的充分条件，
也不会把活跃维度比例等同于密码子修改率；最终仍要求 CE 与固定 loss 改善。

`summary.json` 包含 V1 验收结果、各档差值、五折最终设置和完整外折 V2 对照。
`fold_*/selection.json`、`final_candidates.json` 保存内部选择依据。
`evaluation_candidate.json` 与 `evaluation_V1.json` 含逐序列逐掩码配对统计和隐空间诊断。

工作区只删除本轮已完成试验的 optimizer 临时状态和未入选权重，原 15 次训练文件全部保留。
各折只保留入选的新 checkpoint；未入选试验保留配方、指标、权重 SHA，后续如需该权重可按同一配方重跑。
可用空间少于 5 GiB 时安全退出，并保留已有结果；未配置自动删除原训练权重或无人值守扩容。

## 监控

```bash
tail -n 50 -f /root/gpufree-data/vmae_v2_followup_20261022/logs/followup.log
cat /root/gpufree-data/vmae_v2_followup_20261022/results/queue_status.json
```

单元验证：`python -m unittest latent_decoder.test_v2_followup -v`。
包含固定掩码、固定验收公式、整同源组排除、增益起点不变、某一长度退化即失败，
以及服务器 CUDA 上的实际微调、checkpoint 中断恢复和完成后幂等检查。
