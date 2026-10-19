# Masked CE 收敛效率：原始文献与当前参数建议

查询日期：2026-10-18。范围是密码子级变分掩码训练；不将图像 MAE 的高遮挡比例直接迁移到本项目。只读查看新服务器指标，没有修改运行中的训练。

## 检索与来源边界

检索覆盖 variational masked autoencoder、VarMAE、masked language modeling 的 prediction/corruption rate，以及 LSCO 原文。采用作者论文/ACL Anthology/arXiv 原文作为证据；搜索结果中的转载、聚合页及自动摘要没有作为参数依据。这是针对实现问题的简要查证，不是穷尽式系统综述。

- [VarMAE，EMNLP Findings 2022](https://aclanthology.org/2022.findings-emnlp.468.pdf)：§2.3 区分 masked/unmasked 目标及 KL 权重；§3.3 的默认遮挡策略实验中 15% 最佳。其网络和损失归一化不同，论文的 λ 不能直接套作当前 beta_kl。
- [Should You Mask 15% in Masked Language Modeling?，EACL 2023](https://aclanthology.org/2023.eacl-main.217.pdf)：§3、§6 区分更多预测训练信号与更严重输入破坏；40% 在其 BERT-large 实验中有收益，但最优比例依赖模型、任务及训练方案。高比例不代表更低的原始重建 CE。§7 指出 unchanged-token 预测贡献较小。这些结果来自自然语言，迁移到 codon 模型需实验验证。
- [BERT，NAACL 2019](https://aclanthology.org/N19-1423.pdf)：§3.1 的 MLM 只预测选中位置，而非对整条输入做同权重重建。支持将计算预算分配给遮挡预测的方向，不意味着本 VMAE 应删去完整重建目标。
- [LSCO，arXiv:2610.03098v1](https://arxiv.org/html/2610.03098v1#AppB)：Appendix B 的 mRNA VMAE 用 15% mask、lr=1e-4、40 epochs、10% LR warm-up 后 cosine decay。它不是提高 mask 到 75% 的直接先例；规模、语料和训练方案与本项目不同，不能按 epoch 数直接比较收敛快慢。

## 当前代码与实测

固定配置为 clean_batch_probability=0.25、mask_probability=0.15、masked_loss_weight=1、full_loss_weight=1、beta_kl=0.001、lr=1e-4。

新服务器已完成的前四轮验证 masked CE：1.007193、1.000572、0.996918、0.992294；频率基线 1.144230。clean 可变位准确率均为 100%，活跃 latent 维度为 256、256、255、254。第四轮 masked-input KL/token=12.67426，对应固定 β×KL≈0.01267；这只是损失值尺度，不能代替梯度贡献测量。证据来自服务器 `runs/vmae_lr_0.0001_5ep/metrics.jsonl`。

masked_loss_weight 乘的是已按选中位置求均值的 CE，不是尚未归一化的 15% 位置总和。当前 masked CE 也已占训练总 loss 的大部分；因此没有依据把权重翻几倍等同于收敛加速几倍。它会改变各项相对权重及优化行为，仍需验证。

另外，日志中的 masked_ce 是窗口内所有 microbatch 的平均（含 clean batch 的零值）；调整 clean 概率后它会因统计组成而上升。应比较固定 val 上的 masked_mutable_ce，不能据窗口字段数值直接判断改动优劣。

## 最小、可解释的下一轮对照

**第一项只改 `training.clean_batch_probability: 0.25 -> 0.0`，其余维持当前设置。** 这是结合上述文献方向和本模型已掌握完整输入重建后的工程假设，不是论文证明的本模型最优值。

每轮有遮挡的 batch 从期望 75% 增至 100%，增加约 33.3%。按可变密码子计，预期遮挡训练点从 0.75×0.15=11.25% 增至 15%。不会增加同一次输入的破坏程度。保留 Full CE 和对 clean 重建、z 消融的验证，防止只关注 masked CE 而损害潜在生成接口。训练点增加比例不是已验证的收敛速度或用时收益。

第二项才单独比较 `mask_probability=0.15` 与 `0.20/0.25`，不要一开始同时改 KL、LR、层数和损失权重。新比例只是探索值；VarMAE/LSCO 的 15% 结果及大模型高比例的反例都说明没有通用最佳比例。

**公平验证所需代码注意点：** 当前 train.py 将 training.mask_probability 同时传给训练与 evaluate。因此改为 0.25 前，必须先把验证 mask 概率作为独立参数固定在 0.15。当前没有现成可生效的 evaluation_mask_probability 配置项，仅向 JSON 添加这个字段不会改变代码。主训练 epoch 验证及显式 --evaluate 路径都需适配；固定评估种子、划分及 β。

其他参数先不改：lr=1e-4、mask_weight=1、full_weight=1、β=0.001、latent=256。当前 latent 维度仍活跃，不能将此次 masked CE 缓慢下降直接判定为 KL 塌缩，也没有证据优先通过降低 β 解决。

候选配置片段（建议，不是已部署改动）：

```json
{
  "clean_batch_probability": 0.0,
  "mask_probability": 0.15,
  "learning_rate": 0.0001,
  "masked_loss_weight": 1.0,
  "full_loss_weight": 1.0,
  "beta_kl": 0.001
}
```

改配置须另开实验记录；不能用不同配置直接 --resume 旧 checkpoint，因为现有入口校验配置完全一致。是否改善，要按相同 optimizer-step/样本预算和固定验证 CE 判断。无保证能将 masked CE 降至零；数据规模、可用条件信息和原始序列的不确定性也会限制可预测性。
