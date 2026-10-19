# LSCO 学习率方案，重新训练 20 epochs

用户于 2026-10-18 要求采用 LSCO 学习率并重新训练 20 轮。本次只对齐学习率策略，仍使用按既定文档构建的 DS-RiboDecode VMAE；不意味着严格复现 LSCO 的模型、数据或所有训练细节。

依据：[LSCO Appendix B，PDF 第 10 页](https://arxiv.org/pdf/2610.03098#page=10)：AdamW，初始/峰值 lr=1e-4，前 10% 训练更新步线性 warm-up，随后 cosine decay。本次轮数按用户指定设为 20，而非原文的 40。

## 配置与验证

- 配置：`latent_decoder/config_lsco_lr_20ep.json`。
- 从同一 seed=20261016 重新初始化，不使用旧权重续训。
- 相对前一次全 batch 遮挡的 5 轮配置，仅改 `epochs: 5 -> 20`、`lr_schedule: constant -> warmup_cosine`、`warmup_fraction: 0 -> 0.1`；learning_rate 本身仍为 0.0001。
- 模型仍为 Encoder / Decoder 各 6 层，hidden 512，heads 8，FFN 2048，latent 256，RoPE，AA 条件；FP32、batch 4、累积 8，关闭早停。
- clean_batch_probability=0.0，mask_probability=0.15，Full/Masked CE 权重均为 1，beta_kl=0.001。
- `kl_warmup_fraction=0.2` 保持不变；由于总训练时长增加，KL warm-up 对应前 4464 次更新（4 轮），不是原先 5 轮实验中的 1 轮。该项是本项目规则，不是原文披露的 LSCO KL 调度。
- 固定联合数据 train/val/test=35710/3949/4342，验证为固定种子的 15% mask；test 不参与训练或模型选择。

按当前数据和 batch 计算，每轮 1116 次更新，共 22320 次更新。学习率 warm-up 为 2232 次更新（约 2 轮），峰值为 1e-4。当前实现首个更新 lr≈4.48029e-8；最后一次更新所用 lr 接近零，之后 scheduler 状态到零。

已直接取 train.py 中实际 schedule 函数做数值检查：warm-up 单调递增、峰值、cosine 单调递减及末端零值均通过。远端确认除上述三项外配置完全一致；启动脚本通过 bash 语法检查。

## 服务器和启动

连接：`ssh -p 32412 root@connect.westc.seetacloud.com`。口令不保存。

项目目录：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18`。

原全 batch 遮挡实验已经完整完成 5 轮，最终验证 masked CE=0.989037，旧进程已退出。原权重和日志保留在原目录。

2026-10-18 18:29 +0800，新任务后台启动 PID=14366，记录于 `vmae_lsco_lr_20ep.pid`。

启动命令：`nohup bash latent_decoder/run_server_lsco_lr_20ep.sh > vmae_lsco_lr_20ep.log 2>&1 < /dev/null &`。

日志完整路径：`/root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18/vmae_lsco_lr_20ep.log`。

结果目录：项目内 `runs/vmae_lsco_lr_20ep_20261018/`，含 config.json、steps.jsonl、每轮 metrics.jsonl、last.pt 和验证最佳 best.pt。

```bash
cd /root/autodl-tmp/ds_ribodecode_lr_0001_5ep_2026-10-18
tail -f vmae_lsco_lr_20ep.log
```

已有输出目录拒绝重新开训；需要恢复时使用新 run 自身的 last.pt 和配置。记录是启动凭据，不表示 20 轮已经完成，也不保证调度一定比恒定学习率有更好的验证结果。

启动核验：2026-10-18 18:30:48 +0800，最长 1500 codons / batch 4 GPU 前后向预检通过，已实际更新到 step 50。日志 lr 从首个更新 4.48029e-8 升至 step 50 的 2.24014e-6，符合 warm-up；梯度有限非零，clean_batches=0。

21:07 +0800 状态：完成 12/20 轮，第 13 轮 step 14030、batch 5104/8928，进程 14366 正常运行，显存约 15300 MiB。当前 lr≈3.64634e-5，已进入 cosine 衰减。最近完整轮（12）验证 masked CE=0.981318，较条件频率基线 1.144230 改善 14.24%，为已完成轮中的最低值；首轮为 1.060992。原始快照保存在 `lsco_lr_20ep_metrics_snapshot_2026-10-18.jsonl`，图在 `lsco_lr_20ep_progress_2026-10-18_2107.png` / `.svg`。

潜变量状态：活跃维度从 256 降为 86，按当前 μ 方差阈值统计；但最近 clean CE=0.001291，zero-z CE=2.133454，shuffle-z CE=5.246591，说明重建仍强烈依赖 z，不能仅凭活跃维度下降判为塌缩。clean 重建准确率和整条 exact rate 均为 100%，蛋白错误率为零。仍需继续监测后续轮次及独立生成/集成验收。

前 12 轮平均约 749 秒/轮，按相近负载估计剩余约 1.5～1.7 小时。该估计不是完成保证。

22:18 +0800 更新：完成 18/20 轮，第 19 轮 step 20280、batch 1536/8928（约 17%），PID 14366 运行正常；lr≈2.52558e-6。第 18 轮验证 masked CE=0.976923，masked accuracy=55.7064%，较频率基线改善 14.62%。截至当时最低 masked CE 为第 17 轮的 0.976870，最近两轮接近平台；该最低 masked CE 轮次不直接等于按 clean+masked CE 选择的 best.pt 轮次。

第 18 轮 clean CE=0.001040，zero-z CE=2.925721，shuffle-z CE=5.654459，重建仍依赖 z；活跃维度降到 28（方差阈值 0.01）。clean accuracy 和 exact sequence rate 均为 100%，蛋白错误率为零。最近 checkpoint 已保存；原始指标快照为 `lsco_lr_20ep_metrics_snapshot_2026-10-18_2218.jsonl`。按前 18 轮平均约 753 秒/轮，剩余预计约 23～25 分钟，尚未完成训练。
