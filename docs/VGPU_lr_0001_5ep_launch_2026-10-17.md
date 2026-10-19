# LR=0.0001，独立 5 轮实验

用户要求暂停当前实验，改用 lr=0.0001 重新试 5 epochs。新实验从同一 seed 初始化，不从旧权重续训；与原 0.001 配置逐项比较，仅 learning_rate 不同。

服务器：`ssh -p 31463 root@connect.nmb1.seetacloud.com`。口令不保存。

项目路径：`/root/autodl-tmp/ds_ribodecode_lr_sweep_5ep_2026-10-17`。

## 旧实验暂停状态

2026-10-17 23:21:06 +0800，对旧 sweep 控制进程 PID 1687 和当前 0.001 训练子进程 PID 5005 发送 SIGSTOP，均确认状态为 T (stopped)。0.01 和 0.1 各已完成 5 轮，0.001 暂停在第 2 轮。旧结果保留，运行中的内存状态也保留。

暂停记录：`runs/vmae_lr_sweep_5ep/pause_manifest.json`，旧 0.001 状态文件已标为 paused。SIGSTOP 不释放其约 12GB VRAM，服务器 32GB 实测仍可运行新实验。旧任务不会自动恢复；恢复时应核对 PID/命令，并在新任务结束后处理。实例重启会丢失尚未落盘的内存状态，可从旧 last.pt 恢复到最近完整 epoch。

后续用户明确要求删除旧进程：已核验命令及项目目录后，以 SIGKILL 终止 PID 1687 和 5005，旧 0.001 状态改为 terminated。旧任务内存状态已释放，只能从最近完整 epoch 的 last.pt 重新启动；旧日志、配置和权重均保留。新 PID 5464 未受影响，确认仍运行至 step 190，GPU 总显存占用降至 11728 MiB，剩余 20504 MiB。

暂停前 0.001 首轮已有改善：masked CE=1.076929，频率基线=1.144230，相对改善约 5.88%，active latent dimensions=253。其完整第一轮权重保存在原目录。

## 新实验

- 配置：`latent_decoder/lr_sweep_configs/lr_0.0001.json`。
- 固定 lr=0.0001，5 epochs，无 early stopping，无 LR warm-up/cosine；KL warm-up、β、模型、seed、batch/累积和数据与原组相同。
- 训练命令：`python -u -m latent_decoder.train --data dataset --allow-development-split --config latent_decoder/lr_sweep_configs/lr_0.0001.json --output runs/vmae_lr_0.0001_5ep`。
- 后台启动 PID：5464；PID 记录：`lr_0.0001_5ep.pid`。
- 总日志：`lr_0.0001_5ep.log`。
- 指标与权重：`runs/vmae_lr_0.0001_5ep/{steps.jsonl,metrics.jsonl,last.pt,best.pt,config.json}`，完整 epoch 结束后写 metrics/checkpoint。

截至 23:22:48，最长 batch 4、1500 codons FP32 前后向预检通过，已确认 step 20 的实际参数更新，lr=0.0001。窗口训练 loss 从 step 1 的 2.26610 降到 step 10 的 1.64524、step 20 的 1.15071；该比较含不同 batch/遮挡，不代表固定样本或最终验证结果。旧进程仍为 stopped，新进程运行中，GPU 总占用约 24261 MiB（含暂停任务），未出现 OOM。

查看新实验：

```bash
cd /root/autodl-tmp/ds_ribodecode_lr_sweep_5ep_2026-10-17
tail -f lr_0.0001_5ep.log
```

本记录为启动确认，尚未完成 5 轮。原三组汇总脚本仍针对原三组，新实验指标位于上述独立目录，不混入原 sweep。
